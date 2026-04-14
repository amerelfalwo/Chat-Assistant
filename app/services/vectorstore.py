from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from functools import lru_cache
from threading import Lock
import logging
import time

logger = logging.getLogger(__name__)
_index_initialized = False
_index_init_lock = Lock()
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSION = 768

@lru_cache(maxsize=1)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY
    )

def init_pinecone_index():
    global _index_initialized

    with _index_init_lock:
        if _index_initialized:
            return

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        listed_indexes = pc.list_indexes()
        names_attr = getattr(listed_indexes, "names", None)
        if callable(names_attr):
            existing_indexes = set(names_attr())
        elif names_attr is not None:
            existing_indexes = set(names_attr)
        else:
            existing_indexes = set()
            for idx in listed_indexes:
                if isinstance(idx, dict) and idx.get("name"):
                    existing_indexes.add(idx["name"])
                elif getattr(idx, "name", None):
                    existing_indexes.add(idx.name)
                else:
                    logger.warning("Unexpected Pinecone index entry format: %r", idx)
        
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENV)
            )
            while not pc.describe_index(settings.PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)
        _index_initialized = True

def get_vectorstore(namespace: str = None):
    # Make sure index exists (optional, could be done once on startup)
    init_pinecone_index()
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=settings.PINECONE_API_KEY,
        namespace=namespace
    )
