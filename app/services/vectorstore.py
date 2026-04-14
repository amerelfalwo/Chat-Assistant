from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from functools import lru_cache
from threading import Lock
import time

_index_initialized = False
_index_init_lock = Lock()

@lru_cache(maxsize=1)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )

def init_pinecone_index():
    global _index_initialized
    if _index_initialized:
        return

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
            existing_indexes = {
                idx["name"] if isinstance(idx, dict) else getattr(idx, "name", None)
                for idx in listed_indexes
            }
            existing_indexes.discard(None)
        
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=768, # assuming gemini-embedding-001 outputs 768 dims
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
