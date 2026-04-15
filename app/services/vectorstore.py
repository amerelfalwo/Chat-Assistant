from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
import time
import logging

logger = logging.getLogger(__name__)

# ── Singleton: load the ONNX model once, reuse everywhere ───────────────
_embeddings_instance = None

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX-optimized
EMBED_DIM = 384


def get_embeddings():
    """Return a cached FastEmbed instance (lightweight ONNX, no PyTorch)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading FastEmbed model: {EMBED_MODEL}")
        _embeddings_instance = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return _embeddings_instance


def init_pinecone_index():
    """
    Called once on server startup from main.py.
    Creates the Pinecone index if it doesn't exist,
    or recreates it if the dimension doesn't match.
    """
    logger.info("Checking Pinecone index status...")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = {i["name"]: i for i in pc.list_indexes()}

    if settings.PINECONE_INDEX_NAME in existing_indexes:
        # Verify dimension matches the current embedding model
        index_info = pc.describe_index(settings.PINECONE_INDEX_NAME)
        current_dim = index_info.dimension
        if current_dim != EMBED_DIM:
            logger.warning(
                f"Index dimension mismatch: index={current_dim}, model={EMBED_DIM}. "
                f"Deleting and recreating index..."
            )
            pc.delete_index(settings.PINECONE_INDEX_NAME)
            _create_index(pc)
        else:
            logger.info("Pinecone index exists with correct dimension.")
    else:
        _create_index(pc)


def _create_index(pc: Pinecone):
    """Helper to create a new Pinecone index with the correct dimension."""
    logger.info(f"Creating new Pinecone index: {settings.PINECONE_INDEX_NAME} (dim={EMBED_DIM})")
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENV),
    )
    while not pc.describe_index(settings.PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
    logger.info("Pinecone index is ready.")


def get_vectorstore(namespace: str = None):
    """Return a PineconeVectorStore connection (fast, no index check)."""
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=settings.PINECONE_API_KEY,
        namespace=namespace,
    )