"""
src/retrieval/embedder.py
Wraps sentence-transformers for local, free embedding generation.
No API key or internet connection required after first model download.
"""

import logging
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — model is loaded once and reused
_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns the singleton HuggingFaceEmbeddings instance.
    Downloads the model on first call (~90MB, cached locally after that).

    Model: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Very fast (CPU-friendly)
    - Strong semantic similarity performance for retrieval tasks
    """
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,  # Cosine similarity via dot product
                "batch_size": 64,
            },
        )
        logger.info("Embedding model loaded successfully.")
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of text strings. Returns list of embedding vectors."""
    embedder = get_embedder()
    return embedder.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    embedder = get_embedder()
    return embedder.embed_query(query)
