"""
src/retrieval/embedder.py
Local sentence-transformer embeddings. No API key needed.
"""

import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        logger.info("Embedding model loaded.")
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed_documents(texts)


def embed_query(query: str) -> List[float]:
    return get_embedder().embed_query(query)
