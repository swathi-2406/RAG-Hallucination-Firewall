"""
src/retrieval/retriever.py
FAISS index + MMR retrieval.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import INDEX_DIR, TOP_K, MMR_DIVERSITY
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)
INDEX_PATH = str(INDEX_DIR / "faiss_store")


def build_index(chunks: List[Document], save: bool = True) -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided.")
    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    embedder = get_embedder()
    start = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embedder)
    logger.info(f"Index built in {(time.perf_counter()-start)*1000:.1f}ms")
    if save:
        vectorstore.save_local(INDEX_PATH)
        logger.info(f"Index saved to {INDEX_PATH}")
    return vectorstore


def load_index() -> FAISS:
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"No FAISS index at {INDEX_PATH}. Run: python scripts/ingest_docs.py"
        )
    embedder = get_embedder()
    vs = FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    logger.info(f"FAISS index loaded. Vectors: {vs.index.ntotal}")
    return vs


def retrieve(
    query: str,
    vectorstore: FAISS,
    top_k: int = TOP_K,
    diversity: float = MMR_DIVERSITY,
) -> Tuple[List[Document], float]:
    start = time.perf_counter()
    chunks = vectorstore.max_marginal_relevance_search(
        query, k=top_k, fetch_k=top_k * 4, lambda_mult=diversity,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return chunks, latency_ms


def get_context_string(chunks: List[Document]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[Chunk {i} | Source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)
