"""
src/retrieval/retriever.py
Builds and manages the FAISS vector store. Exposes an MMR-based
retriever that reduces redundancy in retrieved chunks.
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
    """
    Build a FAISS index from document chunks and optionally save to disk.

    Uses IndexFlatIP (inner product) with normalized embeddings,
    which is equivalent to cosine similarity — fast and accurate for dense retrieval.
    """
    if not chunks:
        raise ValueError("No chunks provided to build_index. Run ingest_docs.py first.")

    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    embedder = get_embedder()

    start = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embedder)
    elapsed = (time.perf_counter() - start) * 1000

    logger.info(f"Index built in {elapsed:.1f}ms")

    if save:
        vectorstore.save_local(INDEX_PATH)
        logger.info(f"Index saved to {INDEX_PATH}")

    return vectorstore


def load_index() -> FAISS:
    """Load a pre-built FAISS index from disk."""
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. "
            "Run: python scripts/ingest_docs.py"
        )

    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embedder,
        allow_dangerous_deserialization=True,  # Safe: we created this index ourselves
    )
    logger.info(f"FAISS index loaded. Total vectors: {vectorstore.index.ntotal}")
    return vectorstore


def retrieve(
    query: str,
    vectorstore: FAISS,
    top_k: int = TOP_K,
    diversity: float = MMR_DIVERSITY,
) -> Tuple[List[Document], float]:
    """
    Retrieve top-k relevant chunks using Maximal Marginal Relevance (MMR).

    MMR balances relevance and diversity:
        MMR = argmax [ λ·sim(q, d) - (1-λ)·max_{r∈R} sim(r, d) ]

    Args:
        query:       User's question
        vectorstore: FAISS index
        top_k:       Number of chunks to return
        diversity:   MMR lambda (0=max diversity, 1=max relevance)

    Returns:
        (chunks, latency_ms)
    """
    start = time.perf_counter()

    chunks = vectorstore.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=top_k * 4,   # Fetch 4x more candidates for MMR reranking
        lambda_mult=diversity,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.debug(f"Retrieval: {len(chunks)} chunks in {latency_ms:.1f}ms")

    return chunks, latency_ms


def get_context_string(chunks: List[Document]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[Chunk {i} | Source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)
