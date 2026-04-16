"""
src/evaluation/metrics.py
Custom GenAI evaluation metrics inspired by RAGAS.

Metrics implemented:
  1. Context Precision  — Are the retrieved chunks relevant to the query?
  2. Answer Faithfulness — Does the answer stay grounded in the context?
  3. Answer Relevancy   — Does the answer address the question?

These are computed locally (no external API needed) using cosine similarity
of sentence embeddings as a proxy for semantic relevance.
"""

import logging
from typing import List

import numpy as np

from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit-normalized vectors."""
    return float(np.dot(a, b))


def context_precision(query: str, chunks: List[str], top_k: int = 5) -> float:
    """
    Context Precision: measures how relevant the retrieved chunks are to the query.

    Computed as: mean cosine similarity between the query embedding
    and each chunk embedding.

    Range: [0, 1] where 1 = perfect relevance.
    """
    if not chunks:
        return 0.0

    embedder = get_embedder()
    query_emb = np.array(embedder.embed_query(query))
    chunk_embs = np.array(embedder.embed_documents(chunks[:top_k]))

    similarities = [_cosine_similarity(query_emb, c) for c in chunk_embs]
    score = float(np.mean(similarities))

    logger.debug(f"Context Precision: {score:.3f}")
    return round(score, 4)


def answer_faithfulness(answer: str, chunks: List[str]) -> float:
    """
    Answer Faithfulness: measures how well the answer is grounded in context.

    Computed as: max cosine similarity between the answer embedding
    and any context chunk embedding.

    High score = answer content closely matches something in the context.
    Low score  = answer may be fabricated or out-of-distribution.

    Range: [0, 1].
    """
    if not chunks or not answer.strip():
        return 0.0

    embedder = get_embedder()
    answer_emb = np.array(embedder.embed_query(answer))
    chunk_embs = np.array(embedder.embed_documents(chunks))

    similarities = [_cosine_similarity(answer_emb, c) for c in chunk_embs]
    score = float(np.max(similarities))  # Best matching chunk

    logger.debug(f"Answer Faithfulness: {score:.3f}")
    return round(score, 4)


def answer_relevancy(query: str, answer: str) -> float:
    """
    Answer Relevancy: does the answer actually address the question?

    Computed as cosine similarity between query and answer embeddings.

    Range: [0, 1].
    """
    if not answer.strip():
        return 0.0

    embedder = get_embedder()
    query_emb = np.array(embedder.embed_query(query))
    answer_emb = np.array(embedder.embed_query(answer))

    score = _cosine_similarity(query_emb, answer_emb)
    logger.debug(f"Answer Relevancy: {score:.3f}")
    return round(score, 4)


def compute_all_metrics(
    query: str,
    answer: str,
    chunks: List[str],
) -> dict:
    """Compute all evaluation metrics and return as a dict."""
    return {
        "context_precision": context_precision(query, chunks),
        "answer_faithfulness": answer_faithfulness(answer, chunks),
        "answer_relevancy": answer_relevancy(query, answer),
    }
