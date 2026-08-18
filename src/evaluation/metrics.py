"""
src/evaluation/metrics.py
RAGAS-style metrics via local embedding cosine similarity.
"""

import logging
from typing import List
import numpy as np
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)


def _cos(a, b):
    return float(np.dot(a, b))


def context_precision(query: str, chunks: List[str], top_k: int = 5) -> float:
    if not chunks: return 0.0
    emb = get_embedder()
    q = np.array(emb.embed_query(query))
    cs = np.array(emb.embed_documents(chunks[:top_k]))
    return round(float(np.mean([_cos(q, c) for c in cs])), 4)


def answer_faithfulness(answer: str, chunks: List[str]) -> float:
    if not chunks or not answer.strip(): return 0.0
    emb = get_embedder()
    a = np.array(emb.embed_query(answer))
    cs = np.array(emb.embed_documents(chunks))
    return round(float(np.max([_cos(a, c) for c in cs])), 4)


def answer_relevancy(query: str, answer: str) -> float:
    if not answer.strip(): return 0.0
    emb = get_embedder()
    q = np.array(emb.embed_query(query))
    a = np.array(emb.embed_query(answer))
    return round(_cos(q, a), 4)


def compute_all_metrics(query: str, answer: str, chunks: List[str]) -> dict:
    return {
        "context_precision":    context_precision(query, chunks),
        "answer_faithfulness":  answer_faithfulness(answer, chunks),
        "answer_relevancy":     answer_relevancy(query, answer),
    }
