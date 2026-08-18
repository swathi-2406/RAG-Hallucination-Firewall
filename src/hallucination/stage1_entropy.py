"""
src/hallucination/stage1_entropy.py
Stage 1: Semantic Entropy — samples N outputs, measures mean pairwise cosine distance.
Uses DeepSeek V4 Flash via unified LLM client.
"""

import logging
import time
from typing import List, Tuple

import numpy as np

from config.settings import ENTROPY_SAMPLES, ENTROPY_TEMP, ENTROPY_THRESHOLD
from src.llm_client import call_llm, build_rag_messages
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)


def sample_outputs(
    query: str,
    context: str,
    n: int = ENTROPY_SAMPLES, prefer_eval: bool = False,
    temperature: float = ENTROPY_TEMP,
) -> Tuple[List[str], float, str]:
    """
    Sample N stochastic LLM outputs.
    Returns (outputs, latency_ms, backend_used)
    """
    messages = build_rag_messages(query, context)
    outputs = []
    start = time.perf_counter()
    backend_used = "unknown"

    for i in range(n):
        try:
            text, backend, _ = call_llm(messages, temperature=temperature, max_tokens=256)
            outputs.append(text)
            backend_used = backend
        except Exception as e:
            logger.warning(f"Entropy sample {i} failed: {e}")
            outputs.append("")

    latency_ms = (time.perf_counter() - start) * 1000
    return [o for o in outputs if o], latency_ms, backend_used


def compute_semantic_entropy(outputs: List[str]) -> float:
    if len(outputs) < 2:
        return 0.0
    embedder = get_embedder()
    embeddings = np.array(embedder.embed_documents(outputs))
    n = len(embeddings)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(1.0 - float(np.dot(embeddings[i], embeddings[j])))
    return float(np.mean(distances))


def run_stage1(query: str, context: str, deterministic_answer: str, prefer_eval: bool = False) -> dict:
    samples, latency_ms, backend = sample_outputs(query, context)
    all_outputs = [deterministic_answer] + samples
    score = compute_semantic_entropy(all_outputs)
    flagged = score > ENTROPY_THRESHOLD

    logger.info(f"[Stage 1 | {backend}] Entropy={score:.3f} Flagged={flagged} {latency_ms:.0f}ms")
    return {
        "score": score,
        "flagged": flagged,
        "samples": samples,
        "threshold": ENTROPY_THRESHOLD,
        "latency_ms": latency_ms,
        "backend": backend,
    }
