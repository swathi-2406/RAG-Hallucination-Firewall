"""
src/hallucination/stage1_entropy.py
Stage 1 of the Hallucination Firewall: Semantic Entropy Scoring.

Approach:
  1. Sample N outputs from the LLM at temperature > 0
  2. Embed each output using the same sentence-transformer
  3. Compute the mean pairwise cosine distance across all samples
  4. High distance = high semantic variance = the model is uncertain = hallucination risk

This is inspired by the "Semantic Entropy" paper (Farquhar et al., 2023).
"""

import logging
import time
from typing import List, Tuple

import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, ENTROPY_SAMPLES,
    ENTROPY_TEMP, ENTROPY_THRESHOLD,
)
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the question using ONLY "
    "the provided context. Be concise and factual."
)


def _build_prompt(query: str, context: str) -> List:
    """Build LangChain message list for a RAG query."""
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the context above:"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def sample_outputs(
    query: str,
    context: str,
    n: int = ENTROPY_SAMPLES,
    temperature: float = ENTROPY_TEMP,
) -> Tuple[List[str], float]:
    """
    Sample N stochastic LLM outputs for a given query+context.

    Returns:
        (list_of_outputs, latency_ms)
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=256,
    )

    messages = _build_prompt(query, context)
    outputs = []
    start = time.perf_counter()

    for i in range(n):
        try:
            response = llm.invoke(messages)
            outputs.append(response.content.strip())
        except Exception as e:
            logger.warning(f"Entropy sample {i} failed: {e}")
            outputs.append("")

    latency_ms = (time.perf_counter() - start) * 1000
    return [o for o in outputs if o], latency_ms


def compute_semantic_entropy(outputs: List[str]) -> float:
    """
    Compute semantic entropy as mean pairwise cosine distance.

    Returns a score in [0, 1]:
      - Near 0: all outputs are semantically identical → low uncertainty
      - Near 1: outputs are semantically diverse → high uncertainty / hallucination risk
    """
    if len(outputs) < 2:
        return 0.0

    embedder = get_embedder()
    embeddings = np.array(embedder.embed_documents(outputs))

    # Embeddings are L2-normalized (normalize_embeddings=True in embedder.py)
    # So dot product == cosine similarity
    n = len(embeddings)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = float(np.dot(embeddings[i], embeddings[j]))
            distances.append(1.0 - cos_sim)  # cosine distance

    return float(np.mean(distances))


def run_stage1(
    query: str,
    context: str,
    deterministic_answer: str,
) -> dict:
    """
    Execute Stage 1: Semantic Entropy.

    Returns a result dict with:
      - score:     entropy value (0–1)
      - flagged:   bool (True if score > ENTROPY_THRESHOLD)
      - samples:   the N sampled outputs
      - latency_ms
    """
    samples, latency_ms = sample_outputs(query, context)

    # Include the deterministic answer as one of the data points
    all_outputs = [deterministic_answer] + samples

    score = compute_semantic_entropy(all_outputs)
    flagged = score > ENTROPY_THRESHOLD

    logger.info(
        f"[Stage 1] Entropy={score:.3f} | "
        f"Flagged={flagged} | "
        f"Threshold={ENTROPY_THRESHOLD} | "
        f"Latency={latency_ms:.0f}ms"
    )

    return {
        "score": score,
        "flagged": flagged,
        "samples": samples,
        "threshold": ENTROPY_THRESHOLD,
        "latency_ms": latency_ms,
    }
