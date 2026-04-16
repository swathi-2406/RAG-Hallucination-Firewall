"""
src/hallucination/stage3_nli.py
Stage 3 of the Hallucination Firewall: NLI Cross-Check via DeBERTa.

Approach:
  1. For each retrieved chunk, run NLI with (premise=chunk, hypothesis=answer)
  2. Collect contradiction probabilities across all chunks
  3. If the maximum contradiction probability exceeds the threshold → flag

Using `cross-encoder/nli-deberta-v3-small`:
  - ~85MB download, runs fully locally (CPU)
  - Labels: CONTRADICTION, ENTAILMENT, NEUTRAL
  - Fine-tuned on MNLI + SNLI, strong zero-shot NLI performance

Why cross-encoder NLI?
  A standard LLM might generate a fluent, confident-sounding answer that
  directly contradicts what the retrieved evidence says. NLI catches this
  at the semantic entailment level, not just token overlap.
"""

import logging
import time
from typing import List

import numpy as np
from transformers import pipeline

from config.settings import NLI_MODEL, NLI_THRESHOLD

logger = logging.getLogger(__name__)

# Module-level singleton
_nli_pipeline = None


def get_nli_pipeline():
    """Load the DeBERTa NLI pipeline once and cache it."""
    global _nli_pipeline
    if _nli_pipeline is None:
        logger.info(f"Loading NLI model: {NLI_MODEL} (first run downloads ~85MB)")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,  # CPU
        )
        logger.info("NLI model loaded.")
    return _nli_pipeline


def _get_contradiction_prob(chunk_text: str, answer: str) -> float:
    """
    Run NLI for a single (premise, hypothesis) pair.

    premise:    a retrieved context chunk
    hypothesis: the LLM's answer

    Returns the probability assigned to CONTRADICTION.
    """
    nli = get_nli_pipeline()

    # zero-shot-classification format: classify answer given context as premise
    result = nli(
        answer,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="Based on the context: {}",
    )
    # Result is a dict with 'labels' and 'scores' aligned
    label_to_score = dict(zip(result["labels"], result["scores"]))
    return label_to_score.get("contradiction", 0.0)


def run_stage3(chunks_text: List[str], answer: str) -> dict:
    """
    Execute Stage 3: NLI Cross-Check across all retrieved chunks.

    For each chunk, we check if the answer *contradicts* the chunk.
    We take the max contradiction score across chunks (conservative approach).

    Returns a result dict with:
      - score:            max contradiction probability (0–1)
      - flagged:          bool
      - per_chunk_scores: list of contradiction probs per chunk
      - latency_ms
    """
    start = time.perf_counter()
    per_chunk_scores = []

    for i, chunk in enumerate(chunks_text):
        try:
            prob = _get_contradiction_prob(chunk, answer)
            per_chunk_scores.append(prob)
            logger.debug(f"[Stage 3] Chunk {i+1}: contradiction_prob={prob:.3f}")
        except Exception as e:
            logger.warning(f"[Stage 3] NLI failed for chunk {i+1}: {e}")
            per_chunk_scores.append(0.0)

    # Use max contradiction as the conservative score
    score = float(np.max(per_chunk_scores)) if per_chunk_scores else 0.0
    flagged = score > NLI_THRESHOLD
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"[Stage 3] NLI contradiction_max={score:.3f} | "
        f"Flagged={flagged} | "
        f"Threshold={NLI_THRESHOLD} | "
        f"Latency={latency_ms:.0f}ms"
    )

    return {
        "score": score,
        "flagged": flagged,
        "per_chunk_scores": per_chunk_scores,
        "threshold": NLI_THRESHOLD,
        "latency_ms": latency_ms,
    }
