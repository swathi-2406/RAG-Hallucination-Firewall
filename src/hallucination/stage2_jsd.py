"""
src/hallucination/stage2_jsd.py
Stage 2 of the Hallucination Firewall: Token-Level Jensen-Shannon Divergence.

Approach:
  1. Build a token frequency distribution from the retrieved context
  2. Build a token frequency distribution from the LLM's answer
  3. Compute Jensen-Shannon divergence between the two distributions
  4. High JSD = answer uses tokens not grounded in context = hallucination risk

JSD is a symmetric, bounded (0–1 with log base 2) divergence measure
derived from KL divergence. It's ideal for comparing sparse distributions.
"""

import logging
import re
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from config.settings import JSD_THRESHOLD

logger = logging.getLogger(__name__)

# Common English stopwords to exclude from token distributions
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "this", "that", "these", "those", "it", "its", "as",
    "by", "from", "not", "so", "if", "then", "than", "also", "which", "who",
    "what", "when", "where", "how", "i", "we", "you", "he", "she", "they",
}


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased, stopwords removed."""
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_distribution(tokens: List[str], vocab: List[str]) -> np.ndarray:
    """
    Build a normalized frequency distribution over a shared vocabulary.

    Args:
        tokens: list of tokens from a document
        vocab:  shared vocabulary (union of context + answer tokens)

    Returns:
        Probability vector (sums to 1) aligned to vocab order.
    """
    counts = Counter(tokens)
    total = sum(counts[w] for w in vocab) or 1
    dist = np.array([counts[w] / total for w in vocab], dtype=np.float64)

    # Laplace smoothing to avoid zero probabilities
    dist += 1e-9
    dist /= dist.sum()
    return dist


def compute_jsd(context: str, answer: str) -> Tuple[float, dict]:
    """
    Compute Jensen-Shannon Divergence between context and answer token distributions.

    Returns:
        (jsd_score, debug_info)
        jsd_score: float in [0, 1] (0 = identical distributions)
    """
    context_tokens = tokenize(context)
    answer_tokens = tokenize(answer)

    # Build shared vocabulary
    vocab = sorted(set(context_tokens) | set(answer_tokens))
    if not vocab:
        return 0.0, {"context_tokens": 0, "answer_tokens": 0, "vocab_size": 0}

    p = build_distribution(context_tokens, vocab)
    q = build_distribution(answer_tokens, vocab)

    # scipy's jensenshannon returns sqrt(JSD), so we square it for the raw JSD
    jsd_sqrt = float(jensenshannon(p, q, base=2))
    jsd_score = jsd_sqrt ** 2  # True JSD in [0, 1]

    debug = {
        "context_tokens": len(context_tokens),
        "answer_tokens": len(answer_tokens),
        "vocab_size": len(vocab),
        "top_context_tokens": Counter(context_tokens).most_common(5),
        "top_answer_tokens": Counter(answer_tokens).most_common(5),
    }

    return jsd_score, debug


def run_stage2(context: str, answer: str) -> dict:
    """
    Execute Stage 2: Jensen-Shannon Divergence.

    Returns a result dict with:
      - score:    JSD value (0–1)
      - flagged:  bool (True if score > JSD_THRESHOLD)
      - debug:    token distribution details
      - latency_ms
    """
    start = time.perf_counter()
    score, debug = compute_jsd(context, answer)
    latency_ms = (time.perf_counter() - start) * 1000

    flagged = score > JSD_THRESHOLD

    logger.info(
        f"[Stage 2] JSD={score:.3f} | "
        f"Flagged={flagged} | "
        f"Threshold={JSD_THRESHOLD} | "
        f"Latency={latency_ms:.1f}ms"
    )

    return {
        "score": score,
        "flagged": flagged,
        "debug": debug,
        "threshold": JSD_THRESHOLD,
        "latency_ms": latency_ms,
    }
