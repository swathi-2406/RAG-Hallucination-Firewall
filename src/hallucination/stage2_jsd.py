"""
src/hallucination/stage2_jsd.py
Stage 2: Jensen-Shannon Divergence — token-level vocabulary comparison.
Pure Python/numpy, no LLM needed.
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

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","can","this","that","these",
    "those","it","its","as","by","from","not","so","if","then","than","also",
    "which","who","what","when","where","how","i","we","you","he","she","they",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_distribution(tokens: List[str], vocab: List[str]) -> np.ndarray:
    counts = Counter(tokens)
    total = sum(counts[w] for w in vocab) or 1
    dist = np.array([counts[w] / total for w in vocab], dtype=np.float64)
    dist += 1e-9
    dist /= dist.sum()
    return dist


def compute_jsd(context: str, answer: str) -> Tuple[float, dict]:
    context_tokens = tokenize(context)
    answer_tokens = tokenize(answer)
    vocab = sorted(set(context_tokens) | set(answer_tokens))
    if not vocab:
        return 0.0, {"context_tokens": 0, "answer_tokens": 0, "vocab_size": 0}
    p = build_distribution(context_tokens, vocab)
    q = build_distribution(answer_tokens, vocab)
    jsd_score = float(jensenshannon(p, q, base=2)) ** 2
    return jsd_score, {
        "context_tokens": len(context_tokens),
        "answer_tokens": len(answer_tokens),
        "vocab_size": len(vocab),
        "top_context_tokens": Counter(context_tokens).most_common(5),
        "top_answer_tokens": Counter(answer_tokens).most_common(5),
    }


def run_stage2(context: str, answer: str) -> dict:
    start = time.perf_counter()
    score, debug = compute_jsd(context, answer)
    latency_ms = (time.perf_counter() - start) * 1000
    flagged = score > JSD_THRESHOLD
    logger.info(f"[Stage 2] JSD={score:.3f} Flagged={flagged} {latency_ms:.1f}ms")
    return {
        "score": score,
        "flagged": flagged,
        "debug": debug,
        "threshold": JSD_THRESHOLD,
        "latency_ms": latency_ms,
    }
