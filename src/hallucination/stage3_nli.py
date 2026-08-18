"""
src/hallucination/stage3_nli.py
Stage 3: DeBERTa NLI cross-check — runs locally, no API needed.
"""

import logging
import time
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import NLI_MODEL, NLI_THRESHOLD

logger = logging.getLogger(__name__)
_nli_tokenizer = None
_nli_model = None
_contradiction_idx = None


def get_nli_model():
    global _nli_tokenizer, _nli_model, _contradiction_idx
    if _nli_model is None:
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model.eval()
        # Read the contradiction label's index from the model's own config
        # rather than assuming a fixed order -- cross-encoder NLI checkpoints
        # aren't all labeled [contradiction, entailment, neutral] in the same
        # order, so trust id2label instead of a hardcoded index.
        id2label = {i: str(l).lower() for i, l in _nli_model.config.id2label.items()}
        match = [i for i, l in id2label.items() if "contra" in l]
        if not match:
            raise ValueError(f"No 'contradiction' label found in {NLI_MODEL} id2label: {id2label}")
        _contradiction_idx = match[0]
        logger.info(f"NLI model loaded. id2label={id2label}, contradiction_idx={_contradiction_idx}")
    return _nli_tokenizer, _nli_model, _contradiction_idx


def _get_contradiction_prob(chunk_text: str, answer: str) -> float:
    """
    Proper premise/hypothesis NLI: premise = the retrieved context chunk,
    hypothesis = the generated answer. A high score means the answer states
    something that directly conflicts with what THIS chunk says -- the
    actual hallucination signal this stage exists to catch.

    Bug this replaces: the previous version ran the zero-shot-classification
    pipeline on `answer` alone, with candidate_labels=["entailment","neutral",
    "contradiction"] plugged into a hypothesis template -- `chunk_text` was
    accepted as a parameter but never referenced in the function body, so
    the context was never compared against at all. The model was asked to
    zero-shot classify the answer against the literal strings "entailment" /
    "neutral" / "contradiction" as if they were topic labels, which is close
    to meaningless for an NLI checkpoint and this task. That explains both
    the "You must include at least one label and at least one sequence"
    crashes under certain input shapes AND, more importantly, why Stage 3's
    detection behavior has looked erratic/near-random across every eval run
    so far -- it was never doing premise/hypothesis entailment checking.
    """
    tokenizer, model, contradiction_idx = get_nli_model()
    premise = chunk_text[:2000]  # guard against pathologically long chunks
    inputs = tokenizer(premise, answer, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[contradiction_idx])


def run_stage3(chunks_text: List[str], answer: str) -> dict:
    start = time.perf_counter()
    per_chunk_scores = []
    for i, chunk in enumerate(chunks_text):
        try:
            per_chunk_scores.append(_get_contradiction_prob(chunk, answer))
        except Exception as e:
            logger.warning(f"[Stage 3] Chunk {i+1} failed: {e}")
            per_chunk_scores.append(0.0)

    score = float(np.max(per_chunk_scores)) if per_chunk_scores else 0.0
    flagged = score > NLI_THRESHOLD
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[Stage 3] NLI={score:.3f} Flagged={flagged} {latency_ms:.0f}ms")
    return {
        "score": score,
        "flagged": flagged,
        "per_chunk_scores": per_chunk_scores,
        "threshold": NLI_THRESHOLD,
        "latency_ms": latency_ms,
    }
