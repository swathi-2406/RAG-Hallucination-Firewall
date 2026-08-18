"""
src/hallucination/firewall.py
Orchestrates all three hallucination detection stages.
Uses DeepSeek V4 Flash for answer generation.
"""

import logging
import time
from typing import List

from langchain_core.documents import Document

from config.settings import LLM_TEMP, LLM_MAX_TOKENS, RISK_WEIGHTS
from src.llm_client import call_llm, build_rag_messages
from src.hallucination.stage1_entropy import run_stage1
from src.hallucination.stage2_jsd import run_stage2
from src.hallucination.stage3_nli import run_stage3
from src.retrieval.retriever import get_context_string

logger = logging.getLogger(__name__)


def generate_answer(query: str, context: str, temperature: float = None, stress: bool = False) -> tuple:
    """Generate answer via DeepSeek V4 Flash. temperature=None uses the deterministic default.
    stress=True uses SYSTEM_PROMPT_STRESS (no anti-fabrication instruction) -- eval only."""
    messages = build_rag_messages(query, context, stress=stress)
    temp = LLM_TEMP if temperature is None else temperature
    text, backend, latency_ms = call_llm(messages, temperature=temp, max_tokens=LLM_MAX_TOKENS)
    logger.info(f"Answer [{backend}] temp={temp} stress={stress} in {latency_ms:.0f}ms")
    return text, latency_ms, backend


def compute_composite_risk(
    entropy_flagged: bool, jsd_flagged: bool, nli_flagged: bool,
    run_entropy: bool = True, run_jsd: bool = True, run_nli: bool = True,
) -> float:
    """
    Weighted vote across whichever stages are active, using each stage's
    OWN already-correctly-calibrated `flagged` boolean (score > that
    stage's ENTROPY_THRESHOLD/JSD_THRESHOLD/NLI_THRESHOLD, computed inside
    each stage module) rather than trying to rescale raw, differently-
    distributed continuous scores onto one shared numeric axis.

    Bug this replaces: the previous version divided each raw score by its
    own threshold (score/threshold) and combined those on the assumption
    that "half of a stage's calibrated threshold" is a meaningful universal
    safe/risky boundary. It isn't, for every stage. JSD's benign range
    (0.5-0.95, see the JSD false-positive-rate note above) sits close
    enough to its own threshold (0.80) that even HALF of that threshold
    (0.40) is already below where ordinary, correct answers score -- so
    continuous rescaling still flagged almost everything, just via a
    different arithmetic path than the original bug. Voting on each stage's
    own already-correct binary verdict sidesteps rescaling entirely: a
    stage's threshold only ever needs to mean one thing (crossed it or
    didn't), which is exactly what each stage module already computes.
    """
    active_weight = (
        (RISK_WEIGHTS["entropy"] if run_entropy else 0.0) +
        (RISK_WEIGHTS["jsd"]     if run_jsd     else 0.0) +
        (RISK_WEIGHTS["nli"]     if run_nli     else 0.0)
    )
    if active_weight == 0:
        return 0.0
    raw = (
        (RISK_WEIGHTS["entropy"] * (1.0 if entropy_flagged else 0.0) if run_entropy else 0.0) +
        (RISK_WEIGHTS["jsd"]     * (1.0 if jsd_flagged     else 0.0) if run_jsd     else 0.0) +
        (RISK_WEIGHTS["nli"]     * (1.0 if nli_flagged     else 0.0) if run_nli     else 0.0)
    )
    return round(raw / active_weight, 4)


def risk_label(score: float) -> str:
    """
    Operates on the weighted-vote composite from compute_composite_risk
    (0 = no active stage flagged, 1 = every active stage flagged). With
    RISK_WEIGHTS = {entropy:.35, jsd:.20, nli:.45}, the possible values when
    all 3 stages are active are: 0, .20 (jsd alone), .35 (entropy alone),
    .45 (nli alone), .55 (entropy+jsd), .65 (jsd+nli), .80 (entropy+nli),
    1.0 (all three). Cutoffs below place JSD-alone (the lowest-trust,
    highest-false-positive-rate signal) at LOW, any single higher-trust
    stage at MEDIUM, and either full agreement or the two most reliable
    stages agreeing at HIGH. For single-stage ablation conditions, the
    composite is always exactly 0 or 1 (one voter), so it always resolves
    cleanly to LOW or HIGH -- that stage's own calibrated verdict, with no
    rescaling distortion.
    """
    if score < 0.35:  return "LOW"
    elif score < 0.80: return "MEDIUM"
    else:              return "HIGH"


def score_firewall(
    query: str,
    chunks: List[Document],
    answer: str,
    answer_latency: float = 0.0,
    answer_backend: str = "cached",
    run_entropy: bool = True,
    run_jsd: bool = True,
    run_nli: bool = True,
) -> dict:
    """
    Run the 3 detection stages against an ALREADY-GENERATED answer.

    Split out from run_firewall so that ablation/eval code can generate the
    answer once per question and score every condition (baseline, NLI-only,
    entropy-only, JSD-only, full) against that identical answer. Previously
    each condition called run_firewall independently, which regenerated the
    answer fresh every time -- meaning NLI-only and JSD-only weren't
    necessarily judging the same underlying answer, let alone the same
    hallucination (or lack thereof). That confound is on top of DeepSeek not
    being perfectly deterministic even at temperature=0, so re-running this
    stage-by-stage on a shared, fixed answer removes it entirely and makes
    the per-stage ablation a controlled comparison as intended.
    """
    pipeline_start = time.perf_counter()
    context = get_context_string(chunks)
    chunks_text = [c.page_content for c in chunks]

    stage1 = run_stage1(query, context, answer) if run_entropy else \
        {"score": 0.0, "flagged": False, "samples": [], "latency_ms": 0, "backend": "skipped"}

    stage2 = run_stage2(context, answer) if run_jsd else \
        {"score": 0.0, "flagged": False, "latency_ms": 0}

    stage3 = run_stage3(chunks_text, answer) if run_nli else \
        {"score": 0.0, "flagged": False, "per_chunk_scores": [], "latency_ms": 0}

    composite = compute_composite_risk(
        stage1["flagged"], stage2["flagged"], stage3["flagged"],
        run_entropy=run_entropy, run_jsd=run_jsd, run_nli=run_nli,
    )
    label = risk_label(composite)
    total_latency = (time.perf_counter() - pipeline_start) * 1000 + answer_latency

    logger.info(f"Pipeline done | Risk={label} ({composite:.3f}) | {total_latency:.0f}ms")

    return {
        "query": query,
        "answer": answer,
        "answer_backend": answer_backend,
        "context": context,
        "chunks": [{"content": c.page_content, "source": c.metadata.get("source", "")} for c in chunks],
        "composite_risk_score": composite,
        "risk_label": label,
        "stages_flagged": sum([stage1.get("flagged", False), stage2.get("flagged", False), stage3.get("flagged", False)]),
        "stage1_entropy": stage1,
        "stage2_jsd": stage2,
        "stage3_nli": stage3,
        "latency": {
            "answer_ms":  round(answer_latency, 1),
            "stage1_ms":  round(stage1.get("latency_ms", 0), 1),
            "stage2_ms":  round(stage2.get("latency_ms", 0), 1),
            "stage3_ms":  round(stage3.get("latency_ms", 0), 1),
            "total_ms":   round(total_latency, 1),
        },
    }


def run_firewall(
    query: str,
    chunks: List[Document],
    run_entropy: bool = True,
    run_jsd: bool = True,
    run_nli: bool = True,
    prefer_eval: bool = False,
    answer_temperature: float = None,
) -> dict:
    """Convenience wrapper: generate an answer, then score it. Used by app.py
    (single live query -- there's no shared-answer concern outside eval)."""
    context = get_context_string(chunks)
    answer, answer_latency, answer_backend = generate_answer(query, context, temperature=answer_temperature)
    return score_firewall(query, chunks, answer, answer_latency, answer_backend,
                           run_entropy=run_entropy, run_jsd=run_jsd, run_nli=run_nli)
