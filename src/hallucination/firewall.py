"""
src/hallucination/firewall.py
Orchestrates the three-stage hallucination detection pipeline.

Pipeline:
  Query + Context
      │
      ├─▶ [Stage 1] Semantic Entropy   (LLM sampling)
      ├─▶ [Stage 2] JSD Token Divergence
      └─▶ [Stage 3] NLI Cross-Check    (DeBERTa)
              │
              └─▶ Composite Risk Score (weighted average)
                        │
                        └─▶ PASS / WARN / BLOCK
"""

import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, LLM_TEMP, LLM_MAX_TOKENS, RISK_WEIGHTS,
)
from src.hallucination.stage1_entropy import run_stage1
from src.hallucination.stage2_jsd import run_stage2
from src.hallucination.stage3_nli import run_stage3
from src.retrieval.retriever import get_context_string

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the question using ONLY "
    "the provided context. If the context doesn't contain enough information, "
    "say so explicitly. Do not fabricate information."
)


def generate_answer(query: str, context: str) -> tuple[str, float]:
    """
    Generate a deterministic answer using the LLM (temp=0).

    Returns:
        (answer_text, latency_ms)
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOKENS,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
    ]

    start = time.perf_counter()
    response = llm.invoke(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    return response.content.strip(), latency_ms


def compute_composite_risk(
    entropy_score: float,
    jsd_score: float,
    nli_score: float,
) -> float:
    """
    Compute weighted composite risk score in [0, 1].

    Weights are defined in config/settings.py under RISK_WEIGHTS.
    """
    score = (
        RISK_WEIGHTS["entropy"] * entropy_score
        + RISK_WEIGHTS["jsd"] * jsd_score
        + RISK_WEIGHTS["nli"] * nli_score
    )
    return round(float(score), 4)


def risk_label(score: float) -> str:
    """Map composite score to a human-readable risk label."""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


def run_firewall(
    query: str,
    chunks: List[Document],
    run_entropy: bool = True,
    run_jsd: bool = True,
    run_nli: bool = True,
) -> dict:
    """
    Full RAG pipeline with three-stage hallucination firewall.

    Args:
        query:        User's question
        chunks:       Retrieved document chunks from FAISS
        run_entropy:  Enable Stage 1 (can be disabled for speed)
        run_jsd:      Enable Stage 2
        run_nli:      Enable Stage 3 (slowest stage)

    Returns:
        Comprehensive result dict with answer, scores, flags, and latencies.
    """
    pipeline_start = time.perf_counter()

    # Build context string from chunks
    context = get_context_string(chunks)
    chunks_text = [c.page_content for c in chunks]

    # ── Generate deterministic answer ─────────────────────────────────────────
    answer, answer_latency = generate_answer(query, context)
    logger.info(f"Answer generated in {answer_latency:.0f}ms")

    # ── Stage 1: Semantic Entropy ─────────────────────────────────────────────
    stage1 = {}
    if run_entropy:
        stage1 = run_stage1(query, context, answer)
    else:
        stage1 = {"score": 0.0, "flagged": False, "samples": [], "latency_ms": 0}

    # ── Stage 2: Jensen-Shannon Divergence ────────────────────────────────────
    stage2 = {}
    if run_jsd:
        stage2 = run_stage2(context, answer)
    else:
        stage2 = {"score": 0.0, "flagged": False, "latency_ms": 0}

    # ── Stage 3: NLI Cross-Check ──────────────────────────────────────────────
    stage3 = {}
    if run_nli:
        stage3 = run_stage3(chunks_text, answer)
    else:
        stage3 = {"score": 0.0, "flagged": False, "per_chunk_scores": [], "latency_ms": 0}

    # ── Composite Risk Score ──────────────────────────────────────────────────
    composite = compute_composite_risk(
        stage1["score"], stage2["score"], stage3["score"]
    )
    label = risk_label(composite)
    total_latency = (time.perf_counter() - pipeline_start) * 1000

    result = {
        "query": query,
        "answer": answer,
        "context": context,
        "chunks": [
            {"content": c.page_content, "source": c.metadata.get("source", "")}
            for c in chunks
        ],
        # Risk
        "composite_risk_score": composite,
        "risk_label": label,
        "stages_flagged": sum([
            stage1.get("flagged", False),
            stage2.get("flagged", False),
            stage3.get("flagged", False),
        ]),
        # Stage details
        "stage1_entropy": stage1,
        "stage2_jsd": stage2,
        "stage3_nli": stage3,
        # Latency
        "latency": {
            "answer_ms": round(answer_latency, 1),
            "stage1_ms": round(stage1.get("latency_ms", 0), 1),
            "stage2_ms": round(stage2.get("latency_ms", 0), 1),
            "stage3_ms": round(stage3.get("latency_ms", 0), 1),
            "total_ms": round(total_latency, 1),
        },
    }

    logger.info(
        f"Pipeline complete | Risk={label} ({composite:.3f}) | "
        f"Total={total_latency:.0f}ms"
    )

    return result
