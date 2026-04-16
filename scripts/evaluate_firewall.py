"""
scripts/evaluate_firewall.py
────────────────────────────────────────────────────────────────────────────────
Automated evaluation of the RAG Hallucination Firewall.

Runs 20 questions through the pipeline twice:
  1. BASELINE  — no hallucination firewall (pure RAG answer)
  2. FIREWALL  — full three-stage detection enabled

Measures:
  - Firewall catch rate (how often it correctly flags uncertain answers)
  - False positive rate (how often it flags answers that are actually fine)
  - Average risk scores per question category
  - Per-stage contribution analysis
  - RAGAS metric averages across both conditions

Outputs:
  - Console summary table
  - data/evaluation/eval_report.json  (full results)
  - data/evaluation/eval_summary.txt  (copy-paste for LinkedIn/resume)

Run with:
    python scripts/evaluate_firewall.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs during eval

from config.settings import INDEX_DIR, GROQ_API_KEY
from src.retrieval.retriever import load_index, retrieve
from src.hallucination.firewall import run_firewall
from src.evaluation.metrics import compute_all_metrics

# ── Output directory ──────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Evaluation questions ──────────────────────────────────────────────────────
# Organized into 3 categories:
#   IN_SCOPE   — well-covered by the arXiv corpus (should answer confidently)
#   PARTIAL    — partially covered (may need to hedge)
#   OUT_SCOPE  — not in the corpus at all (model should say it doesn't know)
#
# Ground truth labels:
#   "answerable"   — corpus contains enough info; correct answer = grounded, specific
#   "partial"      — corpus has related info but not a complete answer
#   "unanswerable" — corpus has no relevant info; correct = model says "I don't know"

EVAL_QUESTIONS = [
    # ── IN-SCOPE (well covered by AI/ML arXiv corpus) ─────────────────────────
    {
        "id": "q01", "category": "in_scope", "expected": "answerable",
        "question": "What is retrieval augmented generation (RAG)?",
        "keywords": ["retrieval", "generation", "knowledge", "language model"],
    },
    {
        "id": "q02", "category": "in_scope", "expected": "answerable",
        "question": "How does the transformer self-attention mechanism work?",
        "keywords": ["attention", "query", "key", "value", "softmax"],
    },
    {
        "id": "q03", "category": "in_scope", "expected": "answerable",
        "question": "What is RLHF and how is it used to align language models?",
        "keywords": ["reinforcement", "human feedback", "reward", "alignment"],
    },
    {
        "id": "q04", "category": "in_scope", "expected": "answerable",
        "question": "What is chain-of-thought prompting?",
        "keywords": ["reasoning", "step", "thought", "prompt"],
    },
    {
        "id": "q05", "category": "in_scope", "expected": "answerable",
        "question": "How do diffusion models generate images?",
        "keywords": ["noise", "denoising", "diffusion", "generation"],
    },
    {
        "id": "q06", "category": "in_scope", "expected": "answerable",
        "question": "What is federated learning and why is it used for privacy?",
        "keywords": ["federated", "local", "privacy", "distributed"],
    },
    {
        "id": "q07", "category": "in_scope", "expected": "answerable",
        "question": "What is the difference between BERT and GPT architectures?",
        "keywords": ["encoder", "decoder", "bidirectional", "autoregressive"],
    },
    {
        "id": "q08", "category": "in_scope", "expected": "answerable",
        "question": "How do graph neural networks work?",
        "keywords": ["graph", "node", "edge", "aggregation", "message"],
    },
    # ── PARTIAL (corpus has related info, not a direct answer) ────────────────
    {
        "id": "q09", "category": "partial", "expected": "partial",
        "question": "What are the top three open source RAG frameworks ranked by GitHub stars?",
        "keywords": ["github", "stars", "framework", "open source"],
    },
    {
        "id": "q10", "category": "partial", "expected": "partial",
        "question": "What specific accuracy numbers did RAG achieve on the Natural Questions benchmark?",
        "keywords": ["natural questions", "accuracy", "benchmark", "percent"],
    },
    {
        "id": "q11", "category": "partial", "expected": "partial",
        "question": "How many parameters does the original GPT-3 model have and what was its training cost?",
        "keywords": ["175 billion", "parameters", "training cost", "compute"],
    },
    {
        "id": "q12", "category": "partial", "expected": "partial",
        "question": "What is the ROUGE score of the best summarization model in 2024?",
        "keywords": ["rouge", "summarization", "score", "2024"],
    },
    # ── OUT-OF-SCOPE (not in the corpus — model should refuse or hedge) ────────
    {
        "id": "q13", "category": "out_scope", "expected": "unanswerable",
        "question": "What is the recipe for chocolate chip cookies?",
        "keywords": ["flour", "sugar", "butter", "chocolate", "bake"],
    },
    {
        "id": "q14", "category": "out_scope", "expected": "unanswerable",
        "question": "Who won the FIFA World Cup in 2022 and what was the final score?",
        "keywords": ["argentina", "france", "world cup", "final", "penalty"],
    },
    {
        "id": "q15", "category": "out_scope", "expected": "unanswerable",
        "question": "What are the best tourist attractions in Tokyo, Japan?",
        "keywords": ["tokyo", "temple", "shrine", "shibuya", "tourism"],
    },
    {
        "id": "q16", "category": "out_scope", "expected": "unanswerable",
        "question": "How do you change the oil in a 2020 Honda Civic?",
        "keywords": ["oil", "honda", "civic", "drain", "filter", "mechanic"],
    },
    # ── HALLUCINATION TRAPS (leading questions likely to cause fabrication) ────
    {
        "id": "q17", "category": "trap", "expected": "partial",
        "question": "What did the paper 'Attention Is All You Need' say about the exact BLEU score improvement over previous models?",
        "keywords": ["bleu", "attention", "transformer", "score", "improvement"],
    },
    {
        "id": "q18", "category": "trap", "expected": "partial",
        "question": "According to the papers in your knowledge base, what is the exact percentage of LLM outputs that contain hallucinations?",
        "keywords": ["percent", "percentage", "hallucination", "rate", "study"],
    },
    {
        "id": "q19", "category": "trap", "expected": "partial",
        "question": "What were the names of all the researchers who invented the RAG technique and at which institution?",
        "keywords": ["lewis", "perez", "facebook", "meta", "researcher"],
    },
    {
        "id": "q20", "category": "trap", "expected": "partial",
        "question": "What specific hyperparameters should I use to fine-tune BERT on a medical question answering dataset?",
        "keywords": ["learning rate", "batch size", "epochs", "medical", "hyperparameter"],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_answer_quality(answer: str, expected: str, keywords: list) -> dict:
    """
    Heuristic answer quality scoring (no human needed).

    Returns:
        quality: 'good' | 'hedged' | 'hallucinated'
        score:   0.0 – 1.0
        reason:  explanation
    """
    answer_lower = answer.lower()

    # Detect hedging / refusal
    hedge_phrases = [
        "does not contain", "not enough information", "cannot be determined",
        "not specified", "not mentioned", "i don't know", "no information",
        "context does not", "not provided", "cannot answer", "don't have",
        "not available", "insufficient", "not explicitly",
    ]
    is_hedging = any(p in answer_lower for p in hedge_phrases)

    # Keyword coverage
    kw_hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    kw_coverage = kw_hits / len(keywords) if keywords else 0

    if expected == "answerable":
        if is_hedging and kw_coverage < 0.3:
            return {"quality": "poor", "score": 0.2, "reason": "Refused answerable question"}
        elif kw_coverage >= 0.5:
            return {"quality": "good", "score": 0.9, "reason": f"Covered {kw_hits}/{len(keywords)} keywords"}
        else:
            return {"quality": "partial", "score": 0.6, "reason": f"Low keyword coverage {kw_coverage:.0%}"}

    elif expected == "unanswerable":
        if is_hedging:
            return {"quality": "good", "score": 1.0, "reason": "Correctly refused out-of-scope question"}
        elif kw_coverage > 0.4:
            return {"quality": "hallucinated", "score": 0.0, "reason": "Gave specific answer to out-of-scope question"}
        else:
            return {"quality": "partial", "score": 0.5, "reason": "Vague response to out-of-scope question"}

    else:  # partial / trap
        if is_hedging:
            return {"quality": "good", "score": 0.85, "reason": "Correctly hedged on partial/trap question"}
        elif kw_coverage >= 0.5:
            return {"quality": "hallucinated", "score": 0.1, "reason": "Gave overconfident answer to trap question"}
        else:
            return {"quality": "partial", "score": 0.5, "reason": "Generic answer to partial question"}


def run_single_query(question, vectorstore, use_firewall=True):
    """Run one question through the pipeline and return results."""
    chunks, ret_latency = retrieve(question, vectorstore)
    chunks_text = [c.page_content for c in chunks]

    result = run_firewall(
        question, chunks,
        run_entropy=use_firewall,
        run_jsd=use_firewall,
        run_nli=use_firewall,
    )

    metrics = compute_all_metrics(question, result["answer"], chunks_text)

    return {
        "answer": result["answer"],
        "composite_risk": result["composite_risk_score"],
        "risk_label": result["risk_label"],
        "entropy_score": result["stage1_entropy"].get("score", 0),
        "jsd_score": result["stage2_jsd"].get("score", 0),
        "nli_score": result["stage3_nli"].get("score", 0),
        "stages_flagged": result["stages_flagged"],
        "context_precision": metrics["context_precision"],
        "answer_faithfulness": metrics["answer_faithfulness"],
        "answer_relevancy": metrics["answer_relevancy"],
        "retrieval_latency_ms": ret_latency,
        "total_latency_ms": result["latency"]["total_ms"],
    }


def print_progress(i, total, q_id, category):
    bar_len = 30
    filled = int(bar_len * i / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {i}/{total}  {q_id} ({category})   ", end="", flush=True)


# ── Main Evaluation ───────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  RAG HALLUCINATION FIREWALL — AUTOMATED EVALUATION")
    print("=" * 70)

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    print("\n📂 Loading FAISS index...")
    vs = load_index()
    print(f"   ✓ Index loaded ({vs.index.ntotal} vectors)")

    results = []
    total = len(EVAL_QUESTIONS)

    # ── FIREWALL ON ───────────────────────────────────────────────────────────
    print(f"\n🔥 Running {total} queries WITH firewall (all 3 stages)...\n")
    firewall_results = []

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print_progress(i, total, q["id"], q["category"])
        try:
            r = run_single_query(q["question"], vs, use_firewall=True)
            quality = score_answer_quality(r["answer"], q["expected"], q["keywords"])
            firewall_results.append({**q, **r, **quality, "condition": "firewall"})
        except Exception as e:
            print(f"\n  ⚠ {q['id']} failed: {e}")
            firewall_results.append({**q, "condition": "firewall", "error": str(e),
                                     "composite_risk": 0.5, "risk_label": "UNKNOWN",
                                     "quality": "error", "score": 0.5})
        time.sleep(0.5)  # Rate limit courtesy

    print(f"\n\n✅ Firewall condition complete.")

    # ── FIREWALL OFF (BASELINE) ───────────────────────────────────────────────
    print(f"\n⚙️  Running {total} queries WITHOUT firewall (baseline)...\n")
    baseline_results = []

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print_progress(i, total, q["id"], q["category"])
        try:
            r = run_single_query(q["question"], vs, use_firewall=False)
            quality = score_answer_quality(r["answer"], q["expected"], q["keywords"])
            baseline_results.append({**q, **r, **quality, "condition": "baseline"})
        except Exception as e:
            print(f"\n  ⚠ {q['id']} failed: {e}")
            baseline_results.append({**q, "condition": "baseline", "error": str(e),
                                     "composite_risk": 0.0, "risk_label": "UNKNOWN",
                                     "quality": "error", "score": 0.5})
        time.sleep(0.5)

    print(f"\n\n✅ Baseline condition complete.")

    # ── COMPUTE STATISTICS ────────────────────────────────────────────────────
    print("\n📊 Computing statistics...\n")

    def avg(lst, key):
        vals = [x[key] for x in lst if key in x and not isinstance(x.get(key), str)]
        return sum(vals) / len(vals) if vals else 0

    def pct(lst, condition):
        matches = sum(1 for x in lst if condition(x))
        return matches / len(lst) * 100 if lst else 0

    # Firewall catch rate: for out-of-scope and trap questions,
    # how often does the firewall flag them?
    risky_qs_fw = [r for r in firewall_results if r["category"] in ("out_scope", "trap", "partial")]
    catch_rate = pct(risky_qs_fw, lambda x: x.get("stages_flagged", 0) > 0)

    # False positive rate: for in-scope questions,
    # how often does the firewall flag them (should be low)?
    inscope_fw = [r for r in firewall_results if r["category"] == "in_scope"]
    fp_rate = pct(inscope_fw, lambda x: x.get("risk_label") == "HIGH")

    # Answer quality improvement
    fw_quality  = avg(firewall_results, "score")
    base_quality = avg(baseline_results, "score")
    quality_delta = (fw_quality - base_quality) / max(base_quality, 0.001) * 100

    # Out-of-scope hedging rate (did model correctly refuse?)
    oos_fw   = [r for r in firewall_results   if r["category"] == "out_scope"]
    oos_base = [r for r in baseline_results if r["category"] == "out_scope"]
    hedge_fw   = pct(oos_fw,   lambda x: x.get("quality") == "good")
    hedge_base = pct(oos_base, lambda x: x.get("quality") == "good")

    # Trap question overconfidence rate
    trap_fw   = [r for r in firewall_results   if r["category"] == "trap"]
    trap_base = [r for r in baseline_results if r["category"] == "trap"]
    trap_hallu_fw   = pct(trap_fw,   lambda x: x.get("quality") == "hallucinated")
    trap_hallu_base = pct(trap_base, lambda x: x.get("quality") == "hallucinated")

    # RAGAS averages
    fw_precision   = avg(firewall_results,  "context_precision")
    fw_faithful    = avg(firewall_results,  "answer_faithfulness")
    fw_relevancy   = avg(firewall_results,  "answer_relevancy")
    base_precision = avg(baseline_results, "context_precision")
    base_faithful  = avg(baseline_results, "answer_faithfulness")
    base_relevancy = avg(baseline_results, "answer_relevancy")

    # Risk score distribution
    low_pct    = pct(firewall_results, lambda x: x.get("risk_label") == "LOW")
    medium_pct = pct(firewall_results, lambda x: x.get("risk_label") == "MEDIUM")
    high_pct   = pct(firewall_results, lambda x: x.get("risk_label") == "HIGH")

    avg_risk   = avg(firewall_results, "composite_risk")
    avg_ret_ms = avg(firewall_results, "retrieval_latency_ms")

    # ── PRINT RESULTS ─────────────────────────────────────────────────────────
    SEP = "─" * 70

    print(SEP)
    print("  EVALUATION RESULTS — RAG HALLUCINATION FIREWALL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} | {total} questions | Corpus: 198 arXiv papers")
    print(SEP)

    print("\n  RETRIEVAL PERFORMANCE")
    print(f"    Average retrieval latency:      {avg_ret_ms:.0f}ms  (target: <200ms) {'✓' if avg_ret_ms < 200 else '✗'}")
    print(f"    Context Precision (avg):         {fw_precision:.3f}")
    print(f"    Answer Relevancy (avg):          {fw_relevancy:.3f}")

    print("\n  HALLUCINATION FIREWALL PERFORMANCE")
    print(f"    Firewall catch rate*:            {catch_rate:.0f}%  (flagged risky queries)")
    print(f"    False positive rate:             {fp_rate:.0f}%   (wrongly flagged safe queries)")
    print(f"    Out-of-scope hedging (firewall): {hedge_fw:.0f}%  correct refusals")
    print(f"    Out-of-scope hedging (baseline): {hedge_base:.0f}%  correct refusals")
    print(f"    Trap question hallucination:     {trap_hallu_fw:.0f}%  (firewall) vs {trap_hallu_base:.0f}% (baseline)")

    print("\n  ANSWER QUALITY")
    print(f"    Faithfulness — firewall:         {fw_faithful:.3f}")
    print(f"    Faithfulness — baseline:         {base_faithful:.3f}")
    faithfulness_delta = (fw_faithful - base_faithful) / max(base_faithful, 0.001) * 100
    print(f"    Faithfulness improvement:        {faithfulness_delta:+.1f}%")
    print(f"    Overall quality score — firewall: {fw_quality:.3f}")
    print(f"    Overall quality score — baseline: {base_quality:.3f}")
    print(f"    Quality improvement:             {quality_delta:+.1f}%")

    print("\n  RISK DISTRIBUTION (firewall condition)")
    print(f"    LOW risk:    {low_pct:.0f}%  of queries")
    print(f"    MEDIUM risk: {medium_pct:.0f}%  of queries")
    print(f"    HIGH risk:   {high_pct:.0f}%  of queries")
    print(f"    Average composite risk score:    {avg_risk:.3f}")

    print("\n  PER-CATEGORY BREAKDOWN")
    for cat in ["in_scope", "partial", "out_scope", "trap"]:
        cat_fw = [r for r in firewall_results if r["category"] == cat]
        if cat_fw:
            cat_risk = avg(cat_fw, "composite_risk")
            cat_qual = avg(cat_fw, "score")
            print(f"    {cat:<12} n={len(cat_fw)}  avg_risk={cat_risk:.3f}  avg_quality={cat_qual:.3f}")

    print("\n" + SEP)

    # ── RESUME BULLETS ────────────────────────────────────────────────────────
    resume_lines = [
        "RESUME / LINKEDIN BULLETS (use whichever are accurate for your results)",
        "=" * 70,
        "",
        f"• Built RAG middleware in Python (LangChain + FAISS) achieving sub-{avg_ret_ms:.0f}ms",
        f"  retrieval latency on a {vs.index.ntotal}-vector corpus of 198 AI/ML arXiv papers.",
        "",
        f"• Implemented three-stage hallucination firewall (semantic entropy,",
        f"  Jensen-Shannon divergence, DeBERTa NLI) achieving {catch_rate:.0f}% detection rate",
        f"  on ambiguous and out-of-scope queries with {fp_rate:.0f}% false positive rate.",
        "",
        f"• Improved answer faithfulness by {faithfulness_delta:+.1f}% vs. baseline RAG pipeline",
        f"  ({fw_faithful:.2f} vs {base_faithful:.2f}) across a 20-question evaluation suite",
        f"  spanning in-scope, out-of-scope, and adversarial hallucination trap queries.",
        "",
        f"• System correctly refused {hedge_fw:.0f}% of out-of-scope queries (vs {hedge_base:.0f}% baseline),",
        f"  demonstrating grounded answer generation over confident hallucination.",
        "",
        "LINKEDIN POST STATS TO MENTION:",
        f"  - {vs.index.ntotal} document chunks indexed",
        f"  - {avg_ret_ms:.0f}ms average retrieval latency",
        f"  - {catch_rate:.0f}% hallucination catch rate on risky queries",
        f"  - {fw_faithful:.2f} average answer faithfulness (RAGAS-style)",
        f"  - {low_pct:.0f}% of queries rated LOW risk by composite firewall score",
    ]

    summary_text = "\n".join(resume_lines)
    print("\n" + summary_text)
    print("\n" + SEP)

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "corpus_size": vs.index.ntotal,
        "n_questions": total,
        "summary": {
            "avg_retrieval_latency_ms": round(avg_ret_ms, 1),
            "catch_rate_pct": round(catch_rate, 1),
            "false_positive_rate_pct": round(fp_rate, 1),
            "out_of_scope_hedging_firewall_pct": round(hedge_fw, 1),
            "out_of_scope_hedging_baseline_pct": round(hedge_base, 1),
            "faithfulness_firewall": round(fw_faithful, 3),
            "faithfulness_baseline": round(base_faithful, 3),
            "faithfulness_improvement_pct": round(faithfulness_delta, 1),
            "quality_improvement_pct": round(quality_delta, 1),
            "context_precision": round(fw_precision, 3),
            "answer_relevancy": round(fw_relevancy, 3),
            "risk_distribution": {"low": round(low_pct, 1), "medium": round(medium_pct, 1), "high": round(high_pct, 1)},
            "avg_composite_risk": round(avg_risk, 3),
        },
        "firewall_results": firewall_results,
        "baseline_results": baseline_results,
    }

    json_path = EVAL_DIR / "eval_report.json"
    txt_path  = EVAL_DIR / "eval_summary.txt"

    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    with open(txt_path, "w") as f:
        f.write(summary_text)

    print(f"\n💾 Full results saved to: {json_path}")
    print(f"📝 Resume bullets saved to: {txt_path}")
    print("\nDone! ✓\n")


if __name__ == "__main__":
    main()
