"""
scripts/evaluate_stress.py
Stress-test evaluation: elicits hallucinations deliberately by (a) removing
the deployed anti-fabrication instruction (SYSTEM_PROMPT_STRESS) and (b)
raising sampling temperature, so the firewall's per-stage recall/precision
can be measured against a statistically meaningful number of actual
hallucinations.

Why both levers: an earlier version of this script raised temperature alone
and got ZERO hallucinations across 34 stressed questions even at temp=0.9.
That's because the deployed system prompt tells the model, in plain
language, "If the context does not contain enough information, say so
explicitly. Do not fabricate information" -- and that's an explicit
instruction an instruction-tuned model follows regardless of sampling
temperature. Temperature governs word-choice diversity, not whether the
model obeys a direct directive. To get hallucinations to evaluate against,
this script also swaps in a prompt that asks the model to answer confidently
and fill gaps from general knowledge (SYSTEM_PROMPT_STRESS in
src/llm_client.py) -- NEVER used in the deployed app, only here.

Why this script exists at all: evaluate_firewall.py runs the deterministic,
properly-guarded generator, which -- with the real grounding prompt --
only hallucinates on ~1-2 of 50 questions. That's a real and useful finding
(the prompt-level guardrail works, and is robust to temperature), but it
means recall/FNR computed from evaluate_firewall.py's results rests on 1-2
positive examples and is not something you can defend as a headline number.

This script re-runs the same 5 conditions x same question set at
STRESS_TEMP with the no-hedge prompt, with STRESS_SAMPLES generations per
question per condition (default 1; raise it if you want more positives at
the cost of more calls), and reports the same confusion-matrix metrics as
evaluate_firewall.py.

Report evaluate_firewall.py (organic, guarded prompt, temp=0), this script
(stress, no-hedge prompt, temp=STRESS_TEMP) as two distinct, clearly-labeled
operating points in your writeup -- don't merge them into one number, and be
explicit that the stress condition is a synthetic adversarial setup used to
get enough positive examples to evaluate detector recall, not a claim about
how the deployed system behaves.

Run: python scripts/evaluate_stress.py
"""

import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.retrieval.retriever import load_index, retrieve, get_context_string
from src.hallucination.firewall import score_firewall, generate_answer
from scripts.evaluate_firewall import (
    EVAL_QUESTIONS, CONDITIONS, score_answer, should_flag,
    risk_correct, mcnemar_test, run_with_retry,
)

STRESS_TEMP = 0.9
STRESS_SAMPLES = 1  # generations per question per condition; raise for more positives
# With the no-hedge SYSTEM_PROMPT_STRESS active, in-scope questions can also
# produce fabricated specifics (e.g. invented numbers not in the abstract),
# so include everything rather than restricting to adversarial categories.
STRESS_CATEGORIES = None

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"\n{'='*70}\n  STRESS-TEST EVAL | temp={STRESS_TEMP} x{STRESS_SAMPLES} | DeepSeek {DEEPSEEK_MODEL}\n{'='*70}")
    if not DEEPSEEK_API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY in .env"); sys.exit(1)

    vs = load_index()
    print(f"  {vs.index.ntotal} vectors")

    questions = [q for q in EVAL_QUESTIONS if STRESS_CATEGORIES is None or q["category"] in STRESS_CATEGORIES]
    print(f"  {len(questions)}/{len(EVAL_QUESTIONS)} questions selected for stress testing")

    all_results = {c["name"]: [] for c in CONDITIONS}
    for i, q in enumerate(questions, 1):
        print(f"\r  {i}/{len(questions)} {q['id']}   ", end="", flush=True)
        for s in range(STRESS_SAMPLES):
            try:
                def gen_q():
                    chunks, ret_ms = retrieve(q["question"], vs)
                    context = get_context_string(chunks)
                    answer, answer_latency, answer_backend = generate_answer(q["question"], context, temperature=STRESS_TEMP, stress=True)
                    return chunks, ret_ms, answer, answer_latency, answer_backend
                chunks, ret_ms, answer, answer_latency, answer_backend = run_with_retry(gen_q)
                quality = score_answer(answer, q["expected"], q["keywords"])
            except Exception as e:
                print(f"\n  WARN {q['id']} sample {s} (generation): {str(e)[:60]}")
                continue

            for cond in CONDITIONS:
                cname = cond["name"]
                try:
                    def score_q():
                        return score_firewall(q["question"], chunks, answer, answer_latency, answer_backend,
                            run_entropy=cond["run_entropy"], run_jsd=cond["run_jsd"], run_nli=cond["run_nli"])
                    result = run_with_retry(score_q)
                    correct = risk_correct(result, quality["quality"], cond)
                    all_results[cname].append({**q, "condition": cname, "sample": s,
                        "answer": result["answer"], "risk_label": result["risk_label"],
                        "composite_risk": result["composite_risk_score"],
                        "total_latency_ms": result["latency"]["total_ms"],
                        **quality, "classification_correct": correct})
                except Exception as e:
                    print(f"\n  WARN {q['id']}/{cname} sample {s}: {str(e)[:60]}")
            time.sleep(0.5)
    print(f"\n  Done.")

    def confusion(r):
        tp=fp=tn=fn=0
        for x in r:
            if x.get("classification_correct") is None: continue
            sf = should_flag(x["quality"])
            flagged = x.get("risk_label") in ("MEDIUM","HIGH")
            if sf and flagged: tp+=1
            elif sf and not flagged: fn+=1
            elif not sf and flagged: fp+=1
            else: tn+=1
        return tp,fp,tn,fn
    def frac(k,n):
        return {"k":k,"n":n,"pct":round(k/n*100)} if n else None
    def fmt(d):
        return f"{d['k']}/{d['n']} (~{d['pct']}%)" if d else "N/A"

    stats={}
    for cname, results in all_results.items():
        lbl=next(c["label"] for c in CONDITIONS if c["name"]==cname)
        tp,fp,tn,fn = confusion(results)
        n_positive = tp+fn
        stats[cname]={"label":lbl,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"n_positive":n_positive,
            "accuracy_pct":frac(tp+tn, tp+fp+tn+fn),
            "fpr":frac(fp, fp+tn),
            "fnr":frac(fn, n_positive),
            "precision":frac(tp, tp+fp)}

    mcnemar={}
    if "nli_only" in all_results and "full" in all_results:
        mcnemar["nli_vs_full"] = mcnemar_test(all_results["nli_only"], all_results["full"])

    print(f"\n\n{'='*70}\n  STRESS RESULTS | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*70}")
    print(f"\n  {'Condition':<35} {'Acc':>14} {'FPR':>14} {'FNR (miss)':>14} {'Precision':>14} {'n_pos':>6}")
    for cn in ["baseline","nli_only","entropy_only","jsd_only","full"]:
        if cn not in stats: continue
        s=stats[cn]
        print(f"  {s['label']:<35} {fmt(s['accuracy_pct']):>14} {fmt(s['fpr']):>14} "
              f"{fmt(s['fnr']):>14} {fmt(s['precision']):>14} {s['n_positive']:>6}")

    if "nli_vs_full" in mcnemar:
        m = mcnemar["nli_vs_full"]
        print(f"\n  McNEMAR (NLI-only vs Full): chi2={m['chi2']} p={m['p_value']} {'SIGNIFICANT' if m['significant'] else 'not significant'}")

    total_positive = max((s['n_positive'] for s in stats.values() if s.get('n_positive') is not None), default=0)
    print(f"\n  Positive-class size this run: up to {total_positive} real hallucinations across "
          f"{len(questions)} stressed questions.")
    if total_positive < 10:
        print(f"  Still thin. Raise STRESS_SAMPLES or STRESS_TEMP, or widen STRESS_CATEGORIES, and re-run.")

    with open(EVAL_DIR/"eval_stress_report.json","w") as f:
        json.dump({"timestamp":datetime.now().isoformat(),"stress_temp":STRESS_TEMP,
                   "stress_samples":STRESS_SAMPLES,"n_questions_stressed":len(questions),
                   "stats":stats,"mcnemar":mcnemar,"all_results":all_results}, f, indent=2, default=str)
    print(f"\nSaved: {EVAL_DIR}/eval_stress_report.json")


if __name__ == "__main__":
    main()
