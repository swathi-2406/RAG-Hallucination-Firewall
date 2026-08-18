"""
scripts/evaluate_firewall.py
Comprehensive evaluation using DeepSeek V4 Flash.
5 conditions x 20 questions: ablation, baseline, latency, McNemar test.
Run: python scripts/evaluate_firewall.py
"""

import sys, json, time, logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

import numpy as np
from scipy.stats import chi2

from config.settings import INDEX_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.retrieval.retriever import load_index, retrieve, get_context_string
from src.hallucination.firewall import score_firewall, generate_answer
from src.evaluation.metrics import compute_all_metrics

FAST_MODE = False
EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_QUESTIONS = [
    {"id":"q01","category":"in_scope","expected":"answerable","question":"What is retrieval augmented generation (RAG)?","keywords":["retrieval","generation","knowledge","language model"]},
    {"id":"q02","category":"in_scope","expected":"answerable","question":"How does the transformer self-attention mechanism work?","keywords":["attention","query","key","value","softmax"]},
    {"id":"q03","category":"in_scope","expected":"answerable","question":"What is RLHF and how is it used to align language models?","keywords":["reinforcement","human feedback","reward","alignment"]},
    {"id":"q04","category":"in_scope","expected":"answerable","question":"What is chain-of-thought prompting?","keywords":["reasoning","step","thought","prompt"]},
    {"id":"q05","category":"in_scope","expected":"answerable","question":"How do diffusion models generate images?","keywords":["noise","denoising","diffusion","generation"]},
    {"id":"q06","category":"in_scope","expected":"answerable","question":"What is federated learning and why is it used for privacy?","keywords":["federated","local","privacy","distributed"]},
    {"id":"q07","category":"in_scope","expected":"answerable","question":"What is the difference between BERT and GPT architectures?","keywords":["encoder","decoder","bidirectional","autoregressive"]},
    {"id":"q08","category":"in_scope","expected":"answerable","question":"How do graph neural networks work?","keywords":["graph","node","edge","aggregation","message"]},
    {"id":"q09","category":"partial","expected":"partial","question":"What are the top three open source RAG frameworks ranked by GitHub stars?","keywords":["github","stars","framework","open source"]},
    {"id":"q10","category":"partial","expected":"partial","question":"What specific accuracy numbers did RAG achieve on the Natural Questions benchmark?","keywords":["natural questions","accuracy","benchmark","percent"]},
    {"id":"q11","category":"partial","expected":"partial","question":"How many parameters does the original GPT-3 model have and what was its training cost?","keywords":["175 billion","parameters","training cost","compute"]},
    {"id":"q12","category":"partial","expected":"partial","question":"What is the ROUGE score of the best summarization model in 2024?","keywords":["rouge","summarization","score","2024"]},
    {"id":"q13","category":"out_scope","expected":"unanswerable","question":"What is the recipe for chocolate chip cookies?","keywords":["flour","sugar","butter","chocolate","bake"]},
    {"id":"q14","category":"out_scope","expected":"unanswerable","question":"Who won the FIFA World Cup in 2022 and what was the final score?","keywords":["argentina","france","world cup","final","penalty"]},
    {"id":"q15","category":"out_scope","expected":"unanswerable","question":"What are the best tourist attractions in Tokyo, Japan?","keywords":["tokyo","temple","shrine","shibuya","tourism"]},
    {"id":"q16","category":"out_scope","expected":"unanswerable","question":"How do you change the oil in a 2020 Honda Civic?","keywords":["oil","honda","civic","drain","filter","mechanic"]},
    {"id":"q17","category":"trap","expected":"partial","question":"What did the paper Attention Is All You Need say about the exact BLEU score improvement over previous models?","keywords":["bleu","attention","transformer","score","improvement"]},
    {"id":"q18","category":"trap","expected":"partial","question":"According to the papers in your knowledge base, what is the exact percentage of LLM outputs that contain hallucinations?","keywords":["percent","percentage","hallucination","rate","study"]},
    {"id":"q19","category":"trap","expected":"partial","question":"What were the names of all the researchers who invented the RAG technique and at which institution?","keywords":["lewis","perez","facebook","meta","researcher"]},
    {"id":"q20","category":"trap","expected":"partial","question":"What specific hyperparameters should I use to fine-tune BERT on a medical question answering dataset?","keywords":["learning rate","batch size","epochs","medical","hyperparameter"]},

    # ── Expansion batch (added to reach n=50; every in_scope/partial/trap item below is
    #    grounded in an actual title+abstract present in data/sample_docs/ — spot-check
    #    against the source file before trusting keyword coverage on a new corpus build) ──
    {"id":"q21","category":"in_scope","expected":"answerable","question":"What is knowledge distillation and how does TinyBERT use it to compress BERT?","keywords":["distillation","teacher","student","transformer","tinybert"]},
    {"id":"q22","category":"in_scope","expected":"answerable","question":"What is contrastive learning and how does it structure the embedding space?","keywords":["contrastive","positive","negative","embedding","pull"]},
    {"id":"q23","category":"in_scope","expected":"answerable","question":"What is differential privacy and how is it applied to fine-tuning language models?","keywords":["differential privacy","fine-tuning","noise","utility","private"]},
    {"id":"q24","category":"in_scope","expected":"answerable","question":"What is federated learning with heterogeneous client data and why is aggregation difficult?","keywords":["federated","heterogeneous","aggregation","client","data"]},
    {"id":"q25","category":"in_scope","expected":"answerable","question":"How does self-attention enable long-range coherence in music generation, as in the Music Transformer?","keywords":["self-attention","music","transformer","relative","position"]},
    {"id":"q26","category":"in_scope","expected":"answerable","question":"What is few-shot learning and what does 'partial fine-tuning' mean in that context?","keywords":["few-shot","fine-tuning","base","novel","classifier"]},
    {"id":"q27","category":"in_scope","expected":"answerable","question":"How can large language models assist in designing graph neural network architectures?","keywords":["graph neural network","large language model","architecture","design"]},
    {"id":"q28","category":"in_scope","expected":"answerable","question":"What is digital watermarking and how can it be applied to relational databases?","keywords":["watermark","database","copyright","embed","relational"]},
    {"id":"q29","category":"partial","expected":"partial","question":"What exact compression ratio and inference speedup does TinyBERT achieve compared to BERT-base?","keywords":["compression","speedup","inference","tinybert","ratio"]},
    {"id":"q30","category":"partial","expected":"partial","question":"What exact accuracy does KoreALBERT achieve on Korean NLU benchmarks compared to multilingual BERT?","keywords":["koreALBERT","accuracy","benchmark","korean","multilingual"]},
    {"id":"q31","category":"partial","expected":"partial","question":"What specific convergence rate does the federated superquantile algorithm guarantee in the strongly convex case?","keywords":["convergence","superquantile","strongly convex","rate","federated"]},
    {"id":"q32","category":"partial","expected":"partial","question":"How many parameters does EstBERT have and how long did it take to pretrain?","keywords":["estbert","parameters","pretrain","estonian","training"]},
    {"id":"q33","category":"partial","expected":"partial","question":"What privacy budget (epsilon) do the differentially private fine-tuning methods achieve on MNLI?","keywords":["epsilon","privacy budget","mnli","differential privacy"]},
    {"id":"q34","category":"partial","expected":"partial","question":"What is the exact percentage improvement contrastive chain-of-thought gives over standard chain-of-thought prompting?","keywords":["contrastive","chain-of-thought","improvement","percentage"]},
    {"id":"q35","category":"partial","expected":"partial","question":"What batch size threshold triggers stronger negative-pair separation in contrastive learning, according to the paper?","keywords":["batch size","negative pair","separation","contrastive","threshold"]},
    {"id":"q36","category":"out_scope","expected":"unanswerable","question":"What's the best workout routine for building muscle in 30 days?","keywords":["workout","muscle","exercise","routine","fitness"]},
    {"id":"q37","category":"out_scope","expected":"unanswerable","question":"How do I file my taxes as a freelancer in the United States?","keywords":["taxes","freelancer","irs","deduction","filing"]},
    {"id":"q38","category":"out_scope","expected":"unanswerable","question":"What are the symptoms of the common cold versus the flu?","keywords":["cold","flu","symptoms","fever","virus"]},
    {"id":"q39","category":"out_scope","expected":"unanswerable","question":"What's a good beginner recipe for homemade sourdough bread?","keywords":["sourdough","bread","yeast","starter","bake"]},
    {"id":"q40","category":"out_scope","expected":"unanswerable","question":"Which houseplants are safe to keep around cats?","keywords":["houseplant","cats","toxic","safe","pet"]},
    {"id":"q41","category":"out_scope","expected":"unanswerable","question":"What's the current mortgage interest rate for a 30-year fixed loan?","keywords":["mortgage","interest rate","30-year","loan","fixed"]},
    {"id":"q42","category":"out_scope","expected":"unanswerable","question":"How do I train for my first marathon in six months?","keywords":["marathon","training","running","plan","race"]},
    {"id":"q43","category":"out_scope","expected":"unanswerable","question":"What's the best way to remove a red wine stain from carpet?","keywords":["wine stain","carpet","remove","stain","cleaning"]},
    {"id":"q44","category":"trap","expected":"partial","question":"According to the TinyBERT paper, exactly how many GPU-hours were used for the two-stage distillation training?","keywords":["gpu-hours","tinybert","training","distillation","compute"]},
    {"id":"q45","category":"trap","expected":"partial","question":"What was the exact F1 score EstBERT achieved on named entity recognition, down to the decimal?","keywords":["f1 score","estbert","named entity","decimal","recognition"]},
    {"id":"q46","category":"trap","expected":"partial","question":"Which specific company or lab funded the Music Transformer research, and what was the total grant amount?","keywords":["funding","grant","music transformer","company","lab"]},
    {"id":"q47","category":"trap","expected":"partial","question":"What is the exact wall-clock training time in hours for the federated superquantile experiments in the paper?","keywords":["wall-clock","training time","hours","superquantile","federated"]},
    {"id":"q48","category":"trap","expected":"partial","question":"According to the differentially private fine-tuning paper, what exact GPU model and memory were used for all experiments?","keywords":["gpu model","memory","differentially private","hardware","experiments"]},
    {"id":"q49","category":"trap","expected":"partial","question":"What are the full names and institutional affiliations of every author on the contrastive chain-of-thought paper, in order?","keywords":["author","affiliation","institution","contrastive","chain-of-thought"]},
    {"id":"q50","category":"trap","expected":"partial","question":"What was the exact dataset size in number of examples used to pretrain KoreALBERT, down to the exact count?","keywords":["dataset size","examples","koreALBERT","pretrain","count"]},
]

CONDITIONS = [
    {"name":"baseline","label":"Baseline (No Detection)","run_entropy":False,"run_jsd":False,"run_nli":False},
    {"name":"nli_only","label":"NLI Only (Stage 3)","run_entropy":False,"run_jsd":False,"run_nli":True},
    {"name":"entropy_only","label":"Entropy Only (Stage 1)","run_entropy":True,"run_jsd":False,"run_nli":False,"skip_fast":True},
    {"name":"jsd_only","label":"JSD Only (Stage 2)","run_entropy":False,"run_jsd":True,"run_nli":False,"skip_fast":True},
    {"name":"full","label":"Full Firewall (All 3 Stages)","run_entropy":True,"run_jsd":True,"run_nli":True},
]


def score_answer(answer, expected, keywords):
    al = answer.lower()
    hedges = ["does not contain","not enough information","cannot be determined","not specified",
              "not mentioned","i don't know","no information","context does not","not provided",
              "cannot answer","don't have","not available","insufficient","not explicitly"]
    hedging = any(p in al for p in hedges)
    kw_hits = sum(1 for kw in keywords if kw.lower() in al)
    kw_cov = kw_hits/len(keywords) if keywords else 0
    if expected == "answerable":
        if hedging and kw_cov < 0.3: return {"quality":"poor","score":0.2,"reason":"Refused answerable question"}
        elif kw_cov >= 0.5:          return {"quality":"good","score":0.9,"reason":f"Covered {kw_hits}/{len(keywords)} keywords"}
        else:                         return {"quality":"partial","score":0.6,"reason":f"Low coverage {kw_cov:.0%}"}
    elif expected == "unanswerable":
        if hedging:        return {"quality":"good","score":1.0,"reason":"Correctly refused"}
        elif kw_cov > 0.4: return {"quality":"hallucinated","score":0.0,"reason":"Gave answer to out-of-scope"}
        else:              return {"quality":"partial","score":0.5,"reason":"Vague response"}
    else:
        if hedging:        return {"quality":"good","score":0.85,"reason":"Correctly hedged"}
        elif kw_cov>=0.5:  return {"quality":"hallucinated","score":0.1,"reason":"Overconfident on trap"}
        else:              return {"quality":"partial","score":0.5,"reason":"Generic answer"}


def should_flag(quality_label: str) -> bool:
    """
    Ground truth for 'the firewall should raise MEDIUM/HIGH risk'.

    This is deliberately based on whether the generated ANSWER actually
    hallucinated (quality == 'hallucinated'), not on whether the question's
    topic/category was adversarial. A correctly-hedged refusal on an
    out-of-scope or trap question is a GOOD answer and should not count as
    a missed detection -- there was nothing to catch. The earlier version of
    this eval treated every partial/out_scope/trap question as something
    that must be flagged regardless of what the model actually said, which
    is why single-stage ablations looked artificially terrible: they were
    being penalized for correctly staying quiet on answers that were fine.
    """
    return quality_label == "hallucinated"


def risk_correct(result, quality_label, cond):
    if not any([cond["run_entropy"],cond["run_jsd"],cond["run_nli"]]):
        return None
    label = result.get("risk_label","LOW")
    flagged = label in ("MEDIUM","HIGH")
    return flagged == should_flag(quality_label)


def mcnemar_test(results_a, results_b):
    ca = {r["id"]:r.get("classification_correct") for r in results_a}
    cb = {r["id"]:r.get("classification_correct") for r in results_b}
    b = sum(1 for qid in ca if ca[qid] and not cb.get(qid))
    c = sum(1 for qid in ca if not ca[qid] and cb.get(qid))
    if b+c == 0:
        return {"b":b,"c":c,"chi2":0.0,"p_value":1.0,"significant":False}
    stat = (abs(b-c)-1)**2/(b+c)
    pval = 1-chi2.cdf(stat,df=1)
    return {"b":b,"c":c,"chi2":round(stat,4),"p_value":round(pval,4),"significant":pval<0.05}


def run_with_retry(fn, max_retries=3, wait=60):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < max_retries-1:
                print(f"\n  Retry {attempt+1}/{max_retries} after {wait}s: {str(e)[:60]}")
                time.sleep(wait)
            else:
                raise


def main():
    llm_info = f"DeepSeek {DEEPSEEK_MODEL}"
    print(f"\n{'='*70}\n  RAG FIREWALL EVAL | LLM: {llm_info}\n{'='*70}")

    if not DEEPSEEK_API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY in .env (free at platform.deepseek.com)"); sys.exit(1)

    print("\nLoading FAISS index...")
    vs = load_index()
    print(f"  {vs.index.ntotal} vectors")

    active = [c for c in CONDITIONS if not (FAST_MODE and c.get("skip_fast"))]
    all_results = {c["name"]: [] for c in active}

    # Generate-once, score-per-condition: retrieve + generate the answer ONE
    # time per question, then run every condition's detection stages against
    # that SAME answer. This is what makes the ablation a controlled
    # comparison -- otherwise each condition judges its own independently
    # regenerated answer, and differences between conditions get confounded
    # with plain LLM run-to-run variance (DeepSeek isn't perfectly
    # deterministic even at temperature=0).
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        bar = "█"*int(25*i/len(EVAL_QUESTIONS)) + "░"*(25-int(25*i/len(EVAL_QUESTIONS)))
        print(f"\r  [{bar}] {i}/{len(EVAL_QUESTIONS)} {q['id']}   ", end="", flush=True)
        try:
            def gen_q():
                chunks, ret_ms = retrieve(q["question"], vs)
                context = get_context_string(chunks)
                answer, answer_latency, answer_backend = generate_answer(q["question"], context)
                metrics = compute_all_metrics(q["question"], answer, [c.page_content for c in chunks])
                return chunks, ret_ms, answer, answer_latency, answer_backend, metrics
            chunks, ret_ms, answer, answer_latency, answer_backend, metrics = run_with_retry(gen_q)
            quality = score_answer(answer, q["expected"], q["keywords"])
        except Exception as e:
            print(f"\n  WARN {q['id']} (generation): {str(e)[:60]}")
            for cond in active:
                all_results[cond["name"]].append({**q,"condition":cond["name"],"error":str(e),
                    "composite_risk":0.5,"risk_label":"UNKNOWN","quality":"error","score":0.5,
                    "classification_correct":False,"retrieval_latency_ms":0,"total_latency_ms":0,
                    "stage1_latency_ms":0,"stage2_latency_ms":0,"stage3_latency_ms":0,"answer_latency_ms":0})
            continue

        for cond in active:
            cname = cond["name"]
            try:
                def score_q():
                    return score_firewall(q["question"], chunks, answer, answer_latency, answer_backend,
                        run_entropy=cond["run_entropy"], run_jsd=cond["run_jsd"], run_nli=cond["run_nli"])
                result = run_with_retry(score_q)
                correct = risk_correct(result, quality["quality"], cond)
                all_results[cname].append({**q,"condition":cname,
                    "answer":result["answer"],"answer_backend":result.get("answer_backend","unknown"),
                    "composite_risk":result["composite_risk_score"],"risk_label":result["risk_label"],
                    "entropy_score":result["stage1_entropy"].get("score",0),
                    "jsd_score":result["stage2_jsd"].get("score",0),
                    "nli_score":result["stage3_nli"].get("score",0),
                    "stages_flagged":result["stages_flagged"],
                    "context_precision":metrics["context_precision"],
                    "answer_faithfulness":metrics["answer_faithfulness"],
                    "answer_relevancy":metrics["answer_relevancy"],
                    "retrieval_latency_ms":ret_ms,"answer_latency_ms":result["latency"]["answer_ms"],
                    "stage1_latency_ms":result["latency"]["stage1_ms"],
                    "stage2_latency_ms":result["latency"]["stage2_ms"],
                    "stage3_latency_ms":result["latency"]["stage3_ms"],
                    "total_latency_ms":result["latency"]["total_ms"],
                    **quality,"classification_correct":correct})
            except Exception as e:
                print(f"\n  WARN {q['id']}/{cname}: {str(e)[:60]}")
                all_results[cname].append({**q,"condition":cname,"error":str(e),
                    "composite_risk":0.5,"risk_label":"UNKNOWN","quality":"error","score":0.5,
                    "classification_correct":False,"retrieval_latency_ms":0,"total_latency_ms":0,
                    "stage1_latency_ms":0,"stage2_latency_ms":0,"stage3_latency_ms":0,"answer_latency_ms":0})
        time.sleep(0.5)

    print(f"\n  Done.")

    def avg(lst,k):
        v=[x[k] for x in lst if k in x and isinstance(x.get(k),(int,float))]
        return round(sum(v)/len(v),3) if v else 0
    def confusion(r):
        tp=fp=tn=fn=0
        for x in r:
            if x.get("classification_correct") is None or "quality" not in x:
                continue
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
        total = tp+fp+tn+fn
        n_positive = tp+fn  # how many real hallucinations existed to catch
        stats[cname]={"label":lbl,
            "tp":tp,"fp":fp,"tn":tn,"fn":fn,"n_positive":n_positive,
            "accuracy_pct":frac(tp+tn, total),
            "fpr":frac(fp, fp+tn),          # false-alarm rate on non-hallucinated answers
            "fnr":frac(fn, n_positive),      # miss rate on actual hallucinations (recall = 1 - fnr)
            "precision":frac(tp, tp+fp),
            "avg_faithfulness":avg(results,"answer_faithfulness"),
            "avg_precision":avg(results,"context_precision"),
            "avg_relevancy":avg(results,"answer_relevancy"),
            "avg_retrieval_ms":avg(results,"retrieval_latency_ms"),
            "avg_answer_ms":avg(results,"answer_latency_ms"),
            "avg_stage1_ms":avg(results,"stage1_latency_ms"),
            "avg_stage2_ms":avg(results,"stage2_latency_ms"),
            "avg_stage3_ms":avg(results,"stage3_latency_ms"),
            "avg_total_ms":avg(results,"total_latency_ms")}

    mcnemar={}
    if "nli_only" in all_results and "full" in all_results:
        mcnemar["nli_vs_full"]=mcnemar_test(all_results["nli_only"],all_results["full"])

    SEP="="*70
    print(f"\n\n{SEP}\n  RESULTS | {datetime.now().strftime('%Y-%m-%d %H:%M')} | {llm_info}\n{SEP}")
    print(f"\n  {'Condition':<35} {'Acc':>14} {'FPR':>14} {'FNR (miss)':>14} {'Precision':>14} {'ms':>8}")
    print(f"  {'─'*35} {'─'*14} {'─'*14} {'─'*14} {'─'*14} {'─'*8}")
    for cn in ["baseline","nli_only","entropy_only","jsd_only","full"]:
        if cn not in stats: continue
        s=stats[cn]
        print(f"  {s['label']:<35} "
              f"{fmt(s['accuracy_pct']):>14} "
              f"{fmt(s['fpr']):>14} "
              f"{fmt(s['fnr']):>14} "
              f"{fmt(s['precision']):>14} "
              f"{s['avg_total_ms']:>7.0f}ms")

    small_n = [(cn, s['n_positive']) for cn, s in stats.items() if cn != "baseline" and s.get('n_positive') is not None]
    if small_n and min(n for _, n in small_n) < 5:
        print(f"\n  ⚠ CAUTION: only {min(n for _, n in small_n)}-{max(n for _, n in small_n)} real hallucinations")
        print(f"    occurred across {len(EVAL_QUESTIONS)} questions (see n_positive per condition in the")
        print(f"    JSON). FNR/recall computed over this few positives is NOT statistically reliable —")
        print(f"    report it as a case study, not a headline rate. Run evaluate_stress.py for a")
        print(f"    higher-temperature condition that elicits more hallucinations to measure against.")

    if "full" in stats:
        s=stats["full"]; total=s["avg_total_ms"]
        print(f"\n  PER-STAGE LATENCY (Full):")
        for name,ms in [("Retrieval",s["avg_retrieval_ms"]),("Answer Gen",s["avg_answer_ms"]),
                         ("S1 Entropy",s["avg_stage1_ms"]),("S2 JSD",s["avg_stage2_ms"]),("S3 NLI",s["avg_stage3_ms"])]:
            print(f"    {name:<15} {ms:>8.1f}ms  {ms/total*100 if total else 0:>5.1f}%")
        print(f"    {'TOTAL':<15} {total:>8.1f}ms  100.0%")

    if "nli_vs_full" in mcnemar:
        m=mcnemar["nli_vs_full"]
        print(f"\n  McNEMAR (NLI-only vs Full): chi2={m['chi2']} p={m['p_value']} {'SIGNIFICANT' if m['significant'] else 'not significant'}")

    print(f"\n{SEP}")

    full=stats.get("full",{}); nli=stats.get("nli_only",{}); m=mcnemar.get("nli_vs_full",{})
    n_pos = full.get("n_positive", 0)
    bullets=[
        "RESUME BULLETS","="*70,
        f"Built RAG middleware (LangChain + FAISS) achieving {full.get('avg_retrieval_ms',0):.0f}ms avg retrieval",
        f"latency on {vs.index.ntotal} chunks from 198 arXiv AI/ML papers.",
        f"Three-stage hallucination firewall (semantic entropy, JSD, DeBERTa NLI), evaluated against",
        f"answer-level hallucination ground truth (not just question category) across {len(EVAL_QUESTIONS)} queries:",
        f"  {fmt(full.get('accuracy_pct'))} accuracy, {fmt(full.get('fpr'))} false-alarm rate on correct answers,",
        f"  {fmt(full.get('fnr'))} miss rate on the {n_pos} actual hallucinations observed",
        f"  (n={n_pos} positives -- treat as a case study pending the stress-test run, not a headline rate).",
        f"Per-stage ablation: NLI-alone {fmt(nli.get('accuracy_pct'))} vs full system {fmt(full.get('accuracy_pct'))}",
        f"  (McNemar chi2={m.get('chi2','N/A')}, p={m.get('p_value','N/A')}).",
        f"LLM: DeepSeek V4 Flash. NLI: DeBERTa-v3-small (local).",
    ]
    with open(EVAL_DIR/"eval_summary.txt","w") as f:
        f.write("\n".join(bullets))
    with open(EVAL_DIR/"eval_report.json","w") as f:
        json.dump({"timestamp":datetime.now().isoformat(),"llm":llm_info,
                   "corpus_size":vs.index.ntotal,"n_questions":len(EVAL_QUESTIONS),
                   "stats":stats,"mcnemar":mcnemar,"all_results":all_results},f,indent=2,default=str)

    print("\n".join(bullets))
    print(f"\nSaved: {EVAL_DIR}/eval_report.json")


if __name__=="__main__":
    main()
