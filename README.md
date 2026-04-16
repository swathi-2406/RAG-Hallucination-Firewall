# 🔥 RAG Hallucination Firewall — Control-Dial System

A production-grade RAG (Retrieval-Augmented Generation) middleware with a **three-stage hallucination detection pipeline**, built with LangChain, FAISS, and Groq. Features a real-time Streamlit evaluation dashboard tracking GenAI quality metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Free)-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎬 Demo
![RAG Hallucination Firewall Demo](assets\demo.gif)

[![RAG Hallucination Firewall Demo](https://img.youtube.com/vi/tsqRcIyXGTw/maxresdefault.jpg)](https://youtu.be/tsqRcIyXGTw)

*Click to watch the full pipeline demo — live query, risk gauge, three-stage firewall, and evaluation dashboard*

---

## 🖼️ Screenshots

### Query Interface — Live Result with Risk Gauge
![Query Interface](assests\screenshots\query_interface.png)

### Firewall Stages — PASS/FLAG Badges & Latency Breakdown
![Firewall Stages](assets\screenshots\firewall_stages.png)

### Evaluation Dashboard — 6 Real-Time Charts
![Evaluation Dashboard](assets\screenshots\dashboard.png)

### Query Logs — Full History with JSON Drill-Down
![Query Logs](assets\screenshots\query_logs.png)

---

## 📊 Evaluation Results

Evaluated across **20 questions** spanning 4 adversarial categories on a 198-paper arXiv AI/ML corpus:

| Metric | Result |
|---|---|
| **Retrieval Latency** | **162ms** avg (target: <200ms ✅) |
| **Risk Classification Accuracy** | **100%** — correct LOW/MEDIUM/HIGH on all 20 queries |
| **False Positive Rate** | **0%** — no well-grounded answers wrongly flagged |
| **Out-of-Scope Refusal Rate** | **100%** — model correctly refused all irrelevant queries |
| **Answer Relevancy** | **0.825** avg |
| **Answer Faithfulness** | **0.563** avg |
| **Context Precision** | **0.414** avg (abstracts-only corpus; full papers would push >0.70) |

> **Note on context precision:** The corpus uses paper *abstracts* only, which limits retrieval depth on technical mechanistic questions. Ingesting full papers would significantly improve this metric.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE                                  │
│  RecursiveCharacterTextSplitter (512 chars, 50 overlap)
│  all-MiniLM-L6-v2 embeddings (384-dim, local)        │
│  FAISS IndexFlatIP + MMR (λ=0.3) — sub-200ms         │
└──────────────────────────┬───────────────────────────┘
                           │  Top-K Chunks (MMR)
                           ▼
┌──────────────────────────────────────────────────────┐
│  LLM GENERATION                                      │
│  Groq API · llama-3.3-70b-versatile · temp=0.0       │
└──────────────────────────┬───────────────────────────┘
                           │  Deterministic Answer
                           ▼
┌──────────────────────────────────────────────────────┐
│  THREE-STAGE HALLUCINATION FIREWALL                  │
│                                                      │
│  Stage 1 ── Semantic Entropy                         │
│             5 LLM samples at temp=0.7                │
│             mean pairwise cosine distance            │
│             threshold: 0.35                          │
│                                                      │
│  Stage 2 ── Jensen-Shannon Divergence                │
│             token frequency distributions            │
│             context vs answer vocabulary             │
│             threshold: 0.45                          │
│                                                      │
│  Stage 3 ── DeBERTa NLI Cross-Check                  │
│             cross-encoder/nli-deberta-v3-small       │
│             max contradiction prob across chunks     │
│             threshold: 0.50                          │
│                                                      │
│  composite = 0.35·S1 + 0.30·S2 + 0.35·S3            │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
       🟢 LOW (0–0.3) · 🟡 MEDIUM (0.3–0.6) · 🔴 HIGH (0.6–1.0)
            + Streamlit Dashboard · Query Logs · RAGAS Metrics
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Chunking** | `RecursiveCharacterTextSplitter` (512 chars, 50 overlap, hierarchy: paragraph→sentence→word→char) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — local, free, L2-normalized 384-dim |
| **Vector Store** | FAISS `IndexFlatIP` + MMR retrieval (4× candidate fetch, λ=0.3) |
| **LLM** | Groq API — `llama-3.3-70b-versatile` (free tier) |
| **Stage 1** | Semantic entropy: mean pairwise cosine distance across N=5 stochastic samples |
| **Stage 2** | Jensen-Shannon divergence between context and answer token distributions |
| **Stage 3** | `cross-encoder/nli-deberta-v3-small` NLI — max contradiction prob across chunks |
| **Composite Score** | Weighted: 35% entropy + 30% JSD + 35% NLI |
| **Dashboard** | 3-tab Streamlit: Query, Dashboard (6 charts), Query Logs |
| **Evaluation** | Automated 20-question eval script with 4 adversarial categories |
| **Tests** | 37 unit tests, 100% passing |
| **Zero Cost** | All models run locally; only Groq API call is external (free tier) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (Anaconda recommended on Windows)
- Free Groq API key from [console.groq.com](https://console.groq.com) — no credit card needed

### 1. Clone & Install

```bash
git clone https://github.com/swathi-2406/rag-hallucination-firewall.git
cd rag-hallucination-firewall
```

**Windows (Anaconda — recommended):**
```powershell
conda activate base

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install langchain langchain-community langchain-groq langchain-text-splitters langchain-core langchain-huggingface faiss-cpu sentence-transformers transformers scipy numpy ragas datasets arxiv pypdf streamlit plotly pandas python-dotenv pydantic tenacity tqdm
```

**Mac/Linux:**
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
# Edit .env and add your GROQ_API_KEY
```

### 3. Ingest Documents

```bash
python scripts/ingest_docs.py
```

Downloads ~200 AI/ML paper abstracts from arXiv and builds the FAISS index.
Takes 3–5 minutes on first run (downloads embedding model ~90MB, cached after).

### 4. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. (Optional) Run Automated Evaluation

```bash
python scripts/evaluate_firewall.py
```

Runs 20 questions through the pipeline twice (firewall on vs off) and generates resume-ready metrics in `data/evaluation/eval_summary.txt`. Takes ~20 minutes.

---

## 📁 Project Structure

```
rag-hallucination-firewall/
├── app.py                              # Streamlit dashboard (3 tabs)
├── requirements.txt
├── .env.example
├── setup.bat                           # Windows one-click setup
├── pytest.ini
├── config/
│   └── settings.py                     # All tuneable parameters
├── src/
│   ├── retrieval/
│   │   ├── chunker.py                  # RecursiveCharacterTextSplitter
│   │   ├── embedder.py                 # all-MiniLM-L6-v2 singleton
│   │   └── retriever.py                # FAISS + MMR, latency tracking
│   ├── hallucination/
│   │   ├── firewall.py                 # Orchestrates all 3 stages
│   │   ├── stage1_entropy.py           # Semantic entropy scoring
│   │   ├── stage2_jsd.py               # Jensen-Shannon divergence
│   │   └── stage3_nli.py               # DeBERTa NLI cross-check
│   └── evaluation/
│       ├── metrics.py                  # Context precision, faithfulness, relevancy
│       └── logger.py                   # JSONL query log persistence
├── scripts/
│   ├── ingest_docs.py                  # arXiv download + FAISS index build
│   └── evaluate_firewall.py            # Automated 20-question evaluation
├── assets/
│   └── screenshots/                    # README screenshots
├── data/
│   ├── sample_docs/                    # Auto-populated by ingest_docs.py
│   ├── faiss_index/                    # Auto-generated FAISS index
│   └── evaluation/                     # eval_report.json + eval_summary.txt
└── tests/
    ├── test_retrieval.py               # Chunker, embedder, retriever tests
    └── test_hallucination.py           # Firewall, metrics, logger tests
```

---

## 💡 How the Hallucination Firewall Works

### Stage 1 — Semantic Entropy
Samples N=5 outputs from the LLM at temperature=0.7. Embeds each using the same sentence-transformer. Computes mean pairwise cosine distance:

```
entropy = mean({ 1 - cos_sim(e_i, e_j) | i < j })
```

Near 0 = model is consistent → low risk. Near 1 = outputs diverge → high uncertainty → hallucination risk. Inspired by *Semantic Entropy* (Farquhar et al., 2023).

### Stage 2 — Jensen-Shannon Divergence
Tokenizes context and answer (stopwords removed, Laplace smoothed). Computes JSD between token frequency distributions:

```
JSD(P||Q) = ½·KL(P||M) + ½·KL(Q||M),   M = ½(P+Q)
```

Symmetric, bounded [0,1], always finite. High JSD = answer uses vocabulary not grounded in the retrieved context.

### Stage 3 — NLI Cross-Check
Runs `cross-encoder/nli-deberta-v3-small` (~85MB, runs locally) on each (context chunk, answer) pair. Takes the **maximum** contradiction probability:

```
Stage3_score = max(P(CONTRADICTION | chunk_i, answer))
```

Catches answers that are fluent and confident but directly contradict retrieved evidence.

### Composite Risk Score
```
risk = 0.35 × entropy + 0.30 × JSD + 0.35 × NLI_contradiction
```

| Band | Score | Meaning |
|---|---|---|
| 🟢 LOW | 0.00 – 0.30 | Answer is grounded and consistent |
| 🟡 MEDIUM | 0.30 – 0.60 | Uncertainty detected; review recommended |
| 🔴 HIGH | 0.60 – 1.00 | High hallucination probability |

---

## 📊 Dashboard

**Query Tab**
- Answer display with source attribution
- Risk gauge dial (composite score 0–100%)
- Stage score bar chart with threshold reference lines
- RAGAS metric cards (context precision, faithfulness, latency)
- Firewall PASS/FLAG badges per stage
- Expandable: latency breakdown, retrieved chunks, entropy samples

**Dashboard Tab**
- KPI row: total queries, avg risk, flagged rate, avg faithfulness, avg latency
- Risk score over time (line chart with LOW/MEDIUM threshold lines)
- RAGAS metrics over time (context precision, faithfulness, relevancy)
- Average stage scores bar chart
- Risk label distribution pie chart
- Per-stage latency averages bar chart

**Query Logs Tab**
- Sortable history with progress bar columns
- Per-entry drill-down with JSON score breakdown
- Clear log button

---

## ⚙️ Configuration

All parameters in `config/settings.py`:

```python
# Retrieval
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K = 5                 # chunks to retrieve
MMR_DIVERSITY = 0.3       # 0=max diversity, 1=max relevance

# LLM
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMP = 0.0            # deterministic final answer

# Stage 1 — Entropy
ENTROPY_SAMPLES = 5       # stochastic samples
ENTROPY_TEMP = 0.7        # sampling temperature
ENTROPY_THRESHOLD = 0.35  # flag above this

# Stage 2 — JSD
JSD_THRESHOLD = 0.45      # flag above this

# Stage 3 — NLI
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_THRESHOLD = 0.50      # flag above this

# Composite weights (must sum to 1.0)
RISK_WEIGHTS = {"entropy": 0.35, "jsd": 0.30, "nli": 0.35}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

37 unit tests across 9 test classes, 100% passing:

| Class | What Is Tested |
|---|---|
| `TestChunker` | Splitting, metadata preservation, overlap |
| `TestEmbedder` | Dimensionality, normalization, similarity ordering, singleton |
| `TestRetriever` | Context string formatting, separators |
| `TestSemanticEntropy` | Identical=low entropy, diverse=high, bounded [0,1] |
| `TestJensenShannon` | Zero JSD identity, symmetry, stopword removal, debug info |
| `TestNLI` | Structure validation, high/low contradiction (mocked) |
| `TestCompositeRisk` | Weight correctness, boundary values, risk label thresholds |
| `TestEvaluationMetrics` | Score ranges, grounded vs unrelated answer scoring |
| `TestLogger` | Log/load/clear JSONL with temporary path monkeypatching |

---

## 🐛 Known Issues & Fixes

### Windows — PyTorch won't install from pip
PyTorch is not on standard PyPI. Use the official CPU wheel:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### LangChain 1.x import changes
Imports moved in LangChain 1.x. If you see `ModuleNotFoundError`:
```python
# Old (breaks in LangChain 1.x)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Correct
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### Groq model decommissioned
`llama3-8b-8192` was removed. Update `config/settings.py`:
```python
GROQ_MODEL = "llama-3.3-70b-versatile"
```

### FAISS saves as folder not files
FAISS 1.12+ saves the index as a folder (`faiss_store/`). Check `Path(INDEX_PATH).exists()` not `Path(INDEX_PATH + ".faiss").exists()`.

### Plotly alpha hex colors
Plotly rejects 8-digit hex colors. Use `rgba()`:
```python
"rgba(34,197,94,0.09)"   # instead of "#22c55e18"
```

---

## 📈 Evaluation Methodology

`scripts/evaluate_firewall.py` runs a structured evaluation across 4 question categories:

| Category | Count | Description |
|---|---|---|
| `in_scope` | 8 | Well-covered topics (RAG, transformers, BERT, GNNs, diffusion models...) |
| `partial` | 4 | Partially covered (specific benchmark numbers, parameter counts...) |
| `out_scope` | 4 | Completely irrelevant (recipes, sports, tourism, car maintenance...) |
| `trap` | 4 | Adversarial leading questions designed to induce hallucination |

Each question runs in both **firewall-enabled** and **baseline** conditions. Results saved to:
- `data/evaluation/eval_report.json` — full per-question results
- `data/evaluation/eval_summary.txt` — pre-written resume bullets with your actual numbers

**Key finding from evaluation:** Abstract-only corpus limits context precision (0.414). Technical "how does X work" questions retrieve topically-related but shallow chunks. Full-paper ingestion is the primary improvement lever.

---

## 🗺️ Roadmap

- [ ] Async parallel entropy sampling (cut Stage 1 from ~5s to ~1s)
- [ ] Cross-encoder reranking before firewall (target context precision >0.70)
- [ ] Full-paper PDF ingestion for richer retrieval
- [ ] UMAP embedding space visualization in dashboard
- [ ] Answer citation: map each sentence back to its source chunk
- [ ] FastAPI REST wrapper for programmatic access
- [ ] Docker container for reproducible deployment

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | LangChain 1.x | Pipeline, document loaders |
| Text Splitting | langchain-text-splitters | RecursiveCharacterTextSplitter |
| Vector Store | FAISS CPU 1.12+ | In-memory ANN, no server needed |
| Embeddings | sentence-transformers | Local MiniLM, 384-dim vectors |
| LLM | Groq API (free) | llama-3.3-70b-versatile |
| NLI Model | DeBERTa-v3-small | Local cross-encoder, ~85MB |
| Statistics | scipy + numpy | JSD computation |
| Dashboard | Streamlit 1.38+ | Interactive web UI |
| Charts | Plotly 5.x | Gauge, bar, line, pie charts |
| Data | pandas | Query log aggregation |
| Ingestion | arxiv 3.0 | Free arXiv API, no key needed |
| Testing | pytest | Unit tests + monkeypatching |

---

## 📄 License

MIT

---

## 🙏 Acknowledgements

- [Farquhar et al. (2023)](https://arxiv.org/abs/2302.09664) — Semantic Entropy paper
- [LangChain](https://github.com/langchain-ai/langchain) — Pipeline orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [Groq](https://console.groq.com) — Free LLM inference
- [arXiv](https://arxiv.org) — Free research paper corpus