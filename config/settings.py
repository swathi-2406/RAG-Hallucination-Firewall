"""
config/settings.py
All tuneable parameters for the RAG Hallucination Firewall.
Edit these values to experiment with different configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
INDEX_DIR = DATA_DIR / "faiss_index"
LOG_DIR   = DATA_DIR / "query_logs"

# Ensure dirs exist
for d in [DOCS_DIR, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM (Groq) ────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"   # Free-tier model, very fast
LLM_TEMP      = 0.0                # Deterministic for final answer
LLM_MAX_TOKENS = 512

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local, free, fast

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters (approx 128 tokens for MiniLM)
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 5     # Number of chunks to retrieve
MMR_DIVERSITY = 0.3   # MMR lambda: 0 = max diversity, 1 = max relevance

# ── Hallucination Firewall ────────────────────────────────────────────────────

# Stage 1 — Semantic Entropy
ENTROPY_SAMPLES   = 5     # Number of LLM samples to draw
ENTROPY_TEMP      = 0.7   # Temperature for sampling
ENTROPY_THRESHOLD = 0.35  # Flag if mean pairwise distance > this

# Stage 2 — Jensen-Shannon Divergence
JSD_THRESHOLD = 0.45      # Flag if JSD > this (range 0–1)

# Stage 3 — NLI Cross-Check
NLI_MODEL     = "cross-encoder/nli-deberta-v3-small"  # Local, ~85MB
NLI_THRESHOLD = 0.5       # Flag if contradiction logit probability > this

# Composite risk weights (must sum to 1.0)
RISK_WEIGHTS = {
    "entropy": 0.35,
    "jsd":     0.30,
    "nli":     0.35,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
CONTEXT_PRECISION_TOP_K = 5   # Chunks to evaluate for precision
LOG_FILE = LOG_DIR / "query_log.jsonl"
