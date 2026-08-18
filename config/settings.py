"""
config/settings.py
All tuneable parameters for the RAG Hallucination Firewall.
LLM: DeepSeek V4 Flash (api.deepseek.com)
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

for d in [DOCS_DIR, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM: DeepSeek V4 Flash ────────────────────────────────────────────────────
# Sign up free at platform.deepseek.com — 5M free tokens, no credit card
# $0.14/1M input tokens, $0.28/1M output tokens after free grant
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-v4-flash"

LLM_TEMP       = 0.0
LLM_MAX_TOKENS = 512

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 5
MMR_DIVERSITY = 0.3

# ── Hallucination Firewall ────────────────────────────────────────────────────
ENTROPY_SAMPLES   = 5
ENTROPY_TEMP      = 0.7
ENTROPY_THRESHOLD = 0.35
JSD_THRESHOLD     = 0.80  # recalibrated from 0.45 — see README "Why JSD has a high false positive rate"
NLI_MODEL         = "cross-encoder/nli-deberta-v3-small"
NLI_THRESHOLD     = 0.5

RISK_WEIGHTS = {"entropy": 0.35, "jsd": 0.20, "nli": 0.45}  # matches README-documented reweighting

# ── Evaluation ────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "query_log.jsonl"
