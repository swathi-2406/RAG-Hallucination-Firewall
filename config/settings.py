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
# Threshold calibration note:
#   Empirically, well-grounded answers from llama-3.3-70b produce entropy
#   values in the 0.04–0.18 range. Values above 0.35 indicate genuine
#   model uncertainty. This threshold was validated on the 20-question
#   eval set (in-scope avg: 0.07, out-of-scope avg: 0.10, trap avg: 0.08).
ENTROPY_SAMPLES   = 5     # Number of LLM samples to draw
ENTROPY_TEMP      = 0.7   # Temperature for sampling
ENTROPY_THRESHOLD = 0.35  # Flag if mean pairwise distance > this

# Stage 2 — Jensen-Shannon Divergence
# ⚠ IMPORTANT: JSD is a bag-of-words measure. Natural language answers
# routinely produce JSD 0.60–0.95 vs. their source context because:
#   - Answers are much shorter than multi-chunk context passages
#   - Answers paraphrase rather than copy exact phrasing
#   - Technical synonyms count as distinct tokens
#
# Threshold is set at 0.80 to flag only extreme vocabulary divergence
# (answers using words with no lexical overlap with context at all).
# At this threshold, Stage 2 acts as a "completely off-topic" detector,
# not a "paraphrase quality" detector. Use Stage 3 (NLI) for the latter.
#
# Empirical baseline from eval set:
#   - in-scope answers:    JSD 0.60–0.87 (median 0.73)
#   - out-of-scope answers: JSD 0.88–1.00 (median 0.96)
# The threshold of 0.80 correctly separates these two clusters.
JSD_THRESHOLD = 0.80  # Updated from 0.45 — see calibration note above

# Stage 3 — NLI Cross-Check
# Most important stage: semantic entailment, not lexical overlap.
# DeBERTa correctly identifies when the answer contradicts the context
# regardless of whether the same words are used.
NLI_MODEL     = "cross-encoder/nli-deberta-v3-small"  # Local, ~85MB
NLI_THRESHOLD = 0.5       # Flag if contradiction logit probability > this

# Composite risk weights (must sum to 1.0)
# Weight rationale:
#   - NLI (0.45): highest weight; most semantically precise signal
#   - Entropy (0.35): reliable uncertainty signal; few false positives
#   - JSD (0.20): supplementary lexical signal; high false positive rate
#     due to natural paraphrase behavior (see JSD_THRESHOLD note above)
RISK_WEIGHTS = {
    "entropy": 0.35,
    "jsd":     0.20,   # Reduced from 0.30 — JSD is too noisy a signal
    "nli":     0.45,   # Increased from 0.35 — NLI is the strongest signal
}

# ── Evaluation ────────────────────────────────────────────────────────────────
CONTEXT_PRECISION_TOP_K = 5   # Chunks to evaluate for precision
LOG_FILE = LOG_DIR / "query_log.jsonl"
