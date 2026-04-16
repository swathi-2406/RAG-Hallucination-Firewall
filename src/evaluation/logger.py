"""
src/evaluation/logger.py
Persists query logs to a JSONL file for the Streamlit dashboard.
Each line is one complete query run with all metrics and scores.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.settings import LOG_FILE

logger = logging.getLogger(__name__)


def log_query(
    query: str,
    answer: str,
    firewall_result: dict,
    eval_metrics: dict,
    retrieval_latency_ms: float,
) -> dict:
    """
    Build a log entry and append it to the JSONL log file.

    Returns the log entry dict.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        # Risk
        "composite_risk_score": firewall_result["composite_risk_score"],
        "risk_label": firewall_result["risk_label"],
        "stages_flagged": firewall_result["stages_flagged"],
        # Stage scores
        "entropy_score": firewall_result["stage1_entropy"].get("score", 0),
        "jsd_score": firewall_result["stage2_jsd"].get("score", 0),
        "nli_score": firewall_result["stage3_nli"].get("score", 0),
        # RAGAS-style metrics
        "context_precision": eval_metrics.get("context_precision", 0),
        "answer_faithfulness": eval_metrics.get("answer_faithfulness", 0),
        "answer_relevancy": eval_metrics.get("answer_relevancy", 0),
        # Latency
        "retrieval_latency_ms": retrieval_latency_ms,
        "answer_latency_ms": firewall_result["latency"]["answer_ms"],
        "stage1_latency_ms": firewall_result["latency"]["stage1_ms"],
        "stage2_latency_ms": firewall_result["latency"]["stage2_ms"],
        "stage3_latency_ms": firewall_result["latency"]["stage3_ms"],
        "total_latency_ms": firewall_result["latency"]["total_ms"],
        # Sources
        "sources": [c["source"] for c in firewall_result.get("chunks", [])],
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write query log: {e}")

    return entry


def load_logs(log_file: Path = LOG_FILE) -> List[dict]:
    """Load all query logs from the JSONL file."""
    if not Path(log_file).exists():
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return logs


def clear_logs(log_file: Path = LOG_FILE) -> None:
    """Clear the query log file."""
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")
    logger.info("Query log cleared.")
