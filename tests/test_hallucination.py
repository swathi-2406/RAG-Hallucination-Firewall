"""
tests/test_hallucination.py
Unit tests for the three-stage hallucination detection pipeline.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ── Stage 1: Semantic Entropy ─────────────────────────────────────────────────

class TestSemanticEntropy:
    def test_identical_outputs_low_entropy(self):
        from src.hallucination.stage1_entropy import compute_semantic_entropy
        outputs = ["The transformer uses multi-head attention."] * 5
        score = compute_semantic_entropy(outputs)
        assert score < 0.05, "Identical outputs should have near-zero entropy"

    def test_diverse_outputs_high_entropy(self):
        from src.hallucination.stage1_entropy import compute_semantic_entropy
        outputs = [
            "Transformers use self-attention mechanisms.",
            "The sky is blue because of Rayleigh scattering.",
            "Python is a programming language.",
            "The capital of France is Paris.",
            "Machine learning requires large datasets.",
        ]
        score = compute_semantic_entropy(outputs)
        assert score > 0.1, "Diverse outputs should have higher entropy"

    def test_entropy_bounded(self):
        from src.hallucination.stage1_entropy import compute_semantic_entropy
        outputs = [
            "RAG is great.", "Hallucinations are bad.",
            "BERT is a language model.", "Attention helps transformers.",
        ]
        score = compute_semantic_entropy(outputs)
        assert 0.0 <= score <= 1.0, "Entropy score should be in [0, 1]"

    def test_single_output(self):
        from src.hallucination.stage1_entropy import compute_semantic_entropy
        score = compute_semantic_entropy(["Only one answer."])
        assert score == 0.0

    def test_empty_outputs(self):
        from src.hallucination.stage1_entropy import compute_semantic_entropy
        score = compute_semantic_entropy([])
        assert score == 0.0


# ── Stage 2: Jensen-Shannon Divergence ───────────────────────────────────────

class TestJensenShannon:
    def test_identical_text_zero_jsd(self):
        from src.hallucination.stage2_jsd import compute_jsd
        text = "retrieval augmented generation combines documents with language models"
        score, _ = compute_jsd(text, text)
        assert score < 0.05, "Identical texts should have near-zero JSD"

    def test_completely_different_high_jsd(self):
        from src.hallucination.stage2_jsd import compute_jsd
        context = "transformers use attention mechanisms neural network architecture encoder decoder"
        answer = "sunshine rainbows butterflies meadows springtime flowers gardens picnic"
        score, _ = compute_jsd(context, answer)
        assert score > 0.3, "Completely different texts should have high JSD"

    def test_jsd_bounded(self):
        from src.hallucination.stage2_jsd import compute_jsd
        score, _ = compute_jsd("RAG helps reduce hallucinations", "The model hallucinated facts")
        assert 0.0 <= score <= 1.0

    def test_jsd_symmetric(self):
        from src.hallucination.stage2_jsd import compute_jsd
        a = "language model pretraining fine-tuning transfer learning"
        b = "attention mechanism encoder decoder architecture"
        score_ab, _ = compute_jsd(a, b)
        score_ba, _ = compute_jsd(b, a)
        assert abs(score_ab - score_ba) < 0.01, "JSD should be approximately symmetric"

    def test_tokenizer_removes_stopwords(self):
        from src.hallucination.stage2_jsd import tokenize, STOPWORDS
        tokens = tokenize("the quick brown fox jumps over the lazy dog")
        for t in tokens:
            assert t not in STOPWORDS

    def test_debug_info_returned(self):
        from src.hallucination.stage2_jsd import compute_jsd
        _, debug = compute_jsd("neural network training", "backpropagation gradient descent")
        assert "vocab_size" in debug
        assert "context_tokens" in debug
        assert "answer_tokens" in debug

    def test_run_stage2_returns_dict(self):
        from src.hallucination.stage2_jsd import run_stage2
        result = run_stage2("RAG retrieval augmented generation", "The answer involves retrieval")
        assert "score" in result
        assert "flagged" in result
        assert "latency_ms" in result
        assert isinstance(result["flagged"], bool)


# ── Stage 3: NLI (lightweight test — no model download) ──────────────────────

class TestNLI:
    def test_run_stage3_structure(self, monkeypatch):
        """Test that run_stage3 returns the correct structure (mocked NLI)."""
        from src.hallucination import stage3_nli

        # Mock the NLI pipeline to avoid downloading the model in CI
        def mock_contradiction(chunk, answer):
            return 0.1  # Always return low contradiction

        monkeypatch.setattr(stage3_nli, "_get_contradiction_prob", mock_contradiction)

        result = stage3_nli.run_stage3(
            ["Transformers use attention.", "BERT is pretrained on masked LM."],
            "Transformers are based on attention mechanisms.",
        )
        assert "score" in result
        assert "flagged" in result
        assert "per_chunk_scores" in result
        assert "latency_ms" in result
        assert len(result["per_chunk_scores"]) == 2

    def test_high_contradiction_flagged(self, monkeypatch):
        from src.hallucination import stage3_nli
        from config.settings import NLI_THRESHOLD

        def mock_high_contradiction(chunk, answer):
            return 0.9

        monkeypatch.setattr(stage3_nli, "_get_contradiction_prob", mock_high_contradiction)

        result = stage3_nli.run_stage3(["Any context"], "Any answer")
        assert result["flagged"] is True
        assert result["score"] > NLI_THRESHOLD

    def test_low_contradiction_passes(self, monkeypatch):
        from src.hallucination import stage3_nli

        def mock_low_contradiction(chunk, answer):
            return 0.05

        monkeypatch.setattr(stage3_nli, "_get_contradiction_prob", mock_low_contradiction)

        result = stage3_nli.run_stage3(["Context chunk"], "Consistent answer")
        assert result["flagged"] is False


# ── Composite Risk ────────────────────────────────────────────────────────────

class TestCompositeRisk:
    def test_all_zero_is_zero(self):
        from src.hallucination.firewall import compute_composite_risk
        assert compute_composite_risk(0.0, 0.0, 0.0) == 0.0

    def test_all_one_is_one(self):
        from src.hallucination.firewall import compute_composite_risk
        score = compute_composite_risk(1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-9

    def test_weighted_correctly(self):
        from src.hallucination.firewall import compute_composite_risk
        from config.settings import RISK_WEIGHTS
        expected = (
            RISK_WEIGHTS["entropy"] * 0.5
            + RISK_WEIGHTS["jsd"] * 0.5
            + RISK_WEIGHTS["nli"] * 0.5
        )
        assert abs(compute_composite_risk(0.5, 0.5, 0.5) - expected) < 1e-9

    def test_risk_label_boundaries(self):
        from src.hallucination.firewall import risk_label
        assert risk_label(0.0)  == "LOW"
        assert risk_label(0.29) == "LOW"
        assert risk_label(0.30) == "MEDIUM"
        assert risk_label(0.59) == "MEDIUM"
        assert risk_label(0.60) == "HIGH"
        assert risk_label(1.0)  == "HIGH"


# ── Evaluation Metrics ────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_context_precision_range(self):
        from src.evaluation.metrics import context_precision
        score = context_precision(
            "What is RAG?",
            ["RAG retrieval augmented generation", "Unrelated content about cooking"],
        )
        assert 0.0 <= score <= 1.0

    def test_answer_faithfulness_high_when_grounded(self):
        from src.evaluation.metrics import answer_faithfulness
        context = ["Retrieval augmented generation reduces hallucinations by grounding answers in documents."]
        answer = "RAG reduces hallucinations by grounding LLM answers in retrieved documents."
        score = answer_faithfulness(answer, context)
        assert score > 0.6, "Grounded answer should have high faithfulness"

    def test_answer_faithfulness_low_when_unrelated(self):
        from src.evaluation.metrics import answer_faithfulness
        context = ["Quantum computing uses qubits for superposition-based computation."]
        answer = "The recipe calls for two cups of flour and one egg."
        score = answer_faithfulness(answer, context)
        assert score < 0.5, "Unrelated answer should have low faithfulness"

    def test_answer_relevancy_range(self):
        from src.evaluation.metrics import answer_relevancy
        score = answer_relevancy("What is BERT?", "BERT is a transformer-based language model.")
        assert 0.0 <= score <= 1.0

    def test_compute_all_metrics_returns_all_keys(self):
        from src.evaluation.metrics import compute_all_metrics
        result = compute_all_metrics(
            "What is attention?",
            "Attention allows the model to focus on relevant tokens.",
            ["Attention mechanisms weigh token importance in transformers."],
        )
        assert "context_precision" in result
        assert "answer_faithfulness" in result
        assert "answer_relevancy" in result


# ── Logger ────────────────────────────────────────────────────────────────────

class TestLogger:
    def test_log_and_load(self, tmp_path, monkeypatch):
        from src.evaluation import logger as log_module
        test_log = tmp_path / "test_log.jsonl"
        monkeypatch.setattr(log_module, "LOG_FILE", test_log)

        fake_firewall = {
            "composite_risk_score": 0.25,
            "risk_label": "LOW",
            "stages_flagged": 0,
            "stage1_entropy": {"score": 0.1, "latency_ms": 100},
            "stage2_jsd": {"score": 0.2, "latency_ms": 5},
            "stage3_nli": {"score": 0.3, "latency_ms": 800},
            "latency": {"answer_ms": 200, "stage1_ms": 100, "stage2_ms": 5, "stage3_ms": 800, "total_ms": 1200},
            "chunks": [{"source": "paper.txt", "content": "text"}],
        }
        fake_metrics = {"context_precision": 0.8, "answer_faithfulness": 0.9, "answer_relevancy": 0.85}

        log_module.log_query("Test query?", "Test answer.", fake_firewall, fake_metrics, 50.0)
        logs = log_module.load_logs(test_log)

        assert len(logs) == 1
        assert logs[0]["query"] == "Test query?"
        assert logs[0]["risk_label"] == "LOW"

    def test_clear_logs(self, tmp_path, monkeypatch):
        from src.evaluation import logger as log_module
        test_log = tmp_path / "test_log.jsonl"
        test_log.write_text('{"query": "old"}\n')
        monkeypatch.setattr(log_module, "LOG_FILE", test_log)

        log_module.clear_logs(test_log)
        logs = log_module.load_logs(test_log)
        assert logs == []
