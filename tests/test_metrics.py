"""Tests for musicwhy.evaluation.metrics — metric calculations."""

from __future__ import annotations

import numpy as np
import pytest

from musicwhy.evaluation.metrics import (
    bootstrap_ci,
    mc_accuracy,
    mean_judge_score,
    pearson_with_lag,
    performance_gap,
)


class TestMcAccuracy:
    def test_perfect(self):
        preds = ["A", "B", "C"]
        truth = ["A", "B", "C"]
        assert mc_accuracy(preds, truth) == 1.0

    def test_zero(self):
        preds = ["A", "A", "A"]
        truth = ["B", "B", "B"]
        assert mc_accuracy(preds, truth) == 0.0

    def test_partial(self):
        preds = ["A", "B", "C", "D"]
        truth = ["A", "B", "X", "Y"]
        assert mc_accuracy(preds, truth) == 0.5

    def test_case_insensitive(self):
        preds = ["a", "B"]
        truth = ["A", "b"]
        assert mc_accuracy(preds, truth) == 1.0

    def test_empty(self):
        assert mc_accuracy([], []) == 0.0


class TestPearsonWithLag:
    def test_perfect_correlation_lag0(self):
        a = np.arange(20, dtype=float)
        b = np.arange(20, dtype=float)
        results = pearson_with_lag(a, b, max_lag=0)
        assert 0 in results
        assert abs(results[0]["r"] - 1.0) < 1e-10
        assert results[0]["p"] < 0.001

    def test_negative_lags(self):
        a = np.random.randn(50)
        b = np.random.randn(50)
        results = pearson_with_lag(a, b, max_lag=2)
        assert -2 in results
        assert -1 in results
        assert 0 in results
        assert 1 in results
        assert 2 in results


class TestMeanJudgeScore:
    def test_basic(self):
        results = [{"score": 3}, {"score": 5}, {"score": 4}]
        assert mean_judge_score(results) == 4.0

    def test_empty(self):
        assert mean_judge_score([]) == 0.0


class TestBootstrapCi:
    def test_returns_dict(self):
        scores = np.random.rand(100).tolist()
        result = bootstrap_ci(scores, n_bootstrap=100, ci=0.95)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_single_value(self):
        result = bootstrap_ci([5.0, 5.0, 5.0], n_bootstrap=50)
        assert result["mean"] == 5.0


class TestPerformanceGap:
    def test_identical(self):
        a = [0.8, 0.7, 0.9]
        b = [0.8, 0.7, 0.9]
        result = performance_gap(a, b)
        assert result["gap"] == 0.0

    def test_significant_gap(self):
        np.random.seed(42)
        a = np.random.normal(0.8, 0.05, 100).tolist()
        b = np.random.normal(0.5, 0.05, 100).tolist()
        result = performance_gap(a, b)
        assert result["gap"] > 0
        assert "p_value" in result


class TestExtractMcAnswer:
    """The old parser took the first a/b/c/d anywhere, so prose answers were misread."""

    def test_cases(self):
        from musicwhy.evaluation.models import _extract_mc_answer

        cases = {
            "C": "C",
            "**B**": "B",
            "(D)": "D",
            "The answer is B.": "B",
            "Answer: C) horn sections": "C",
            "Based on the bright timbre, the answer is D": "D",
            "B) The percussive texture puts sharp transients up front": "B",
            "I would choose option c because the drums are loud": "C",
            "The song features a horn section, so (A) fits best.": "A",
            "Sounds like the 1980s, which is C).": "C",
            "I cannot tell from this clip.": "X",
            "": "X",
        }
        for text, want in cases.items():
            assert _extract_mc_answer(text) == want, text

    def test_last_response_is_kept(self):
        from musicwhy.evaluation.models import _extract_mc_answer, last_mc_response

        _extract_mc_answer("The answer is B.")
        assert last_mc_response() == "The answer is B."
