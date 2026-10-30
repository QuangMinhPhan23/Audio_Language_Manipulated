"""Tests for musicwhy.analysis.statistics — statistical analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicwhy.analysis.statistics import (
    bootstrap_all_cis,
    compute_all_correlations,
    effect_sizes,
    significance_tests,
)


@pytest.fixture
def feature_trends():
    np.random.seed(42)
    years = list(range(1960, 2020))
    return pd.DataFrame({
        "year": years,
        "avg_tempo": np.random.normal(120, 10, len(years)),
        "avg_brightness": np.random.normal(2500, 500, len(years)),
    })


@pytest.fixture
def economic_data():
    np.random.seed(42)
    years = list(range(1960, 2020))
    return pd.DataFrame({
        "year": years,
        "gdp_growth": np.random.normal(2.5, 1.5, len(years)),
        "unemployment": np.random.normal(6.0, 2.0, len(years)),
    })


class TestComputeAllCorrelations:
    def test_returns_dict(self, feature_trends, economic_data):
        results = compute_all_correlations(feature_trends, economic_data, max_lag=2)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_lag_range(self, feature_trends, economic_data):
        results = compute_all_correlations(feature_trends, economic_data, max_lag=3)
        for key, lag_dict in results.items():
            lags = set(lag_dict.keys())
            assert -3 in lags
            assert 0 in lags
            assert 3 in lags

    def test_r_and_p_values(self, feature_trends, economic_data):
        results = compute_all_correlations(feature_trends, economic_data, max_lag=1)
        for key, lag_dict in results.items():
            for lag, (r, p) in lag_dict.items():
                assert -1.0 <= r <= 1.0
                assert 0.0 <= p <= 1.0


class TestSignificanceTests:
    def test_returns_dict(self):
        model_results = {
            "model_a": [0.8, 0.7, 0.9, 0.85, 0.75],
            "model_b": [0.6, 0.5, 0.7, 0.65, 0.55],
            "model_c": [0.7, 0.6, 0.8, 0.75, 0.65],
        }
        results = significance_tests(model_results)
        assert isinstance(results, dict)
        assert len(results) == 3  # 3 pairs

    def test_identical_models(self):
        scores = [0.8, 0.7, 0.9, 0.85, 0.75]
        model_results = {"a": scores, "b": scores.copy()}
        results = significance_tests(model_results)
        for model_a, inner in results.items():
            for model_b, (stat, p) in inner.items():
                assert p >= 0.05 or np.isnan(p)


class TestEffectSizes:
    def test_zero_effect(self):
        a = [0.5] * 20
        b = [0.5] * 20
        d = effect_sizes(a, b)
        assert abs(d) < 0.01

    def test_large_effect(self):
        np.random.seed(42)
        a = np.random.normal(0.8, 0.1, 50).tolist()
        b = np.random.normal(0.3, 0.1, 50).tolist()
        d = effect_sizes(a, b)
        assert d > 1.0


class TestBootstrapAllCis:
    def test_structure(self):
        results = {
            "model_a": {"task_1": [0.8, 0.7, 0.9], "task_2": [0.6, 0.5, 0.7]},
            "model_b": {"task_1": [0.6, 0.5, 0.7], "task_2": [0.4, 0.3, 0.5]},
        }
        cis = bootstrap_all_cis(results, n_boot=100)
        assert "model_a" in cis
        assert "task_1" in cis["model_a"]
        ci_tuple = cis["model_a"]["task_1"]
        assert len(ci_tuple) >= 2
        assert ci_tuple[0] <= ci_tuple[-1]
