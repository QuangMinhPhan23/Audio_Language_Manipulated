"""
Statistical analysis utilities for MusicWhy experiments.

Functions
---------
compute_all_correlations
    Time-lagged Pearson correlations between audio-feature trends and
    economic indicators.
significance_tests
    Pairwise Wilcoxon signed-rank tests across models.
effect_sizes
    Cohen's d between two result vectors.
bootstrap_all_cis
    Non-parametric bootstrap confidence intervals per model per task.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Time-lagged correlations
# ---------------------------------------------------------------------------

def compute_all_correlations(
    feature_trends: pd.DataFrame,
    economic_data: pd.DataFrame,
    max_lag: int = 5,
    feature_cols: Optional[List[str]] = None,
    economic_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Compute time-lagged Pearson correlations between audio features and
    economic indicators.

    Parameters
    ----------
    feature_trends : pd.DataFrame
        Yearly audio-feature averages.  Must contain a ``year`` column plus
        one column per audio feature (e.g. ``avg_tempo``, ``avg_brightness``,
        ``avg_energy``).
    economic_data : pd.DataFrame
        Economic indicators indexed or containing a ``year`` column.
        Columns such as ``gdp_growth``, ``unemployment``, ``inflation``.
    max_lag : int, default 5
        Maximum lag (in years) to test in both directions.
        Negative lags mean audio leads economics; positive lags mean
        economics leads audio.
    feature_cols : list[str] | None
        Audio-feature columns to use.  ``None`` auto-detects numeric columns
        excluding ``year``.
    economic_cols : list[str] | None
        Economic columns to use.  ``None`` auto-detects numeric columns
        excluding ``year``.

    Returns
    -------
    dict[str, dict[int, tuple[float, float]]]
        Mapping ``"{feature}_vs_{indicator}"`` -> ``{lag: (r, p)}``.
    """
    # Normalise the year index ------------------------------------------------
    ft = feature_trends.copy()
    ec = economic_data.copy()

    if "year" in ft.columns:
        ft = ft.set_index("year")
    if "year" in ec.columns:
        ec = ec.set_index("year")

    ft = ft.sort_index()
    ec = ec.sort_index()

    if feature_cols is None:
        feature_cols = [c for c in ft.select_dtypes(include="number").columns]
    if economic_cols is None:
        economic_cols = [c for c in ec.select_dtypes(include="number").columns]

    results: Dict[str, Dict[int, Tuple[float, float]]] = {}

    for feat, econ in itertools.product(feature_cols, economic_cols):
        key = f"{feat}_vs_{econ}"
        lag_results: Dict[int, Tuple[float, float]] = {}
        for lag in range(-max_lag, max_lag + 1):
            # Shift the economic series by *lag* years.
            # lag > 0  =>  economic data from *lag* years earlier predicts
            #              current audio features (economy leads music).
            # lag < 0  =>  audio features from |lag| years earlier predict
            #              current economic data (music leads economy).
            shifted_econ = ec[econ].shift(lag)
            merged = pd.DataFrame({"feature": ft[feat], "econ": shifted_econ}).dropna()
            if len(merged) < 5:
                continue
            r, p = stats.pearsonr(merged["feature"], merged["econ"])
            lag_results[lag] = (float(r), float(p))
        results[key] = lag_results

    return results


# ---------------------------------------------------------------------------
# Significance tests (model comparisons)
# ---------------------------------------------------------------------------

def significance_tests(
    model_results: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Pairwise Wilcoxon signed-rank tests between models.

    Parameters
    ----------
    model_results : dict[str, np.ndarray]
        Mapping ``model_name`` -> 1-D array of per-item scores (e.g. per-
        question accuracy, 0/1 or continuous).  All arrays must have the
        same length (paired on the same questions).

    Returns
    -------
    dict[str, dict[str, tuple[float, float]]]
        Nested mapping ``model_a`` -> ``model_b`` -> ``(statistic, p_value)``.

    Notes
    -----
    Uses the two-sided Wilcoxon signed-rank test (non-parametric, paired).
    If two models produce identical scores on every item the test is
    undefined; the entry is recorded as ``(nan, 1.0)``.
    """
    models = sorted(model_results.keys())
    pairwise: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for i, ma in enumerate(models):
        pairwise[ma] = {}
        for mb in models[i + 1 :]:
            diff = np.asarray(model_results[ma]) - np.asarray(model_results[mb])
            # Wilcoxon requires at least one nonzero difference.
            if np.all(diff == 0):
                pairwise[ma][mb] = (float("nan"), 1.0)
                continue
            stat, p = stats.wilcoxon(
                model_results[ma], model_results[mb], alternative="two-sided"
            )
            pairwise[ma][mb] = (float(stat), float(p))

    return pairwise


# ---------------------------------------------------------------------------
# Effect size (Cohen's d)
# ---------------------------------------------------------------------------

def effect_sizes(
    results_a: Union[np.ndarray, Sequence[float]],
    results_b: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Cohen's d (pooled standard deviation) between two samples.

    Parameters
    ----------
    results_a, results_b : array-like
        Score arrays for two conditions / models.

    Returns
    -------
    float
        Cohen's d.  Positive when ``results_a`` > ``results_b`` on average.
    """
    a = np.asarray(results_a, dtype=float)
    b = np.asarray(results_b, dtype=float)

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Each sample must contain at least 2 observations.")

    mean_diff = float(np.mean(a) - np.mean(b))
    # Pooled standard deviation (Hedges-style denominator).
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))

    if pooled_std == 0.0:
        return 0.0

    return mean_diff / pooled_std


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_all_cis(
    results: Dict[str, Dict[str, np.ndarray]],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Non-parametric bootstrap confidence intervals for each model x task.

    Parameters
    ----------
    results : dict[str, dict[str, np.ndarray]]
        ``{model_name: {task_name: scores_array}}``.
    n_boot : int, default 1000
        Number of bootstrap resamples.
    ci : float, default 0.95
        Confidence level (e.g. 0.95 for a 95 % CI).
    seed : int | None, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, tuple[float, float, float]]]
        ``{model: {task: (point_estimate, ci_lower, ci_upper)}}``.
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci

    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for model, tasks in results.items():
        out[model] = {}
        for task, scores in tasks.items():
            arr = np.asarray(scores, dtype=float)
            n = len(arr)
            point = float(np.mean(arr))

            boot_means = np.empty(n_boot)
            for b in range(n_boot):
                sample = rng.choice(arr, size=n, replace=True)
                boot_means[b] = np.mean(sample)

            lower = float(np.percentile(boot_means, 100 * alpha / 2))
            upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
            out[model][task] = (point, lower, upper)

    return out
