"""Metric calculations for MusicWhy evaluation.

Includes accuracy metrics, inter-annotator agreement, correlation analysis,
bootstrap confidence intervals, and significance-tested performance gaps.
"""

from __future__ import annotations

import logging
import math
import random as _random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiple-choice accuracy
# ---------------------------------------------------------------------------

def mc_accuracy(
    predictions: Sequence[str],
    ground_truth: Sequence[str],
) -> float:
    """Compute multiple-choice accuracy.

    Parameters
    ----------
    predictions : sequence of str
        Predicted option keys (e.g., ``["A", "C", "B", ...]``).
    ground_truth : sequence of str
        Correct option keys.

    Returns
    -------
    float
        Accuracy in [0, 1].

    Raises
    ------
    ValueError
        If the sequences have different lengths.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truth."
        )
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if str(p).strip().upper() == str(g).strip().upper()
    )
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------

def krippendorff_alpha(
    annotations: List[Dict[str, Any]],
    value_key: str = "selected_answer",
) -> float:
    """Compute Krippendorff's alpha for inter-annotator agreement.

    This is a standalone version (no dependency on the validator module)
    suitable for use in evaluation pipelines.

    Parameters
    ----------
    annotations : list of dict
        Each dict must have ``question_id``, ``annotator_id``, and the
        key specified by *value_key*.
    value_key : str
        The dict key containing the annotator's response value.

    Returns
    -------
    float
        Alpha coefficient in [-1, 1].
    """
    # Build: {question_id: {annotator_id: value}}
    by_question: Dict[str, Dict[str, Any]] = {}
    for ann in annotations:
        qid = ann.get("question_id", "")
        aid = ann.get("annotator_id", "")
        val = ann.get(value_key)
        if val is None:
            continue
        by_question.setdefault(qid, {})[aid] = val

    # Coincidence matrix approach for nominal data
    value_counts: Dict[Any, float] = {}
    observed_disagreement = 0.0
    total_pairs = 0.0

    for qid, annotator_values in by_question.items():
        values = list(annotator_values.values())
        m = len(values)
        if m < 2:
            continue

        # Pairwise comparisons within this unit
        for i in range(m):
            for j in range(i + 1, m):
                total_pairs += 1.0
                if values[i] != values[j]:
                    observed_disagreement += 1.0

        for v in values:
            value_counts[v] = value_counts.get(v, 0.0) + 1.0

    if total_pairs == 0:
        return 1.0

    d_o = observed_disagreement / total_pairs

    n_total = sum(value_counts.values())
    if n_total < 2:
        return 1.0

    expected_disagreement = 0.0
    all_vals = list(value_counts.keys())
    for i in range(len(all_vals)):
        for j in range(i + 1, len(all_vals)):
            expected_disagreement += value_counts[all_vals[i]] * value_counts[all_vals[j]]

    d_e = expected_disagreement / (n_total * (n_total - 1) / 2)

    if d_e == 0.0:
        return 1.0

    return 1.0 - (d_o / d_e)


# ---------------------------------------------------------------------------
# Pearson correlation with lag
# ---------------------------------------------------------------------------

def pearson_with_lag(
    series_a: Sequence[float],
    series_b: Sequence[float],
    max_lag: int = 5,
) -> Dict[int, Dict[str, float]]:
    """Compute Pearson correlation between two time series at multiple lags.

    A positive lag means *series_a* leads *series_b* by that many steps.

    Parameters
    ----------
    series_a : sequence of float
    series_b : sequence of float
    max_lag : int
        Maximum lag in both directions.

    Returns
    -------
    dict
        ``{lag: {"r": float, "p": float, "n": int}}``
        where ``r`` is Pearson's r, ``p`` the two-tailed p-value, and
        ``n`` the number of paired observations.
    """
    from scipy import stats

    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    results: Dict[int, Dict[str, float]] = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_slice = a[:n - lag] if lag > 0 else a
            b_slice = b[lag:] if lag > 0 else b
        else:
            a_slice = a[-lag:]
            b_slice = b[:n + lag]

        paired_n = min(len(a_slice), len(b_slice))
        if paired_n < 3:
            results[lag] = {"r": float("nan"), "p": float("nan"), "n": paired_n}
            continue

        a_s = a_slice[:paired_n]
        b_s = b_slice[:paired_n]

        # Guard against constant series
        if np.std(a_s) == 0 or np.std(b_s) == 0:
            results[lag] = {"r": 0.0, "p": 1.0, "n": paired_n}
            continue

        r, p = stats.pearsonr(a_s, b_s)
        results[lag] = {"r": float(r), "p": float(p), "n": paired_n}

    return results


# ---------------------------------------------------------------------------
# Mean judge score
# ---------------------------------------------------------------------------

def mean_judge_score(
    judge_results: List[Dict[str, Any]],
    score_key: str = "score",
) -> float:
    """Compute the mean score from a list of LLM judge results.

    Parameters
    ----------
    judge_results : list of dict
        Each dict should contain a numeric score under *score_key*.
    score_key : str
        Key to extract.

    Returns
    -------
    float
        Mean score, or 0.0 if no valid scores.
    """
    scores = []
    for r in judge_results:
        val = r.get(score_key)
        if val is not None:
            try:
                scores.append(float(val))
            except (ValueError, TypeError):
                continue
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a bootstrap confidence interval for the mean.

    Parameters
    ----------
    scores : sequence of float
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{"mean": float, "ci_lower": float, "ci_upper": float,
           "std": float, "n": int}``
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    n = len(arr)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Performance gap with significance test
# ---------------------------------------------------------------------------

def performance_gap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    test: str = "permutation",
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute the performance gap between two score sets with significance.

    Parameters
    ----------
    scores_a : sequence of float
        Scores from condition A (e.g., instrumental questions).
    scores_b : sequence of float
        Scores from condition B (e.g., vocal questions).
    test : ``"permutation"`` or ``"bootstrap"`` or ``"t-test"``
        Statistical test to assess significance.
    n_permutations : int
        Number of permutations/bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{"mean_a": float, "mean_b": float, "gap": float,
           "p_value": float, "significant_at_05": bool,
           "significant_at_01": bool, "test": str,
           "ci_gap": (float, float)}``
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    mean_a = float(np.mean(a)) if len(a) > 0 else 0.0
    mean_b = float(np.mean(b)) if len(b) > 0 else 0.0
    observed_gap = mean_a - mean_b

    p_value: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    if test == "t-test":
        from scipy import stats
        if len(a) < 2 or len(b) < 2:
            p_value = 1.0
        else:
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            p_value = float(p_value)
            # Approximate CI for the difference
            se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
            ci_lower = observed_gap - 1.96 * se
            ci_upper = observed_gap + 1.96 * se

    elif test == "permutation":
        rng = np.random.RandomState(seed)
        combined = np.concatenate([a, b])
        n_a = len(a)
        count_extreme = 0

        for _ in range(n_permutations):
            rng.shuffle(combined)
            perm_gap = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
            if abs(perm_gap) >= abs(observed_gap):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)

        # Bootstrap CI for the gap
        boot_gaps = np.empty(n_permutations)
        for i in range(n_permutations):
            ba = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            boot_gaps[i] = np.mean(ba) - np.mean(bb)
        ci_lower = float(np.percentile(boot_gaps, 2.5))
        ci_upper = float(np.percentile(boot_gaps, 97.5))

    elif test == "bootstrap":
        rng = np.random.RandomState(seed)
        boot_gaps = np.empty(n_permutations)
        for i in range(n_permutations):
            ba = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            boot_gaps[i] = np.mean(ba) - np.mean(bb)
        ci_lower = float(np.percentile(boot_gaps, 2.5))
        ci_upper = float(np.percentile(boot_gaps, 97.5))
        # p-value: proportion of bootstrap gaps that cross zero
        if observed_gap > 0:
            p_value = float(np.mean(boot_gaps <= 0)) * 2  # two-tailed
        else:
            p_value = float(np.mean(boot_gaps >= 0)) * 2
        p_value = min(p_value, 1.0)

    else:
        raise ValueError(f"Unknown test: {test!r}. Use 'permutation', 'bootstrap', or 't-test'.")

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "gap": observed_gap,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "test": test,
        "ci_gap": (ci_lower, ci_upper),
        "n_a": len(a),
        "n_b": len(b),
    }
