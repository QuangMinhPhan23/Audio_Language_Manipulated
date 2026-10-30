"""
MusicWhy Analysis & Visualization Module (Phase 6).

Provides publication-quality figures, LaTeX tables, and statistical
analysis routines for the MusicWhy benchmark paper.
"""

from musicwhy.analysis.visualize import ResultsVisualizer
from musicwhy.analysis.tables import ResultsTables
from musicwhy.analysis.statistics import (
    compute_all_correlations,
    significance_tests,
    effect_sizes,
    bootstrap_all_cis,
)

__all__ = [
    "ResultsVisualizer",
    "ResultsTables",
    "compute_all_correlations",
    "significance_tests",
    "effect_sizes",
    "bootstrap_all_cis",
]
