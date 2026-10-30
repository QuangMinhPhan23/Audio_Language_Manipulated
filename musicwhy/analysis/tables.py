"""
LaTeX table generation for the MusicWhy paper.

Classes
-------
ResultsTables
    Generates publication-ready LaTeX tables (NeurIPS format).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ResultsTables:
    """Generate LaTeX tables suitable for inclusion in a NeurIPS paper.

    All ``generate_*`` methods return a *complete* LaTeX ``tabular``
    environment string (no surrounding ``\\begin{table}``), so the caller
    can wrap it in a ``table`` / ``table*`` float with their own caption
    and label.
    """

    # Ordered list of models (rows) used consistently across tables.
    DEFAULT_MODELS: List[str] = [
        "Qwen2-Audio-7B",
        "Audio-Reasoner-7B",
        "Gemini Pro",
        "GPT-4o",
        "Random baseline",
        "Human expert",
    ]

    # Ordered list of tasks (columns) used in the main results table.
    DEFAULT_TASKS: List[str] = [
        "Reasoning QA",
        "Source Attribution",
        "Relationship",
        "Symbolic",
        "Sound-World",
    ]

    # ------------------------------------------------------------------
    # Main results table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_main_results_table(
        results: Dict[str, Dict[str, Any]],
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
        models: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        bold_best: bool = True,
    ) -> str:
        """Generate the main benchmark results table (models x tasks).

        Parameters
        ----------
        results : dict[str, dict[str, float]]
            ``{model_name: {task_name: score}}``.  Scores should already
            be in the display scale (e.g. 0--100 for accuracy tasks,
            1--5 for Likert-style tasks).
        confidence_intervals : dict | None
            ``{model: {task: (point, lower, upper)}}``.  If given, each
            cell shows ``point $\\pm$ half-width``.
        models : list[str] | None
            Row ordering.  Defaults to :pyattr:`DEFAULT_MODELS`.
        tasks : list[str] | None
            Column ordering.  Defaults to :pyattr:`DEFAULT_TASKS`.
        bold_best : bool, default True
            Bold the best non-human, non-baseline score in each task
            column.

        Returns
        -------
        str
            LaTeX ``tabular`` environment.
        """
        if models is None:
            models = ResultsTables.DEFAULT_MODELS
        if tasks is None:
            tasks = ResultsTables.DEFAULT_TASKS

        n_tasks = len(tasks)

        # Identify best score per task (excluding baselines / humans).
        best: Dict[str, float] = {}
        if bold_best:
            non_special = [
                m for m in models
                if m.lower() not in ("random baseline", "human expert")
            ]
            for t in tasks:
                scores = [
                    results.get(m, {}).get(t, float("-inf"))
                    for m in non_special
                ]
                best[t] = max(scores) if scores else float("-inf")

        # Build LaTeX -------------------------------------------------------
        col_spec = "l" + "c" * n_tasks
        lines: List[str] = []
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header_cells = ["Model"] + [f"\\textbf{{{t}}}" for t in tasks]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for model in models:
            cells: List[str] = [model]

            # Horizontal rule before Random / Human blocks.
            if model.lower() == "random baseline":
                lines.append("\\midrule")

            for task in tasks:
                score = results.get(model, {}).get(task)
                if score is None:
                    cells.append("--")
                    continue

                # Format the cell.
                if confidence_intervals and model in confidence_intervals:
                    ci = confidence_intervals[model].get(task)
                    if ci is not None:
                        _, lo, hi = ci
                        half = (hi - lo) / 2.0
                        cell = f"{score:.1f} $\\pm$ {half:.1f}"
                    else:
                        cell = f"{score:.1f}"
                else:
                    cell = f"{score:.1f}"

                # Bold best.
                if bold_best and task in best and abs(score - best[task]) < 1e-9:
                    cell = f"\\textbf{{{cell}}}"

                cells.append(cell)

            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Correlation table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_correlation_table(
        correlations: Dict[str, Dict[int, Tuple[float, float]]],
        selected_pairs: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        significance_threshold: float = 0.05,
    ) -> str:
        """Generate a LaTeX table of time-lagged correlations.

        Parameters
        ----------
        correlations : dict
            Output of :func:`compute_all_correlations`.
        selected_pairs : list[str] | None
            Feature-vs-indicator keys to include (default: all).
        lags : list[int] | None
            Which lags to show (default: [-2, -1, 0, 1, 2]).
        significance_threshold : float
            p-values below this threshold are starred.

        Returns
        -------
        str
            LaTeX ``tabular`` environment.
        """
        if selected_pairs is None:
            selected_pairs = sorted(correlations.keys())
        if lags is None:
            lags = [-2, -1, 0, 1, 2]

        col_spec = "l" + "c" * len(lags)
        lines: List[str] = []
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header = ["Feature pair"] + [f"Lag {l:+d}" for l in lags]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        for pair in selected_pairs:
            row_data = correlations.get(pair, {})
            # Pretty-print the pair name.
            display_name = pair.replace("_vs_", " vs. ").replace("_", " ")
            cells: List[str] = [display_name]
            for lag in lags:
                entry = row_data.get(lag)
                if entry is None:
                    cells.append("--")
                else:
                    r, p = entry
                    star = "*" if p < significance_threshold else ""
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    cells.append(f"{r:.3f}{star}")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ablation table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_ablation_table(
        ablation_results: Dict[str, Dict[str, float]],
        stems: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
    ) -> str:
        """Generate a LaTeX table for stem-removal ablation results.

        Parameters
        ----------
        ablation_results : dict[str, dict[str, float]]
            ``{emotion: {removed_stem: magnitude_of_change}}``.
        stems : list[str] | None
            Column ordering.  Defaults to sorted keys of the first
            emotion entry.
        emotions : list[str] | None
            Row ordering.  Defaults to sorted keys of *ablation_results*.

        Returns
        -------
        str
            LaTeX ``tabular`` environment.
        """
        if emotions is None:
            emotions = sorted(ablation_results.keys())
        if stems is None:
            first = next(iter(ablation_results.values()), {})
            stems = sorted(first.keys())

        col_spec = "l" + "c" * len(stems)
        lines: List[str] = []
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header = ["Emotion"] + [f"$-${s.capitalize()}" for s in stems]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        for emo in emotions:
            cells: List[str] = [emo.capitalize()]
            for stem in stems:
                val = ablation_results.get(emo, {}).get(stem)
                if val is None:
                    cells.append("--")
                else:
                    cells.append(f"{val:.3f}")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    @staticmethod
    def export_all_tables(
        results_dir: str,
        output_dir: str,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
        correlations: Optional[Dict[str, Dict[int, Tuple[float, float]]]] = None,
        ablation_results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[str]:
        """Write all tables as individual ``.tex`` files.

        If data dicts are ``None`` the method tries to load NumPy
        ``.npz`` files from *results_dir* using the conventions:

        - ``main_results.npz``  (keys: model names; values: task-score dicts)
        - ``correlations.npz``
        - ``ablation.npz``

        Parameters
        ----------
        results_dir : str
            Directory containing serialised result files.
        output_dir : str
            Directory to write ``.tex`` files into (created if absent).
        results, confidence_intervals, correlations, ablation_results
            Pre-loaded data.  Overrides file-based loading.

        Returns
        -------
        list[str]
            Paths of the written ``.tex`` files.
        """
        os.makedirs(output_dir, exist_ok=True)
        written: List[str] = []

        # --- Main results ------------------------------------------------
        if results is None:
            main_path = os.path.join(results_dir, "main_results.npz")
            if os.path.isfile(main_path):
                loaded = np.load(main_path, allow_pickle=True)
                results = {k: loaded[k].item() for k in loaded.files}

        if results is not None:
            tex = ResultsTables.generate_main_results_table(
                results, confidence_intervals=confidence_intervals
            )
            out_path = os.path.join(output_dir, "table_main_results.tex")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(tex)
            written.append(out_path)

        # --- Correlations ------------------------------------------------
        if correlations is None:
            corr_path = os.path.join(results_dir, "correlations.npz")
            if os.path.isfile(corr_path):
                loaded = np.load(corr_path, allow_pickle=True)
                correlations = {k: loaded[k].item() for k in loaded.files}

        if correlations is not None:
            tex = ResultsTables.generate_correlation_table(correlations)
            out_path = os.path.join(output_dir, "table_correlations.tex")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(tex)
            written.append(out_path)

        # --- Ablation ----------------------------------------------------
        if ablation_results is None:
            abl_path = os.path.join(results_dir, "ablation.npz")
            if os.path.isfile(abl_path):
                loaded = np.load(abl_path, allow_pickle=True)
                ablation_results = {k: loaded[k].item() for k in loaded.files}

        if ablation_results is not None:
            tex = ResultsTables.generate_ablation_table(ablation_results)
            out_path = os.path.join(output_dir, "table_ablation.tex")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(tex)
            written.append(out_path)

        return written
