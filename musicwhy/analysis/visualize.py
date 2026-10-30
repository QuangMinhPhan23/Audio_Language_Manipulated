"""
Publication-quality visualization for the MusicWhy paper.

Produces the five key figures described in the experiment plan (Phase 6).
All plots use a colour-blind-friendly palette and are saved as both PDF
(vector) and PNG (300 dpi), sized for NeurIPS single- or double-column
layout.

Classes
-------
ResultsVisualizer
    Stateless helper whose class methods each produce one figure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Module-level style configuration (NeurIPS-friendly)
# ---------------------------------------------------------------------------

# Colour-blind-friendly palette (Paul Tol "bright" scheme, 7 colours).
CB_PALETTE: List[str] = [
    "#4477AA",  # blue
    "#EE6677",  # red/pink
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey (baselines)
]

# NeurIPS column widths in inches.
SINGLE_COL_WIDTH = 5.5
DOUBLE_COL_WIDTH = 11.0
GOLDEN_RATIO = 1.618

_RC_PARAMS: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,   # TrueType in PDFs (editable text)
    "ps.fonttype": 42,
}


def _apply_style() -> None:
    """Apply the module-level Matplotlib RC overrides."""
    mpl.rcParams.update(_RC_PARAMS)


def _save(fig: mpl.figure.Figure, output_path: str) -> None:
    """Save *fig* as both PDF and PNG next to *output_path*.

    *output_path* may end in any extension (or none); both ``.pdf`` and
    ``.png`` variants are always written.
    """
    base, _ = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(f"{base}.pdf", format="pdf")
    fig.savefig(f"{base}.png", format="png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ResultsVisualizer
# ---------------------------------------------------------------------------

class ResultsVisualizer:
    """Stateless helper producing all key figures for the MusicWhy paper.

    Every public method is a ``@staticmethod`` that takes data and an
    *output_path*, produces one figure, and returns the
    :class:`~matplotlib.figure.Figure` (already saved and closed).
    """

    # Default model order and display names.
    DEFAULT_MODELS: List[str] = [
        "Qwen2-Audio-7B",
        "Audio-Reasoner-7B",
        "Gemini Pro",
        "GPT-4o",
        "Human expert",
        "Random baseline",
    ]

    DEFAULT_TASKS: List[str] = [
        "Reasoning QA",
        "Source Attribution",
        "Relationship",
        "Symbolic",
        "Sound-World",
    ]

    # ------------------------------------------------------------------ #
    # Figure 1: Spider / radar chart                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_radar_chart(
        results_by_model: Dict[str, List[float]],
        tasks: Optional[List[str]] = None,
        output_path: str = "fig1_radar",
        title: str = "Model Performance Across Tasks",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> mpl.figure.Figure:
        """Spider / radar chart of model scores across the five tasks.

        Parameters
        ----------
        results_by_model : dict[str, list[float]]
            ``{model_name: [score_task1, ..., score_taskN]}``.
            All score lists must have the same length as *tasks*.
        tasks : list[str] | None
            Axis labels.  Defaults to :pyattr:`DEFAULT_TASKS`.
        output_path : str
            File stem (extensions added automatically).
        title : str
            Figure title.
        figsize : tuple | None
            ``(width, height)`` in inches.  Default adapts to column width.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _apply_style()

        if tasks is None:
            tasks = list(ResultsVisualizer.DEFAULT_TASKS)

        n_tasks = len(tasks)
        angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        if figsize is None:
            figsize = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH)

        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"polar": True}
        )

        # Style the grid.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(
            np.degrees(angles[:-1]), labels=tasks, fontsize=8
        )
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="grey")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc")

        model_names = list(results_by_model.keys())
        for idx, (model, scores) in enumerate(results_by_model.items()):
            values = list(scores) + [scores[0]]  # close
            color = CB_PALETTE[idx % len(CB_PALETTE)]

            # Baselines get dashed lines.
            ls = "--" if model.lower() in ("random baseline", "human expert") else "-"
            lw = 1.5 if ls == "-" else 1.2

            ax.plot(angles, values, linewidth=lw, linestyle=ls, color=color, label=model)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=20)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.12),
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            fontsize=7,
        )

        fig.tight_layout()
        _save(fig, output_path)
        return fig

    # ------------------------------------------------------------------ #
    # Figure 2: Feature trend plots with economic overlay                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_feature_trends(
        yearly_features: pd.DataFrame,
        economic_data: pd.DataFrame,
        output_path: str = "fig2_feature_trends",
        feature_cols: Optional[List[str]] = None,
        economic_cols: Optional[List[str]] = None,
        highlight_events: Optional[Dict[int, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> mpl.figure.Figure:
        """Dual-axis line plots of audio features and economic indicators.

        Parameters
        ----------
        yearly_features : pd.DataFrame
            Must contain ``year`` and numeric feature columns (e.g.
            ``avg_tempo``, ``avg_brightness``, ``avg_energy``).
        economic_data : pd.DataFrame
            Must contain ``year`` and numeric indicator columns (e.g.
            ``gdp_growth``, ``unemployment``).
        output_path : str
            File stem.
        feature_cols : list[str] | None
            Audio feature columns to plot.  ``None`` auto-detects up to 3.
        economic_cols : list[str] | None
            Economic columns to plot.  ``None`` auto-detects up to 2.
        highlight_events : dict[int, str] | None
            ``{year: label}`` of notable events to annotate with vertical
            lines.  Defaults to major US recessions and cultural shifts.
        figsize : tuple | None
            Figure dimensions in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _apply_style()

        ft = yearly_features.copy().sort_values("year")
        ec = economic_data.copy().sort_values("year")

        if feature_cols is None:
            candidates = [
                c for c in ft.select_dtypes(include="number").columns
                if c != "year"
            ]
            feature_cols = candidates[:3]

        if economic_cols is None:
            candidates = [
                c for c in ec.select_dtypes(include="number").columns
                if c != "year"
            ]
            economic_cols = candidates[:2]

        if highlight_events is None:
            highlight_events = {
                1973: "Oil crisis",
                1981: "Recession",
                1991: "Recession",
                2001: "Dot-com bust",
                2008: "Financial crisis",
                2020: "COVID-19",
            }

        n_features = len(feature_cols)
        if figsize is None:
            figsize = (DOUBLE_COL_WIDTH, 2.6 * n_features)

        fig, axes = plt.subplots(
            n_features, 1, figsize=figsize, sharex=True
        )
        if n_features == 1:
            axes = [axes]

        for i, feat in enumerate(feature_cols):
            ax_left: mpl.axes.Axes = axes[i]

            # --- Audio feature (left y-axis) ---
            color_feat = CB_PALETTE[i % len(CB_PALETTE)]
            ax_left.plot(
                ft["year"], ft[feat],
                color=color_feat, linewidth=1.5, label=feat.replace("_", " ").title(),
            )
            ax_left.set_ylabel(
                feat.replace("_", " ").title(),
                color=color_feat, fontsize=8,
            )
            ax_left.tick_params(axis="y", labelcolor=color_feat, labelsize=7)

            # --- Economic overlay (right y-axis) ---
            if economic_cols:
                ax_right = ax_left.twinx()
                for j, econ in enumerate(economic_cols):
                    color_econ = CB_PALETTE[(n_features + j) % len(CB_PALETTE)]
                    ax_right.plot(
                        ec["year"], ec[econ],
                        color=color_econ, linewidth=1.0, linestyle="--",
                        alpha=0.7, label=econ.replace("_", " ").title(),
                    )
                ax_right.set_ylabel("Economic indicator", fontsize=8, color="grey")
                ax_right.tick_params(axis="y", labelcolor="grey", labelsize=7)
                ax_right.spines["right"].set_visible(True)
                ax_right.spines["right"].set_color("#cccccc")

            # --- Event highlights ---
            for yr, label in highlight_events.items():
                if ft["year"].min() <= yr <= ft["year"].max():
                    ax_left.axvline(
                        yr, color="#999999", linewidth=0.7,
                        linestyle=":", alpha=0.6, zorder=0,
                    )
                    if i == 0:
                        ax_left.annotate(
                            label, xy=(yr, ax_left.get_ylim()[1]),
                            xytext=(4, 2), textcoords="offset points",
                            fontsize=6, color="#666666", rotation=45,
                            ha="left", va="bottom",
                        )

            # Combined legend for this subplot.
            lines_left, labels_left = ax_left.get_legend_handles_labels()
            if economic_cols:
                lines_right, labels_right = ax_right.get_legend_handles_labels()
            else:
                lines_right, labels_right = [], []
            ax_left.legend(
                lines_left + lines_right,
                labels_left + labels_right,
                loc="upper left", fontsize=7, framealpha=0.9,
            )

        axes[-1].set_xlabel("Year", fontsize=9)
        fig.suptitle(
            "Audio Feature Trends of Chart Music (1960--2024)\nwith Economic Indicators",
            fontsize=10, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, output_path)
        return fig

    # ------------------------------------------------------------------ #
    # Figure 3: Ablation heatmap                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_ablation_heatmap(
        ablation_results: pd.DataFrame,
        output_path: str = "fig3_ablation_heatmap",
        title: str = "Emotional Change When Stems Are Removed",
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "RdBu_r",
    ) -> mpl.figure.Figure:
        """Heatmap of emotional change magnitudes from stem-removal ablation.

        Parameters
        ----------
        ablation_results : pd.DataFrame
            Rows = emotions, columns = removed stems.  Cell values are
            signed magnitudes of emotional change (positive = increase,
            negative = decrease).
        output_path : str
            File stem.
        title : str
            Figure title.
        figsize : tuple | None
            Figure dimensions.
        cmap : str
            Matplotlib colormap name.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _apply_style()

        if figsize is None:
            w = max(SINGLE_COL_WIDTH, 0.9 * ablation_results.shape[1] + 1.5)
            h = max(3.0, 0.5 * ablation_results.shape[0] + 1.0)
            figsize = (w, h)

        fig, ax = plt.subplots(figsize=figsize)

        # Symmetric colour limits centred on zero.
        vmax = float(np.abs(ablation_results.values).max())

        sns.heatmap(
            ablation_results,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Magnitude of emotional change", "shrink": 0.8},
            annot_kws={"fontsize": 7},
        )

        ax.set_xlabel("Removed Stem", fontsize=9)
        ax.set_ylabel("Emotion", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=12)

        # Rotate labels for readability.
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=30, ha="right", fontsize=8
        )
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=0, fontsize=8
        )

        fig.tight_layout()
        _save(fig, output_path)
        return fig

    # ------------------------------------------------------------------ #
    # Figure 4: t-SNE comparison (raw features vs CLAP embeddings)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_tsne_comparison(
        raw_features: np.ndarray,
        clap_embeddings: np.ndarray,
        emotion_labels: Sequence[str],
        output_path: str = "fig4_tsne_comparison",
        perplexity: float = 30.0,
        seed: int = 42,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> mpl.figure.Figure:
        """Side-by-side t-SNE plots coloured by human emotion labels.

        Parameters
        ----------
        raw_features : np.ndarray, shape (n, d_raw)
            Raw audio feature vectors (MFCCs, spectral, etc.).
        clap_embeddings : np.ndarray, shape (n, d_clap)
            CLAP (or MERT) embeddings for the same songs.
        emotion_labels : sequence of str
            Human-assigned emotion label per song.
        output_path : str
            File stem.
        perplexity : float
            t-SNE perplexity.
        seed : int
            Random state.
        figsize : tuple | None
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from sklearn.manifold import TSNE  # deferred so the module loads without sklearn

        _apply_style()

        if figsize is None:
            figsize = (DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.55)

        # Run t-SNE on both representations.
        tsne_raw = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca"
        ).fit_transform(raw_features)
        tsne_clap = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca"
        ).fit_transform(clap_embeddings)

        # Map labels to integers for colouring.
        unique_labels = sorted(set(emotion_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        colour_indices = np.array([label_to_idx[l] for l in emotion_labels])

        # Build a discrete colourmap from our CB palette, cycling if needed.
        n_labels = len(unique_labels)
        colours = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(n_labels)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for ax, coords, subtitle in [
            (ax1, tsne_raw, "Raw Audio Features"),
            (ax2, tsne_clap, "CLAP Embeddings"),
        ]:
            for i, label in enumerate(unique_labels):
                mask = colour_indices == i
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=colours[i], label=label, s=12, alpha=0.65,
                    edgecolors="none",
                )
            ax.set_title(subtitle, fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Shared legend below the figure.
        handles = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colours[i], markersize=6,
                label=unique_labels[i],
            )
            for i in range(n_labels)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(n_labels, 6),
            fontsize=7,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(
            "t-SNE: Raw Features vs CLAP Embeddings (coloured by emotion)",
            fontsize=10, fontweight="bold", y=1.03,
        )
        fig.tight_layout()
        _save(fig, output_path)
        return fig

    # ------------------------------------------------------------------ #
    # Figure 5: Instrumental vs. Vocal performance gap                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_instrumental_vocal_gap(
        results: Dict[str, Dict[str, float]],
        output_path: str = "fig5_instrumental_vocal_gap",
        models: Optional[List[str]] = None,
        title: str = "Instrumental vs. Vocal Performance per Model",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> mpl.figure.Figure:
        """Grouped bar chart comparing vocal and instrumental scores.

        Parameters
        ----------
        results : dict[str, dict[str, float]]
            ``{model_name: {"vocal": score, "instrumental": score}}``.
        output_path : str
            File stem.
        models : list[str] | None
            Model ordering.  ``None`` uses the keys of *results* in their
            insertion order.
        title : str
            Figure title.
        figsize : tuple | None
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _apply_style()

        if models is None:
            models = list(results.keys())

        if figsize is None:
            figsize = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH / GOLDEN_RATIO)

        vocal_scores = [results[m].get("vocal", 0) for m in models]
        inst_scores = [results[m].get("instrumental", 0) for m in models]

        x = np.arange(len(models))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=figsize)

        bars_vocal = ax.bar(
            x - bar_width / 2, vocal_scores, bar_width,
            label="Vocal", color=CB_PALETTE[0], edgecolor="white", linewidth=0.5,
        )
        bars_inst = ax.bar(
            x + bar_width / 2, inst_scores, bar_width,
            label="Instrumental", color=CB_PALETTE[1], edgecolor="white", linewidth=0.5,
        )

        # Value labels on bars.
        for bars in (bars_vocal, bars_inst):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

        # Gap annotations (arrows).
        for i in range(len(models)):
            gap = vocal_scores[i] - inst_scores[i]
            if abs(gap) > 0.5:
                mid_x = x[i] + bar_width / 2
                mid_y = (vocal_scores[i] + inst_scores[i]) / 2.0
                ax.annotate(
                    f"$\\Delta${gap:+.1f}",
                    xy=(mid_x + 0.22, mid_y),
                    fontsize=6, color="#555555", ha="left", va="center",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.legend(fontsize=8, frameon=True, edgecolor="#cccccc")
        ax.set_ylim(0, max(max(vocal_scores), max(inst_scores)) * 1.18)

        fig.tight_layout()
        _save(fig, output_path)
        return fig
