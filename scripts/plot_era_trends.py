"""Figure: decade trends in expert-rated sound across all MGPHot Billboard Hot 100 songs (1958-2022).

Small multiples on one shared 0-1 scale, one series per panel (so no legend), shaded 95% interval
of each decade mean, first and last values labelled. Runs locally where the MGPHot files are.
  python -I scripts/plot_era_trends.py --out figures/era_trends
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from musicwhy.data.mgphot import by_decade, load_mgphot  # noqa: E402

PANELS = [
    ("Synthetic Drums", "Drum machines"),
    ("Synthetic Sonority", "Electronic sound"),
    ("Synthesizer", "Synthesizer"),
    ("Electric Guitar Distortion", "Distorted guitar"),
    ("Horn Ensemble", "Horn section"),
    ("String Ensemble", "String section"),
    ("Minor / Major Key Tonality", "Major key"),
    ("Swing Feel", "Swing rhythm"),
]
SERIES = "#2a78d6"      # categorical slot 1 of the dataviz reference palette
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE, GRID = "#fcfcfb", "#e4e3df"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mgphot-dir", default="D:/WMM_external/MGPHot-audio")
    p.add_argument("--out", default="figures/era_trends")
    args = p.parse_args()

    groups = by_decade(load_mgphot(args.mgphot_dir))
    decades = list(groups)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (gene, label) in zip(axes.flat, PANELS):
        means, lo, hi = [], [], []
        for d in decades:
            v = [s["genes"][gene] for s in groups[d]]
            m = sum(v) / len(v)
            se = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))
            means.append(m)
            lo.append(m - 1.96 * se)
            hi.append(m + 1.96 * se)
        ax.set_facecolor(SURFACE)
        ax.fill_between(decades, lo, hi, color=SERIES, alpha=0.18, linewidth=0)
        ax.plot(decades, means, color=SERIES, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1)
        for i in (0, len(decades) - 1):  # label only the ends
            ax.annotate(f"{means[i]:.2f}", (decades[i], means[i]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=INK_2)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks(decades[::2])
        ax.set_xticklabels([f"{d}s" for d in decades[::2]], fontsize=8, color=INK_2)
        ax.tick_params(axis="y", labelsize=8, colors=INK_2, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    n = sum(len(v) for v in groups.values())
    fig.suptitle(f"Expert-rated sound of Billboard Hot 100 hits by decade (n = {n:,}; mean and 95% interval)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Ratings 0-1 from the Music Genome Project (MGPHot). Major key: 1 = major, 0 = minor.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"Saved {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
