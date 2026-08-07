"""
plot_distribution_context.py — Figure S1: UMAP + kNN density context
======================================================================
(a) UMAP projection (literature fit, experiment transform)
(b) High-dimensional kNN distance percentile for experimental points

Reads: Results/Distribution/umap_coordinates.csv
       Results/Distribution/experimental_knn_density.csv
"""
import os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Distribution")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 13
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_LIT = "#5DA5DA"
COLOR_EXP = "#C91511"


def plot():
    umap_path = os.path.join(RESULTS_DIR, "umap_coordinates.csv")
    density_path = os.path.join(RESULTS_DIR, "experimental_knn_density.csv")

    if not os.path.exists(umap_path):
        print("[WARN] umap_coordinates.csv not found, run fair_holdout_comparison.py first")
        return

    umap_df = pd.read_csv(umap_path)
    lit = umap_df[umap_df["source"] == "Literature"]
    exp = umap_df[umap_df["source"] == "Experiment"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # ── (a) UMAP projection ──
    ax1.scatter(lit["umap_1"], lit["umap_2"],
                c=COLOR_LIT, s=5, alpha=0.2, edgecolors="none",
                label=f"Literature (n={len(lit)})")
    ax1.scatter(exp["umap_1"], exp["umap_2"],
                c=COLOR_EXP, s=60, alpha=0.85, marker="o",
                edgecolors="black", linewidth=0.6,
                label=f"Experiment (n={len(exp)})")
    ax1.set_xlabel("UMAP Dimension 1 (a.u.)", fontweight="bold")
    ax1.set_ylabel("UMAP Dimension 2 (a.u.)", fontweight="bold")
    ax1.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax1.text(-0.10, 1.02, "(a)", transform=ax1.transAxes, fontsize=FS + 4, fontweight="bold")
    ax1.set_title("Chemical Structure Space (Descriptors)", fontweight="bold")

    # ── (b) kNN density percentile ──
    if os.path.exists(density_path):
        dens = pd.read_csv(density_path)
        x_idx = np.arange(len(dens))
        colors = [COLOR_EXP if v else COLOR_LIT for v in dens["above_p90"].values]

        ax2.bar(x_idx, dens["literature_distance_percentile"] * 100,
                color=colors, edgecolor="white", linewidth=0.3, width=0.7)
        ax2.axhline(90, color=COLOR_EXP, linestyle="--", linewidth=0.8, alpha=0.5,
                    label="90th percentile")
        ax2.axhline(50, color="#8C92AC", linestyle=":", linewidth=0.8, alpha=0.5,
                    label="Median")
        ax2.set_xlabel("Experimental Sample Index", fontweight="bold")
        ax2.set_ylabel("Distance Percentile vs Literature (%)", fontweight="bold")
        ax2.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
        ax2.set_ylim(0, 105)
        n_above = dens["above_p90"].sum()
        ax2.set_title(f"kNN Density: {n_above}/{len(dens)} above P90", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "kNN density data not found", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=FS)

    ax2.text(-0.10, 1.02, "(b)", transform=ax2.transAxes, fontsize=FS + 4, fontweight="bold")

    fig.suptitle(
        "Dataset Distribution Context\n"
        "(Not used as evidence of out-of-domain sampling or extrapolative generalization)",
        fontsize=FS + 1, fontweight="bold", y=1.02)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fp = os.path.join(SAVE_DIR, f"Distribution_Context.{fmt}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Distribution_Context saved (png/pdf/svg)")


if __name__ == "__main__":
    plot()
