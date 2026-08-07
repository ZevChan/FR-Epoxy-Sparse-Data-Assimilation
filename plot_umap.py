"""
================================================================================
 plot_umap.py — Chemical-space UMAP projection
================================================================================
 Reads the UMAP coordinates saved by distribution.py
 (Results/Distribution/umap_coordinates.csv),
 distinguishes Literature / Experiment by the Source column.
 Outputs PNG / PDF / SVG.

 Colors:
   Literature — #5DA5DA (steel blue)
   Experiment — #C91511 (dark red, highlighted)
================================================================================
"""

import os
BASE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap

# ==================== NC-journal global plot settings ====================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ==================== Color definitions ====================
COLOR_LIT = "#5DA5DA"  # steel blue — literature
COLOR_EXP = "#C91511"  # dark red — experiment highlighted

# ==================== Path configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_umap():
    umap_file = os.path.join(RESULTS_DIR, "Distribution", "umap_coordinates.csv")
    if not os.path.exists(umap_file):
        print("[WARN] umap_coordinates.csv not found; run fair_holdout_comparison.py first")
        return

    print("Loading UMAP coordinates (output of distribution.py) ...")
    df = pd.read_csv(umap_file)
    df = df.rename(columns={"source": "Source", "umap_1": "UMAP-1", "umap_2": "UMAP-2"})
    df_unique = df.copy()

    lit_mask = df_unique["Source"] == "Literature"
    exp_mask = df_unique["Source"] == "Experiment"
    n_lit = int(lit_mask.sum())
    n_exp = int(exp_mask.sum())
    print(f"  Literature samples: {n_lit}")
    print(f"  Experiment samples: {n_exp}")

    # ---- Main plot ----
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=600)

    # KDE background
    sns.kdeplot(
        data=df_unique[lit_mask],
        x="UMAP-1", y="UMAP-2",
        fill=True, alpha=0.12, color="#7f8c8d",
        levels=8, thresh=0.05, ax=ax,
    )

    # literature scatter
    ax.scatter(
        df_unique.loc[lit_mask, "UMAP-1"],
        df_unique.loc[lit_mask, "UMAP-2"],
        c=COLOR_LIT, s=10, alpha=0.3, edgecolors="none",
        label=f"Literature Data (n={n_lit})",
    )

    # experiment scatter highlighted
    ax.scatter(
        df_unique.loc[exp_mask, "UMAP-1"],
        df_unique.loc[exp_mask, "UMAP-2"],
        c=COLOR_EXP, s=55, alpha=0.95, marker="o",
        edgecolors="black", linewidth=0.6,
        label=f"Experimental Data (n={n_exp})",
    )

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical Space Distribution of EP/FR Composites",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Chemical_Space.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("[OK] UMAP_Chemical_Space saved (png/pdf/svg)")


if __name__ == "__main__":
    plot_umap()
