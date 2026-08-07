"""
plot_shap_stability.py — Figure 3: SHAP decision structure stability
======================================================================
(a) Top-1 SHAP share Before vs After
(b) Effective feature number Before vs After
(c) SHAP entropy Before vs After
(d) Top-10 feature occurrence heatmap (Frozen protocol)
"""
import os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 12
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER = "#C91511"

DISPLAY_NAMES = {
    "LOI": "LOI", "THR": "THR", "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength", "UL94_Rating": "UL-94",
}


def plot():
    sm = pd.read_csv(os.path.join(RESULTS_DIR, "SHAP_Stability", "shap_structure_metrics.csv"))
    
    # Filter frozen protocol only (primary analysis)
    frozen = sm[sm["protocol"] == "frozen"].copy()
    if frozen.empty:
        print("[WARN] No frozen SHAP data, try all protocols")
        frozen = sm

    targets = frozen["target"].unique()
    labels = [DISPLAY_NAMES.get(t, t) for t in targets]

    # Pivot: before vs after per target
    before = frozen[frozen["track"] == "before"]
    after = frozen[frozen["track"] == "after"]

    # ═══════════ Figure ═══════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # ── (a) Top-1 SHAP share ──
    ax = axes[0, 0]
    x = np.arange(len(targets))
    w = 0.32

    b1 = []
    for t in targets:
        vals = before[before["target"] == t]["top1_share"].values
        b1.append(np.median(vals) if len(vals) > 0 else 0)
    a1 = []
    for t in targets:
        vals = after[after["target"] == t]["top1_share"].values
        a1.append(np.median(vals) if len(vals) > 0 else 0)

    ax.bar(x - w/2, b1, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
           label="Before (Literature only)")
    ax.bar(x + w/2, a1, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
           label="After (Literature + Experiment)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Top-1 SHAP Share", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(a)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (b) Effective feature number ──
    ax = axes[0, 1]
    b2 = []
    for t in targets:
        vals = before[before["target"] == t]["effective_feature_number"].values
        b2.append(np.median(vals) if len(vals) > 0 else 0)
    a2 = []
    for t in targets:
        vals = after[after["target"] == t]["effective_feature_number"].values
        a2.append(np.median(vals) if len(vals) > 0 else 0)

    ax.bar(x - w/2, b2, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
           label="Before (Literature only)")
    ax.bar(x + w/2, a2, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
           label="After (Literature + Experiment)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Effective Feature Number", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(b)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (c) SHAP entropy ──
    ax = axes[1, 0]
    b3 = []
    for t in targets:
        vals = before[before["target"] == t]["shap_entropy"].values
        b3.append(np.median(vals) if len(vals) > 0 else 0)
    a3 = []
    for t in targets:
        vals = after[after["target"] == t]["shap_entropy"].values
        a3.append(np.median(vals) if len(vals) > 0 else 0)

    ax.bar(x - w/2, b3, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
           label="Before (Literature only)")
    ax.bar(x + w/2, a3, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
           label="After (Literature + Experiment)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("SHAP Entropy", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(c)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (d) Top-10 Jaccard (from shap_ranking_similarity.csv) ──
    ax = axes[1, 1]
    sim_path = os.path.join(RESULTS_DIR, "SHAP_Stability", "shap_ranking_similarity.csv")
    if os.path.exists(sim_path):
        sim = pd.read_csv(sim_path)
        sim_frozen = sim[sim["protocol"] == "frozen"]
        jaccard_summary = (
            sim_frozen.groupby("target")["top10_jaccard"]
            .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
        )
        jacc_vals = []
        jacc_q1s = []
        jacc_q3s = []
        jacc_labels = []
        for t in targets:
            row = jaccard_summary[jaccard_summary["target"] == t]
            if len(row) > 0:
                jacc_vals.append(row["median"].values[0])
                jacc_q1s.append(row["q1"].values[0])
                jacc_q3s.append(row["q3"].values[0])
            else:
                jacc_vals.append(0)
                jacc_q1s.append(0)
                jacc_q3s.append(0)
            jacc_labels.append(DISPLAY_NAMES.get(t, t))
        xj = np.arange(len(jacc_vals))
        ax.bar(xj, jacc_vals, 0.55, color="#6EAA5E", edgecolor="white", linewidth=0.5)
        # IQR error bars
        yerr_low = [max(0, v - q1) for v, q1 in zip(jacc_vals, jacc_q1s)]
        yerr_high = [q3 - v for v, q3 in zip(jacc_vals, jacc_q3s)]
        ax.errorbar(xj, jacc_vals, yerr=[yerr_low, yerr_high], fmt="none",
                    ecolor="#333333", capsize=3, linewidth=0.8)
    else:
        xj = np.arange(len(targets))
        jacc_vals = [0] * len(targets)
        jacc_labels = labels
        ax.bar(xj, jacc_vals, 0.55, color="#6EAA5E", edgecolor="white", linewidth=0.5)
        print("[WARN] shap_ranking_similarity.csv not found, Jaccard panel shows zeros")

    ax.axhline(0.3, color=COLOR_AFTER, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.7, color=COLOR_BEFORE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(xj)
    ax.set_xticklabels(jacc_labels, rotation=25, ha="right")
    ax.set_ylabel("Top-10 Jaccard Similarity", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(d)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── Save ──
    fig.suptitle("Stability of Model Decision Structures under Controlled Data Addition",
                 fontsize=FS + 2, fontweight="bold", y=0.99)
    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fp = os.path.join(SAVE_DIR, f"SHAP_Stability.{fmt}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] SHAP_Stability saved (png/pdf/svg)")


if __name__ == "__main__":
    plot()
