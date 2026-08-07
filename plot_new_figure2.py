"""
plot_new_figure2.py — 新版 Figure 2: 数据分布背景 + 受控增量效应
=================================================================
(a) UMAP: 2332 literature + 24 experimental
(b) kNN local sparsity: distance percentile distribution
(c) Frozen multi-seed delta: 6 targets x 10 seeds point plot
(d) 95% CI: Frozen protocol bar chart with error bars
(e) Frozen vs Adaptive: protocol comparison boxplot / paired lines

输出到 Graphs/ 子文件夹，同时输出 a-e 各子图。
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from collections import Counter

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Fair_Comparison_PROC_review_REVISED", "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs", "Figure2_Rebuilt")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Style ──
FS = 11
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.unicode_minus": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_LIT = "#5DA5DA"
COLOR_EXP = "#C91511"
COLOR_FROZEN = "#2E86AB"
COLOR_ADAPTIVE = "#A23B72"
COLOR_POS = "#2E86AB"
COLOR_NEG = "#C91511"
GRAY = "#8C92AC"
LIGHT_GRAY = "#E0E0E0"

DISPLAY = {
    "LOI": "LOI", "UL94_Rating": "UL-94 (Bal.Acc)",
    "pHRR": "pHRR", "THR": "THR",
    "TSP": "TSP", "Flexural_Strength": "Flexural Str."
}

# ══════════════════════════════════════════════════════════
def load_data():
    """Load all required data."""
    data = {}

    # UMAP
    umap_path = os.path.join(RESULTS_DIR, "Distribution", "umap_coordinates.csv")
    if os.path.exists(umap_path):
        df = pd.read_csv(umap_path)
        data["umap_lit"] = df[df["source"] == "Literature"]
        data["umap_exp"] = df[df["source"] == "Experiment"]

    # kNN density
    knn_path = os.path.join(RESULTS_DIR, "Distribution", "experimental_knn_density.csv")
    if os.path.exists(knn_path):
        data["knn"] = pd.read_csv(knn_path)

    # Robustness (Multi-seed)
    rob_path = os.path.join(RESULTS_DIR, "Robustness", "multiseed_metrics.csv")
    if os.path.exists(rob_path):
        data["robust"] = pd.read_csv(rob_path)

    return data


def draw_panel_a(ax, data):
    """UMAP projection."""
    lit = data["umap_lit"]
    exp = data["umap_exp"]

    ax.scatter(lit["umap_1"], lit["umap_2"],
               c=COLOR_LIT, s=3, alpha=0.15, edgecolors="none",
               label=f"Literature (n={len(lit)})")
    ax.scatter(exp["umap_1"], exp["umap_2"],
               c=COLOR_EXP, s=55, alpha=0.9, marker="o",
               edgecolors="black", linewidth=0.5,
               label=f"Experiment (n={len(exp)})")
    ax.set_xlabel("UMAP 1 (a.u.)", fontweight="bold")
    ax.set_ylabel("UMAP 2 (a.u.)", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-3,
              loc="upper right", markerscale=1.2)
    ax.text(-0.12, 1.02, "(a)", transform=ax.transAxes, fontsize=FS+5, fontweight="bold")
    ax.set_title("2D UMAP Projection (Descriptor Space)", fontweight="bold", fontsize=FS+1)


def draw_panel_b(ax, data):
    """kNN local sparsity distribution."""
    knn = data["knn"]
    x_idx = np.arange(len(knn))
    pct = knn["literature_distance_percentile"].values * 100
    colors = [COLOR_EXP if v else COLOR_LIT for v in knn["above_p90"].values]

    ax.bar(x_idx, pct, color=colors, edgecolor="white", linewidth=0.3, width=0.65)
    ax.axhline(90, color=COLOR_EXP, linestyle="--", linewidth=0.8, alpha=0.5,
               label="P90 threshold")
    ax.axhline(50, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.5,
               label="Median")
    ax.set_xlabel("Experimental Sample Index", fontweight="bold")
    ax.set_ylabel("kNN Distance Percentile (%)", fontweight="bold")
    n_above = int(knn["above_p90"].sum())
    ax.set_title(f"Local Sparsity: {n_above}/{len(knn)} above P90", fontweight="bold", fontsize=FS+1)
    ax.legend(frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-3, loc="upper left")
    ax.set_ylim(0, 105)
    ax.text(-0.12, 1.02, "(b)", transform=ax.transAxes, fontsize=FS+5, fontweight="bold")


def draw_panel_c(ax, data):
    """Frozen protocol: 6 targets x 10 seeds delta point plot."""
    rob = data["robust"]
    frozen = rob[rob["protocol"] == "frozen"].copy()

    targets = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]
    x_positions = []
    all_deltas = []
    x_labels = []
    primary_map = {}

    for i, t in enumerate(targets):
        sub = frozen[frozen["target"] == t]
        deltas = sub["improvement_delta"].dropna().values
        all_deltas.append(deltas)
        x_positions.append(i)
        x_labels.append(DISPLAY.get(t, t))
        primary_map[t] = sub["primary_metric"].iloc[0] if len(sub) > 0 else ""

    # Boxplot
    bp = ax.boxplot(all_deltas, positions=x_positions, widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor="#E8E8E8", edgecolor="#333", linewidth=0.7),
                    medianprops=dict(color=COLOR_EXP, linewidth=1.5),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    # Overlay individual seed points
    for i, deltas in enumerate(all_deltas):
        rng = np.random.default_rng(i + 42)
        jitter = rng.uniform(-0.08, 0.08, size=len(deltas))
        colors_arr = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]
        ax.scatter(np.full_like(deltas, i) + jitter, deltas,
                   s=22, c=colors_arr, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=5)

    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=FS-1)
    ax.set_ylabel("Delta Metric (After - Before)", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.12, 1.02, "(c)", transform=ax.transAxes, fontsize=FS+5, fontweight="bold")
    ax.set_title("Frozen Protocol: 10-Seed Per-Target Gains", fontweight="bold", fontsize=FS+1)

    # Legend
    leg_elements = [Patch(facecolor=COLOR_POS, edgecolor="white", label="Delta >= 0"),
                    Patch(facecolor=COLOR_NEG, edgecolor="white", label="Delta < 0")]
    ax.legend(handles=leg_elements, frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-3,
              loc="lower left")


def draw_panel_d(ax, data):
    """95% CI bar chart: Frozen protocol, median + IQR error bars."""
    rob = data["robust"]
    frozen = rob[rob["protocol"] == "frozen"].copy()

    targets = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]
    medians = []
    q1s = []
    q3s = []
    labels = []

    for t in targets:
        sub = frozen[frozen["target"] == t]
        deltas = sub["improvement_delta"].dropna().values
        medians.append(np.median(deltas))
        q1s.append(np.percentile(deltas, 25))
        q3s.append(np.percentile(deltas, 75))
        labels.append(DISPLAY.get(t, t))

    x = np.arange(len(targets))
    colors = [COLOR_POS if m >= 0 else COLOR_NEG for m in medians]
    bars = ax.bar(x, medians, 0.55, color=colors, edgecolor="white", linewidth=0.5)

    # Error bars (25th-75th percentile)
    yerr_low = [max(0, m - q1) if m >= 0 else m - q1 for m, q1 in zip(medians, q1s)]
    yerr_high = [q3 - m if m >= 0 else min(0, q3 - m) for m, q3 in zip(medians, q3s)]
    ax.errorbar(x, medians, yerr=[yerr_low, yerr_high], fmt="none",
                ecolor="#333333", capsize=4, linewidth=1.0)

    # Value labels
    for i, (m, q1, q3) in enumerate(zip(medians, q1s, q3s)):
        label_y = m + 0.003 if m >= 0 else m - 0.008
        va = "bottom" if m >= 0 else "top"
        ax.text(i, label_y, f"{m:+.3f}", ha="center", va=va, fontsize=FS-2, fontweight="bold",
                color=COLOR_POS if m >= 0 else COLOR_NEG)

    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=FS-1)
    ax.set_ylabel("Median Delta (After - Before)", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.12, 1.02, "(d)", transform=ax.transAxes, fontsize=FS+5, fontweight="bold")
    ax.set_title("Frozen: 95% CI (10-Seed Median + IQR)", fontweight="bold", fontsize=FS+1)


def draw_panel_e(ax, data):
    """Frozen vs Adaptive protocol comparison: paired boxplot."""
    rob = data["robust"]
    frozen = rob[rob["protocol"] == "frozen"]
    adaptive = rob[rob["protocol"] == "adaptive"]

    targets = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]

    # For each target, get Frozen and Adaptive deltas
    frozen_deltas = []
    adaptive_deltas = []
    labels = []

    for t in targets:
        fd = frozen[frozen["target"] == t]["improvement_delta"].dropna().values
        ad = adaptive[adaptive["target"] == t]["improvement_delta"].dropna().values
        if len(fd) > 0:
            frozen_deltas.append(fd)
        else:
            frozen_deltas.append(np.array([]))
        if len(ad) > 0:
            adaptive_deltas.append(ad)
        else:
            adaptive_deltas.append(np.array([]))
        labels.append(DISPLAY.get(t, t))

    x_f = np.arange(len(targets)) - 0.18
    x_a = np.arange(len(targets)) + 0.18

    # Frozen boxplots
    bp_f = ax.boxplot([fd for fd in frozen_deltas if len(fd) > 0],
                      positions=x_f[[i for i, fd in enumerate(frozen_deltas) if len(fd) > 0]],
                      widths=0.28, patch_artist=True, manage_ticks=False,
                      boxprops=dict(facecolor=COLOR_FROZEN, edgecolor="#1a5c7a", linewidth=0.7, alpha=0.85),
                      medianprops=dict(color="white", linewidth=1.3),
                      whiskerprops=dict(color=COLOR_FROZEN, linewidth=0.7),
                      capprops=dict(color=COLOR_FROZEN, linewidth=0.7),
                      flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor=COLOR_FROZEN))

    # Adaptive boxplots (handle empty gracefully)
    ad_nonempty = [(i, ad) for i, ad in enumerate(adaptive_deltas) if len(ad) > 0]
    if ad_nonempty:
        positions_a = [x_a[i] for i, _ in ad_nonempty]
        data_a = [ad for _, ad in ad_nonempty]
        bp_a = ax.boxplot(data_a, positions=positions_a, widths=0.28,
                          patch_artist=True, manage_ticks=False,
                          boxprops=dict(facecolor=COLOR_ADAPTIVE, edgecolor="#7a1a4a", linewidth=0.7, alpha=0.85),
                          medianprops=dict(color="white", linewidth=1.3),
                          whiskerprops=dict(color=COLOR_ADAPTIVE, linewidth=0.7),
                          capprops=dict(color=COLOR_ADAPTIVE, linewidth=0.7),
                          flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor=COLOR_ADAPTIVE))

    # Paired connecting lines (same seed)
    for t_idx, t in enumerate(targets):
        fd_df = frozen[frozen["target"] == t][["seed", "improvement_delta"]].set_index("seed")
        ad_df = adaptive[adaptive["target"] == t][["seed", "improvement_delta"]].set_index("seed")
        common_seeds = fd_df.index.intersection(ad_df.index)
        for s in common_seeds:
            fv = fd_df.loc[s, "improvement_delta"]
            av = ad_df.loc[s, "improvement_delta"]
            if not (isinstance(fv, pd.Series) or isinstance(av, pd.Series)):
                ax.plot([x_f[t_idx], x_a[t_idx]], [fv, av],
                        color=GRAY, linewidth=0.4, alpha=0.35)

    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=FS-1)
    ax.set_ylabel("Delta Metric (After - Before)", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.12, 1.02, "(e)", transform=ax.transAxes, fontsize=FS+5, fontweight="bold")
    ax.set_title("Frozen vs. Adaptive Protocol Stability", fontweight="bold", fontsize=FS+1)

    leg_elements = [Patch(facecolor=COLOR_FROZEN, edgecolor="#1a5c7a", label="Frozen"),
                    Patch(facecolor=COLOR_ADAPTIVE, edgecolor="#7a1a4a", label="Adaptive")]
    ax.legend(handles=leg_elements, frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-3,
              loc="lower left")


# ══════════════════════════════════════════════════════════
def main():
    data = load_data()

    if not data:
        print("ERROR: No data found. Run fair_holdout_comparison.py first.")
        return

    # ── Combined Figure ──
    fig = plt.figure(figsize=(18, 9), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                          width_ratios=[1.05, 0.95, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    axes = {"a": ax_a, "b": ax_b, "c": ax_c, "d": ax_d, "e": ax_e}

    draw_panel_a(ax_a, data)
    draw_panel_b(ax_b, data)
    draw_panel_c(ax_c, data)
    draw_panel_d(ax_d, data)
    draw_panel_e(ax_e, data)

    fig.suptitle("Data Distribution Context and Controlled Data-Addition Effects",
                 fontsize=FS+4, fontweight="bold", y=1.01)

    # Save combined
    for fmt in ["png", "pdf", "svg"]:
        fp = os.path.join(SAVE_DIR, f"Figure2_Combined.{fmt}")
        fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] Figure2_Combined saved")

    # ── Individual Panels ──
    for panel_id, draw_fn in [("a", draw_panel_a), ("b", draw_panel_b),
                                ("c", draw_panel_c), ("d", draw_panel_d),
                                ("e", draw_panel_e)]:
        fig_i, ax_i = plt.subplots(figsize=(6, 5), dpi=300)
        draw_fn(ax_i, data)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            fp = os.path.join(SAVE_DIR, f"Figure2_Panel{panel_id}.{fmt}")
            fig_i.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_i)
        print(f"[OK] Figure2_Panel{panel_id} saved")

    print(f"\nAll outputs -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
