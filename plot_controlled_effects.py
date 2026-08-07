"""
plot_controlled_effects.py — Figure 2: Task-level data addition effects
=========================================================================
(a) Effective training labels & augmentation ratio
(b) Frozen ΔR² (After − Before)
(c) Paired bootstrap 95% CI forest plot
(d) Multi-seed delta consistency
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

FS = 13
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

PRIMARY_SEED = 42

COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER = "#C91511"
COLOR_DELTA_POS = "#2E86AB"
COLOR_DELTA_NEG = "#A23B72"

DISPLAY_NAMES = {
    "LOI": "LOI", "THR": "THR", "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength", "UL94_Rating": "UL-94",
}


def plot():
    # ── Load data ──
    ss = pd.read_csv(os.path.join(RESULTS_DIR, "SampleSizes", "target_sample_sizes.csv"))
    ss_primary = ss[ss["seed"] == PRIMARY_SEED]

    frozen = pd.read_csv(os.path.join(RESULTS_DIR, "Frozen", "frozen_metrics.csv"))
    reg_frozen = frozen[frozen["target"] != "UL94_Rating"].copy()

    robust = pd.read_csv(os.path.join(RESULTS_DIR, "Robustness", "multiseed_metrics.csv"))
    robust_frozen = robust[robust["protocol"] == "frozen"]

    targets = ss_primary["target"].tolist()
    labels = [DISPLAY_NAMES.get(t, t) for t in targets]

    # ═══════════ Figure ═══════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # ── (a) Training labels & augmentation % ──
    ax = axes[0, 0]
    x = np.arange(len(targets))
    w = 0.35

    b1 = ax.bar(x - w/2, ss_primary["n_literature_train"], w,
                color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                label="Literature training labels")
    b2 = ax.bar(x + w/2, ss_primary["n_experiment"], w,
                color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                label="Experimental labels")

    # 标注 augmentation %
    for i, (_, row) in enumerate(ss_primary.iterrows()):
        pct = row["augmentation_percent"]
        if not np.isnan(pct):
            ax.text(i, max(row["n_literature_train"], row["n_experiment"]) * 1.05,
                    f"+{pct:.0f}%", ha="center", fontsize=9, fontweight="bold", color=COLOR_AFTER)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Number of Labels", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 3)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(a)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (b) Frozen ΔR² ──
    ax = axes[0, 1]
    # Filter to PRIMARY_SEED first, then sort by target
    fs_data = reg_frozen[reg_frozen["seed"] == PRIMARY_SEED].copy()
    fs_data = fs_data.sort_values("target")
    reg_targets = fs_data["target"].tolist()
    reg_labels = [DISPLAY_NAMES.get(t, t) for t in reg_targets]
    x2 = np.arange(len(reg_targets))

    colors = [COLOR_DELTA_POS if v >= 0 else COLOR_DELTA_NEG
              for v in fs_data["delta_r2"].values]
    bars = ax.bar(x2, fs_data["delta_r2"], 0.55, color=colors,
                  edgecolor="white", linewidth=0.5)

    # 数值标注
    for i, (_, row) in enumerate(fs_data.iterrows()):
        v = row["delta_r2"]
        ax.text(i, v + 0.003 * (1 if v >= 0 else -1),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold",
                color=COLOR_DELTA_POS if v >= 0 else COLOR_DELTA_NEG)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(reg_labels, rotation=25, ha="right")
    ax.set_ylabel("ΔR²  (After − Before)", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(b)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (c) Bootstrap 95% CI forest plot ──
    ax = axes[1, 0]
    y_positions = []
    for i, (_, row) in enumerate(fs_data.iterrows()):
        y = len(fs_data) - i - 1
        y_positions.append(y)
        delta = row["delta_r2"]
        ci_low = row.get("r2_ci_low", delta - 0.01)
        ci_high = row.get("r2_ci_high", delta + 0.01)
        color = COLOR_DELTA_POS if delta >= 0 else COLOR_DELTA_NEG

        ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=2.5, alpha=0.7)
        ax.scatter([delta], [y], color=color, s=60, zorder=5,
                   edgecolors="white", linewidth=0.8)

    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(reg_labels)
    ax.set_xlabel("ΔR²  (After − Before)", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(c)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── (d) Multi-seed delta distribution ──
    ax = axes[1, 1]
    robust_reg = robust_frozen[robust_frozen["target"] != "UL94_Rating"]
    reg_targets_r = robust_reg["target"].unique()
    reg_labels_r = [DISPLAY_NAMES.get(t, t) for t in reg_targets_r]

    positions = []
    data_groups = []
    for i, t in enumerate(reg_targets_r):
        deltas = robust_reg[robust_reg["target"] == t]["improvement_delta"].dropna().values
        data_groups.append(deltas)
        positions.append(i)

    bp = ax.boxplot(data_groups, positions=positions, widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor="#E8E8E8", edgecolor="#333333", linewidth=0.8),
                    medianprops=dict(color="#C91511", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    # 叠加每个 seed 的点
    for i, deltas in enumerate(data_groups):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(deltas))
        ax.scatter(np.full_like(deltas, i) + jitter, deltas,
                   s=18, alpha=0.5, color=COLOR_DELTA_POS, edgecolors="none")

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(reg_labels_r, rotation=25, ha="right")
    ax.set_ylabel("ΔR²  (After − Before)", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(d)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # ── Save ──
    fig.suptitle("Target-Dependent Predictive Responses under Controlled Data Addition",
                 fontsize=FS + 2, fontweight="bold", y=0.99)
    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fp = os.path.join(SAVE_DIR, f"Controlled_Effects.{fmt}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Controlled_Effects saved (png/pdf/svg)")


if __name__ == "__main__":
    plot()
