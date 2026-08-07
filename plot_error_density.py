"""
================================================================================
 plot_error_density.py — Prediction-error density before vs after experimental guidance
================================================================================
 Reads the predictions CSVs saved by fair_holdout_comparison.py,
 plots Before/After residual KDE curves on the same fixed test set.
 Outputs PNG / PDF / SVG.
 Colors: Before = #5DA5DA (steel blue), After = #C91511 (dark red)
         zero-error line = #8C92AC (cool slate gray)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import glob
BASE = os.path.dirname(os.path.abspath(__file__))

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
COLOR_BEFORE  = "#5DA5DA"   # steel blue — literature baseline
COLOR_AFTER   = "#C91511"   # dark red — after experimental guidance
COLOR_ZERO    = "#8C92AC"   # cool slate gray — zero-error reference

# regression targets (UL94 excluded; classification does not suit residual density)
REGRESSION_TARGETS = ["LOI", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
}

UNITS = {
    "LOI": "%",
    "pHRR": "kW m⁻²",
    "THR": "MJ m⁻²",
    "TSP": "m²",
    "Flexural_Strength": "MPa",
}

# ==================== Path configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Frozen")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_error_density():
    for target in REGRESSION_TARGETS:
        pred_files = sorted(glob.glob(
            os.path.join(RESULTS_DIR, f"predictions_{target}_seed_*.csv")))
        if not pred_files:
            print(f"  [WARN] skipping {target}: no predictions_{target}_seed_*.csv")
            continue

        df = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        residuals_before = y_true - y_pred_before
        residuals_after = y_true - y_pred_after

        unit = UNITS.get(target, "")

        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=600)

        # KDE curves
        from scipy.stats import gaussian_kde

        try:
            kde_before = gaussian_kde(residuals_before)
            kde_after = gaussian_kde(residuals_after)

            # unified x range
            all_res = np.concatenate([residuals_before, residuals_after])
            x_range = np.linspace(all_res.min() * 1.15, all_res.max() * 1.15, 300)

            ax.fill_between(x_range, kde_before(x_range), alpha=0.25, color=COLOR_BEFORE)
            ax.plot(x_range, kde_before(x_range), color=COLOR_BEFORE, linewidth=1.8,
                    label="Before (Literature only)")

            ax.fill_between(x_range, kde_after(x_range), alpha=0.35, color=COLOR_AFTER)
            ax.plot(x_range, kde_after(x_range), color=COLOR_AFTER, linewidth=1.8,
                    label="After (Literature + Experiment)")
        except Exception:
            # fall back to histograms
            ax.hist(residuals_before, bins=20, density=True, alpha=0.35,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.3,
                    label="Before (Literature only)")
            ax.hist(residuals_after, bins=20, density=True, alpha=0.45,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.3,
                    label="After (Literature + Experiment)")

        # zero-error reference line
        ax.axvline(0, color=COLOR_ZERO, linestyle="--", linewidth=1.0, alpha=0.8)

        display_name = DISPLAY_NAMES.get(target, target)
        ax.set_xlabel(f"Residual ({unit})" if unit else "Residual", fontsize=13)
        ax.set_ylabel("Probability Density", fontsize=13)
        ax.set_title(display_name, fontsize=14, fontweight="bold")

        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)  # only expand the Y upper limit

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Error_Density_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[OK] Error_Density_{target} saved (png/pdf/svg)")

    # ---- UL94 binary: confusion-matrix distribution comparison ----
    ul94_files = sorted(glob.glob(
        os.path.join(RESULTS_DIR, "predictions_UL94_Rating_seed_*.csv")))
    if ul94_files:
        df = pd.concat([pd.read_csv(f) for f in ul94_files], ignore_index=True)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        cm_b = confusion_matrix(y_true, y_pred_before, labels=[0, 1])
        cm_a = confusion_matrix(y_true, y_pred_after, labels=[0, 1])

        ul_labels = [
            "TN\n(True Non-V-0)",
            "FP\n(Non-V-0 → V-0)",
            "FN\n(V-0 → Non-V-0)",
            "TP\n(True V-0)",
        ]
        before_counts = [cm_b[0, 0], cm_b[0, 1], cm_b[1, 0], cm_b[1, 1]]
        after_counts = [cm_a[0, 0], cm_a[0, 1], cm_a[1, 0], cm_a[1, 1]]
        n_total = len(y_true)

        x = np.arange(len(ul_labels))
        width = 0.32

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(x - width/2, before_counts, width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x + width/2, after_counts, width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")

        # numeric labels
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=11, color=COLOR_BEFORE)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=11, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(ul_labels, fontsize=11)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title("UL-94 (V-0 vs Non V-0): Confusion Matrix Distribution",
                     fontsize=14, fontweight="bold")

        all_h = before_counts + after_counts
        ax.set_ylim(0, max(all_h) * 1.25)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Error_Density_UL94_Rating.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Error_Density_UL94_Rating saved (png/pdf/svg)")
    else:
        print("[WARN] skipping UL94_Rating: prediction files not found")


if __name__ == "__main__":
    print("Generating error-density plots ...")
    plot_error_density()
    print("Done.")