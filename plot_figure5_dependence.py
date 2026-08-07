"""
plot_figure5_dependence.py — SHAP dependence panels (5 targets, 5 panels)
===========================================================================
(a) LOI       (b) UL-94       (c) THR
(d) TSP       (e) Flexural

x = raw descriptor value, y = signed SHAP, blue=Before, red=After.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
SHAP_CSV = os.path.join(BASE, "Results", "Knowledge_Assimilation", "SHAP_Relationships", "signed_shap_samples.csv")
SAVE = os.path.join(BASE, "Graphs", "Figure5_Dependence")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; DARK = "#333333"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
FEATURES = {"LOI":"B01[C-P]_FR","UL94_Rating":"F01[C-P]_FR","THR":"Eta_sh_x_Curing",
            "TSP":"Mor32e_FR","Flexural_Strength":"MDEN-22_FR"}

def lowess_trend(x, y, frac=0.3):
    """Simple LOWESS-like trend using rolling median+bins."""
    order = np.argsort(x); xs = x[order]; ys = y[order]
    n = len(xs); window = max(int(n*frac), 10)
    trend = np.array([np.median(ys[max(0,i-window//2):min(n,i+window//2)]) for i in range(n)])
    return xs, trend

def bootstrap_ci(x, y, frac=0.3, n_boot=200):
    """Bootstrap CI for LOWESS trend."""
    n = len(x); window = max(int(n*frac), 10)
    rng = np.random.default_rng(42)
    all_trends = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs_b = x[idx]; ys_b = y[idx]
        order = np.argsort(xs_b)
        trend = np.array([np.median(ys_b[order][max(0,j-window//2):min(n,j+window//2)]) for j in range(n)])
        all_trends.append(trend[np.argsort(order)])
    all_trends = np.array(all_trends)
    lo = np.percentile(all_trends, 2.5, axis=0)
    hi = np.percentile(all_trends, 97.5, axis=0)
    x_sorted = np.sort(x)
    return x_sorted, lo, hi

def main():
    df = pd.read_csv(SHAP_CSV)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), dpi=300)
    axes_flat = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    axes[1,2].set_visible(False)

    for idx, target in enumerate(TARGETS):
        ax = axes_flat[idx]
        feat = FEATURES[target]
        sub = df[(df["target"]==target)&(df["feature"]==feat)]

        before = sub[sub["track"]=="before"]
        after = sub[sub["track"]=="after"]

        # Determine shared x/y range
        x_all = pd.concat([before["feature_value_raw"], after["feature_value_raw"]])
        y_all = pd.concat([before["shap_value"], after["shap_value"]])
        x_min, x_max = x_all.min(), x_all.max()
        x_pad = (x_max-x_min)*0.05 if x_max>x_min else 1
        y_min, y_max = y_all.min(), y_all.max()
        y_pad = max(abs(y_min), abs(y_max))*0.1

        # Before
        if len(before)>0:
            xb = before["feature_value_raw"].values; yb = before["shap_value"].values
            ax.scatter(xb, yb, s=6, c=COLOR_BEFORE, alpha=0.25, edgecolors="none")
            xs, trend = lowess_trend(xb, yb)
            ax.plot(xs, trend, color=COLOR_BEFORE, linewidth=2.0)
            try:
                xs_ci, lo, hi = bootstrap_ci(xb, yb)
                ax.fill_between(xs_ci, lo, hi, color=COLOR_BEFORE, alpha=0.12)
            except: pass

        # After
        if len(after)>0:
            xa = after["feature_value_raw"].values; ya = after["shap_value"].values
            ax.scatter(xa, ya, s=6, c=COLOR_AFTER, alpha=0.25, edgecolors="none")
            xs, trend = lowess_trend(xa, ya)
            ax.plot(xs, trend, color=COLOR_AFTER, linewidth=2.0)
            try:
                xs_ci, lo, hi = bootstrap_ci(xa, ya)
                ax.fill_between(xs_ci, lo, hi, color=COLOR_AFTER, alpha=0.12)
            except: pass

        ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
        ax.set_xlabel(feat, fontweight="bold", fontsize=FS-1)
        ax.set_ylabel("SHAP value", fontweight="bold", fontsize=FS-1)
        ax.set_title(DISP[target], fontweight="bold", fontsize=FS+2)
        ax.grid(linestyle=":", alpha=0.3)
        ax.text(-0.10, 1.02, f"({chr(97+idx)})", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")

        # Legend on first panel only
        if idx==0:
            ax.legend(handles=[Patch(facecolor=COLOR_BEFORE, edgecolor="white", label="Before"),
                               Patch(facecolor=COLOR_AFTER, edgecolor="white", label="After")],
                      frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-2, loc="upper right")

    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE, f"Figure5_Dependence.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved -> {SAVE}")

if __name__ == "__main__":
    main()
