"""
plot_figure6.py — Figure 6: Target-Specific Feature-Attribution Reorganization (seed 42)
===========================================================================================
(a) LOI: relative feature-importance redistribution
(b) UL-94: minor SHAP-share redistribution
(c) THR: feature-importance redistribution
(d) Flexural: feature-importance redistribution
(e) TSP: distribution of normalized SHAP shares

All SHAP values are normalized within each track (sum=1).
Interpretation: relative importance share redistribution, not absolute SHAP magnitude change.
"""
import os, warnings, json
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
SAVE = os.path.join(BASE, "Graphs", "Figure6_Microscopic")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 13
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":2,"ytick.major.size":2,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","Flexural_Strength","TSP"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","Flexural_Strength":"Flexural","TSP":"TSP"}
SUBTITLES = {
    "LOI":"Relative feature-importance redistribution",
    "UL94_Rating":"Minor SHAP-share redistribution",
    "THR":"Feature-importance redistribution",
    "Flexural_Strength":"Feature-importance redistribution",
    "TSP":"Distribution of normalized SHAP shares",
}

def load_shap(target):
    """Load shap_feature_values.csv for seed 42, given target."""
    fp = os.path.join(KA,"WorkUnits","seed_42",target,"shap_feature_values.csv")
    if not os.path.exists(fp):
        fp = os.path.join(KA,"Plasticity","shap_feature_values.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        if "target" in df.columns:
            df = df[df["target"]==target]
        return df
    return None

def get_top_features(df, n=15):
    """Select top-n features by max normalized SHAP across Before and After (avoid truncation bias)."""
    if df is None or len(df)==0: return pd.DataFrame()
    before = df[df["track"]=="before"].copy()
    after = df[df["track"]=="after"].copy()
    if len(before)==0 or len(after)==0: return pd.DataFrame()
    # Merge by feature
    merged = before[["feature","normalized_mean_abs_shap"]].rename(columns={"normalized_mean_abs_shap":"shap_before"})
    merged = merged.merge(
        after[["feature","normalized_mean_abs_shap"]].rename(columns={"normalized_mean_abs_shap":"shap_after"}),
        on="feature", how="outer").fillna(0)
    merged["max_shap"] = merged[["shap_before","shap_after"]].max(axis=1)
    top = merged.nlargest(n, "max_shap")
    return top

def draw_dumbbell(ax, target):
    """Draw before-after dumbbell chart for one target."""
    df = load_shap(target)
    top = get_top_features(df, 15)
    if len(top)==0:
        ax.text(0.5,0.5,"Data not available",ha="center",va="center",transform=ax.transAxes,fontsize=FS)
        return
    y = np.arange(len(top))[::-1]
    for i, (_, row) in enumerate(top.iterrows()):
        yi = y[i]; bv = row["shap_before"]; av = row["shap_after"]
        ax.plot([bv, av], [yi, yi], color=GRAY, linewidth=1.5, alpha=0.7)
        ax.scatter([bv], [yi], s=50, c=COLOR_BEFORE, zorder=5, edgecolors="white", linewidth=0.5)
        ax.scatter([av], [yi], s=50, c=COLOR_AFTER, zorder=5, edgecolors="white", linewidth=0.5)
        # Shortened feature name
        feat = row["feature"]
        if len(feat)>40: feat = feat[:38]+"..."
        ax.text(-0.005, yi, feat, ha="right", va="center", fontsize=FS-4, color=DARK)
    ax.set_yticks([])
    ax.set_xlabel("Normalized |SHAP| share", fontweight="bold", fontsize=FS-2)
    ax.set_title(SUBTITLES.get(target,""), fontsize=FS-1, fontweight="bold", color=DARK)
    leg = [Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Before"),
           Patch(facecolor=COLOR_AFTER,edgecolor="white",label="After")]
    ax.legend(handles=leg, frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower right")

def main():
    fig = plt.figure(figsize=(20, 24), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30)
    panels = []
    for i, t in enumerate(TARGETS):
        r = i//2; c = i%2
        if i==4:  # TSP in last row center
            r=2; c=0
        ax = fig.add_subplot(gs[r,c]) if i<4 else fig.add_subplot(gs[2,:])
        if i<4: ax.set_box_aspect(1)
        draw_dumbbell(ax, t)
        ax.text(-0.08,1.02,f"({chr(97+i)})",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")
        panels.append((chr(97+i), t))

    # Remove empty cell at (2,1)
    empty_ax = fig.add_subplot(gs[2,1])
    empty_ax.set_visible(False)

    fig.text(0.5, 0.01,
        "Seed 42. SHAP values normalized within each track (sum=1). "
        "Interpretation: relative importance redistribution, not absolute magnitude change.",
        ha="center", fontsize=FS-3, fontstyle="italic", color="#555555")

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure6.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure6 saved")

    for pid, t in panels:
        fi, axi = plt.subplots(figsize=(8,8),dpi=300)
        draw_dumbbell(axi, t); axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE,f"Figure6_{pid}.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fi)
    print(f"All -> {SAVE}")

if __name__=="__main__": main()
