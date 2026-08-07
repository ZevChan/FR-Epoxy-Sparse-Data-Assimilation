"""
plot_figure3.py — Figure 3: Dose-Dependent Assimilation (seed 42 diagnostics)
===============================================================================
(a) Regression dose-response: Delta-R2 vs 0% dose, 4 targets
(b) UL-94 dose-response: Delta Balanced Accuracy vs 0% dose
(c) LOEO marginal value: 5 targets, per-sample scatter
(d) LOEO sum vs full utility: non-additivity evidence
(e) Novelty/Surprise/Compatibility correlation heatmap

All results are seed 42 diagnostics. Utility at 100% dose:
LOI +0.00159 | UL-94 +0.00305 | THR −0.00154 | TSP +0.00443 | Flexural +0.02285
Shaded bands: 2.5th–97.5th percentile across 20 random experiment subsets.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
SAVE = os.path.join(BASE, "Graphs", "Figure3_Assimilation")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

REG = ["LOI","THR","TSP","Flexural_Strength"]  # pHRR removed
ALL5 = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
TAB10 = plt.cm.tab10

def load():
    d = {}
    for k,f in [("dose","Dose/assimilation_curve_summary.csv"),
                ("dv","DataValue/experimental_data_value.csv"),
                ("corr","DataValue/data_value_correlations.csv")]:
        fp = os.path.join(KA,f)
        if os.path.exists(fp): d[k] = pd.read_csv(fp)
    return d

# ═══════════ PANELS ═══════════

def panel_a_regression_dose(ax, data):
    """Delta-R2 vs 0% dose for 4 regression targets (seed 42)."""
    dose = data["dose"]; d42 = dose[dose["seed"]==42]
    for i, t in enumerate(REG):
        sub = d42[(d42["target"]==t)].sort_values("dose_percent")
        base_score = sub[sub["dose_percent"]==0]["score_mean"].values[0]
        delta = sub["score_mean"].values - base_score
        delta_lo = sub["score_ci_low"].values - base_score
        delta_hi = sub["score_ci_high"].values - base_score
        c = TAB10(i)
        ax.plot(sub["dose_percent"], delta, marker="o", markersize=5, linewidth=1.8, color=c, label=DISP[t])
        ax.fill_between(sub["dose_percent"], delta_lo, delta_hi, alpha=0.12, color=c)
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xlabel("Experiment dose (%)", fontweight="bold"); ax.set_ylabel("\u0394R\u00b2 (vs 0% dose)", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower right", ncol=2)
    ax.grid(linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(a)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_b_ul94_dose(ax, data):
    """Delta Balanced Accuracy for UL-94 (seed 42)."""
    dose = data["dose"]; d42 = dose[dose["seed"]==42]
    sub = d42[d42["target"]=="UL94_Rating"].sort_values("dose_percent")
    base_score = sub[sub["dose_percent"]==0]["score_mean"].values[0]
    delta = sub["score_mean"].values - base_score
    delta_lo = sub["score_ci_low"].values - base_score
    delta_hi = sub["score_ci_high"].values - base_score
    ax.plot(sub["dose_percent"], delta, marker="s", markersize=6, linewidth=1.8, color=COLOR_AFTER)
    ax.fill_between(sub["dose_percent"], delta_lo, delta_hi, alpha=0.15, color=COLOR_AFTER)
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xlabel("Experiment dose (%)", fontweight="bold"); ax.set_ylabel("\u0394 balanced accuracy (vs 0%)", fontweight="bold")
    ax.grid(linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(b)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_c_loeo(ax, data):
    """LOEO marginal value scatter for 5 targets (seed 42)."""
    dv = data["dv"]; d42 = dv[dv["seed"]==42]
    y_positions = []
    for i, t in enumerate(ALL5):
        sub = d42[d42["target"]==t]
        loeo = sub["marginal_value"].dropna().values
        if len(loeo)>0:
            jitter = np.random.default_rng(i).uniform(-0.12,0.12,size=len(loeo))
            colors_arr = [COLOR_BEFORE if v>=0 else COLOR_AFTER for v in loeo]
            ax.scatter(np.full_like(loeo,i)+jitter, loeo, s=28, c=colors_arr, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=5)
            y_positions.append(i)
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xticks(y_positions); ax.set_xticklabels([DISP[t] for t in ALL5], rotation=25, fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("LOEO Marginal Utility", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Beneficial"),
                       Patch(facecolor=COLOR_AFTER,edgecolor="white",label="Detrimental")],
              frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower left")
    ax.text(-0.08,1.02,"(c)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_d_fractions(ax, data):
    """(d) Beneficial/detrimental experiment fractions — 100% stacked bar."""
    # Seed 42 data (from user)
    fractions = {
        "LOI": (15, 2, 0), "UL94_Rating": (8, 2, 0),
        "THR": (4, 10, 0), "TSP": (12, 2, 0),
        "Flexural_Strength": (8, 11, 0),
    }
    targets = ALL5
    beneficial = []; detrimental = []; neutral = []
    for t in targets:
        b, d, n = fractions.get(t, (0,0,0))
        beneficial.append(b); detrimental.append(d); neutral.append(n)
    y = np.arange(len(targets))[::-1]
    totals = np.array(beneficial)+np.array(detrimental)+np.array(neutral)
    b_pct = np.array(beneficial)/totals*100
    d_pct = np.array(detrimental)/totals*100
    n_pct = np.array(neutral)/totals*100

    ax.barh(y, b_pct, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5, label="Beneficial")
    ax.barh(y, d_pct, left=b_pct, color=COLOR_AFTER, edgecolor="white", linewidth=0.5, label="Detrimental")
    if any(n > 0 for n in neutral):
        ax.barh(y, n_pct, left=b_pct+d_pct, color=GRAY, edgecolor="white", linewidth=0.5, label="Neutral")
    for i in range(len(targets)):
        ax.text(b_pct[i]/2, y[i], f"{beneficial[i]}/{totals[i]}", ha="center", va="center",
                fontsize=FS-3, fontweight="bold", color="white")
        ax.text(b_pct[i]+d_pct[i]/2, y[i], f"{detrimental[i]}/{totals[i]}", ha="center", va="center",
                fontsize=FS-3, fontweight="bold", color="white")
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in targets], fontsize=FS-1, fontweight="bold")
    ax.set_xlabel("Fraction of experiments (%)", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower right")
    ax.text(-0.08,1.02,"(d)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_e_nonadditivity(ax, data):
    """Sum of LOEO vs full utility (seed 42)."""
    dv = data["dv"]; d42 = dv[dv["seed"]==42]
    sum_vals = []; full_vals = []; labels = []
    for t in ALL5:
        sub = d42[d42["target"]==t]
        s = sub["marginal_value"].dropna().sum()
        f = sub["full_after_score"].dropna().values[0] - sub["before_score"].dropna().values[0]
        sum_vals.append(s); full_vals.append(f); labels.append(DISP[t])
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x-w/2, sum_vals, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5, label="Sum of LOEO utilities")
    ax.bar(x+w/2, full_vals, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5, label="Full dataset utility")
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Utility", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(e)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_f_heatmap(ax, data):
    """(f) Correlation heatmap: Novelty/Surprise/Compatibility x 5 targets (seed 42)."""
    corr = data["corr"]; c42 = corr[corr["seed"]==42]
    predictors = ["descriptor_novelty_percentile","label_surprise","local_compatibility"]
    pred_labels = ["Novelty","Surprise","Compatibility"]
    mat = np.zeros((len(ALL5), 3))
    annot = []
    for i, t in enumerate(ALL5):
        row_annot = []
        for j, p in enumerate(predictors):
            r = c42[(c42["target"]==t)&(c42["predictor"]==p)]
            if len(r)>0:
                mat[i,j] = r["spearman_rho"].values[0]
                row_annot.append(f"{mat[i,j]:.2f}")
            else:
                row_annot.append("")
        annot.append(row_annot)
    im = ax.imshow(mat, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(pred_labels, fontsize=FS-1, fontweight="bold", rotation=15)
    ax.set_yticks(range(len(ALL5))); ax.set_yticklabels([DISP[t] for t in ALL5], fontsize=FS-1, fontweight="bold")
    for i in range(len(ALL5)):
        for j in range(3):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=FS-1, fontweight="bold",
                    color="white" if abs(mat[i,j])>0.5 else "black")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho", fontsize=FS-2)
    ax.text(-0.08,1.02,"(f)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

# ═══════════ MAIN ═══════════
def main():
    data = load()
    fig = plt.figure(figsize=(24, 16), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
    for ax in axes: ax.set_box_aspect(1)

    panel_a_regression_dose(axes[0], data); panel_b_ul94_dose(axes[1], data)
    panel_c_loeo(axes[2], data); panel_d_fractions(axes[3], data)
    panel_e_nonadditivity(axes[4], data); panel_f_heatmap(axes[5], data)

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure3.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure3 saved")

    panels = [("a",panel_a_regression_dose),("b",panel_b_ul94_dose),
              ("c",panel_c_loeo),("d",panel_d_fractions),
              ("e",panel_e_nonadditivity),("f",panel_f_heatmap)]
    for pid, fn in panels:
        fi, axi = plt.subplots(figsize=(7,7),dpi=300); fn(axi,data); axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE,f"Figure3_{pid}.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fi)
    print(f"All -> {SAVE}")

if __name__=="__main__": main()
