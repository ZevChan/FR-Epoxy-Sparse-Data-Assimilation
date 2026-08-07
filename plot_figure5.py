"""
plot_figure5.py — Figure 5: Utility–Plasticity Phase Map & Model Sensitivity (2x2)
=====================================================================================
(a) Phase map: XGBoost (PHASE plasticity) + ExtraTrees (SENS plasticity)
(b) Model-family utility: XGBoost (Frozen) vs ExtraTrees (SENS)
(c) Top-10 Jaccard median + IQR
(d) Spearman rank stability median + IQR
"""
import os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
R1 = os.path.join(BASE, "Results")
SAVE = os.path.join(BASE, "Graphs", "Figure5_PhaseMap")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
TOLERANCE = {"LOI":0.01,"UL94_Rating":0.02,"THR":0.01,"TSP":0.01,"Flexural_Strength":0.01}

def load():
    d = {}
    fp = os.path.join(KA,"Plasticity","utility_plasticity_phase_map.csv")
    if os.path.exists(fp): d["phase"] = pd.read_csv(fp)
    fp = os.path.join(KA,"ModelSensitivity","model_family_sensitivity.csv")
    if os.path.exists(fp): d["sens"] = pd.read_csv(fp)
    fp = os.path.join(R1,"SHAP_Stability","shap_ranking_similarity.csv")
    if os.path.exists(fp): d["sim"] = pd.read_csv(fp)
    fp = os.path.join(R1,"Frozen","frozen_metrics.csv")
    if os.path.exists(fp): d["fm"] = pd.read_csv(fp)
    return d

def panel_a_phase(ax, data):
    """Phase map: XGBoost (PHASE) vs ExtraTrees (SENS). Vertical state zones."""
    phase = data["phase"]; p42 = phase[phase["seed"]==42]
    sens = data["sens"]; s42 = sens[sens["seed"]==42]

    # XGBoost points from PHASE
    for _, row in p42.iterrows():
        t = row["target"]
        if t not in TARGETS: continue
        norm_u = row["utility"] / TOLERANCE.get(t, 0.01)
        plast = row["plasticity"]
        ax.scatter(norm_u, plast, s=140, c=COLOR_BEFORE, marker="o",
                   edgecolors="white", linewidth=1, zorder=5)
        offsets_xgb = {"LOI":(0.08,-0.04),"UL94_Rating":(-0.05,-0.04),"THR":(0.08,0.015),
                       "TSP":(-0.05,-0.04),"Flexural_Strength":(0.08,0.015)}
        dx, dy = offsets_xgb.get(t, (0.08,0.015))
        ax.annotate(DISP[t], (norm_u+dx, plast+dy), fontsize=FS-3, color=COLOR_BEFORE, fontweight="bold")

    # ExtraTrees points from SENS (use SENS's own plasticity)
    for _, row in s42.iterrows():
        t = row["target"]
        if t not in TARGETS: continue
        norm_u = row["utility"] / TOLERANCE.get(t, 0.01)
        plast = row["plasticity"]  # SENS has its own plasticity column
        ax.scatter(norm_u, plast, s=140, c=COLOR_AFTER, marker="s",
                   edgecolors="white", linewidth=1, zorder=5)
        ax.annotate(DISP[t], (norm_u+0.08, plast+0.015), fontsize=FS-3, color=COLOR_AFTER, fontweight="bold")

    # Vertical state zones
    ax.axvline(-1, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(1, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0.5, color=GRAY, linewidth=0.6, linestyle=":", alpha=0.5)
    # Zone labels inside plot
    ax.text(-0.6, 0.55, "Destructive", ha="center", fontsize=FS-3, color=GRAY, fontstyle="italic")
    ax.text(0, 0.55, "Inert\nupdate", ha="center", fontsize=FS-3, color=DARK, fontweight="bold")
    ax.text(1.8, 0.25, "Stable\nassimilation", ha="center", fontsize=FS-3, color=COLOR_BEFORE, fontweight="bold")
    ax.text(1.8, 0.57, "Plastic\nassimilation", ha="center", fontsize=FS-3, color=COLOR_AFTER, fontweight="bold")

    ax.set_xlim(-1.5, 2.8); ax.set_ylim(0, 0.65)
    ax.set_xlabel("Normalized utility (utility / target tolerance)", fontweight="bold", fontsize=FS-1)
    ax.set_ylabel("Plasticity (1 - Top-10 Jaccard)", fontweight="bold", fontsize=FS-1)
    ax.grid(linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(a)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COLOR_BEFORE,markersize=10,label="XGBoost"),
                       Line2D([0],[0],marker='s',color='w',markerfacecolor=COLOR_AFTER,markersize=10,label="ExtraTrees")],
              frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-3,loc="upper left")

def panel_b_model_family(ax, data):
    """XGBoost (Frozen) vs ExtraTrees (SENS) utility for 5 targets."""
    fm = data["fm"]; sens = data["sens"]
    fm42 = fm[(fm["seed"]==42)&(fm["protocol"]=="frozen")]; s42 = sens[sens["seed"]==42]
    xgb_u = []; et_u = []; labels = []
    for t in TARGETS:
        xrow = fm42[fm42["target"]==t]
        erow = s42[(s42["target"]==t)&(s42["model_family"]=="ExtraTrees")]
        if len(xrow)>0 and len(erow)>0:
            xgb_u.append(xrow["improvement_delta"].values[0])
            et_u.append(erow["utility"].values[0])
            labels.append(DISP[t])
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x-w/2, xgb_u, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5, label="XGBoost")
    ax.bar(x+w/2, et_u, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5, label="ExtraTrees")
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=FS-1, fontweight="bold")
    ax.set_ylabel("Delta primary metric", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-3, loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(b)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_c_jaccard(ax, data):
    """10-seed Top-10 Jaccard median + IQR per target."""
    sim = data["sim"]; frozen = sim[sim["protocol"]=="frozen"]
    meds=[]; q1s=[]; q3s=[]; labels=[]
    for t in TARGETS:
        sub = frozen[frozen["target"]==t]; jac = sub["top10_jaccard"].dropna().values
        if len(jac)>0:
            meds.append(np.median(jac)); q1s.append(np.percentile(jac,25))
            q3s.append(np.percentile(jac,75)); labels.append(DISP[t])
    x = np.arange(len(labels))
    ax.bar(x,meds,0.55,color=COLOR_BEFORE,edgecolor="white",linewidth=0.5)
    ax.errorbar(x,meds,yerr=[[m-q1 for m,q1 in zip(meds,q1s)],[q3-m for m,q3 in zip(meds,q3s)]],
                fmt="none",ecolor=DARK,capsize=4,linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=FS-1,fontweight="bold")
    ax.set_ylabel("Before-after Top-10 Jaccard",fontweight="bold"); ax.set_ylim(0,1.05)
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(c)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_d_spearman(ax, data):
    """10-seed Spearman rho median + IQR per target."""
    sim = data["sim"]; frozen = sim[sim["protocol"]=="frozen"]
    meds=[]; q1s=[]; q3s=[]; labels=[]
    for t in TARGETS:
        sub = frozen[frozen["target"]==t]; rho = sub["spearman_rho"].dropna().values
        if len(rho)>0:
            meds.append(np.median(rho)); q1s.append(np.percentile(rho,25))
            q3s.append(np.percentile(rho,75)); labels.append(DISP[t])
    x = np.arange(len(labels))
    ax.bar(x,meds,0.55,color=COLOR_AFTER,edgecolor="white",linewidth=0.5)
    ax.errorbar(x,meds,yerr=[[m-q1 for m,q1 in zip(meds,q1s)],[q3-m for m,q3 in zip(meds,q3s)]],
                fmt="none",ecolor=DARK,capsize=4,linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=FS-1,fontweight="bold")
    ax.set_ylabel("Before-after Spearman rank corr.",fontweight="bold"); ax.set_ylim(0,1.05)
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(d)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def main():
    data = load()
    fig = plt.figure(figsize=(20, 20), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    for ax in axes: ax.set_box_aspect(1)

    panel_a_phase(axes[0], data); panel_b_model_family(axes[1], data)
    panel_c_jaccard(axes[2], data); panel_d_spearman(axes[3], data)

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure5.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure5 saved")

    panels = [("a",panel_a_phase),("b",panel_b_model_family),("c",panel_c_jaccard),("d",panel_d_spearman)]
    for pid, fn in panels:
        fi, axi = plt.subplots(figsize=(7,7),dpi=300); fn(axi,data); axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE,f"Figure5_{pid}.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fi)
    print(f"All -> {SAVE}")

if __name__=="__main__": main()
