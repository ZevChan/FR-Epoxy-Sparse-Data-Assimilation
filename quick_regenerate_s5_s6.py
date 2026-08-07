"""quick_regenerate_s5_s6.py — Regenerate S5 (prediction diagnostics) and S6 (UL-94)"""
import os, warnings, glob
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

R1 = os.path.join(BASE, "Results")
SI = os.path.join(BASE, "Graphs", "SI")
C_B = "#5DA5DA"; C_A = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 12
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.direction":"in","ytick.direction":"in","pdf.fonttype":42,"ps.fonttype":42})
T4 = ["LOI","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}

# S5: Prediction-error diagnostics
fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
for ti, t in enumerate(T4):
    fp = os.path.join(R1, "Frozen", f"predictions_{t}_seed_42.csv")
    if not os.path.exists(fp): continue
    pred = pd.read_csv(fp)
    yt = pred["y_true"].values; pb = pred["y_pred_before"].values; pa = pred["y_pred_after"].values
    # Row 1: abs error
    ax = axes[0,ti]; mx = max(pred["abs_error_before"].max(),pred["abs_error_after"].max())*1.1
    ax.scatter(pred["abs_error_before"],pred["abs_error_after"],s=10,alpha=0.4,c=GRAY,edgecolors="none")
    ax.plot([0,mx],[0,mx],color=DARK,linewidth=0.7,linestyle="--")
    ax.set_xlabel("Abs error Before"); ax.set_ylabel("Abs error After" if ti==0 else "")
    ax.set_title(DISP[t],fontweight="bold"); ax.grid(linestyle=":",alpha=0.3)
    # Row 2: parity
    ax = axes[1,ti]; rng = [min(yt),max(yt)]
    ax.scatter(yt,pb,s=10,alpha=0.4,c=C_B,edgecolors="none",label="Before")
    ax.scatter(yt,pa,s=10,alpha=0.4,c=C_A,edgecolors="none",label="After")
    ax.plot(rng,rng,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted" if ti==0 else "")
    if ti==0: ax.legend(frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-3)
    ax.grid(linestyle=":",alpha=0.3)
plt.tight_layout()
for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S5.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
plt.close(fig); print("[OK] S5")

# S6: UL-94 class-specific diagnostics
fm = pd.read_csv(os.path.join(R1,"Frozen","frozen_metrics.csv"))
ul = fm[(fm["target"]=="UL94_Rating")&(fm["protocol"]=="frozen")]
seeds_ul = sorted(ul["seed"].unique())
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
# (a) V-0 recall
ax = axes[0,0]
for s in seeds_ul:
    row = ul[ul["seed"]==s]
    if len(row)>0:
        ax.scatter([0],[row["before_v0_recall"].values[0]],s=40,c=C_B,alpha=0.7,edgecolors="white",zorder=5)
        ax.scatter([1],[row["after_v0_recall"].values[0]],s=40,c=C_A,alpha=0.7,edgecolors="white",zorder=5)
        ax.plot([0,1],[row["before_v0_recall"].values[0],row["after_v0_recall"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
ax.set_xticks([0,1]); ax.set_xticklabels(["Before","After"],fontweight="bold")
ax.set_ylabel("V-0 Recall"); ax.grid(axis="y",linestyle=":",alpha=0.3); ax.set_title("(a)",fontweight="bold")
# (b) Non-V-0 recall
ax = axes[0,1]
for s in seeds_ul:
    row = ul[ul["seed"]==s]
    if len(row)>0:
        ax.scatter([0],[row["before_non_v0_recall"].values[0]],s=40,c=C_B,alpha=0.7,edgecolors="white",zorder=5)
        ax.scatter([1],[row["after_non_v0_recall"].values[0]],s=40,c=C_A,alpha=0.7,edgecolors="white",zorder=5)
        ax.plot([0,1],[row["before_non_v0_recall"].values[0],row["after_non_v0_recall"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
ax.set_xticks([0,1]); ax.set_xticklabels(["Before","After"],fontweight="bold")
ax.set_ylabel("Non-V-0 Recall"); ax.grid(axis="y",linestyle=":",alpha=0.3); ax.set_title("(b)",fontweight="bold")
# (c) MCC + F1
ax = axes[0,2]
for s in seeds_ul:
    row = ul[ul["seed"]==s]
    if len(row)>0:
        ax.scatter([0],[row["before_mcc"].values[0]],s=40,c=C_B,alpha=0.7,edgecolors="white",zorder=5)
        ax.scatter([1],[row["after_mcc"].values[0]],s=40,c=C_A,alpha=0.7,edgecolors="white",zorder=5)
        ax.plot([0,1],[row["before_mcc"].values[0],row["after_mcc"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
        ax.scatter([2],[row["before_f1"].values[0]],s=40,c=C_B,alpha=0.7,edgecolors="white",zorder=5)
        ax.scatter([3],[row["after_f1"].values[0]],s=40,c=C_A,alpha=0.7,edgecolors="white",zorder=5)
        ax.plot([2,3],[row["before_f1"].values[0],row["after_f1"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["MCC-B","MCC-A","F1-B","F1-A"],rotation=20,fontweight="bold")
ax.set_ylabel("Score"); ax.grid(axis="y",linestyle=":",alpha=0.3); ax.set_title("(c)",fontweight="bold")
# (d) McNemar discordant
ax = axes[1,0]
disc = ul["n_discordant"].dropna().values
ax.bar(range(len(disc)),disc,color=C_B,edgecolor="white",linewidth=0.5)
ax.set_xlabel("Seed index"); ax.set_ylabel("Discordant pairs"); ax.grid(axis="y",linestyle=":",alpha=0.3); ax.set_title("(d)",fontweight="bold")
# (e) Confusion matrices
ax = axes[1,1]
pred_path = os.path.join(R1,"Frozen","predictions_UL94_Rating_seed_42.csv")
if os.path.exists(pred_path):
    pred = pd.read_csv(pred_path); yt = pred["y_true"].astype(int); pb = pred["y_pred_before"].astype(int)
    cm = confusion_matrix(yt,pb)
    ax.imshow(cm,cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=FS+4,fontweight="bold",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Non-V-0","Pred V-0"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["True Non-V-0","True V-0"])
    ax.set_title("(e) CM Before (seed 42)",fontweight="bold")
# (f) McNemar p
ax = axes[1,2]
mcp = ul["mcnemar_p"].dropna().values
ax.bar(range(len(mcp)),mcp,color=C_A if np.median(mcp)<0.05 else GRAY,edgecolor="white",linewidth=0.5)
ax.axhline(0.05,color=DARK,linewidth=0.7,linestyle="--",alpha=0.6)
ax.set_xlabel("Seed index"); ax.set_ylabel("McNemar p"); ax.grid(axis="y",linestyle=":",alpha=0.3); ax.set_title("(f)",fontweight="bold")
plt.tight_layout()
for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S6.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
plt.close(fig); print("[OK] S6")
print("Done")
