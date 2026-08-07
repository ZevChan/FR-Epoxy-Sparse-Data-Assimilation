"""
plot_figure5_dependence_v1.py — Figure 5 v1: 6 panels (2x3) — FIXED
========================================================================
(a)-(e): Signed SHAP dependence, per variable type
(f): Empirical alignment
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

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; DARK = "#333333"; GRAY = "#8C92AC"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
FEATURES = {"LOI":"B01[C-P]_FR","UL94_Rating":"F01[C-P]_FR","THR":"Eta_sh_x_Curing",
            "TSP":"Mor32e_FR","Flexural_Strength":"MDEN-22_FR"}

def load_data():
    return pd.read_csv(SHAP_CSV)

def draw_dependence(ax, df, target):
    feat = FEATURES[target]
    sub = df[(df["target"]==target)&(df["feature"]==feat)]
    before = sub[sub["track"]=="before"]; after = sub[sub["track"]=="after"]
    xb = before["feature_value_raw"].values; yb = before["shap_value"].values
    xa = after["feature_value_raw"].values; ya = after["shap_value"].values
    x_all = np.concatenate([xb, xa]) if len(xb)>0 and len(xa)>0 else (xb if len(xb)>0 else xa)
    x_range = np.ptp(x_all) if len(x_all)>1 else 1.0
    offset = max(x_range * 0.015, 0.005)
    unique_vals = len(set(np.round(x_all, 6)))
    is_binary = (unique_vals <= 2)
    # Detect zero-inflated
    zero_frac = np.mean(x_all == 0) if len(x_all)>0 else 0

    if is_binary:
        for i, (xvals, yvals, color, off) in enumerate(
            [(xb,yb,COLOR_BEFORE,-offset),(xa,ya,COLOR_AFTER,+offset)]):
            jitter = np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off, yvals, s=10, c=color, alpha=0.3, edgecolors="none")
            for val in sorted(set(xvals)):
                mask = xvals==val; n = mask.sum()
                if n>=3:
                    med = np.median(yvals[mask])
                    bs = [np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    ci_lo, ci_hi = np.percentile(bs,[2.5,97.5])
                    w = offset*0.8
                    ax.plot([val+off-w,val+off+w],[med,med],color=color,linewidth=2.5)
                    ax.plot([val+off,val+off],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
    elif zero_frac > 0.3:
        # Zero-inflated: x=0 group separate, non-zero binned. Only dodge medians, not scatter.
        for i, (xvals, yvals, color) in enumerate([(xb,yb,COLOR_BEFORE),(xa,ya,COLOR_AFTER)]):
            dodge = -offset if i==0 else +offset
            jitter = np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            # Scatter at true x (no global offset, only jitter)
            ax.scatter(xvals + jitter, yvals, s=6, c=color, alpha=0.2, edgecolors="none")
            # Zero group
            mask0 = xvals==0; n0 = mask0.sum()
            if n0>=3:
                med = np.median(yvals[mask0])
                bs = [np.median(yvals[mask0][np.random.default_rng(j).integers(0,n0,n0)]) for j in range(500)]
                ci_lo,ci_hi = np.percentile(bs,[2.5,97.5])
                w = offset*0.8
                ax.plot([dodge-w,dodge+w],[med,med],color=color,linewidth=2.5)
                ax.plot([dodge,dodge],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
            nz = xvals[xvals>0]
            y_nz = yvals[xvals>0]
            if len(nz)>=10:
                deciles = np.percentile(nz, np.arange(0,101,20))
                bx=[]; bm=[]; blo=[]; bhi=[]
                for bi in range(len(deciles)-1):
                    mask = (nz>=deciles[bi])&(nz<deciles[bi+1]); n=mask.sum()
                    if n>=3:
                        bx.append((deciles[bi]+deciles[bi+1])/2)
                        m=np.median(y_nz[mask])
                        bs=[np.median(y_nz[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                        bm.append(m); blo.append(np.percentile(bs,2.5)); bhi.append(np.percentile(bs,97.5))
                if len(bx)>1:
                    ax.plot(bx,bm,color=color,linewidth=2.0)
                    ax.fill_between(bx,blo,bhi,color=color,alpha=0.1)
    elif unique_vals <= 30:
        for i, (xvals, yvals, color, off) in enumerate(
            [(xb,yb,COLOR_BEFORE,-offset),(xa,ya,COLOR_AFTER,+offset)]):
            jitter = np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off, yvals, s=6, c=color, alpha=0.25, edgecolors="none")
            uniq = sorted(set(xvals))
            for val in uniq:
                mask = xvals==val; n = mask.sum()
                if n>=3:
                    med = np.median(yvals[mask])
                    bs = [np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    ci_lo, ci_hi = np.percentile(bs,[2.5,97.5])
                    w = offset*0.8
                    ax.plot([val+off-w,val+off+w],[med,med],color=color,linewidth=2.5)
                    ax.plot([val+off,val+off],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
    else:
        for i, (xvals, yvals, color) in enumerate([(xb,yb,COLOR_BEFORE),(xa,ya,COLOR_AFTER)]):
            jitter = np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter, yvals, s=5, c=color, alpha=0.2, edgecolors="none")
            deciles = np.percentile(xvals, np.arange(0,101,10))
            bx=[]; bm=[]; blo=[]; bhi=[]
            for bi in range(len(deciles)-1):
                mask = (xvals>=deciles[bi])&(xvals<deciles[bi+1]); n=mask.sum()
                if n>=3:
                    bx.append((deciles[bi]+deciles[bi+1])/2)
                    m=np.median(yvals[mask])
                    bs=[np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    bm.append(m); blo.append(np.percentile(bs,2.5)); bhi.append(np.percentile(bs,97.5))
            if len(bx)>1:
                ax.plot(bx,bm,color=color,linewidth=2.0)
                ax.fill_between(bx,blo,bhi,color=color,alpha=0.1)
    ax.axhline(0,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_xlabel(feat,fontweight="bold",fontsize=FS-1)
    ylabel = "SHAP value (V-0 log-odds)" if target=="UL94_Rating" else "SHAP value"
    ax.set_ylabel(ylabel,fontweight="bold",fontsize=FS-1)
    # LOI: show only 0/1 ticks
    if target=="LOI":
        ax.set_xticks([0,1])
        ax.set_xticklabels(["0 (absent)","1 (present)"])

def panel_f_alignment(ax, df):
    """SHAP-stratified pairwise outcome concordance per target (After-assimilation)."""
    # Metric: P(high-SHAP group y_true > low-SHAP group y_true) + 0.5*P(=).
    # Uniform direction across all targets (no higher/lower-is-better special cases):
    # SHAP sign describes predicted-value shifts, not material-quality direction.
    results = []
    for target in TARGETS:
        feat = FEATURES[target]
        sub = df[(df["target"]==target)&(df["feature"]==feat)&(df["track"]=="after")]
        if len(sub)==0: continue
        # Clean NaN/Inf
        shap = pd.to_numeric(sub["shap_value"], errors="coerce").to_numpy()
        yt = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy()
        valid = np.isfinite(shap) & np.isfinite(yt)
        shap = shap[valid]; yt = yt[valid]
        if len(shap) < 20: continue
        # Median split
        med = np.median(shap)
        high = yt[shap >= med]; low = yt[shap < med]
        n_high = len(high); n_low = len(low)
        if min(n_high, n_low) < 10: continue
        # Observed concordance
        prob = float(np.mean((high[:,None] > low[None,:]).flatten())
                     + 0.5*np.mean((high[:,None] == low[None,:]).flatten()))
        # Bootstrap (single RNG, stratified for UL-94)
        rng = np.random.default_rng(42)
        bs = []
        is_cls = (target == "UL94_Rating")
        for _ in range(5000):
            if is_cls:
                idx0 = np.where(yt==0)[0]; idx1 = np.where(yt==1)[0]
                if len(idx0)==0 or len(idx1)==0: continue
                idx = np.concatenate([rng.choice(idx0,size=len(idx0),replace=True),
                                      rng.choice(idx1,size=len(idx1),replace=True)])
            else:
                idx = rng.integers(0,len(yt),size=len(yt))
            sb = shap[idx]; yb = yt[idx]
            mb = np.median(sb)
            hb = yb[sb>=mb]; lb = yb[sb<mb]
            if len(hb)>0 and len(lb)>0:
                b = float(np.mean((hb[:,None]>lb[None,:]).flatten())
                          + 0.5*np.mean((hb[:,None]==lb[None,:]).flatten()))
                bs.append(b)
        if len(bs) < 20: continue
        ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
        results.append({"target":target,"prob":prob,"ci_lo":ci_lo,"ci_hi":ci_hi,
                        "n_high":n_high,"n_low":n_low,"n_boot_valid":len(bs)})
    res = pd.DataFrame(results)
    y = np.arange(len(res))[::-1]
    for i,(_,row) in enumerate(res.iterrows()):
        # CI-based three-way color (no implied significance claim)
        if row["ci_lo"] > 0.5: c = COLOR_BEFORE
        elif row["ci_hi"] < 0.5: c = COLOR_AFTER
        else: c = GRAY
        ax.plot([row["ci_lo"],row["ci_hi"]],[y[i],y[i]],color=c,linewidth=2.5,alpha=0.7)
        ax.scatter([row["prob"]],[y[i]],s=80,c=c,zorder=5,edgecolors="white",linewidth=1)
    ax.axvline(0.5,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in res["target"]],fontsize=FS-1,fontweight="bold")
    ax.set_xlabel("SHAP-stratified pairwise outcome concordance",fontweight="bold",fontsize=FS-1)
    ax.set_xlim(0.25, max(0.85, res["ci_hi"].max()+0.05) if len(res)>0 else 0.85)
    ax.grid(axis="x",linestyle=":",alpha=0.3)
    ax.set_title("Empirical alignment",fontweight="bold",fontsize=FS+1,color=DARK)

def main():
    df = load_data()
    fig = plt.figure(figsize=(24, 16), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
    for ax in axes: ax.set_box_aspect(1)

    # New layout: (a) = pairwise concordance, (b)-(f) = dependence panels
    # (a) first cell — CI-based color legend
    panel_f_alignment(axes[0], df)
    axes[0].text(-0.10,1.02,"(a)",transform=axes[0].transAxes,fontsize=FS+4,fontweight="bold")
    axes[0].legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="CI above 0.5"),
                            Patch(facecolor=GRAY,edgecolor="white",label="CI crosses 0.5"),
                            Patch(facecolor=COLOR_AFTER,edgecolor="white",label="CI below 0.5")],
                   frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-3,loc="upper left")

    # (b)-(f): dependence for LOI, UL-94, THR, TSP, Flexural (shifted by 1)
    for idx, target in enumerate(TARGETS):
        ax_i = axes[idx+1]
        draw_dependence(ax_i, df, target)
        ax_i.set_title(DISP[target], fontweight="bold", fontsize=FS+2)
        ax_i.text(-0.10,1.02,f"({chr(98+idx)})",transform=ax_i.transAxes,fontsize=FS+4,fontweight="bold")
        # Before/After legend on every dependence panel
        ax_i.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Before"),
                             Patch(facecolor=COLOR_AFTER,edgecolor="white",label="After")],
                    frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-2,loc="lower right")

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure5_v1.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"[OK] v1 -> {SAVE}")

if __name__=="__main__": main()
