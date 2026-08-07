"""
plot_si_S37_S42.py — New SI Figures S37–S42
===============================================
S37: Dose curve decomposition (20 random subsets)
S38: 5×3 scatter matrix (utility vs novelty/surprise/compatibility)
S39: Sample-level loss improvement vs distance percentile
S40: Top-10 feature frequency vs median rank heatmap
S41: SHAP structure metrics 10-seed paired boxplots
S42: Top-5 features alignment probability forest plot
"""
import os, warnings, glob
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
R1 = os.path.join(BASE, "Results")
SI = os.path.join(BASE, "Graphs", "SI")
os.makedirs(SI, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 12
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":2,"ytick.major.size":2,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
TARGETS_REG = ["LOI","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}

# ═══════════ S37: Dose curve decomposition ═══════════
def figure_S37():
    fp = os.path.join(KA, "Dose", "assimilation_curve_raw.csv")
    if not os.path.exists(fp): print("[SKIP] S37: no raw dose data"); return
    raw = pd.read_csv(fp)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    axes_flat = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    axes[1,2].set_visible(False)
    for idx, t in enumerate(TARGETS):
        ax = axes_flat[idx]; sub = raw[raw["target"]==t]
        if len(sub)==0: continue
        doses = sorted(sub["dose_percent"].unique())
        for rep in sorted(sub["repeat"].unique()):
            s = sub[sub["repeat"]==rep].sort_values("dose_percent")
            ax.plot(s["dose_percent"], s["score"], color=GRAY, linewidth=0.3, alpha=0.4)
        # Mean ± CI
        summary = sub.groupby("dose_percent")["score"].agg(["mean", lambda x: np.percentile(x,2.5), lambda x: np.percentile(x,97.5)])
        ax.plot(summary.index, summary["mean"], color=COLOR_AFTER, linewidth=2.0)
        ax.fill_between(summary.index, summary["<lambda_0>"], summary["<lambda_1>"], color=COLOR_AFTER, alpha=0.12)
        ax.set_title(DISP[t], fontweight="bold")
        ax.set_xlabel("Dose (%)"); ax.set_ylabel("Score" if idx==0 else "")
        ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S37.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S37")

# ═══════════ S38: 5×3 scatter matrix ═══════════
def figure_S38():
    fp = os.path.join(KA, "DataValue", "experimental_data_value.csv")
    if not os.path.exists(fp): print("[SKIP] S38"); return
    dv = pd.read_csv(fp); d42 = dv[dv["seed"]==42]
    pairs = [("descriptor_novelty_percentile","Novelty"),("label_surprise","Surprise"),("local_compatibility","Compatibility")]
    fig, axes = plt.subplots(5, 3, figsize=(16, 24), dpi=300)
    for ti, t in enumerate(TARGETS):
        sub = d42[d42["target"]==t]
        for pi, (col, label) in enumerate(pairs):
            ax = axes[ti, pi]
            x = sub[col].dropna().values; y = sub["marginal_value"].dropna().values
            if len(x)>0 and len(y)>0:
                ax.scatter(x, y, s=15, c=COLOR_BEFORE, alpha=0.6, edgecolors="none")
                if len(x)>=3:
                    rho, p = stats.spearmanr(x, y)
                    ax.set_title(f"{DISP[t]}: r={rho:.2f}, p={p:.3f}", fontsize=FS-2)
            ax.set_xlabel(label if ti==4 else "")
            ax.set_ylabel("Utility" if pi==0 else "")
            ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S38.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S38")

# ═══════════ S39: Sample loss improvement vs distance ═══════════
def figure_S39():
    fp = os.path.join(KA, "Propagation", "propagation_samples.csv")
    if not os.path.exists(fp): print("[SKIP] S39"); return
    ps = pd.read_csv(fp); s42 = ps[ps["seed"]==42]
    reg_t = ["LOI","THR","TSP","Flexural_Strength"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    for idx, t in enumerate(reg_t):
        ax = axes[idx//2, idx%2]; sub = s42[s42["target"]==t]
        dist = sub["distance_percentile"].values; loss = sub["loss_improvement"].values
        ax.scatter(dist, loss, s=5, c=COLOR_BEFORE, alpha=0.3, edgecolors="none")
        # Binned median
        bins = np.percentile(dist, np.arange(0,101,10)); bx=[]; bm=[]
        for bi in range(len(bins)-1):
            mask = (dist>=bins[bi])&(dist<bins[bi+1])
            if mask.sum()>3:
                bx.append((bins[bi]+bins[bi+1])/2); bm.append(np.median(loss[mask]))
        if len(bx)>1: ax.plot(bx, bm, color=COLOR_AFTER, linewidth=2.5, marker='o', markersize=4)
        ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
        ax.set_title(DISP[t], fontweight="bold"); ax.set_xlabel("Distance percentile"); ax.set_ylabel("Loss improvement")
        ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S39.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S39")

# ═══════════ S40: Top-10 feature frequency vs median rank ═══════════
def figure_S40():
    fp = os.path.join(R1, "SHAP_Stability", "shap_feature_stability.csv")
    if not os.path.exists(fp): print("[SKIP] S40"); return
    sf = pd.read_csv(fp); frozen = sf[sf["protocol"]=="frozen"]
    # Get top-15 features by median frequency across targets
    top_feats = frozen.groupby("feature")["top10_frequency"].median().nlargest(15).index
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    axes_flat = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    axes[1,2].set_visible(False)
    for idx, t in enumerate(TARGETS):
        ax = axes_flat[idx]; sub = frozen[frozen["target"]==t]
        sub = sub[sub["feature"].isin(top_feats)]
        if len(sub)==0: continue
        before = sub[sub["track"]=="before"]; after = sub[sub["track"]=="after"]
        for _, row in before.iterrows():
            ax.scatter(row["top10_frequency"], row["median_rank"], s=30, c=COLOR_BEFORE, alpha=0.6, edgecolors="white", zorder=3)
        for _, row in after.iterrows():
            ax.scatter(row["top10_frequency"], row["median_rank"], s=30, c=COLOR_AFTER, alpha=0.6, edgecolors="white", zorder=4)
        ax.invert_yaxis(); ax.set_title(DISP[t], fontweight="bold")
        ax.set_xlabel("Top-10 frequency"); ax.set_ylabel("Median rank" if idx%3==0 else "")
        ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S40.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S40")

# ═══════════ S41: SHAP structure metrics paired boxplots ═══════════
def figure_S41():
    fp = os.path.join(R1, "SHAP_Stability", "shap_structure_metrics.csv")
    if not os.path.exists(fp): print("[SKIP] S41"); return
    sm = pd.read_csv(fp); frozen = sm[sm["protocol"]=="frozen"]
    metrics = ["top1_share","top5_share","effective_feature_number","shap_entropy"]
    titles = ["Top-1 SHAP share","Top-5 SHAP share","Effective feature N","SHAP entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    for mi, (col, title) in enumerate(zip(metrics, titles)):
        ax = axes[mi//2, mi%2]
        before = frozen[frozen["track"]=="before"]; after = frozen[frozen["track"]=="after"]
        for i, t in enumerate(TARGETS):
            bv = before[before["target"]==t][col].dropna().values
            av = after[after["target"]==t][col].dropna().values
            if len(bv)>0 and len(av)>0:
                ax.boxplot([bv], positions=[i*2-0.3], widths=0.5, patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=COLOR_BEFORE,edgecolor="#3a7bb5",alpha=0.85),
                           medianprops=dict(color="white",linewidth=1.5), flierprops=dict(markersize=3))
                ax.boxplot([av], positions=[i*2+0.3], widths=0.5, patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=COLOR_AFTER,edgecolor="#a0100d",alpha=0.85),
                           medianprops=dict(color="white",linewidth=1.5), flierprops=dict(markersize=3))
        ax.set_xticks(range(0,len(TARGETS)*2,2))
        ax.set_xticklabels([DISP[t] for t in TARGETS], rotation=20, fontsize=FS-2, fontweight="bold")
        ax.set_ylabel(col); ax.set_title(title, fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S41.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S41")

# ═══════════ S42: Top-5 features alignment forest plot ═══════════
def figure_S42():
    fp = os.path.join(KA, "SHAP_Relationships", "signed_shap_samples.csv")
    if not os.path.exists(fp): print("[SKIP] S42"); return
    df = pd.read_csv(fp)
    feat_path = os.path.join(KA, "Plasticity", "shap_feature_values.csv")
    top_feats = {}
    if os.path.exists(feat_path):
        fv = pd.read_csv(feat_path)
        for t in TARGETS:
            sub = fv[(fv["target"]==t)&(fv["seed"]==42)]
            top5 = sub.nlargest(5, "normalized_mean_abs_shap")["feature"].tolist()
            top_feats[t] = top5
    if not top_feats:
        # Fallback: use known representative features
        top_feats = {t: [f] for t, f in {"LOI":"B01[C-P]_FR","UL94_Rating":"F01[C-P]_FR",
                     "THR":"Eta_sh_x_Curing","TSP":"Mor32e_FR","Flexural_Strength":"MDEN-22_FR"}.items()}

    fig, axes = plt.subplots(5, 1, figsize=(10, 20), dpi=300)
    for ti, t in enumerate(TARGETS):
        ax = axes[ti]; features = top_feats.get(t, [])
        results = []
        for feat in features[:5]:
            sub = df[(df["target"]==t)&(df["feature"]==feat)&(df["track"]=="after")]
            if len(sub)==0: continue
            med = sub["shap_value"].median()
            high = sub[sub["shap_value"]>=med]["y_true"].values
            low = sub[sub["shap_value"]<med]["y_true"].values
            if len(high)==0 or len(low)==0: continue
            if t=="UL94_Rating":
                prob = np.mean((high[:,None]>low[None,:]).flatten()) + 0.5*np.mean((high[:,None]==low[None,:]).flatten())
            elif t in ["THR","TSP"]:
                prob = np.mean((low[:,None]<high[None,:]).flatten())
            else:
                prob = np.mean((high[:,None]>low[None,:]).flatten())
            bs = []; rng = np.random.default_rng(42); n = len(sub)
            for _ in range(500):
                idx = rng.integers(0,n,n); hs = sub.iloc[idx]; hm = hs["shap_value"].median()
                hh = hs[hs["shap_value"]>=hm]["y_true"].values; ll = hs[hs["shap_value"]<hm]["y_true"].values
                if len(hh)>0 and len(ll)>0:
                    if t=="UL94_Rating": b = np.mean((hh[:,None]>ll[None,:]).flatten())+0.5*np.mean((hh[:,None]==ll[None,:]).flatten())
                    elif t in ["THR","TSP"]: b = np.mean((ll[:,None]<hh[None,:]).flatten())
                    else: b = np.mean((hh[:,None]>ll[None,:]).flatten()); bs.append(b)
            ci_lo, ci_hi = np.percentile(bs,[2.5,97.5]) if len(bs)>0 else (0.5,0.5)
            feat_short = feat[:50]
            results.append({"feature":feat_short,"prob":prob,"ci_lo":ci_lo,"ci_hi":ci_hi})
        if not results: continue
        res = pd.DataFrame(results); y = np.arange(len(res))[::-1]
        for i, (_, row) in enumerate(res.iterrows()):
            c = COLOR_BEFORE if row["prob"]>=0.5 else COLOR_AFTER
            ax.plot([row["ci_lo"],row["ci_hi"]],[y[i],y[i]],color=c,linewidth=2.5,alpha=0.7)
            ax.scatter([row["prob"]],[y[i]],s=80,c=c,zorder=5,edgecolors="white",linewidth=1)
        ax.axvline(0.5, color=DARK, linewidth=0.7, linestyle="--")
        ax.set_yticks(y); ax.set_yticklabels(res["feature"], fontsize=FS-2)
        ax.set_title(DISP[t], fontweight="bold"); ax.set_xlim(0.3, 0.8); ax.grid(axis="x", linestyle=":", alpha=0.3)
        if ti==4: ax.set_xlabel("P(observed perf. aligns with SHAP)")
    plt.tight_layout()
    for fmt in ["png","pdf"]: fig.savefig(os.path.join(SI,f"Figure_S42.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure_S42")

def main():
    figure_S37(); figure_S38(); figure_S39()
    figure_S40(); figure_S41(); figure_S42()
    print(f"\nAll SI -> {SI}")

if __name__=="__main__": main()
