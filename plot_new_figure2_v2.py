"""
plot_new_figure2_v2.py — Figure 2: 3x2 portrait layout, corrected semantics
=============================================================================
Row 1: (a) Sample composition   (b) UMAP
Row 2: (c) kNN sparsity        (d) Forest plot
Row 3: (e) Frozen vs Adaptive   (f) UL-94 metrics
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVISED_DIR = BASE_DIR
RESULTS_DIR = os.path.join(REVISED_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs", "Figure2_Rebuilt_v2")
os.makedirs(SAVE_DIR, exist_ok=True)

# Colors — blue=Literature/Before/Frozen, red=Experiment/After/Adaptive
COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER  = "#C91511"
GRAY = "#8C92AC"
LIGHT_GRAY = "#D0D0D0"
DARK = "#333333"
FS = 18  # large for 3x2 layout — after scaling ~8pt
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

TARGETS_5 = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]
DISPLAY = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural strength"}

# ═══════════════ DATA ═══════════════
def load_data():
    d = {}
    for key, fname in [("umap","Distribution/umap_coordinates.csv"),
                       ("ss","SampleSizes/target_sample_sizes.csv"),
                       ("robust","Robustness/multiseed_metrics.csv"),
                       ("fm","Frozen/frozen_metrics.csv")]:
        fp = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fp): d[key] = pd.read_csv(fp)
    # PCA kNN from SI directory (Table S5)
    pca_path = os.path.join(BASE_DIR, "Graphs", "SI", "Table_S5_PCA.csv")
    if os.path.exists(pca_path): d["knn"] = pd.read_csv(pca_path)
    else:
        # fallback to old full-space data
        fp = os.path.join(RESULTS_DIR, "Distribution", "experimental_knn_density.csv")
        if os.path.exists(fp): d["knn"] = pd.read_csv(fp)
    if "umap" in d:
        d["umap_lit"] = d["umap"][d["umap"]["source"]=="Literature"]
        d["umap_exp"] = d["umap"][d["umap"]["source"]=="Experiment"]
    return d

def compute_forest_stats(data):
    rob = data["robust"]; frozen = rob[rob["protocol"]=="frozen"]
    rows = []
    for t in TARGETS_5:
        sub = frozen[frozen["target"]==t]; deltas = sub["improvement_delta"].dropna().values
        n = len(deltas)
        if n<3: rows.append({"target":t,"mean":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_raw":np.nan,"q_bh":np.nan}); continue
        mean_d = np.mean(deltas); se = stats.sem(deltas)
        ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_d, scale=se)
        _, p_raw = stats.ttest_1samp(deltas, 0)
        rows.append({"target":t,"mean":mean_d,"ci_low":ci_low,"ci_high":ci_high,"p_raw":p_raw,"n":n})
    df = pd.DataFrame(rows).dropna(subset=["p_raw"]); m = len(df)
    df = df.sort_values("p_raw").reset_index(drop=True); df["rank"] = np.arange(1,m+1)
    df["q_bh"] = df["p_raw"]*m/df["rank"]
    for i in range(m-2,-1,-1): df.iloc[i, df.columns.get_loc("q_bh")] = min(df.iloc[i]["q_bh"], df.iloc[i+1]["q_bh"])
    df["q_bh"] = df["q_bh"].clip(upper=1.0)
    df["target_cat"] = pd.Categorical(df["target"], categories=TARGETS_5, ordered=True)
    return df.sort_values("target_cat").reset_index(drop=True)

# ═══════════════ PANELS ═══════════════

def panel_a_sample(ax, data):
    """(a) Target-wise sample composition."""
    ss = data["ss"]
    ss_pri = ss[ss["seed"]==42].copy()
    if len(ss_pri) == 0:
        # seed 42 removed from formal evaluation; sample sizes are seed-invariant,
        # so fall back to the first formal seed per target
        ss_pri = ss.groupby("target").first().reset_index()
    targets = TARGETS_5
    lit_tr=[]; exp_n=[]; test_n=[]; aug_pct=[]; labels=[]
    for t in targets:
        r = ss_pri[ss_pri["target"]==t]
        if len(r)>0:
            lit_tr.append(int(r["n_literature_train"].values[0])); exp_n.append(int(r["n_experiment"].values[0]))
            test_n.append(int(r["n_test"].values[0])); aug_pct.append(r["augmentation_percent"].values[0])
            labels.append(DISPLAY.get(t,t))
    x = np.arange(len(targets)); w = 0.55
    ax.bar(x, lit_tr, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5, label="Literature training")
    ax.bar(x, exp_n, w, bottom=lit_tr, color=COLOR_AFTER, edgecolor="white", linewidth=0.5, label="Experiment")
    btm = np.array(lit_tr)+np.array(exp_n)
    ax.bar(x, test_n, w, bottom=btm, color=LIGHT_GRAY, edgecolor="white", linewidth=0.5, alpha=0.6, label="Literature holdout")
    for i, (pct, exp) in enumerate(zip(aug_pct, exp_n)):
        ax.text(i, btm[i]+test_n[i]+max(lit_tr)*0.03, f"n={exp} (+{pct:.1f}%)", ha="center",
                fontsize=FS-4, fontweight="bold", color=COLOR_AFTER)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Number of samples", fontweight="bold", fontsize=FS-1)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-5, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(a)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")


def panel_b_umap(ax, data):
    """(b) UMAP global context."""
    import seaborn as sns
    lit = data["umap_lit"]; exp = data["umap_exp"]
    sns.kdeplot(x=lit["umap_1"], y=lit["umap_2"], fill=True, alpha=0.15, color="#7f8c8d", levels=8, thresh=0.05, ax=ax)
    ax.scatter(lit["umap_1"], lit["umap_2"], c=COLOR_BEFORE, s=12, alpha=0.35, edgecolors="none",
               label=f"Literature (n={len(lit)})")
    ax.scatter(exp["umap_1"], exp["umap_2"], c=COLOR_AFTER, s=60, alpha=0.95, marker="o",
               edgecolors="black", linewidth=0.6, label=f"Experiment (n={len(exp)})")
    ax.set_xlabel("UMAP 1 (a.u.)", fontweight="bold", fontsize=FS-1)
    ax.set_ylabel("UMAP 2 (a.u.)", fontweight="bold", fontsize=FS-1)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="upper right", markerscale=1.2)
    ax.text(-0.10, 1.02, "(b)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")


def panel_c_knn(ax, data):
    """(c) kNN local sparsity — PCA-space distances (Table S5). P95=red, P90=blue, rest gray."""
    knn = data["knn"].copy()
    knn["idx"] = np.arange(1, len(knn)+1)
    pct = knn["Literature_distance_percentile"].values
    n_p90 = int((knn["Above_P90"]=="Yes").sum()); n_p95 = int((knn["Above_P95"]=="Yes").sum())
    # Color logic: P95=red, P90-P95=blue, below P90=gray
    colors = []
    for _, row in knn.iterrows():
        if row["Above_P95"] == "Yes": colors.append(COLOR_AFTER)
        elif row["Above_P90"] == "Yes": colors.append(COLOR_BEFORE)
        else: colors.append(LIGHT_GRAY)
    ax.bar(knn["idx"].values, pct, color=colors, edgecolor="white", linewidth=0.3, width=0.7, alpha=0.9)
    ax.axhline(90, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(95, color=DARK, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(50, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Experimental sample index", fontweight="bold", fontsize=FS-1)
    ax.set_ylabel("Distance percentile vs. literature (PCA, %)", fontweight="bold", fontsize=FS-1)
    ax.set_ylim(0, 118)
    # legend with color explanation
    from matplotlib.lines import Line2D
    leg = [Patch(facecolor=COLOR_AFTER, edgecolor="white", label=f">P95 (n={n_p95})"),
           Patch(facecolor=COLOR_BEFORE, edgecolor="white", label=f"P90-P95 (n={n_p90-n_p95})"),
           Patch(facecolor=LIGHT_GRAY, edgecolor="white", label=f"<=P90 (n={24-n_p90})"),
           Line2D([0],[0],color=GRAY,linestyle="--",lw=1.0,label="P90"),
           Line2D([0],[0],color=DARK,linestyle="--",lw=1.0,label="P95")]
    ax.legend(handles=leg, frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-5, loc="upper left", ncol=2)
    ax.text(-0.10, 1.02, "(c)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")


def panel_d_forest(ax, data):
    """(d) Horizontal dot-and-whisker forest plot: neutral gray, q-values right-aligned."""
    df = compute_forest_stats(data); n = len(df)
    y = np.arange(n)[::-1]
    for i, (_, row) in enumerate(df.iterrows()):
        yi = y[i]; c = COLOR_BEFORE if row["mean"]>=0 else COLOR_AFTER
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=c, linewidth=2.8, alpha=0.7)
        ax.scatter([row["mean"]], [yi], s=100, c=c, zorder=5, edgecolors="white", linewidth=1)
        q_str = f"BH q = {row['q_bh']:.3f}"
        ax.text(row["ci_high"] + abs(row["ci_high"])*0.04, yi, q_str, va="center",
                fontsize=FS-4, color="#555555", fontstyle="italic")
    ax.axvline(0, color=DARK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([DISPLAY.get(t,t) for t in df["target"]], fontsize=FS-1, fontweight="bold")
    ax.set_xlabel("Change in primary score (After \u2212 Before)", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(d)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")


def panel_e_frozen_vs_adaptive(ax, data):
    """(e) Frozen vs Adaptive: show all 10 seed endpoints."""
    rob = data["robust"]; frozen = rob[rob["protocol"]=="frozen"]; adaptive = rob[rob["protocol"]=="adaptive"]
    targets = TARGETS_5
    x_f = np.arange(len(targets))-0.18; x_a = np.arange(len(targets))+0.18

    # Boxplots
    for i, t in enumerate(targets):
        fd = frozen[frozen["target"]==t]["improvement_delta"].dropna().values
        ad = adaptive[adaptive["target"]==t]["improvement_delta"].dropna().values
        if len(fd)>0:
            ax.boxplot([fd], positions=[x_f[i]], widths=0.24, patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=COLOR_BEFORE,edgecolor="#3a7bb5",linewidth=0.7,alpha=0.85),
                       medianprops=dict(color="white",linewidth=1.5),
                       whiskerprops=dict(color=COLOR_BEFORE,linewidth=0.7),
                       capprops=dict(color=COLOR_BEFORE,linewidth=0.7),
                       flierprops=dict(marker='', markersize=0))
        if len(ad)>0:
            ax.boxplot([ad], positions=[x_a[i]], widths=0.24, patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=COLOR_AFTER,edgecolor="#a0100d",linewidth=0.7,alpha=0.85),
                       medianprops=dict(color="white",linewidth=1.5),
                       whiskerprops=dict(color=COLOR_AFTER,linewidth=0.7),
                       capprops=dict(color=COLOR_AFTER,linewidth=0.7),
                       flierprops=dict(marker='', markersize=0))

    # Seed dots
    for i, t in enumerate(targets):
        fd = frozen[frozen["target"]==t]["improvement_delta"].dropna().values
        ad = adaptive[adaptive["target"]==t]["improvement_delta"].dropna().values
        jf = np.random.default_rng(i).uniform(-0.06,0.06,size=len(fd))
        ja = np.random.default_rng(i+10).uniform(-0.06,0.06,size=len(ad))
        ax.scatter(np.full_like(fd,x_f[i])+jf, fd, s=18, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=5)
        ax.scatter(np.full_like(ad,x_a[i])+ja, ad, s=18, c=COLOR_AFTER, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=5)
        # Paired lines
        for s in range(min(len(fd), len(ad))):
            ax.plot([x_f[i], x_a[i]], [fd[s], ad[s]], color=GRAY, linewidth=0.4, alpha=0.3)

    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([DISPLAY.get(t,t) for t in targets], rotation=25, fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Delta (After \u2212 Before)", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(e)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="#3a7bb5",label="Frozen"),
                       Patch(facecolor=COLOR_AFTER,edgecolor="#a0100d",label="Adaptive")],
              frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-5, loc="lower left")


def panel_f_ul94(ax, data):
    """(f) UL-94: Accuracy, Balanced accuracy, MCC — paired Before/After."""
    fm = data["fm"]; ul = fm[(fm["target"]=="UL94_Rating")&(fm["protocol"]=="frozen")]
    pairs = [("before_accuracy","after_accuracy","Accuracy"),
             ("before_balanced_accuracy","after_balanced_accuracy","Balanced\naccuracy"),
             ("before_mcc","after_mcc","MCC")]
    for i,(col_b,col_a,_) in enumerate(pairs):
        vals_b = ul[col_b].dropna().values; vals_a = ul[col_a].dropna().values
        seeds_vals = ul["seed"].values[:len(vals_b)]
        if len(vals_b)>0:
            jb = np.random.default_rng(i*3).uniform(-0.06,0.06,size=len(vals_b))
            ax.scatter(np.full_like(vals_b,i-0.18)+jb,vals_b,s=36,c=COLOR_BEFORE,alpha=0.75,edgecolors="white",linewidth=0.3,zorder=5)
            ax.plot([i-0.28,i-0.08],[np.median(vals_b)]*2,color=COLOR_BEFORE,linewidth=2.5)
        if len(vals_a)>0:
            ja = np.random.default_rng(i*3+1).uniform(-0.06,0.06,size=len(vals_a))
            ax.scatter(np.full_like(vals_a,i+0.18)+ja,vals_a,s=36,c=COLOR_AFTER,alpha=0.75,edgecolors="white",linewidth=0.3,zorder=5)
            ax.plot([i+0.08,i+0.28],[np.median(vals_a)]*2,color=COLOR_AFTER,linewidth=2.5)
        # Connect paired seeds
        for s in range(min(len(vals_b), len(vals_a))):
            ax.plot([i-0.18+0.01*s%2*0.04, i+0.18+0.01*(s+1)%2*0.04], [vals_b[s], vals_a[s]],
                    color=GRAY, linewidth=0.3, alpha=0.25)

    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Accuracy","Balanced\naccuracy","MCC"], fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold", fontsize=FS-1)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.text(-0.10, 1.02, "(f)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Before"),
                       Patch(facecolor=COLOR_AFTER,edgecolor="white",label="After")],
              frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-5, loc="lower left")

# ═══════════════ MAIN ═══════════════
def main():
    data = load_data()
    if not data: print("ERROR: No data"); return

    # 2 rows × 3 columns
    fig = plt.figure(figsize=(26, 18), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes: ax.set_box_aspect(1)

    panel_a_sample(axes[0], data); panel_b_umap(axes[1], data); panel_c_knn(axes[2], data)
    panel_d_forest(axes[3], data); panel_e_frozen_vs_adaptive(axes[4], data); panel_f_ul94(axes[5], data)

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE_DIR, f"Figure2.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure2 saved")

    # Individual panels
    fns = [("a",panel_a_sample),("b",panel_b_umap),("c",panel_c_knn),
           ("d",panel_d_forest),("e",panel_e_frozen_vs_adaptive),("f",panel_f_ul94)]
    for pid, fn in fns:
        fi, axi = plt.subplots(figsize=(8,8), dpi=300); fn(axi, data); axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE_DIR, f"Figure2_{pid}.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fi); print(f"[OK] Figure2_{pid} saved")
    print(f"All -> {SAVE_DIR}")

if __name__ == "__main__": main()
