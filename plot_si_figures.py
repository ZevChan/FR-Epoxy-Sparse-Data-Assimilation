"""
plot_si_figures.py — SI Figures for Figure 2
=============================================
S1: 24x5 experimental label missingness heatmap
S2: Secondary metric robustness (RMSE/MAE relative delta + bootstrap)
S3: Prediction-error diagnostics (residuals, absolute errors, parity)
S4: Configuration-selection sensitivity (K, CV, features, HPO trajectories)
S5: UL-94 class-specific diagnostics (recall, MCC, confusion matrix, McNemar)
T2: kNN density table (CSV)
T3: Five-target complete statistical inference (CSV)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVISED_DIR = BASE_DIR
RESULTS_DIR = os.path.join(REVISED_DIR, "Results")
TARGET_CSV = os.path.join(BASE_DIR, "data", "BasicData", "Target.csv")
SI_DIR = os.path.join(BASE_DIR, "Graphs", "SI")
os.makedirs(SI_DIR, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER  = "#C91511"
GRAY = "#8C92AC"
FS = 11
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

TARGETS_5 = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]
TARGETS_REG = ["LOI", "THR", "TSP", "Flexural_Strength"]
DISPLAY = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural strength"}

def load_data():
    d = {}
    d["fm"] = pd.read_csv(os.path.join(RESULTS_DIR, "Frozen", "frozen_metrics.csv"))
    d["am"] = pd.read_csv(os.path.join(RESULTS_DIR, "Adaptive", "adaptive_metrics.csv"))
    d["robust"] = pd.read_csv(os.path.join(RESULTS_DIR, "Robustness", "multiseed_metrics.csv"))
    d["ss"] = pd.read_csv(os.path.join(RESULTS_DIR, "SampleSizes", "target_sample_sizes.csv"))
    d["knn"] = pd.read_csv(os.path.join(RESULTS_DIR, "Distribution", "experimental_knn_density.csv"))
    return d

# ═══════════════ S1: 24x5 experimental label missingness heatmap ═══════════════
def figure_s1():
    fp = TARGET_CSV
    if not os.path.exists(fp):
        print("[WARN] S1: Target.csv not found"); return
    # Read and detect encoding
    import chardet
    with open(fp, "rb") as f: enc = chardet.detect(f.read())["encoding"]
    df = pd.read_csv(fp, encoding=enc or "utf-8")
    # Last 24 rows, only target columns
    targets_found = [c for c in TARGETS_5 if c in df.columns]
    exp_rows = df.iloc[-24:][targets_found].copy()
    # Convert to binary: 1=measured, 0=missing
    mask = exp_rows.notna().astype(int)
    mask.index = range(1, 25)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    # Discrete two-color grid: blue=measured, red=not measured
    from matplotlib.colors import ListedColormap
    cmap_s1 = ListedColormap([COLOR_AFTER, COLOR_BEFORE])
    im = ax.imshow(mask.T, cmap=cmap_s1, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(1, 25), fontsize=FS-2, rotation=90)
    ax.set_yticks(range(len(targets_found)))
    ax.set_yticklabels([DISPLAY.get(t, t) for t in targets_found], fontsize=FS-1, fontweight="bold")
    ax.set_xlabel("Experimental Sample Index", fontweight="bold")

    for i in range(len(targets_found)):
        for j in range(24):
            v = mask.iloc[j, i]
            ax.text(j, i, "O" if v else "X", ha="center", va="center",
                    fontsize=FS-2, fontweight="bold", color="black" if v else "#990000")
    # Discrete legend instead of continuous colorbar
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=COLOR_BEFORE, edgecolor="white", label="O = Measured"),
        Patch(facecolor=COLOR_AFTER, edgecolor="white", label="X = Not measured"),
    ]
    ax.legend(handles=legend_patches, frameon=True, fancybox=False, edgecolor="#777",
              fontsize=FS-1, loc="lower right")
    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SI_DIR, f"Figure_S1.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure S1 saved")

# ═══════════════ T2: kNN density table ═══════════════
def table_s2(data):
    knn = data["knn"].copy()
    knn["experiment_index"] = knn["experiment_index"] + 1  # 1-indexed
    out_cols = ["experiment_index", "mean_knn_distance", "literature_distance_percentile", "above_p90", "above_p95"]
    knn[out_cols].to_csv(os.path.join(SI_DIR, "Table_S2_knn_density.csv"), index=False)
    print("[OK] Table S2 saved")

# ═══════════════ S2: Secondary metric robustness ═══════════════
def figure_s2(data):
    fm = data["fm"]; frozen = fm[fm["protocol"]=="frozen"]
    targets = TARGETS_REG
    metrics_info = [("delta_rmse", "rmse_ci_low", "rmse_ci_high", "RMSE"),
                    ("delta_mae", "mae_ci_low", "mae_ci_high", "MAE")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for ax_idx, (col_d, col_lo, col_hi, label) in enumerate(metrics_info):
        ax = axes[ax_idx]; x = np.arange(len(targets))
        medians, q1s, q3s = [], [], []
        for t in targets:
            sub = frozen[frozen["target"]==t]
            d = sub[col_d].dropna().values
            medians.append(np.median(d) if len(d)>0 else 0)
            q1s.append(np.percentile(d, 25) if len(d)>0 else 0)
            q3s.append(np.percentile(d, 75) if len(d)>0 else 0)
        colors = [COLOR_BEFORE if m>=0 else COLOR_AFTER for m in medians]
        ax.bar(x, medians, 0.55, color=colors, edgecolor="white", linewidth=0.5)
        yerr_low = [max(0,m-q1) if m>=0 else m-q1 for m,q1 in zip(medians,q1s)]
        yerr_high = [q3-m if m>=0 else min(0,q3-m) for m,q3 in zip(medians,q3s)]
        ax.errorbar(x, medians, yerr=[yerr_low, yerr_high], fmt="none", ecolor="#333", capsize=4, linewidth=1)
        ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY.get(t,t) for t in targets], rotation=20, fontsize=FS, fontweight="bold")
        ax.set_ylabel(f"Delta {label} (After-Before)", fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
    
    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SI_DIR, f"Figure_S2.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure S2 saved")

# ═══════════════ S3: Prediction-error diagnostics ═══════════════
def figure_s3(data):
    import glob
    targets = TARGETS_REG
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    for ti, t in enumerate(targets):
        fps = glob.glob(os.path.join(RESULTS_DIR, "Frozen", f"predictions_{t}_seed_*.csv"))
        if not fps: continue
        # use seed 42
        pred_path = os.path.join(RESULTS_DIR, "Frozen", f"predictions_{t}_seed_42.csv")
        if not os.path.exists(pred_path) and fps: pred_path = fps[0]
        if not os.path.exists(pred_path): continue
        pred = pd.read_csv(pred_path)
        yt = pred["y_true"].values; pb = pred["y_pred_before"].values; pa = pred["y_pred_after"].values

        # Row 1: absolute error
        ax_err = axes[0, ti]
        ax_err.scatter(pred["abs_error_before"], pred["abs_error_after"], s=12, alpha=0.5, c=GRAY, edgecolors="none")
        mx = max(pred["abs_error_before"].max(), pred["abs_error_after"].max()) * 1.1
        ax_err.plot([0,mx],[0,mx], color="#333", linewidth=0.7, linestyle="--")
        ax_err.set_xlabel("Abs Error Before", fontsize=FS-2); ax_err.set_ylabel("Abs Error After", fontsize=FS-2)
        ax_err.set_title(DISPLAY.get(t,t), fontweight="bold", fontsize=FS)
        ax_err.grid(linestyle=":", alpha=0.3)

        # Row 2: parity plot
        ax_par = axes[1, ti]
        ax_par.scatter(yt, pb, s=10, alpha=0.4, c=COLOR_BEFORE, edgecolors="none", label="Before")
        ax_par.scatter(yt, pa, s=10, alpha=0.4, c=COLOR_AFTER, edgecolors="none", label="After")
        rng = [min(yt), max(yt)]; ax_par.plot(rng, rng, color="#333", linewidth=0.7, linestyle="--")
        ax_par.set_xlabel("True", fontsize=FS-2); ax_par.set_ylabel("Predicted", fontsize=FS-2)
        ax_par.legend(frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-3, loc="upper left")
        ax_par.grid(linestyle=":", alpha=0.3)


    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SI_DIR, f"Figure_S3.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure S3 saved")


# ═══════════════ T3: Five-target complete statistical inference ═══════════════
def table_s3(data):
    fm = data["fm"]; frozen = fm[fm["protocol"]=="frozen"]
    rows = []
    for t in TARGETS_5:
        sub = frozen[frozen["target"]==t]
        deltas = sub["improvement_delta"].dropna().values
        n = len(deltas)
        if n < 3: continue
        mean_d = np.mean(deltas); sd_d = np.std(deltas, ddof=1)
        se = stats.sem(deltas)
        ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_d, scale=se)
        t_stat, p_ttest = stats.ttest_1samp(deltas, 0)
        # Wilcoxon
        try: _, p_wilcox = stats.wilcoxon(deltas, zero_method="zsplit")
        except: p_wilcox = np.nan
        pos = int((deltas>0).sum()); neg = int((deltas<0).sum())
        rows.append({"target":t, "n_seeds":n, "mean_delta":mean_d, "sd_delta":sd_d,
                     "ci_low":ci_low, "ci_high":ci_high, "positive_seeds":pos, "negative_seeds":neg,
                     "p_ttest":p_ttest, "p_wilcoxon":p_wilcox})
    df_inf = pd.DataFrame(rows)
    # BH correction
    df_inf = df_inf.sort_values("p_ttest").reset_index(drop=True)
    m = len(df_inf); df_inf["rank"] = np.arange(1,m+1)
    df_inf["q_bh_ttest"] = df_inf["p_ttest"]*m/df_inf["rank"]
    for i in range(m-2,-1,-1): df_inf.iloc[i, df_inf.columns.get_loc("q_bh_ttest")] = min(df_inf.iloc[i]["q_bh_ttest"], df_inf.iloc[i+1]["q_bh_ttest"])
    df_inf["q_bh_ttest"] = df_inf["q_bh_ttest"].clip(upper=1.0)
    df_inf = df_inf.sort_values("target").reset_index(drop=True)
    df_inf.to_csv(os.path.join(SI_DIR, "Table_S3_inference.csv"), index=False, float_format="%.6f")
    print("[OK] Table S3 saved")

# ═══════════════ S4: Configuration-selection sensitivity ═══════════════
def figure_s4(data):
    am = data["am"]; fm = data["fm"]; frozen = fm[fm["protocol"]=="frozen"]
    targets = TARGETS_REG

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # (a) Before/After K comparison
    ax = axes[0,0]
    for i, t in enumerate(targets):
        subs = frozen[frozen["target"]==t]
        k_frozen = subs["k"].values[0] if len(subs)>0 else np.nan
        sub_a = am[am["target"]==t]
        k_before = sub_a["before_k"].values; k_after = sub_a["after_k"].values
        ax.scatter(np.full_like(k_before, i-0.18), k_before, s=30, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", zorder=5)
        ax.scatter(np.full_like(k_after, i+0.18), k_after, s=30, c=COLOR_AFTER, alpha=0.7, edgecolors="white", zorder=5)
        if not np.isnan(k_frozen):
            ax.axhline(k_frozen, xmin=i/4-0.06, xmax=i/4+0.06, color=GRAY, linewidth=1.5, linestyle="--")
    ax.axhline(np.nan, color=GRAY, linewidth=1.5, linestyle="--", label="Frozen K")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([DISPLAY.get(t,t) for t in targets], rotation=20, fontsize=FS, fontweight="bold")
    ax.set_ylabel("Selected K", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-2)
    ax.set_title("(a) Before/After K vs Frozen K", fontweight="bold")

    # (b) CV score Before/After
    ax = axes[0,1]
    for i, t in enumerate(targets):
        sub_a = am[am["target"]==t]
        cv_b = sub_a["before_cv_score"].values; cv_a = sub_a["after_cv_score"].values
        ax.scatter(np.full_like(cv_b, i-0.18), cv_b, s=30, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", zorder=5)
        ax.scatter(np.full_like(cv_a, i+0.18), cv_a, s=30, c=COLOR_AFTER, alpha=0.7, edgecolors="white", zorder=5)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([DISPLAY.get(t,t) for t in targets], rotation=20, fontsize=FS, fontweight="bold")
    ax.set_ylabel("Inner-CV Score", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(b) Adaptive Before/After CV Score", fontweight="bold")

    # (c) Feature Jaccard Before vs After
    ax = axes[1,0]
    import glob, json
    jaccards = {}
    for t in targets:
        jac_list = []
        for seed in [42,7,13,19,29,37,43,53,61,71]:
            fp_b = os.path.join(RESULTS_DIR, "Adaptive", f"before_{t}_seed_{seed}_features.json")
            fp_a = os.path.join(RESULTS_DIR, "Adaptive", f"after_{t}_seed_{seed}_features.json")
            if os.path.exists(fp_b) and os.path.exists(fp_a):
                with open(fp_b) as fb: fb_feat = set(json.load(fb))
                with open(fp_a) as fa: fa_feat = set(json.load(fa))
                union = fb_feat | fa_feat
                jac = len(fb_feat & fa_feat)/len(union) if union else 0
                jac_list.append(jac)
        jaccards[t] = jac_list
    for i, t in enumerate(targets):
        vals = jaccards.get(t, [])
        if vals:
            jitter = np.random.default_rng(i).uniform(-0.08,0.08,size=len(vals))
            ax.scatter(np.full_like(vals,i)+jitter, vals, s=30, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", zorder=5)
            ax.plot([i-0.2,i+0.2],[np.median(vals)]*2, color=COLOR_AFTER, linewidth=2)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([DISPLAY.get(t,t) for t in targets], rotation=20, fontsize=FS, fontweight="bold")
    ax.set_ylabel("Before/After Feature Jaccard", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(c) Feature Overlap Before vs After", fontweight="bold")

    # (d) Representative CV trajectory (seed=42, LOI)
    ax = axes[1,1]
    traj_files = [
        (os.path.join(RESULTS_DIR, "Frozen", "trajectory_LOI_seed_42.csv"), "Frozen (LOI)", COLOR_BEFORE),
        (os.path.join(RESULTS_DIR, "Adaptive", "trajectory_before_LOI_seed_42.csv"), "Adaptive Before", COLOR_BEFORE),
        (os.path.join(RESULTS_DIR, "Adaptive", "trajectory_after_LOI_seed_42.csv"), "Adaptive After", COLOR_AFTER),
    ]
    for fp, label, color in traj_files:
        if os.path.exists(fp):
            traj = pd.read_csv(fp)
            ax.plot(traj["k"], traj["cv_score"], marker=".", color=color, alpha=0.8, linewidth=1.2, label=label, markersize=4)
    ax.set_xlabel("K", fontweight="bold"); ax.set_ylabel("CV Score", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#555", fontsize=FS-2)
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_title("(d) CV Score vs K (LOI, seed=42)", fontweight="bold")


    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SI_DIR, f"Figure_S4.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure S4 saved")

# ═══════════════ S5: UL-94 class-specific diagnostics ═══════════════
def figure_s5(data):
    fm = data["fm"]; ul = fm[(fm["target"]=="UL94_Rating")&(fm["protocol"]=="frozen")]
    import glob

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)

    # (a) V-0 recall Before vs After per seed
    ax = axes[0,0]
    seeds = ul["seed"].values
    for s in sorted(set(seeds)):
        row = ul[ul["seed"]==s]
        if len(row)>0:
            ax.scatter([0], [row["before_v0_recall"].values[0]], s=40, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", zorder=5)
            ax.scatter([1], [row["after_v0_recall"].values[0]], s=40, c=COLOR_AFTER, alpha=0.7, edgecolors="white", zorder=5)
            ax.plot([0,1], [row["before_v0_recall"].values[0], row["after_v0_recall"].values[0]],
                    color=GRAY, linewidth=0.5, alpha=0.4)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Before","After"], fontweight="bold")
    ax.set_ylabel("V-0 Recall", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(a) V-0 Recall per Seed", fontweight="bold")

    # (b) Non-V-0 recall per seed
    ax = axes[0,1]
    for s in sorted(set(seeds)):
        row = ul[ul["seed"]==s]
        if len(row)>0:
            ax.scatter([0], [row["before_non_v0_recall"].values[0]], s=40, c=COLOR_BEFORE, alpha=0.7, edgecolors="white", zorder=5)
            ax.scatter([1], [row["after_non_v0_recall"].values[0]], s=40, c=COLOR_AFTER, alpha=0.7, edgecolors="white", zorder=5)
            ax.plot([0,1], [row["before_non_v0_recall"].values[0], row["after_non_v0_recall"].values[0]],
                    color=GRAY, linewidth=0.5, alpha=0.4)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Before","After"], fontweight="bold")
    ax.set_ylabel("Non-V-0 Recall", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(b) Non-V-0 Recall per Seed", fontweight="bold")

    # (c) MCC + F1 per seed
    ax = axes[0,2]
    for s in sorted(set(seeds)):
        row = ul[ul["seed"]==s]
        if len(row)>0:
            ax.scatter([0],[row["before_mcc"].values[0]],s=40,c=COLOR_BEFORE,alpha=0.7,edgecolors="white",zorder=5)
            ax.scatter([1],[row["after_mcc"].values[0]],s=40,c=COLOR_AFTER,alpha=0.7,edgecolors="white",zorder=5)
            ax.plot([0,1],[row["before_mcc"].values[0],row["after_mcc"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
            ax.scatter([2],[row["before_f1"].values[0]],s=40,c=COLOR_BEFORE,alpha=0.7,edgecolors="white",zorder=5)
            ax.scatter([3],[row["after_f1"].values[0]],s=40,c=COLOR_AFTER,alpha=0.7,edgecolors="white",zorder=5)
            ax.plot([2,3],[row["before_f1"].values[0],row["after_f1"].values[0]],color=GRAY,linewidth=0.5,alpha=0.4)
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["MCC-B","MCC-A","F1-B","F1-A"], rotation=20, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(c) MCC & F1 per Seed", fontweight="bold")

    # (d) McNemar discordant pairs per seed
    ax = axes[1,0]
    disc = ul["n_discordant"].dropna().values
    seeds_disc = ul["seed"].values[:len(disc)]
    ax.bar(range(len(disc)), disc, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Seed Index", fontweight="bold"); ax.set_ylabel("Discordant Pairs", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(d) McNemar Discordant Pairs", fontweight="bold")

    # (e) Confusion matrix seed=42
    ax = axes[1,1]
    pred_path = os.path.join(RESULTS_DIR, "Frozen", "predictions_UL94_Rating_seed_42.csv")
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path)
        yt = pred["y_true"].values.astype(int); pb = pred["y_pred_before"].values.astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yt, pb)
        ax.imshow(cm, cmap=plt.cm.Blues, vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=FS+2, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Non-V-0","Pred V-0"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["True Non-V-0","True V-0"])
        ax.set_title("(e) Confusion Matrix (Before, seed=42)", fontweight="bold")

    # (f) McNemar p per seed
    ax = axes[1,2]
    mcp = ul["mcnemar_p"].dropna().values
    ax.bar(range(len(mcp)), mcp, color=COLOR_AFTER if len(mcp)>0 and np.median(mcp)<0.05 else GRAY, edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="#333", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Seed Index", fontweight="bold"); ax.set_ylabel("McNemar p", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title("(f) McNemar p per Seed", fontweight="bold")


    plt.tight_layout()
    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SI_DIR, f"Figure_S5.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("[OK] Figure S5 saved")


# ═══════════════ MAIN ═══════════════
def main():
    data = load_data()
    figure_s1()
    table_s2(data)
    figure_s2(data)
    figure_s3(data)
    table_s3(data)
    figure_s4(data)
    figure_s5(data)
    print(f"\nAll outputs -> {SI_DIR}")

if __name__ == "__main__":
    main()
