"""
================================================================================
 plot_umap.py — 化学空间 UMAP 投影
================================================================================
 直接读取 fair_holdout_comparison.py 保存的 umap_data_MASTER.csv，
 根据 Source 列区分 Literature / Experiment，降维到 2D。
 输出 PNG / PDF / SVG。

 颜色:
   文献数据 — #5DA5DA (钢蓝色)
   实验数据 — #C91511 (深红色，高亮)
================================================================================
"""

import os
BASE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap

# ==================== NC 期刊绘图全局设置 ====================
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

# ==================== 颜色定义 ====================
COLOR_LIT = "#5DA5DA"  # 钢蓝色 — 文献数据
COLOR_EXP = "#C91511"  # 深红色 — 实验数据高亮

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_umap():
    umap_file = os.path.join(RESULTS_DIR, "umap_data_MASTER.csv")
    if not os.path.exists(umap_file):
        print("[WARN] umap_data_MASTER.csv 不存在，请先运行 fair_holdout_comparison.py")
        return

    print("加载 UMAP 数据 ...")
    df = pd.read_csv(umap_file)

    # 分离特征列（排除 Target, Target_Value, Source）
    meta_cols = ["Target", "Target_Value", "Source"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # 去重（同一行可能出现在多个 target 中）
    df_unique = df.drop_duplicates(subset=feature_cols).copy()

    n_lit = (df_unique["Source"] == "Literature").sum()
    n_exp = (df_unique["Source"] == "Experiment").sum()
    print(f"  文献样本: {n_lit}")
    print(f"  实验样本: {n_exp}")

    X = df_unique[feature_cols].values
    lit_mask = df_unique["Source"] == "Literature"
    exp_mask = df_unique["Source"] == "Experiment"

    # ---- 只用文献数据 fit，实验数据 transform ----
    print("执行 UMAP 降维（文献数据 fit，实验数据嵌入）...")
    selector = VarianceThreshold()
    X_lit_sel = selector.fit_transform(X[lit_mask])
    X_exp_sel = selector.transform(X[exp_mask])

    scaler = StandardScaler()
    X_lit_scaled = scaler.fit_transform(X_lit_sel)
    X_exp_scaled = scaler.transform(X_exp_sel)

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, verbose=False,
    )
    lit_embedding = reducer.fit_transform(X_lit_scaled)
    exp_embedding = reducer.transform(X_exp_scaled)

    df_unique["UMAP-1"] = np.nan
    df_unique["UMAP-2"] = np.nan
    df_unique.loc[lit_mask, "UMAP-1"] = lit_embedding[:, 0]
    df_unique.loc[lit_mask, "UMAP-2"] = lit_embedding[:, 1]
    df_unique.loc[exp_mask, "UMAP-1"] = exp_embedding[:, 0]
    df_unique.loc[exp_mask, "UMAP-2"] = exp_embedding[:, 1]

    # ---- 主图 ----
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=600)

    # KDE 背景
    sns.kdeplot(
        data=df_unique[lit_mask],
        x="UMAP-1", y="UMAP-2",
        fill=True, alpha=0.12, color="#7f8c8d",
        levels=8, thresh=0.05, ax=ax,
    )

    # 文献散点
    ax.scatter(
        df_unique.loc[lit_mask, "UMAP-1"],
        df_unique.loc[lit_mask, "UMAP-2"],
        c=COLOR_LIT, s=10, alpha=0.3, edgecolors="none",
        label=f"Literature Data (n={n_lit})",
    )

    # 实验散点高亮
    ax.scatter(
        df_unique.loc[exp_mask, "UMAP-1"],
        df_unique.loc[exp_mask, "UMAP-2"],
        c=COLOR_EXP, s=55, alpha=0.95, marker="o",
        edgecolors="black", linewidth=0.6,
        label=f"Experimental Data (n={n_exp})",
    )

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical Space Distribution of EP/FR Composites",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Chemical_Space.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("[OK] UMAP_Chemical_Space 已保存 (png/pdf/svg)")


if __name__ == "__main__":
    plot_umap()
