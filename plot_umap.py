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
    umap_file = os.path.join(RESULTS_DIR, "Distribution", "umap_coordinates.csv")
    if not os.path.exists(umap_file):
        print("[WARN] umap_coordinates.csv 不存在，请先运行 fair_holdout_comparison.py")
        return

    print("加载 UMAP 坐标（由 distribution.py 输出）...")
    df = pd.read_csv(umap_file)
    df = df.rename(columns={"source": "Source", "umap_1": "UMAP-1", "umap_2": "UMAP-2"})
    df_unique = df.copy()

    lit_mask = df_unique["Source"] == "Literature"
    exp_mask = df_unique["Source"] == "Experiment"
    n_lit = int(lit_mask.sum())
    n_exp = int(exp_mask.sum())
    print(f"  文献样本: {n_lit}")
    print(f"  实验样本: {n_exp}")

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
