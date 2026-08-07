"""
================================================================================
 plot_error_density.py — 实验引导前后预测误差密度分布对比
================================================================================
 读取 fair_holdout_comparison.py 保存的 predictions CSV，
 在同一固定测试集上绘制 Before / After 的残差 KDE 曲线。
 输出 PNG / PDF / SVG。
 颜色: Before = #5DA5DA (钢蓝), After = #C91511 (深红)
       零误差线 = #8C92AC (冷石板灰)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
BASE = os.path.dirname(os.path.abspath(__file__))

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
COLOR_BEFORE  = "#5DA5DA"   # 钢蓝色 — 文献基准
COLOR_AFTER   = "#C91511"   # 深红色 — 实验引导后
COLOR_ZERO    = "#8C92AC"   # 冷石板灰 — 零误差参考线

# 回归目标（排除 UL94，分类不适合残差密度）
REGRESSION_TARGETS = ["LOI", "pHRR", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
}

UNITS = {
    "LOI": "%",
    "pHRR": "kW m⁻²",
    "THR": "MJ m⁻²",
    "TSP": "m²",
    "Flexural_Strength": "MPa",
}

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Frozen")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_error_density():
    for target in REGRESSION_TARGETS:
        pred_file = os.path.join(RESULTS_DIR, f"predictions_{target}.csv")
        if not os.path.exists(pred_file):
            print(f"  [WARN] 跳过 {target}: 预测文件不存在")
            continue

        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        residuals_before = y_true - y_pred_before
        residuals_after = y_true - y_pred_after

        unit = UNITS.get(target, "")

        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=600)

        # KDE 曲线
        from scipy.stats import gaussian_kde

        try:
            kde_before = gaussian_kde(residuals_before)
            kde_after = gaussian_kde(residuals_after)

            # 统一 x 范围
            all_res = np.concatenate([residuals_before, residuals_after])
            x_range = np.linspace(all_res.min() * 1.15, all_res.max() * 1.15, 300)

            ax.fill_between(x_range, kde_before(x_range), alpha=0.25, color=COLOR_BEFORE)
            ax.plot(x_range, kde_before(x_range), color=COLOR_BEFORE, linewidth=1.8,
                    label="Before (Literature only)")

            ax.fill_between(x_range, kde_after(x_range), alpha=0.35, color=COLOR_AFTER)
            ax.plot(x_range, kde_after(x_range), color=COLOR_AFTER, linewidth=1.8,
                    label="After (Literature + Experiment)")
        except Exception:
            # 回退到直方图
            ax.hist(residuals_before, bins=20, density=True, alpha=0.35,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.3,
                    label="Before (Literature only)")
            ax.hist(residuals_after, bins=20, density=True, alpha=0.45,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.3,
                    label="After (Literature + Experiment)")

        # 零误差参考线
        ax.axvline(0, color=COLOR_ZERO, linestyle="--", linewidth=1.0, alpha=0.8)

        display_name = DISPLAY_NAMES.get(target, target)
        ax.set_xlabel(f"Residual ({unit})" if unit else "Residual", fontsize=13)
        ax.set_ylabel("Probability Density", fontsize=13)
        ax.set_title(display_name, fontsize=14, fontweight="bold")

        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)  # 仅扩大Y轴上限

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Error_Density_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[OK] Error_Density_{target} 已保存 (png/pdf/svg)")

    # ---- UL94 二分类：混淆矩阵分布对比 ----
    ul94_file = os.path.join(RESULTS_DIR, "predictions_UL94_Rating.csv")
    if os.path.exists(ul94_file):
        df = pd.read_csv(ul94_file)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        cm_b = confusion_matrix(y_true, y_pred_before, labels=[0, 1])
        cm_a = confusion_matrix(y_true, y_pred_after, labels=[0, 1])

        ul_labels = [
            "TN\n(True Non-V-0)",
            "FP\n(Non-V-0 → V-0)",
            "FN\n(V-0 → Non-V-0)",
            "TP\n(True V-0)",
        ]
        before_counts = [cm_b[0, 0], cm_b[0, 1], cm_b[1, 0], cm_b[1, 1]]
        after_counts = [cm_a[0, 0], cm_a[0, 1], cm_a[1, 0], cm_a[1, 1]]
        n_total = len(y_true)

        x = np.arange(len(ul_labels))
        width = 0.32

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(x - width/2, before_counts, width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x + width/2, after_counts, width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")

        # 数字标签
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=11, color=COLOR_BEFORE)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=11, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(ul_labels, fontsize=11)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title("UL-94 (V-0 vs Non V-0): Confusion Matrix Distribution",
                     fontsize=14, fontweight="bold")

        all_h = before_counts + after_counts
        ax.set_ylim(0, max(all_h) * 1.25)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Error_Density_UL94_Rating.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Error_Density_UL94_Rating 已保存 (png/pdf/svg)")
    else:
        print("[WARN] 跳过 UL94_Rating: 预测文件不存在")


if __name__ == "__main__":
    print("生成误差密度分布图 ...")
    plot_error_density()
    print("完成。")