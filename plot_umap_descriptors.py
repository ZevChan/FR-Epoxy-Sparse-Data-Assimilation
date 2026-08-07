"""
================================================================================
 plot_umap_descriptors.py — 文献 vs 实验 UMAP（分子描述符，纯散点）
================================================================================
 用 RDKit 217 维 2D 分子描述符替代 Morgan 指纹，
 对 5 个化学组分各算描述符并拼接，
 文献数据 fit → 实验数据 transform，红色点缩小 + 半透明。
================================================================================
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# ==================== 绘图设置 ====================
plt.rcParams.update({
    "font.family": "Arial", "font.size": 12,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_LIT = "#5DA5DA"
COLOR_EXP = "#C91511"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)

# 5 个 SMILES 列
SMILES_COLS = [
    "EPOXY STRUCTURE",   # 环氧树脂
    "Flame_retardant",   # 阻燃剂
    "Curing_agent ",     # 固化剂
    "Other_Material_1",  # 助剂1
    "Other_Material_2",  # 助剂2
]

# ==================== 描述符计算 ====================
_cache = {}  # 缓存，避免重复计算相同 SMILES


def smiles_to_descriptors(smiles):
    """把一个 SMILES 转成 217 维描述符向量，失败返 zero 向量。"""
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    s = smiles.strip()
    if s in _cache:
        return _cache[s]
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        _cache[s] = None
        return None
    d = Descriptors.CalcMolDescriptors(mol)
    vec = np.array(list(d.values()), dtype=np.float64)
    _cache[s] = vec
    return vec


def encode_column(series):
    """对一列 SMILES 编码，替换缺失/无效为列均值。"""
    vecs = []
    for v in series:
        arr = smiles_to_descriptors(v)
        if arr is not None:
            vecs.append(arr)
        else:
            vecs.append(None)
    # 找 None 的索引
    none_idx = [i for i, x in enumerate(vecs) if x is None]
    # 先算有效值的均值
    valid = [x for x in vecs if x is not None]
    if len(valid) == 0:
        # 全缺失 → zero 向量，长度从第一个非 None 缓存推断
        sample = next((v for v in _cache.values() if v is not None), np.zeros(217))
        mean_vec = np.zeros_like(sample)
    else:
        mean_vec = np.mean(valid, axis=0)
    for i in none_idx:
        vecs[i] = mean_vec
    return np.vstack(vecs)


# ==================== 主函数 ====================
def plot():
    print("读取数据 ...")
    exp = pd.read_csv(os.path.join(BASE_DIR, "实验数据.csv"))
    lit = pd.read_csv(os.path.join(BASE_DIR, "文献数据.csv"))
    if "Unnamed: 0" in exp.columns:
        exp = exp.drop(columns=["Unnamed: 0"])

    exp["Source"] = "Experiment"
    lit["Source"] = "Literature"
    df = pd.concat([lit, exp], ignore_index=True)
    print(f"文献: {len(lit)}  实验: {len(exp)}  总计: {len(df)}")

    # 每个组分算描述符，拼接
    print("计算分子描述符 (5 组分 × 217 维) ...")
    all_encoded = []
    for col in SMILES_COLS:
        print(f"  -> {col}")
        all_encoded.append(encode_column(df[col]))
    X_all = np.hstack(all_encoded)
    n_desc = X_all.shape[1] // len(SMILES_COLS)
    print(f"特征维度: {X_all.shape[1]}  ({len(SMILES_COLS)} 组分 × {n_desc} 描述符)")

    # 清理 NaN / inf
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # 分离文献/实验
    lit_idx = (df["Source"] == "Literature").values
    exp_idx = (df["Source"] == "Experiment").values

    print("降维 (文献 fit, 实验 transform) ...")
    selector = VarianceThreshold()
    X_lit_sel = selector.fit_transform(X_all[lit_idx])
    X_exp_sel = selector.transform(X_all[exp_idx])

    scaler = StandardScaler()
    X_lit_scaled = scaler.fit_transform(X_lit_sel)
    X_exp_scaled = scaler.transform(X_exp_sel)

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, verbose=False,
    )
    lit_emb = reducer.fit_transform(X_lit_scaled)
    exp_emb = reducer.transform(X_exp_scaled)

    # ---- 绘图：文献 KDE 等高线 + 实验散点 ----
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=600)

    # 文献 KDE 渐变填充等高线
    sns.kdeplot(
        x=lit_emb[:, 0], y=lit_emb[:, 1],
        fill=True, cmap="Blues", alpha=0.28,
        levels=10, thresh=0.02, ax=ax,
        label="Literature Density",
    )

    # 文献散点
    ax.scatter(lit_emb[:, 0], lit_emb[:, 1],
               c=COLOR_LIT, s=14, alpha=0.25, edgecolors="none",
               label=f"Literature (n={len(lit_emb)})")

    # 实验散点
    ax.scatter(exp_emb[:, 0], exp_emb[:, 1],
               c=COLOR_EXP, s=90, alpha=0.75, marker="o",
               edgecolors="black", linewidth=0.6,
               label=f"Experiment (n={len(exp_emb)})")

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical Structure Space (Descriptors): Literature vs Experiment",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Descriptors_Scatter.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("\n[OK] UMAP_Descriptors_Scatter 已保存 (png/pdf/svg)")


if __name__ == "__main__":
    plot()
