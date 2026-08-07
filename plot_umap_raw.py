"""
================================================================================
 plot_umap_raw.py — Literature vs experiment UMAP on raw data
================================================================================
 Reads literature_data.csv and experimental_data.csv directly,
 encodes chemical structures (SMILES) with Morgan fingerprints, plus process parameters,
 literature fit -> experiment transform into the same chemical space.
================================================================================
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
from rdkit import Chem
from rdkit.Chem import AllChem

# ==================== Plot settings ====================
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

# ==================== Morgan fingerprints ====================
def smiles_to_fp(smiles, n_bits=256, radius=2):
    """Convert SMILES to Morgan fingerprint; return zero vector on failure."""
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return np.zeros(n_bits, dtype=np.float64)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float64)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float64)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def encode_smiles_column(series, n_bits=256):
    """Encode a column of SMILES strings into a (N, n_bits) array."""
    fps = np.vstack([smiles_to_fp(s, n_bits) for s in series])
    return fps


# ==================== Main function ====================
def plot():
    # ---- Read data ----
    exp = pd.read_csv(os.path.join(BASE_DIR, "experimental_data.csv"))
    lit = pd.read_csv(os.path.join(BASE_DIR, "literature_data.csv"))
    if "Unnamed: 0" in exp.columns:
        exp = exp.drop(columns=["Unnamed: 0"])

    exp["Source"] = "Experiment"
    lit["Source"] = "Literature"

    df = pd.concat([lit, exp], ignore_index=True)
    print(f"Literature: {len(lit)}  Experiment: {len(exp)}  Total: {len(df)}")

    # ---- Column classification ----
    smiles_cols = [
        "EPOXY STRUCTURE",
        "Flame_retardant",
        "Curing_agent ",
        "Other_Material_1",
        "Other_Material_2",
    ]
    # Other_Material_3 is almost entirely empty; skip
    numeric_cols = [
        "EEW",
        "Flame_retardant_AdditionAmount(wt%)",
        "Curing_agent_AdditionAmount(wt%)",
        "Curing_Tem1", "Curing_Time1",
        "Curing_Tem2", "Curing_Time2",
        "Curing_Tem3", "Curing_Time3",
        "Curing_Tem4", "Curing_Time4",
        "Curing_Tem5", "Curing_Time5",
        "Curing_Tem6", "Curing_Time6",
        "Curing_Tem7", "Curing_Time8",
        "Curing_Tem9", "Curing_Time9",
        "Curing_Tem10", "Curing_Time10",
        "Curing_Pressure", "Curing_UV",
        "Other_Material_AdditionAmount_1 (wt%)",
        "Other_Material_AdditionAmount_2 (wt%)",
        "Other_Material_AdditionAmount_3 (wt%)",
    ]

    # ---- Encode SMILES -> Morgan FP ----
    print("Encoding SMILES fingerprints ...")
    fp_parts = []
    for col in smiles_cols:
        fp_parts.append(encode_smiles_column(df[col], n_bits=256))
    X_fp = np.hstack(fp_parts)  # (N, 5*256)

    # ---- Numeric columns ----
    X_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    # ---- Merge ----
    X_all = np.hstack([X_fp, X_num])
    print(f"Feature dimension: {X_all.shape[1]}  ({X_fp.shape[1]} fingerprints + {X_num.shape[1]} process)")

    # ---- Split literature/experiment ----
    lit_idx = df["Source"] == "Literature"
    exp_idx = df["Source"] == "Experiment"

    # ---- fit on literature, transform experiment ----
    print("Reducing (literature fit, experiment transform) ...")

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

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=600)

    # KDE background - literature
    sns.kdeplot(
        x=lit_emb[:, 0], y=lit_emb[:, 1],
        fill=True, alpha=0.12, color="#7f8c8d",
        levels=8, thresh=0.05, ax=ax,
    )

    # literature scatter
    ax.scatter(
        lit_emb[:, 0], lit_emb[:, 1],
        c=COLOR_LIT, s=8, alpha=0.25, edgecolors="none",
        label=f"Literature (n={len(lit_emb)})",
    )

    # experiment highlighted
    ax.scatter(
        exp_emb[:, 0], exp_emb[:, 1],
        c=COLOR_EXP, s=70, alpha=0.95, marker="o",
        edgecolors="black", linewidth=0.6,
        label=f"Experiment (n={len(exp_emb)})",
    )

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical & Process Space: Literature vs Experiment",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    # full four-sided border
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Raw_Comparison.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("[OK] UMAP_Raw_Comparison saved (png/pdf/svg)")


if __name__ == "__main__":
    plot()
