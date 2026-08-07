"""
================================================================================
 plot_umap_structure_scatter.py — Literature vs experiment UMAP (structure-only, pure scatter)
================================================================================
 Reads literature_data.csv and experimental_data.csv directly,
 encodes chemical structure (SMILES) with Morgan fingerprints only, no KDE background,
 literature fit -> experiment transform.
================================================================================
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
from rdkit import Chem
from rdkit.Chem import AllChem

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


def smiles_to_fp(smiles, n_bits=256, radius=2):
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
    return np.vstack([smiles_to_fp(s, n_bits) for s in series])


def plot():
    exp = pd.read_csv(os.path.join(BASE_DIR, "experimental_data.csv"))
    lit = pd.read_csv(os.path.join(BASE_DIR, "literature_data.csv"))
    if "Unnamed: 0" in exp.columns:
        exp = exp.drop(columns=["Unnamed: 0"])
    exp["Source"] = "Experiment"
    lit["Source"] = "Literature"
    df = pd.concat([lit, exp], ignore_index=True)
    print(f"Literature: {len(lit)}  Experiment: {len(exp)}  Total: {len(df)}")

    smiles_cols = [
        "EPOXY STRUCTURE", "Flame_retardant", "Curing_agent ",
        "Other_Material_1", "Other_Material_2",
    ]
    print("Encoding SMILES fingerprints (structure only) ...")
    X_all = np.hstack([encode_smiles_column(df[c]) for c in smiles_cols])
    print(f"Feature dimension: {X_all.shape[1]}  (5 components x 256-bit)")

    lit_idx = (df["Source"] == "Literature").values
    exp_idx = (df["Source"] == "Experiment").values

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

    # ---- Pure scatter (no KDE background) ----
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=600)

    ax.scatter(lit_emb[:, 0], lit_emb[:, 1],
               c=COLOR_LIT, s=8, alpha=0.25, edgecolors="none",
               label=f"Literature (n={len(lit_emb)})")
    ax.scatter(exp_emb[:, 0], exp_emb[:, 1],
               c=COLOR_EXP, s=70, alpha=0.95, marker="o",
               edgecolors="black", linewidth=0.6,
               label=f"Experiment (n={len(exp_emb)})")

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical Structure Space: Literature vs Experiment",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Structure_Scatter.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("[OK] UMAP_Structure_Scatter saved (png/pdf/svg)")


if __name__ == "__main__":
    plot()
