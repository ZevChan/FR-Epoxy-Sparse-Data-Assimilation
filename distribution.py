"""
distribution.py — Unsupervised UMAP + high-dimensional kNN density analysis
====================================================
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from config import BASE_DIR, OUTPUT_DIR, SMILES_COLS


def build_distribution_representation(X_all, is_literature):
    """Unsupervised representation: all transforms fitted on literature data only."""
    X_lit = X_all.loc[is_literature].copy()
    X_exp = X_all.loc[~is_literature].copy()

    imputer = SimpleImputer(strategy="mean")
    var_filter = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()

    X_lit_i = imputer.fit_transform(X_lit)
    X_exp_i = imputer.transform(X_exp)

    X_lit_v = var_filter.fit_transform(X_lit_i)
    X_exp_v = var_filter.transform(X_exp_i)

    X_lit_s = scaler.fit_transform(X_lit_v)
    X_exp_s = scaler.transform(X_exp_v)

    return X_lit_s, X_exp_s


def compute_umap_from_representation(X_lit_s, X_exp_s, n_neighbors=15, min_dist=0.1, random_state=42):
    """Compute UMAP embeddings from the processed representation. PCA pre-reduction for speed."""
    import umap

    # PCA pre-reduction (fitted on literature data)
    n_pca = min(100, X_lit_s.shape[1], X_lit_s.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_lit_repr = pca.fit_transform(X_lit_s)
    X_exp_repr = pca.transform(X_exp_s)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=False,
    )
    lit_emb = reducer.fit_transform(X_lit_repr)
    exp_emb = reducer.transform(X_exp_repr)

    return lit_emb, exp_emb


def density_percentile(X_lit_repr, X_exp_repr, n_neighbors=10):
    """High-dimensional kNN density percentile analysis."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X_lit_repr)

    lit_dist = nn.kneighbors(X_lit_repr)[0][:, 1:].mean(axis=1)
    exp_dist = nn.kneighbors(X_exp_repr, n_neighbors=n_neighbors)[0].mean(axis=1)

    percentiles = np.array([
        np.mean(lit_dist <= d) for d in exp_dist
    ])

    return pd.DataFrame({
        "experiment_index": np.arange(len(X_exp_repr)),
        "mean_knn_distance": exp_dist,
        "literature_distance_percentile": percentiles,
        "above_p90": percentiles >= 0.90,
        "above_p95": percentiles >= 0.95,
    })


def export_distribution_data(X_all, is_literature):
    """Export UMAP coordinates and kNN density data."""
    dist_dir = os.path.join(OUTPUT_DIR, "Distribution")

    X_lit_s, X_exp_s = build_distribution_representation(X_all, is_literature)

    # UMAP
    lit_emb, exp_emb = compute_umap_from_representation(X_lit_s, X_exp_s)

    umap_df = pd.DataFrame({
        "source": ["Literature"] * len(lit_emb) + ["Experiment"] * len(exp_emb),
        "umap_1": np.concatenate([lit_emb[:, 0], exp_emb[:, 0]]),
        "umap_2": np.concatenate([lit_emb[:, 1], exp_emb[:, 1]]),
    })
    umap_df.to_csv(os.path.join(dist_dir, "umap_coordinates.csv"), index=False)

    # kNN density
    density_df = density_percentile(X_lit_s, X_exp_s)
    density_df.to_csv(os.path.join(dist_dir, "experimental_knn_density.csv"), index=False)

    print(f"  [Distribution] UMAP + kNN density exported")
