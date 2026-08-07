import os
BASE = os.path.dirname(os.path.abspath(__file__))
"""
shap_metrics.py — SHAP decision-structure stability metrics
============================================
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calculate_shap_structure_metrics(shap_values, feature_names):
    """Compute feature-concentration metrics from SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    if total <= 0:
        p = np.zeros_like(mean_abs)
    else:
        p = mean_abs / total

    order = np.argsort(p)[::-1]
    eps = 1e-12

    sq_sum = np.sum(p ** 2)
    eff_n = float(1.0 / sq_sum) if sq_sum > 0 else float(np.nan)

    return {
        "top1_share": float(p[order[0]]),
        "top5_share": float(p[order[:5]].sum()),
        "effective_feature_number": eff_n,
        "shap_entropy": float(-np.sum(p * np.log(p + eps))),
        "ranking": order,
        "normalized_shap": p,
        "feature_names": list(feature_names),
    }


def ranking_similarity(before_scores, after_scores, feature_names):
    """Before/After feature-ranking similarity."""
    bs = np.asarray(before_scores, dtype=float)
    as_ = np.asarray(after_scores, dtype=float)
    names = list(feature_names)

    b_order = np.argsort(bs)[::-1]
    a_order = np.argsort(as_)[::-1]

    top10_b = {names[i] for i in b_order[:10]}
    top10_a = {names[i] for i in a_order[:10]}

    union = top10_b | top10_a
    jaccard = len(top10_b & top10_a) / len(union) if union else 0.0

    if len(bs) >= 3:
        rho, pval = spearmanr(bs, as_)
    else:
        rho, pval = 0.0, 1.0

    return {
        "top10_jaccard": float(jaccard),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
    }


def export_shap_stability(shap_metrics_list, output_dir):
    """Export multi-seed SHAP stability data."""
    metrics_rows = []
    feature_rows = []

    for entry in shap_metrics_list:
        target = entry["target"]
        seed = entry["seed"]
        protocol = entry["protocol"]
        track = entry["track"]
        sm = entry["metrics"]

        metrics_rows.append({
            "target": target,
            "seed": seed,
            "protocol": protocol,
            "track": track,
            "top1_share": sm["top1_share"],
            "top5_share": sm["top5_share"],
            "effective_feature_number": sm["effective_feature_number"],
            "shap_entropy": sm["shap_entropy"],
        })

        for rank, idx in enumerate(sm["ranking"], start=1):
            feature_rows.append({
                "target": target,
                "seed": seed,
                "protocol": protocol,
                "track": track,
                "feature": sm["feature_names"][idx],
                "rank": rank,
                "normalized_mean_abs_shap": sm["normalized_shap"][idx],
                "is_top10": rank <= 10,
            })

    pd.DataFrame(metrics_rows).to_csv(
        f"{output_dir}/shap_structure_metrics.csv", index=False)
    if feature_rows:
        fdf = pd.DataFrame(feature_rows)
        summary = (
            fdf.groupby(["target", "protocol", "track", "feature"])
            .agg(
                top10_frequency=("is_top10", "mean"),
                median_rank=("rank", "median"),
                rank_q1=("rank", lambda x: x.quantile(0.25)),
                rank_q3=("rank", lambda x: x.quantile(0.75)),
                median_normalized_shap=("normalized_mean_abs_shap", "median"),
            )
            .reset_index()
        )
        summary["rank_IQR"] = summary["rank_q3"] - summary["rank_q1"]
        summary.to_csv(f"{output_dir}/shap_feature_stability.csv", index=False)