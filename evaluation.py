"""
evaluation.py — Paired bootstrap + classification statistics
==========================================
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, f1_score,
)
from config import N_BOOTSTRAP


def paired_bootstrap_regression(
    y_true, pred_before, pred_after,
    metric_fn,
    higher_is_better=True,
    n_bootstrap=N_BOOTSTRAP,
    seed=42,
):
    """Paired bootstrap on the fixed test set. improvement_delta > 0 means After is better."""
    y_true = np.asarray(y_true, dtype=float)
    pred_before = np.asarray(pred_before, dtype=float)
    pred_after = np.asarray(pred_after, dtype=float)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            b = metric_fn(y_true[idx], pred_before[idx])
            a = metric_fn(y_true[idx], pred_after[idx])
        except ValueError:
            continue
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        delta = a - b if higher_is_better else b - a
        deltas.append(delta)

    obs_b = metric_fn(y_true, pred_before)
    obs_a = metric_fn(y_true, pred_after)
    obs_delta = obs_a - obs_b if higher_is_better else obs_b - obs_a

    return {
        "before": float(obs_b),
        "after": float(obs_a),
        "improvement_delta": float(obs_delta),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_bootstrap_valid": len(deltas),
        "direction_positive": float(np.mean(np.array(deltas) > 0)),
    }


def regression_metrics_all(y_true, pred_before, pred_after, seed=42):
    """Full regression metric set: R2, RMSE, MAE + bootstrap CI."""
    results = {}
    yt = np.asarray(y_true, dtype=float)
    pb = np.asarray(pred_before, dtype=float)
    pa = np.asarray(pred_after, dtype=float)

    # R²
    r2 = paired_bootstrap_regression(yt, pb, pa, r2_score, higher_is_better=True, seed=seed)
    results["before_r2"] = r2["before"]
    results["after_r2"] = r2["after"]
    results["delta_r2"] = r2["improvement_delta"]
    results["r2_ci_low"] = r2["ci_low"]
    results["r2_ci_high"] = r2["ci_high"]

    # RMSE
    rmse = paired_bootstrap_regression(yt, pb, pa, lambda y,p: np.sqrt(mean_squared_error(y,p)),
                                        higher_is_better=False, seed=seed)
    results["before_rmse"] = rmse["before"]
    results["after_rmse"] = rmse["after"]
    results["delta_rmse"] = rmse["improvement_delta"]
    results["rmse_ci_low"] = rmse["ci_low"]
    results["rmse_ci_high"] = rmse["ci_high"]

    # MAE
    mae = paired_bootstrap_regression(yt, pb, pa, mean_absolute_error, higher_is_better=False, seed=seed)
    results["before_mae"] = mae["before"]
    results["after_mae"] = mae["after"]
    results["delta_mae"] = mae["improvement_delta"]
    results["mae_ci_low"] = mae["ci_low"]
    results["mae_ci_high"] = mae["ci_high"]

    return results


def classification_statistics(y_true, pred_before, pred_after):
    """Full classification metric set: balanced accuracy, MCC, recall, McNemar.
    pos_label=1 -> V-0, pos_label=0 -> Non-V-0."""
    yt = np.asarray(y_true, dtype=int)
    pb = np.asarray(pred_before, dtype=int)
    pa = np.asarray(pred_after, dtype=int)

    # McNemar
    correct_b = (pb == yt)
    correct_a = (pa == yt)
    b = int(np.sum(correct_b & ~correct_a))
    c = int(np.sum(~correct_b & correct_a))
    # exact binomial test
    from scipy.stats import binomtest
    n_discordant = b + c
    if n_discordant > 0:
        mcnemar_p = binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided").pvalue
    else:
        mcnemar_p = 1.0

    return {
        "before_accuracy": float(accuracy_score(yt, pb)),
        "after_accuracy": float(accuracy_score(yt, pa)),
        "before_balanced_accuracy": float(balanced_accuracy_score(yt, pb)),
        "after_balanced_accuracy": float(balanced_accuracy_score(yt, pa)),
        "before_mcc": float(matthews_corrcoef(yt, pb)),
        "after_mcc": float(matthews_corrcoef(yt, pa)),
        "before_v0_recall": float(recall_score(yt, pb, pos_label=1)),
        "after_v0_recall": float(recall_score(yt, pa, pos_label=1)),
        "before_non_v0_recall": float(recall_score(yt, pb, pos_label=0)),
        "after_non_v0_recall": float(recall_score(yt, pa, pos_label=0)),
        "before_f1": float(f1_score(yt, pb, average="binary")),
        "after_f1": float(f1_score(yt, pa, average="binary")),
        "mcnemar_p": float(mcnemar_p),
        "n_discordant": n_discordant,
    }