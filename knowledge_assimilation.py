"""
Low-cost post-hoc analyses for sparse experimental knowledge assimilation.

This module deliberately does NOT call ``select_configuration`` or Optuna.
Every fitted model reuses a Frozen configuration that was already selected by
the long 10-seed/50-trial/5-fold pipeline.  The original result directory is
read-only; all new artifacts are written by ``run_knowledge_assimilation.py``
to ``Results/Knowledge_Assimilation`` in the current project directory.
"""
from __future__ import annotations

import os
BASE = os.path.dirname(os.path.abspath(__file__))


import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from shap_metrics import calculate_shap_structure_metrics, ranking_similarity


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_RESULTS = HERE / "Results"
DEFAULT_OUTPUT_DIR = HERE / "Results" / "Knowledge_Assimilation"
PRIMARY_SEED = 42
DEFAULT_DOSE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
REGRESSION_UTILITY_TOLERANCE = 0.01
CLASSIFICATION_UTILITY_TOLERANCE = 0.02
DEFAULT_PLASTICITY_THRESHOLD = 0.50


def stable_signature(payload: Any) -> str:
    """Return a deterministic short signature for cache validation."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def stable_int_seed(seed: int, target: str, stage: str, repeat: int = 0) -> int:
    """Mix text labels into a deterministic NumPy-compatible random seed."""
    token = f"{seed}|{target}|{stage}|{repeat}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return int((int(seed) + offset) % (2**32 - 1))


def ensure_separate_directories(
    source_results: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    """Reject any configuration that could write into the running source job."""
    source = Path(source_results).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output:
        raise ValueError(
            "The knowledge-assimilation output directory must differ from "
            "the original long-run Results directory."
        )
    allowed_isolated_child = (
        output.parent == source and output.name == "Knowledge_Assimilation"
    )
    if source in output.parents and not allowed_isolated_child:
        raise ValueError(
            "The post-hoc output directory may not be nested inside the "
            "original long-run Results directory except as the isolated "
            "Knowledge_Assimilation child."
        )
    return source, output


def _required_prediction_columns() -> set[str]:
    return {
        "sample_id",
        "target",
        "seed",
        "protocol",
        "y_true",
        "y_pred_before",
        "y_pred_after",
    }


def frozen_config_path(source_results: Path, target: str, seed: int) -> Path:
    return source_results / "Frozen" / f"config_{target}_seed_{seed}.json"


def frozen_prediction_path(source_results: Path, target: str, seed: int) -> Path:
    return source_results / "Frozen" / f"predictions_{target}_seed_{seed}.csv"


def load_frozen_config(
    source_results: str | Path,
    target: str,
    seed: int,
    fallback_seed: int | None = PRIMARY_SEED,
) -> tuple[dict[str, Any], int, Path]:
    """
    Load a completed Frozen configuration.

    A partially written JSON file from a concurrently running source pipeline
    is treated as unavailable.  If requested, seed 42 is then used as the
    explicit fallback.
    """
    source = Path(source_results)
    candidates = [int(seed)]
    if fallback_seed is not None and int(fallback_seed) not in candidates:
        candidates.append(int(fallback_seed))

    errors: list[str] = []
    for candidate in candidates:
        path = frozen_config_path(source, target, candidate)
        if not path.is_file():
            errors.append(f"missing: {path.name}")
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable: {path.name} ({exc})")
            continue
        required = {"k", "features", "params", "cv_score"}
        missing = required - set(config)
        if missing:
            errors.append(f"incomplete: {path.name} ({sorted(missing)})")
            continue
        if not config["features"] or not isinstance(config["params"], dict):
            errors.append(f"invalid: {path.name}")
            continue
        return config, candidate, path

    raise FileNotFoundError(
        f"No usable Frozen configuration for {target}, requested seed={seed}. "
        + "; ".join(errors)
    )


def load_saved_predictions(
    source_results: str | Path,
    target: str,
    seed: int,
    fallback_seed: int | None = PRIMARY_SEED,
) -> tuple[pd.DataFrame, int, Path]:
    """Load and validate a completed endpoint prediction artifact."""
    source = Path(source_results)
    candidates = [int(seed)]
    if fallback_seed is not None and int(fallback_seed) not in candidates:
        candidates.append(int(fallback_seed))

    errors: list[str] = []
    for candidate in candidates:
        path = frozen_prediction_path(source, target, candidate)
        if not path.is_file():
            errors.append(f"missing: {path.name}")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            errors.append(f"unreadable: {path.name} ({exc})")
            continue
        missing = _required_prediction_columns() - set(frame.columns)
        if missing or frame.empty:
            errors.append(f"incomplete: {path.name} ({sorted(missing)})")
            continue
        if frame[list(_required_prediction_columns())].isna().any().any():
            errors.append(f"contains missing values: {path.name}")
            continue
        return frame, candidate, path

    raise FileNotFoundError(
        f"No usable Frozen predictions for {target}, requested seed={seed}. "
        + "; ".join(errors)
    )


def discover_complete_seeds(
    source_results: str | Path,
    targets: Sequence[str],
    candidate_seeds: Sequence[int],
) -> list[int]:
    """Return seeds with readable config and prediction files for every target."""
    complete: list[int] = []
    source = Path(source_results)
    for seed in candidate_seeds:
        ok = True
        for target in targets:
            try:
                load_frozen_config(source, target, seed, fallback_seed=None)
                load_saved_predictions(source, target, seed, fallback_seed=None)
            except (FileNotFoundError, ValueError):
                ok = False
                break
        if ok:
            complete.append(int(seed))
    return complete


def make_frozen_model(
    params: dict[str, Any],
    is_classification: bool,
    seed: int,
    n_jobs: int = 2,
):
    """Create XGBoost with an already selected configuration (no HPO)."""
    model_params = {
        "n_estimators": params.get("n_estimators", 100),
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "reg_alpha": params.get("reg_alpha", 0.0),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "random_state": int(seed),
        "n_jobs": max(1, int(n_jobs)),
        "verbosity": 0,
    }
    if is_classification:
        return XGBClassifier(**model_params, eval_metric="logloss")
    return XGBRegressor(**model_params, objective="reg:squarederror")


def primary_metric_name(is_classification: bool) -> str:
    return "balanced_accuracy" if is_classification else "r2"


def primary_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    is_classification: bool,
) -> float:
    if is_classification:
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def per_sample_losses(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    is_classification: bool,
) -> np.ndarray:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if is_classification:
        return (y.astype(int) != p.astype(int)).astype(float)
    return np.square(y.astype(float) - p.astype(float))


def build_frozen_context(
    data: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    n_jobs: int = 2,
) -> dict[str, Any]:
    """
    Fit the two endpoint models with the Frozen preprocessing/configuration.

    The preprocessor is fit only on literature training rows, exactly as in the
    original Frozen protocol.
    """
    features = list(config["features"])
    missing = [name for name in features if name not in data["X_train_before"]]
    if missing:
        raise KeyError(f"Frozen configuration references missing features: {missing[:10]}")

    preprocessor = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="mean", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    X_lit_raw = data["X_train_before"][features].reset_index(drop=True)
    X_exp_raw = data["X_experiment"][features].reset_index(drop=True)
    X_test_raw = data["X_test"][features].reset_index(drop=True)
    X_lit = preprocessor.fit_transform(X_lit_raw)
    X_exp = preprocessor.transform(X_exp_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_lit = np.asarray(data["y_train_before"])
    y_exp = np.asarray(data["y_experiment"])
    y_test = np.asarray(data["y_test"])
    is_cls = bool(data["is_classification"])

    before = make_frozen_model(config["params"], is_cls, seed, n_jobs=n_jobs)
    before.fit(X_lit, y_lit)
    pred_before = before.predict(X_test)

    after = make_frozen_model(config["params"], is_cls, seed, n_jobs=n_jobs)
    after.fit(
        np.vstack([X_lit, X_exp]),
        np.concatenate([y_lit, y_exp]),
    )
    pred_after = after.predict(X_test)

    return {
        "target": data["target"],
        "seed": int(seed),
        "features": features,
        "params": dict(config["params"]),
        "is_classification": is_cls,
        "preprocessor": preprocessor,
        "X_lit_raw": X_lit_raw,
        "X_exp_raw": X_exp_raw,
        "X_test_raw": X_test_raw,
        "X_lit": X_lit,
        "X_exp": X_exp,
        "X_test": X_test,
        "y_lit": y_lit,
        "y_exp": y_exp,
        "y_test": y_test,
        "model_before": before,
        "model_after": after,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "before_score": primary_score(y_test, pred_before, is_cls),
        "after_score": primary_score(y_test, pred_after, is_cls),
        "metric": primary_metric_name(is_cls),
        "test_sample_ids": np.asarray(data["test_sample_ids"]),
        "experiment_sample_ids": np.asarray(data["experiment_sample_ids"]),
    }


def endpoint_reproduction_audit(
    context: dict[str, Any], saved_predictions: pd.DataFrame
) -> dict[str, Any]:
    """Verify that low-cost endpoint refits reproduce the long-run predictions."""
    expected_ids = pd.Index(context["test_sample_ids"].astype(int))
    saved = saved_predictions.copy()
    saved["sample_id"] = pd.to_numeric(saved["sample_id"], errors="raise").astype(int)
    saved = saved.set_index("sample_id").reindex(expected_ids)
    if saved[["y_pred_before", "y_pred_after"]].isna().any().any():
        raise ValueError("Saved prediction sample IDs do not match the reconstructed split")

    before_diff = np.abs(
        np.asarray(context["pred_before"], dtype=float)
        - saved["y_pred_before"].to_numpy(dtype=float)
    )
    after_diff = np.abs(
        np.asarray(context["pred_after"], dtype=float)
        - saved["y_pred_after"].to_numpy(dtype=float)
    )
    return {
        "n_test": int(len(saved)),
        "max_abs_diff_before": float(before_diff.max(initial=0.0)),
        "max_abs_diff_after": float(after_diff.max(initial=0.0)),
        "mean_abs_diff_before": float(before_diff.mean()) if len(before_diff) else 0.0,
        "mean_abs_diff_after": float(after_diff.mean()) if len(after_diff) else 0.0,
        "exact_class_match_before": bool(
            np.array_equal(
                np.asarray(context["pred_before"]).astype(int),
                saved["y_pred_before"].to_numpy().astype(int),
            )
        )
        if context["is_classification"]
        else None,
        "exact_class_match_after": bool(
            np.array_equal(
                np.asarray(context["pred_after"]).astype(int),
                saved["y_pred_after"].to_numpy().astype(int),
            )
        )
        if context["is_classification"]
        else None,
    }


def _fit_subset_and_score(
    context: dict[str, Any],
    experiment_indices: np.ndarray,
    n_jobs: int,
) -> tuple[float, np.ndarray]:
    X_parts = [context["X_lit"]]
    y_parts = [context["y_lit"]]
    if len(experiment_indices):
        X_parts.append(context["X_exp"][experiment_indices])
        y_parts.append(context["y_exp"][experiment_indices])

    model = make_frozen_model(
        context["params"],
        context["is_classification"],
        context["seed"],
        n_jobs=n_jobs,
    )
    model.fit(np.vstack(X_parts), np.concatenate(y_parts))
    predictions = model.predict(context["X_test"])
    score = primary_score(
        context["y_test"], predictions, context["is_classification"]
    )
    return score, predictions


def compute_assimilation_curve(
    context: dict[str, Any],
    repeats: int = 20,
    fractions: Sequence[float] = DEFAULT_DOSE_FRACTIONS,
    n_jobs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate the dose-response curve using nested random experiment orders.

    The 0% and 100% endpoints are fitted once because repeated permutations
    produce the identical training set.  Intermediate doses use one prefix of
    each independently shuffled order.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    fractions = sorted({float(value) for value in fractions})
    if not fractions or fractions[0] < 0 or fractions[-1] > 1:
        raise ValueError("dose fractions must lie in [0, 1]")
    if 0.0 not in fractions or 1.0 not in fractions:
        raise ValueError("dose fractions must include both 0 and 1")

    n_exp = len(context["X_exp"])
    target = context["target"]
    seed = int(context["seed"])
    rows: list[dict[str, Any]] = []
    endpoint_scores = {
        0: float(context["before_score"]),
        n_exp: float(context["after_score"]),
    }

    counts = {
        fraction: min(
            n_exp,
            max(0, int(math.floor(fraction * n_exp + 0.5))),
        )
        for fraction in fractions
    }

    for repeat in range(repeats):
        rng = np.random.default_rng(
            stable_int_seed(seed, target, "dose_order", repeat)
        )
        order = rng.permutation(n_exp)
        for fraction in fractions:
            k = counts[fraction]
            if k in endpoint_scores:
                if repeat > 0:
                    continue
                score = endpoint_scores[k]
                selected = order[:k]
            else:
                selected = order[:k]
                score, _ = _fit_subset_and_score(
                    context,
                    selected,
                    n_jobs,
                )
            selected_ids = context["experiment_sample_ids"][selected]
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "primary_metric": context["metric"],
                    "dose_fraction": fraction,
                    "dose_percent": 100.0 * fraction,
                    "n_experiments": int(k),
                    "n_experiments_total": int(n_exp),
                    "repeat": int(repeat),
                    "score": float(score),
                    "before_score": float(context["before_score"]),
                    "utility": float(score - context["before_score"]),
                    "selected_experiment_ids": ";".join(
                        str(int(value)) for value in selected_ids
                    ),
                    "configuration_mode": "frozen_no_hpo",
                }
            )

    raw = pd.DataFrame(rows)
    grouped = raw.groupby(
        [
            "target",
            "seed",
            "primary_metric",
            "dose_fraction",
            "dose_percent",
            "n_experiments",
            "n_experiments_total",
        ],
        as_index=False,
    )
    summary = grouped.agg(
        score_mean=("score", "mean"),
        score_sd=("score", "std"),
        score_ci_low=("score", lambda x: x.quantile(0.025)),
        score_ci_high=("score", lambda x: x.quantile(0.975)),
        utility_mean=("utility", "mean"),
        utility_sd=("utility", "std"),
        utility_ci_low=("utility", lambda x: x.quantile(0.025)),
        utility_ci_high=("utility", lambda x: x.quantile(0.975)),
        n_random_orders=("repeat", "nunique"),
        n_unique_subsets=("selected_experiment_ids", "nunique"),
    )
    for column in ["score_sd", "utility_sd"]:
        summary[column] = summary[column].fillna(0.0)
    return raw, summary


def compute_loeo_marginal_values(
    context: dict[str, Any], n_jobs: int = 2
) -> pd.DataFrame:
    """
    Exact leave-one-experiment-out marginal utility on the fixed literature test.

    V_j = M(D_L union D_E) - M(D_L union (D_E minus e_j)).
    Positive values are beneficial; negative values indicate interference.
    """
    n_exp = len(context["X_exp"])
    all_indices = np.arange(n_exp)
    rows: list[dict[str, Any]] = []

    for held_position in range(n_exp):
        keep = all_indices[all_indices != held_position]
        loo_score, _ = _fit_subset_and_score(
            context,
            keep,
            n_jobs,
        )
        value = float(context["after_score"] - loo_score)
        rows.append(
            {
                "target": context["target"],
                "seed": int(context["seed"]),
                "experiment_position": int(held_position),
                "experiment_sample_id": int(
                    context["experiment_sample_ids"][held_position]
                ),
                "primary_metric": context["metric"],
                "before_score": float(context["before_score"]),
                "full_after_score": float(context["after_score"]),
                "loeo_score": float(loo_score),
                "marginal_value": value,
                "value_direction": (
                    "beneficial"
                    if value > 0
                    else "harmful"
                    if value < 0
                    else "redundant"
                ),
                "configuration_mode": "frozen_no_hpo",
            }
        )
    return pd.DataFrame(rows)


def _cross_validated_label_surprise(
    context: dict[str, Any],
    n_folds: int = 5,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """
    Compute label surprise without using the fixed literature test set.

    Regression: |y_exp - yhat_literature| / OOF_RMSE.
    Classification: per-sample negative log likelihood / OOF_log_loss.
    """
    X_lit_raw = context["X_lit_raw"]
    X_exp = context["X_exp"]
    y_lit = np.asarray(context["y_lit"])
    y_exp = np.asarray(context["y_exp"])
    is_cls = context["is_classification"]
    seed = int(context["seed"])
    target = context["target"]

    if is_cls:
        class_counts = pd.Series(y_lit).value_counts()
        folds = min(int(n_folds), int(class_counts.min()))
        if folds < 2:
            raise ValueError("Too few minority-class rows for label-surprise CV")
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        split_iter = splitter.split(X_lit_raw, y_lit)
        oof = np.full(len(y_lit), np.nan, dtype=float)
    else:
        folds = min(int(n_folds), len(y_lit))
        if folds < 2:
            raise ValueError("Too few literature rows for label-surprise CV")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_lit_raw)
        oof = np.full(len(y_lit), np.nan, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(split_iter):
        fold_preprocessor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="mean", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        X_fold_train = fold_preprocessor.fit_transform(
            X_lit_raw.iloc[train_idx]
        )
        X_fold_valid = fold_preprocessor.transform(
            X_lit_raw.iloc[valid_idx]
        )
        model = make_frozen_model(
            context["params"],
            is_cls,
            seed,
            n_jobs=n_jobs,
        )
        model.fit(X_fold_train, y_lit[train_idx])
        if is_cls:
            oof[valid_idx] = model.predict_proba(X_fold_valid)[:, 1]
        else:
            oof[valid_idx] = model.predict(X_fold_valid)

    if np.isnan(oof).any():
        raise RuntimeError("Label-surprise OOF predictions are incomplete")

    baseline_model = context["model_before"]
    if is_cls:
        exp_prediction = baseline_model.predict_proba(X_exp)[:, 1]
        clipped_oof = np.clip(oof, 1e-12, 1 - 1e-12)
        scale = float(log_loss(y_lit.astype(int), clipped_oof, labels=[0, 1]))
        p_observed = np.where(
            y_exp.astype(int) == 1,
            np.clip(exp_prediction, 1e-12, 1 - 1e-12),
            np.clip(1.0 - exp_prediction, 1e-12, 1 - 1e-12),
        )
        raw_surprise = -np.log(p_observed)
        normalized = raw_surprise / max(scale, 1e-12)
        definition = "negative_log_likelihood / OOF_log_loss"
        prediction_kind = "literature_model_probability_V0"
    else:
        exp_prediction = baseline_model.predict(X_exp)
        scale = float(np.sqrt(mean_squared_error(y_lit, oof)))
        raw_surprise = np.abs(y_exp.astype(float) - exp_prediction.astype(float))
        normalized = raw_surprise / max(scale, 1e-12)
        definition = "absolute_residual / OOF_RMSE"
        prediction_kind = "literature_model_prediction"

    return pd.DataFrame(
        {
            "target": target,
            "seed": seed,
            "experiment_position": np.arange(len(y_exp), dtype=int),
            "experiment_sample_id": context["experiment_sample_ids"].astype(int),
            "experimental_label": y_exp,
            "literature_model_prediction": exp_prediction,
            "surprise_raw": raw_surprise,
            "surprise_scale": scale,
            "label_surprise": normalized,
            "surprise_definition": definition,
            "prediction_kind": prediction_kind,
            "cv_folds": folds,
        }
    )


def _target_space_novelty(
    context: dict[str, Any], n_neighbors: int = 10
) -> pd.DataFrame:
    """Low-memory seed-42 fallback novelty in the Frozen feature subspace."""
    X_lit = context["X_lit"]
    X_exp = context["X_exp"]
    k = min(max(1, int(n_neighbors)), max(1, len(X_lit) - 1))

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_lit)
    lit_dist = nn.kneighbors(X_lit)[0][:, 1:].mean(axis=1)
    exp_dist = nn.kneighbors(X_exp, n_neighbors=k)[0].mean(axis=1)
    percentiles = np.asarray(
        [np.mean(lit_dist <= distance) for distance in exp_dist],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "target": context["target"],
            "seed": int(context["seed"]),
            "experiment_position": np.arange(len(X_exp), dtype=int),
            "experiment_sample_id": context["experiment_sample_ids"].astype(int),
            "mean_knn_distance": exp_dist,
            "descriptor_novelty_percentile": percentiles,
            "novelty_k": k,
            "novelty_source": "target_frozen_feature_subspace_fallback",
        }
    )


def load_global_novelty_if_available(
    source_results: str | Path,
    data: dict[str, Any],
    seed: int,
) -> pd.DataFrame | None:
    """
    Reuse the original all-descriptor kNN audit when the long run has finished.

    The original file numbers all 24 experimental rows in their global order.
    Absolute sample IDs are reconstructed from the target-specific IDs.
    """
    path = Path(source_results) / "Distribution" / "experimental_knn_density.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    required = {
        "experiment_index",
        "mean_knn_distance",
        "literature_distance_percentile",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None

    n_literature_all = data.get("n_literature_all")
    if n_literature_all is None:
        return None
    global_id_by_position = {
        int(position): int(n_literature_all + position)
        for position in frame["experiment_index"]
    }
    target_ids = [int(value) for value in data["experiment_sample_ids"]]
    inverse = {
        sample_id: sample_id - int(n_literature_all) for sample_id in target_ids
    }
    selected = frame[
        frame["experiment_index"].astype(int).isin(inverse.values())
    ].copy()
    if len(selected) != len(target_ids):
        return None
    selected["experiment_sample_id"] = selected["experiment_index"].map(
        global_id_by_position
    )
    selected["target"] = data["target"]
    selected["seed"] = int(seed)
    selected["experiment_position"] = selected["experiment_sample_id"].map(
        {sample_id: position for position, sample_id in enumerate(target_ids)}
    )
    selected = selected.rename(
        columns={
            "literature_distance_percentile": "descriptor_novelty_percentile"
        }
    )
    selected["novelty_k"] = np.nan
    selected["novelty_source"] = "original_all_descriptor_knn_audit"
    return selected[
        [
            "target",
            "seed",
            "experiment_position",
            "experiment_sample_id",
            "mean_knn_distance",
            "descriptor_novelty_percentile",
            "novelty_k",
            "novelty_source",
        ]
    ].sort_values("experiment_position")


def _local_compatibility(
    context: dict[str, Any], n_neighbors: int = 10
) -> pd.DataFrame:
    """Measure label compatibility with nearby literature-training samples."""
    X_lit = context["X_lit"]
    X_exp = context["X_exp"]
    y_lit = np.asarray(context["y_lit"])
    y_exp = np.asarray(context["y_exp"])
    k = min(max(1, int(n_neighbors)), len(X_lit))
    nn = NearestNeighbors(n_neighbors=k).fit(X_lit)
    neighbor_idx = nn.kneighbors(X_exp, return_distance=False)
    neighbor_y = y_lit[neighbor_idx]

    if context["is_classification"]:
        observed = y_exp.astype(int)
        local_positive_rate = neighbor_y.mean(axis=1)
        compatibility = np.where(
            observed == 1, local_positive_rate, 1.0 - local_positive_rate
        )
        gap = 1.0 - compatibility
        local_center = local_positive_rate
        definition = "fraction_of_k_neighbors_with_same_class"
    else:
        local_center = np.median(neighbor_y.astype(float), axis=1)
        scale = float(np.std(y_lit.astype(float), ddof=1))
        gap = np.abs(y_exp.astype(float) - local_center) / max(scale, 1e-12)
        compatibility = 1.0 / (1.0 + gap)
        definition = "1 / (1 + abs(label-local_median)/literature_SD)"

    return pd.DataFrame(
        {
            "target": context["target"],
            "seed": int(context["seed"]),
            "experiment_position": np.arange(len(X_exp), dtype=int),
            "experiment_sample_id": context["experiment_sample_ids"].astype(int),
            "local_label_center": local_center,
            "local_label_gap": gap,
            "local_compatibility": compatibility,
            "compatibility_k": k,
            "compatibility_definition": definition,
        }
    )


def enrich_experimental_values(
    loeo: pd.DataFrame,
    context: dict[str, Any],
    data: dict[str, Any],
    source_results: str | Path,
    n_neighbors: int = 10,
    n_folds: int = 5,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """Attach novelty, label surprise, and local compatibility to LOEO values."""
    novelty = load_global_novelty_if_available(
        source_results, data=data, seed=context["seed"]
    )
    if novelty is None:
        novelty = _target_space_novelty(context, n_neighbors=n_neighbors)
    surprise = _cross_validated_label_surprise(
        context, n_folds=n_folds, n_jobs=n_jobs
    )
    compatibility = _local_compatibility(context, n_neighbors=n_neighbors)

    keys = ["target", "seed", "experiment_position", "experiment_sample_id"]
    result = loeo.merge(novelty, on=keys, how="left", validate="one_to_one")
    result = result.merge(surprise, on=keys, how="left", validate="one_to_one")
    result = result.merge(
        compatibility, on=keys, how="left", validate="one_to_one"
    )
    return result


def data_value_correlations(values: pd.DataFrame) -> pd.DataFrame:
    """Spearman associations requested by the knowledge-value analysis."""
    rows: list[dict[str, Any]] = []
    predictors = [
        "descriptor_novelty_percentile",
        "label_surprise",
        "local_compatibility",
    ]
    for (target, seed), group in values.groupby(["target", "seed"]):
        for predictor in predictors:
            clean = group[["marginal_value", predictor]].dropna()
            if len(clean) >= 3 and clean[predictor].nunique() > 1:
                rho, p_value = spearmanr(
                    clean[predictor], clean["marginal_value"]
                )
            else:
                rho, p_value = np.nan, np.nan
            rows.append(
                {
                    "target": target,
                    "seed": int(seed),
                    "predictor": predictor,
                    "outcome": "marginal_value",
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": (
                        float(p_value) if np.isfinite(p_value) else np.nan
                    ),
                    "n_experiments": int(len(clean)),
                    "interpretation_scope": "descriptive_small_n",
                }
            )
    return pd.DataFrame(rows)


def _assign_distance_bins(distances: pd.Series) -> pd.Series:
    """Balanced near/middle/far groups, robust to tied distances."""
    if len(distances) < 3:
        labels = ["near"] * len(distances)
        return pd.Series(labels, index=distances.index, dtype="object")
    ranks = distances.rank(method="first")
    return pd.qcut(
        ranks,
        q=3,
        labels=["near", "middle", "far"],
    ).astype("object")


def compute_propagation(
    context: dict[str, Any],
    saved_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Relate fixed-test loss improvement to distance from target-labelled experiments.

    This stage reuses saved endpoint predictions and therefore performs no fit.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(context["X_exp"])
    distances = nn.kneighbors(context["X_test"])[0][:, 0]

    saved = saved_predictions.copy()
    saved["sample_id"] = pd.to_numeric(saved["sample_id"], errors="raise").astype(int)
    expected_ids = pd.Index(context["test_sample_ids"].astype(int), name="sample_id")
    saved = saved.set_index("sample_id").reindex(expected_ids).reset_index()
    if saved[["y_true", "y_pred_before", "y_pred_after"]].isna().any().any():
        raise ValueError("Saved prediction sample IDs do not match the fixed split")

    is_cls = context["is_classification"]
    loss_before = per_sample_losses(
        saved["y_true"], saved["y_pred_before"], is_cls
    )
    loss_after = per_sample_losses(
        saved["y_true"], saved["y_pred_after"], is_cls
    )

    samples = saved.copy()
    samples["distance_to_nearest_experiment"] = distances
    samples["distance_percentile"] = pd.Series(distances).rank(pct=True).to_numpy()
    samples["neighborhood"] = _assign_distance_bins(
        pd.Series(distances)
    ).to_numpy()
    samples["loss_definition"] = (
        "0_1_error" if is_cls else "squared_error"
    )
    samples["loss_before"] = loss_before
    samples["loss_after"] = loss_after
    samples["loss_improvement"] = loss_before - loss_after
    samples["knowledge_effect"] = np.select(
        [
            samples["loss_improvement"] > 0,
            samples["loss_improvement"] < 0,
        ],
        ["improved", "worsened"],
        default="unchanged",
    )

    region_rows: list[dict[str, Any]] = []
    for region in ["near", "middle", "far"]:
        group = samples[samples["neighborhood"] == region]
        if group.empty:
            continue
        yt = group["y_true"].to_numpy()
        pb = group["y_pred_before"].to_numpy()
        pa = group["y_pred_after"].to_numpy()
        if is_cls:
            both_classes = len(np.unique(yt.astype(int))) >= 2
            if both_classes:
                before_primary = float(balanced_accuracy_score(yt, pb))
                after_primary = float(balanced_accuracy_score(yt, pa))
                metric_defined = True
            else:
                before_primary = float(np.mean(yt.astype(int) == pb.astype(int)))
                after_primary = float(np.mean(yt.astype(int) == pa.astype(int)))
                metric_defined = False
        else:
            metric_defined = len(group) >= 2 and np.var(yt.astype(float)) > 0
            before_primary = (
                float(r2_score(yt, pb)) if metric_defined else np.nan
            )
            after_primary = (
                float(r2_score(yt, pa)) if metric_defined else np.nan
            )
        region_rows.append(
            {
                "target": context["target"],
                "seed": int(context["seed"]),
                "neighborhood": region,
                "n_test_samples": int(len(group)),
                "primary_metric": context["metric"],
                "primary_metric_defined": metric_defined,
                "before_primary": before_primary,
                "after_primary": after_primary,
                "utility": after_primary - before_primary
                if metric_defined
                else np.nan,
                "mean_loss_before": float(group["loss_before"].mean()),
                "mean_loss_after": float(group["loss_after"].mean()),
                "mean_loss_improvement": float(
                    group["loss_improvement"].mean()
                ),
                "fraction_improved": float(
                    (group["loss_improvement"] > 0).mean()
                ),
                "distance_min": float(
                    group["distance_to_nearest_experiment"].min()
                ),
                "distance_median": float(
                    group["distance_to_nearest_experiment"].median()
                ),
                "distance_max": float(
                    group["distance_to_nearest_experiment"].max()
                ),
            }
        )
    return samples, pd.DataFrame(region_rows)


def _coerce_shap_values(values: Any, is_classification: bool) -> np.ndarray:
    """Normalize SHAP's version-dependent binary-class output shape."""
    if isinstance(values, list):
        values = values[1] if is_classification and len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, 1] if is_classification else array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {array.shape}")
    return array


def compute_plasticity(
    context: dict[str, Any],
    utility: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the utility-plasticity coordinates and per-feature SHAP values."""
    import shap

    explainer_before = shap.TreeExplainer(context["model_before"])
    explainer_after = shap.TreeExplainer(context["model_after"])
    shap_before = _coerce_shap_values(
        explainer_before(context["X_test"]).values,
        context["is_classification"],
    )
    shap_after = _coerce_shap_values(
        explainer_after(context["X_test"]).values,
        context["is_classification"],
    )
    before_metrics = calculate_shap_structure_metrics(
        shap_before, context["features"]
    )
    after_metrics = calculate_shap_structure_metrics(
        shap_after, context["features"]
    )
    similarity = ranking_similarity(
        before_metrics["normalized_shap"],
        after_metrics["normalized_shap"],
        context["features"],
    )

    phase = pd.DataFrame(
        [
            {
                "target": context["target"],
                "seed": int(context["seed"]),
                "primary_metric": context["metric"],
                "before_primary": float(context["before_score"]),
                "after_primary": float(context["after_score"]),
                "utility": float(utility),
                "top10_jaccard": similarity["top10_jaccard"],
                "plasticity": 1.0 - similarity["top10_jaccard"],
                "spearman_rho": similarity["spearman_rho"],
                "spearman_p": similarity["spearman_p"],
                "effective_features_before": before_metrics[
                    "effective_feature_number"
                ],
                "effective_features_after": after_metrics[
                    "effective_feature_number"
                ],
                "effective_features_delta": after_metrics[
                    "effective_feature_number"
                ]
                - before_metrics["effective_feature_number"],
                "entropy_before": before_metrics["shap_entropy"],
                "entropy_after": after_metrics["shap_entropy"],
                "entropy_delta": after_metrics["shap_entropy"]
                - before_metrics["shap_entropy"],
                "top1_share_before": before_metrics["top1_share"],
                "top1_share_after": after_metrics["top1_share"],
                "top5_share_before": before_metrics["top5_share"],
                "top5_share_after": after_metrics["top5_share"],
                "shap_source": "reconstructed_from_frozen_seed_models",
            }
        ]
    )

    feature_rows: list[dict[str, Any]] = []
    for track, metrics in [
        ("before", before_metrics),
        ("after", after_metrics),
    ]:
        rank_by_index = np.empty(len(metrics["ranking"]), dtype=int)
        rank_by_index[metrics["ranking"]] = np.arange(
            1, len(metrics["ranking"]) + 1
        )
        for index, feature in enumerate(context["features"]):
            feature_rows.append(
                {
                    "target": context["target"],
                    "seed": int(context["seed"]),
                    "track": track,
                    "feature": feature,
                    "rank": int(rank_by_index[index]),
                    "normalized_mean_abs_shap": float(
                        metrics["normalized_shap"][index]
                    ),
                    "is_top10": bool(rank_by_index[index] <= 10),
                    "shap_source": "reconstructed_from_frozen_seed_models",
                }
            )
    return phase, pd.DataFrame(feature_rows)


def compute_extra_trees_sensitivity(
    context: dict[str, Any],
    n_jobs: int = 2,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """
    Optional fixed-rule model-family sensitivity check.

    ExtraTrees is intentionally not tuned.  It uses the same Frozen feature
    subset, literature-fitted preprocessor, fixed split, and experiment dose as
    XGBoost.  SHAP is used only for the same decision-structure plasticity
    summary, not for a physical-mechanism claim.
    """
    is_cls = context["is_classification"]
    common = {
        "n_estimators": int(n_estimators),
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "random_state": int(context["seed"]),
        "n_jobs": max(1, int(n_jobs)),
    }
    if is_cls:
        before = ExtraTreesClassifier(**common, class_weight="balanced")
        after = ExtraTreesClassifier(**common, class_weight="balanced")
    else:
        before = ExtraTreesRegressor(**common)
        after = ExtraTreesRegressor(**common)

    before.fit(context["X_lit"], context["y_lit"])
    after.fit(
        np.vstack([context["X_lit"], context["X_exp"]]),
        np.concatenate([context["y_lit"], context["y_exp"]]),
    )
    pred_before = before.predict(context["X_test"])
    pred_after = after.predict(context["X_test"])
    before_score = primary_score(context["y_test"], pred_before, is_cls)
    after_score = primary_score(context["y_test"], pred_after, is_cls)

    import shap

    shap_before = _coerce_shap_values(
        shap.TreeExplainer(before)(context["X_test"]).values, is_cls
    )
    shap_after = _coerce_shap_values(
        shap.TreeExplainer(after)(context["X_test"]).values, is_cls
    )
    before_metrics = calculate_shap_structure_metrics(
        shap_before, context["features"]
    )
    after_metrics = calculate_shap_structure_metrics(
        shap_after, context["features"]
    )
    similarity = ranking_similarity(
        before_metrics["normalized_shap"],
        after_metrics["normalized_shap"],
        context["features"],
    )
    return pd.DataFrame(
        [
            {
                "target": context["target"],
                "seed": int(context["seed"]),
                "model_family": "ExtraTrees",
                "configuration_mode": "fixed_rule_no_hpo",
                "primary_metric": context["metric"],
                "before_primary": before_score,
                "after_primary": after_score,
                "utility": after_score - before_score,
                "top10_jaccard": similarity["top10_jaccard"],
                "plasticity": 1.0 - similarity["top10_jaccard"],
                "spearman_rho": similarity["spearman_rho"],
                "effective_features_before": before_metrics[
                    "effective_feature_number"
                ],
                "effective_features_after": after_metrics[
                    "effective_feature_number"
                ],
                "entropy_before": before_metrics["shap_entropy"],
                "entropy_after": after_metrics["shap_entropy"],
                "n_estimators": int(n_estimators),
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "interpretation_scope": "model_family_sensitivity_only",
            }
        ]
    )


def assign_update_states(
    phase_map: pd.DataFrame,
    plasticity_threshold: float = DEFAULT_PLASTICITY_THRESHOLD,
    regression_tolerance: float = REGRESSION_UTILITY_TOLERANCE,
    classification_tolerance: float = CLASSIFICATION_UTILITY_TOLERANCE,
) -> pd.DataFrame:
    """Assign the four pre-declared update states without target-wise tuning."""
    result = phase_map.copy()
    tolerances = np.where(
        result["primary_metric"].eq("balanced_accuracy"),
        float(classification_tolerance),
        float(regression_tolerance),
    )
    result["utility_tolerance"] = tolerances
    result["plasticity_threshold"] = float(plasticity_threshold)
    result["update_outcome"] = np.select(
        [
            result["utility"] > tolerances,
            result["utility"] < -tolerances,
        ],
        ["assimilation", "interference"],
        default="inertness",
    )

    states: list[str] = []
    for row, tolerance in zip(result.to_dict("records"), tolerances):
        if row["utility"] > tolerance:
            state = (
                "Plastic assimilation"
                if row["plasticity"] >= plasticity_threshold
                else "Stable assimilation"
            )
        elif row["utility"] < -tolerance:
            state = "Destructive interference"
        else:
            state = "Inert update"
        states.append(state)
    result["phase_state"] = states
    return result


def endpoint_utility_from_saved_predictions(
    predictions: pd.DataFrame, is_classification: bool
) -> dict[str, float]:
    """Compute the authoritative endpoint utility from the saved long-run file."""
    before = primary_score(
        predictions["y_true"],
        predictions["y_pred_before"],
        is_classification,
    )
    after = primary_score(
        predictions["y_true"],
        predictions["y_pred_after"],
        is_classification,
    )
    return {
        "before_primary": before,
        "after_primary": after,
        "utility": after - before,
    }


def parse_dose_fractions(text: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        values = [float(item) for item in text]
    normalized = tuple(sorted(set(values)))
    if not normalized or normalized[0] < 0 or normalized[-1] > 1:
        raise ValueError("Dose fractions must be comma-separated values in [0, 1]")
    if 0.0 not in normalized or 1.0 not in normalized:
        raise ValueError("Dose fractions must include 0 and 1")
    return normalized


def estimate_fit_count(
    n_experiments: int,
    dose_repeats: int,
    dose_fractions: Sequence[float],
    include_dose: bool = True,
    include_loeo: bool = True,
    include_value_attributes: bool = True,
    include_model_sensitivity: bool = False,
) -> dict[str, int]:
    """Transparent cost estimate; HPO fit count is always zero."""
    intermediate = sum(0.0 < value < 1.0 for value in dose_fractions)
    counts = {
        "endpoint_models": 2,
        "dose_models": int(dose_repeats * intermediate) if include_dose else 0,
        "loeo_models": int(n_experiments) if include_loeo else 0,
        "label_surprise_cv_models": 5 if include_value_attributes else 0,
        "extra_trees_endpoint_models": 2 if include_model_sensitivity else 0,
        "hpo_or_optuna_models": 0,
    }
    counts["total_frozen_fits_estimate"] = int(sum(counts.values()))
    return counts