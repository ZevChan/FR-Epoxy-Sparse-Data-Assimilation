"""Compute sample-level signed SHAP for the formal 10-seed Frozen analysis.

This script reuses each seed/target's saved Frozen configuration.  It never
performs feature selection, Optuna, or hyperparameter optimisation.  Each unit
is cached independently so an interrupted run resumes without refitting units
that are already complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "Results")
FROZEN_DIR = os.path.join(RESULTS, "Frozen")
OUT_DIR = os.path.join(RESULTS, "Knowledge_Assimilation", "SHAP_Relationships")
UNIT_DIR = os.path.join(OUT_DIR, "Units")
STATUS_PATH = os.path.join(OUT_DIR, "signed_shap_status.json")
AUDIT_PATH = os.path.join(OUT_DIR, "signed_shap_reconstruction_audit.csv")
FINAL_PATH = os.path.join(OUT_DIR, "signed_shap_samples.csv")
SELECTION_PATH = os.path.join(OUT_DIR, "signed_shap_feature_selection.csv")

FORMAL_SEEDS = [7, 13, 19, 29, 37, 43, 53, 61, 71, 79]
TARGETS = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]

sys.path.insert(0, BASE)
from fair_holdout_comparison import (  # noqa: E402
    get_train_test_for_target,
    make_model,
    prepare_fixed_split,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def unit_path(target: str, seed: int) -> str:
    return os.path.join(UNIT_DIR, f"signed_shap_{target}_seed_{seed}.csv")


def valid_unit(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        probe = pd.read_csv(path, usecols=["target", "seed", "track", "feature"])
        return len(probe) > 0 and set(probe["track"].unique()) == {"before", "after"}
    except Exception:
        return False


def normalise_shap_values(values, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[1] == 2:
            arr = arr[:, 1, :]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP shape {arr.shape}; expected (*, {n_features})")
    return arr


def build_unit(X_all, y_all, is_literature, target: str, seed: int):
    config_path = os.path.join(FROZEN_DIR, f"config_{target}_seed_{seed}.json")
    prediction_path = os.path.join(FROZEN_DIR, f"predictions_{target}_seed_{seed}.csv")
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    features = list(config["features"])
    params = dict(config["params"])
    is_cls = target == "UL94_Rating"
    data = get_train_test_for_target(X_all, y_all, is_literature, target, seed=seed)
    if data is None:
        raise RuntimeError(f"No valid split for {target}, seed {seed}")

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    X_before = preprocessor.fit_transform(data["X_train_before"][features])
    X_after = preprocessor.transform(data["X_train_after"][features])
    X_test = preprocessor.transform(data["X_test"][features])
    X_test_raw = data["X_test"][features].apply(pd.to_numeric, errors="coerce").to_numpy()

    model_before = make_model(params, is_cls, seed)
    model_after = make_model(params, is_cls, seed)
    model_before.fit(X_before, data["y_train_before"])
    model_after.fit(X_after, data["y_train_after"])

    pred_before = model_before.predict(X_test)
    pred_after = model_after.predict(X_test)
    audit = {
        "target": target,
        "seed": seed,
        "n_test": len(data["y_test"]),
        "before_max_abs_diff": np.nan,
        "after_max_abs_diff": np.nan,
        "before_corr": np.nan,
        "after_corr": np.nan,
        "audit_pass": False,
    }
    if os.path.exists(prediction_path):
        original = pd.read_csv(prediction_path)
        if len(original) == len(pred_before):
            for track, rebuilt in [("before", pred_before), ("after", pred_after)]:
                column = f"y_pred_{track}"
                if column in original:
                    reference = pd.to_numeric(original[column], errors="coerce").to_numpy()
                    diff = np.abs(reference - rebuilt)
                    audit[f"{track}_max_abs_diff"] = float(np.nanmax(diff))
                    audit[f"{track}_corr"] = float(np.corrcoef(reference, rebuilt)[0, 1]) if len(reference) > 1 else 1.0
            audit["audit_pass"] = bool(
                np.isfinite(audit["before_max_abs_diff"])
                and np.isfinite(audit["after_max_abs_diff"])
                and audit["before_max_abs_diff"] < 1e-5
                and audit["after_max_abs_diff"] < 1e-5
            )

    import shap

    frames = []
    y_true = pd.to_numeric(data["y_test"], errors="coerce").to_numpy()
    sample_ids = np.asarray(data["test_sample_ids"])
    for track, model, predictions in [
        ("before", model_before, pred_before),
        ("after", model_after, pred_after),
    ]:
        explainer = shap.TreeExplainer(model)
        shap_values = normalise_shap_values(explainer(X_test).values, len(features))
        n_samples, n_features = shap_values.shape
        frames.append(pd.DataFrame({
            "target": np.repeat(target, n_samples * n_features),
            "seed": np.repeat(seed, n_samples * n_features),
            "track": np.repeat(track, n_samples * n_features),
            "sample_id": np.repeat(sample_ids, n_features),
            "feature": np.tile(features, n_samples),
            "feature_value_raw": X_test_raw.reshape(-1),
            "feature_value_scaled": X_test.reshape(-1),
            "shap_value": shap_values.reshape(-1),
            "y_true": np.repeat(y_true, n_features),
            "y_pred": np.repeat(predictions, n_features),
        }))
    return pd.concat(frames, ignore_index=True), audit


def write_feature_selection() -> pd.DataFrame:
    source = os.path.join(RESULTS, "Knowledge_Assimilation", "Plasticity", "shap_feature_values.csv")
    values = pd.read_csv(source)
    values = values[values["seed"].isin(FORMAL_SEEDS) & values["target"].isin(TARGETS)].copy()
    values["is_top10"] = values["is_top10"].astype(str).str.lower().isin(["true", "1", "yes"])
    rows = []
    for target in TARGETS:
        subset = values[values["target"] == target]
        candidates = []
        for feature, group in subset.groupby("feature"):
            top = group[group["is_top10"]]
            candidates.append({
                "target": target,
                "feature": feature,
                "top10_seed_count": int(top["seed"].nunique()),
                "top10_seed_track_count": int(top[["seed", "track"]].drop_duplicates().shape[0]),
                "median_rank": float(group["rank"].median()),
                "median_normalized_mean_abs_shap": float(group["normalized_mean_abs_shap"].median()),
                "seed_coverage": int(group["seed"].nunique()),
            })
        ranking = pd.DataFrame(candidates).sort_values(
            ["top10_seed_count", "top10_seed_track_count", "median_rank", "median_normalized_mean_abs_shap", "feature"],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)
        ranking["cross_seed_rank"] = np.arange(1, len(ranking) + 1)
        ranking["selected_for_main_figure"] = ranking["cross_seed_rank"] == 1
        rows.append(ranking)
    result = pd.concat(rows, ignore_index=True)
    atomic_csv(result, SELECTION_PATH)
    return result


def aggregate_units(seeds, targets) -> None:
    paths = [unit_path(target, seed) for seed in seeds for target in targets]
    missing = [path for path in paths if not valid_unit(path)]
    if missing:
        raise RuntimeError(f"Cannot aggregate; {len(missing)} units are missing or invalid")
    tmp = FINAL_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as output:
        writer = None
        for path in paths:
            with open(path, newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    os.replace(tmp, FINAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    if any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"Only formal seeds are permitted: {FORMAL_SEEDS}")

    os.makedirs(UNIT_DIR, exist_ok=True)
    expected = len(seeds) * len(targets)
    completed = sum(valid_unit(unit_path(target, seed)) for seed in seeds for target in targets)
    status = {
        "status": "running",
        "started_utc": utc_now(),
        "formal_seeds": seeds,
        "targets": targets,
        "expected_units": expected,
        "completed_units": completed,
        "current": None,
        "failed_units": [],
        "hpo_calls": 0,
    }
    atomic_json(STATUS_PATH, status)
    X_all, y_all, _, is_literature = prepare_fixed_split()
    audits = []
    try:
        for seed in seeds:
            for target in targets:
                path = unit_path(target, seed)
                if valid_unit(path) and not args.force:
                    print(f"[CACHE] seed={seed} target={target}")
                    continue
                status["current"] = {"seed": seed, "target": target, "stage": "fit_and_shap"}
                atomic_json(STATUS_PATH, status)
                print(f"[RUN] seed={seed} target={target}", flush=True)
                try:
                    frame, audit = build_unit(X_all, y_all, is_literature, target, seed)
                    atomic_csv(frame, path)
                    audits.append(audit)
                    status["completed_units"] = sum(
                        valid_unit(unit_path(t, s)) for s in seeds for t in targets
                    )
                    atomic_json(STATUS_PATH, status)
                except Exception as exc:
                    failure = {"seed": seed, "target": target, "error": repr(exc), "traceback": traceback.format_exc()}
                    status["failed_units"].append(failure)
                    status["status"] = "failed"
                    atomic_json(STATUS_PATH, status)
                    raise
        if audits:
            prior = pd.read_csv(AUDIT_PATH) if os.path.exists(AUDIT_PATH) else pd.DataFrame()
            combined = pd.concat([prior, pd.DataFrame(audits)], ignore_index=True)
            combined = combined.drop_duplicates(["target", "seed"], keep="last")
            atomic_csv(combined.sort_values(["seed", "target"]), AUDIT_PATH)
        status["current"] = {"stage": "aggregate"}
        atomic_json(STATUS_PATH, status)
        aggregate_units(seeds, targets)
        selection = write_feature_selection()
        status.update({
            "status": "complete",
            "completed_utc": utc_now(),
            "completed_units": expected,
            "current": {"stage": "complete"},
            "output": FINAL_PATH,
            "feature_selection": SELECTION_PATH,
            "selected_features": selection[selection["selected_for_main_figure"]][["target", "feature"]].to_dict("records"),
        })
        atomic_json(STATUS_PATH, status)
        print(f"[DONE] {expected}/{expected} units; HPO calls=0", flush=True)
    except Exception:
        status["status"] = "failed"
        status["completed_utc"] = utc_now()
        atomic_json(STATUS_PATH, status)
        raise


if __name__ == "__main__":
    main()