"""
Independent post-hoc runner for sparse experimental knowledge assimilation.

Default behavior:
（即当前仓库的 Results/Knowledge_Assimilation）
  * analyzes random seed 42 only
  * never runs Optuna/HPO or Adaptive configuration selection
  * writes only to ./Results/Knowledge_Assimilation
  * checkpoints every target and every analysis stage
"""
from __future__ import annotations


import argparse
import json
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import N_CV_FOLDS, SEEDS, TARGETS
from fair_holdout_comparison import (
    get_train_test_for_target,
    prepare_fixed_split,
)
from knowledge_assimilation import (
    DEFAULT_DOSE_FRACTIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLASTICITY_THRESHOLD,
    DEFAULT_SOURCE_RESULTS,
    PRIMARY_SEED,
    assign_update_states,
    build_frozen_context,
    compute_assimilation_curve,
    compute_loeo_marginal_values,
    compute_extra_trees_sensitivity,
    compute_plasticity,
    compute_propagation,
    data_value_correlations,
    discover_complete_seeds,
    endpoint_reproduction_audit,
    endpoint_utility_from_saved_predictions,
    enrich_experimental_values,
    ensure_separate_directories,
    estimate_fit_count,
    load_frozen_config,
    load_saved_predictions,
    parse_dose_fractions,
    stable_signature,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _cache_valid(meta_path: Path, signature: str, files: Sequence[Path]) -> bool:
    if not meta_path.is_file() or not all(path.is_file() for path in files):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("signature") == signature


def _write_stage_meta(
    meta_path: Path,
    signature: str,
    stage: str,
    files: Sequence[Path],
) -> None:
    _atomic_write_json(
        meta_path,
        {
            "stage": stage,
            "signature": signature,
            "completed_utc": _utc_now(),
            "files": [str(path) for path in files],
            "hpo_or_optuna_calls": 0,
        },
    )


def _prediction_signature(predictions: pd.DataFrame) -> str:
    columns = [
        "sample_id",
        "y_true",
        "y_pred_before",
        "y_pred_after",
    ]
    values = pd.util.hash_pandas_object(
        predictions[columns], index=False
    ).astype("uint64")
    return stable_signature(values.tolist())


def _parse_seed_request(text: str, source_results: Path, targets: list[str]) -> list[int]:
    normalized = text.strip().lower()
    if normalized == "available":
        complete = discover_complete_seeds(source_results, targets, SEEDS)
        if not complete:
            raise FileNotFoundError(
                "No seed currently has complete Frozen config/prediction artifacts "
                "for every requested target."
            )
        return complete
    if normalized == "all":
        return [int(seed) for seed in SEEDS]
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must be 42, a comma-separated list, available, or all")
    return list(dict.fromkeys(seeds))


def _resolve_artifacts(
    source_results: Path,
    target: str,
    requested_seed: int,
    allow_seed42_fallback: bool,
) -> dict[str, Any]:
    fallback = PRIMARY_SEED if allow_seed42_fallback else None
    config, config_seed, config_path = load_frozen_config(
        source_results,
        target,
        requested_seed,
        fallback_seed=fallback,
    )
    predictions, prediction_seed, prediction_path = load_saved_predictions(
        source_results,
        target,
        requested_seed,
        fallback_seed=fallback,
    )
    if config_seed != prediction_seed:
        # Never combine a split/config from one seed with predictions from another.
        config, config_seed, config_path = load_frozen_config(
            source_results,
            target,
            PRIMARY_SEED,
            fallback_seed=None,
        )
        predictions, prediction_seed, prediction_path = load_saved_predictions(
            source_results,
            target,
            PRIMARY_SEED,
            fallback_seed=None,
        )
    return {
        "target": target,
        "requested_seed": int(requested_seed),
        "effective_seed": int(config_seed),
        "used_seed42_fallback": int(config_seed) != int(requested_seed),
        "config": config,
        "config_path": config_path,
        "predictions": predictions,
        "prediction_path": prediction_path,
    }


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_results, output_dir = ensure_separate_directories(
        args.source_results, args.output
    )
    if not source_results.is_dir():
        raise FileNotFoundError(f"Original Results directory not found: {source_results}")

    requested_targets = list(dict.fromkeys(args.targets))
    unknown = sorted(set(requested_targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}; valid targets: {TARGETS}")
    requested_seeds = _parse_seed_request(
        args.seeds, source_results, requested_targets
    )
    fractions = parse_dose_fractions(args.dose_fractions)

    X_all, y_all, n_literature_all, is_literature = prepare_fixed_split()
    resolved_units: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    validation_rows: list[dict[str, Any]] = []

    for requested_seed in requested_seeds:
        for target in requested_targets:
            artifact = _resolve_artifacts(
                source_results,
                target,
                requested_seed,
                allow_seed42_fallback=not args.no_seed42_fallback,
            )
            effective_seed = artifact["effective_seed"]
            data = get_train_test_for_target(
                X_all,
                y_all,
                is_literature,
                target,
                seed=effective_seed,
            )
            if data is None:
                raise ValueError(
                    f"Could not reconstruct the fixed split for {target}, seed={effective_seed}"
                )
            data["n_literature_all"] = int(n_literature_all)
            missing_features = sorted(
                set(artifact["config"]["features"]) - set(data["X_train_before"].columns)
            )
            if missing_features:
                raise KeyError(
                    f"{target}, seed={effective_seed}: missing Frozen features "
                    f"{missing_features[:10]}"
                )
            if len(artifact["predictions"]) != data["n_test"]:
                raise ValueError(
                    f"{target}, seed={effective_seed}: saved prediction rows "
                    f"({len(artifact['predictions'])}) != reconstructed test rows "
                    f"({data['n_test']})"
                )

            fit_count = estimate_fit_count(
                n_experiments=data["n_exp"],
                dose_repeats=args.dose_repeats,
                dose_fractions=fractions,
                include_dose=not args.skip_dose,
                include_loeo=not args.skip_loeo,
                include_value_attributes=(
                    not args.skip_loeo and not args.skip_value_attributes
                ),
                include_model_sensitivity=args.include_extra_trees,
            )
            validation_rows.append(
                {
                    "target": target,
                    "requested_seed": int(requested_seed),
                    "effective_seed": int(effective_seed),
                    "used_seed42_fallback": artifact["used_seed42_fallback"],
                    "n_literature_train": int(data["n_train_before"]),
                    "n_experiments": int(data["n_exp"]),
                    "n_test": int(data["n_test"]),
                    "frozen_k": int(artifact["config"]["k"]),
                    "config_path": str(artifact["config_path"]),
                    "prediction_path": str(artifact["prediction_path"]),
                    **fit_count,
                }
            )
            key = (target, effective_seed)
            if key in seen:
                continue
            seen.add(key)
            resolved_units.append({**artifact, "data": data})

    validation = {
        "validated_utc": _utc_now(),
        "source_results_read_only": str(source_results),
        "output_dir": str(output_dir),
        "targets": requested_targets,
        "requested_seeds": requested_seeds,
        "effective_units": [
            {"target": unit["target"], "seed": unit["effective_seed"]}
            for unit in resolved_units
        ],
        "dose_fractions": fractions,
        "dose_repeats": int(args.dose_repeats),
        "n_jobs_per_xgboost_fit": int(args.n_jobs),
        "configuration_selection_calls": 0,
        "optuna_calls": 0,
        "full_hpo_pipeline_called": False,
        "rows": validation_rows,
    }
    return validation, resolved_units


def _unit_signature(unit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target": unit["target"],
        "effective_seed": unit["effective_seed"],
        "config": unit["config"],
        "prediction_signature": _prediction_signature(unit["predictions"]),
        "dose_repeats": args.dose_repeats,
        "dose_fractions": parse_dose_fractions(args.dose_fractions),
        "knn_neighbors": args.knn_neighbors,
        "cv_folds": args.cv_folds,
        "plasticity_threshold": args.plasticity_threshold,
        # Keep the core-stage cache signature compatible with runs that did not
        # request the optional model-family sensitivity. ExtraTrees has its own
        # stage/family signature below and must not invalidate dose/LOEO/etc.
        "include_extra_trees": False,
        "regression_utility_tolerance": args.regression_utility_tolerance,
        "classification_utility_tolerance": args.classification_utility_tolerance,
        "code_schema": 1,
    }


def run_unit(
    unit: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    target = unit["target"]
    seed = int(unit["effective_seed"])
    data = unit["data"]
    unit_dir = output_dir / "WorkUnits" / f"seed_{seed}" / target
    unit_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {target} | seed {seed} | Frozen post-hoc (no HPO) ---")
    context = build_frozen_context(
        data,
        unit["config"],
        seed,
        n_jobs=args.n_jobs,
    )
    audit = endpoint_reproduction_audit(context, unit["predictions"])
    endpoint_utility = endpoint_utility_from_saved_predictions(
        unit["predictions"], context["is_classification"]
    )
    audit_row = pd.DataFrame(
        [
            {
                "target": target,
                "seed": seed,
                "requested_seed": unit["requested_seed"],
                "used_seed42_fallback": unit["used_seed42_fallback"],
                "primary_metric": context["metric"],
                **endpoint_utility,
                **audit,
            }
        ]
    )
    audit_path = unit_dir / "endpoint_reproduction_audit.csv"
    _write_csv(audit_path, audit_row)

    signature_payload = _unit_signature(unit, args)
    stage_outputs: dict[str, list[str]] = {"audit": [str(audit_path)]}

    if not args.skip_dose:
        raw_path = unit_dir / "assimilation_curve_raw.csv"
        summary_path = unit_dir / "assimilation_curve_summary.csv"
        meta_path = unit_dir / "dose.meta.json"
        signature = stable_signature(
            {**signature_payload, "stage": "dose"}
        )
        if not args.overwrite and _cache_valid(
            meta_path, signature, [raw_path, summary_path]
        ):
            print("  [cache] assimilation curve")
        else:
            print("  [fit] assimilation curve (Frozen configuration only)")
            raw, summary = compute_assimilation_curve(
                context,
                repeats=args.dose_repeats,
                fractions=parse_dose_fractions(args.dose_fractions),
                n_jobs=args.n_jobs,
            )
            _write_csv(raw_path, raw)
            _write_csv(summary_path, summary)
            _write_stage_meta(
                meta_path, signature, "assimilation_curve", [raw_path, summary_path]
            )
        stage_outputs["dose"] = [str(raw_path), str(summary_path)]

    if not args.skip_loeo:
        value_path = unit_dir / "experimental_data_value.csv"
        correlation_path = unit_dir / "data_value_correlations.csv"
        meta_path = unit_dir / "data_value.meta.json"
        signature = stable_signature(
            {
                **signature_payload,
                "stage": "data_value",
                "include_attributes": not args.skip_value_attributes,
            }
        )
        expected_files = [value_path, correlation_path]
        if not args.overwrite and _cache_valid(
            meta_path, signature, expected_files
        ):
            print("  [cache] LOEO data value")
        else:
            print("  [fit] exact LOEO marginal value (Frozen configuration only)")
            values = compute_loeo_marginal_values(
                context, n_jobs=args.n_jobs
            )
            if not args.skip_value_attributes:
                print("  [fit] novelty, label surprise, local compatibility")
                values = enrich_experimental_values(
                    values,
                    context=context,
                    data=data,
                    source_results=args.source_results,
                    n_neighbors=args.knn_neighbors,
                    n_folds=args.cv_folds,
                    n_jobs=args.n_jobs,
                )
                correlations = data_value_correlations(values)
            else:
                correlations = pd.DataFrame(
                    columns=[
                        "target",
                        "seed",
                        "predictor",
                        "outcome",
                        "spearman_rho",
                        "p_value",
                        "n_experiments",
                        "interpretation_scope",
                    ]
                )
            _write_csv(value_path, values)
            _write_csv(correlation_path, correlations)
            _write_stage_meta(
                meta_path, signature, "experimental_data_value", expected_files
            )
        stage_outputs["data_value"] = [str(value_path), str(correlation_path)]

    if not args.skip_propagation:
        samples_path = unit_dir / "propagation_samples.csv"
        regions_path = unit_dir / "propagation_regions.csv"
        meta_path = unit_dir / "propagation.meta.json"
        signature = stable_signature(
            {**signature_payload, "stage": "propagation"}
        )
        if not args.overwrite and _cache_valid(
            meta_path, signature, [samples_path, regions_path]
        ):
            print("  [cache] near/middle/far propagation")
        else:
            print("  [reuse] propagation from saved endpoint predictions")
            samples, regions = compute_propagation(
                context, unit["predictions"]
            )
            _write_csv(samples_path, samples)
            _write_csv(regions_path, regions)
            _write_stage_meta(
                meta_path,
                signature,
                "knowledge_propagation",
                [samples_path, regions_path],
            )
        stage_outputs["propagation"] = [str(samples_path), str(regions_path)]

    if not args.skip_plasticity:
        phase_path = unit_dir / "utility_plasticity.csv"
        shap_path = unit_dir / "shap_feature_values.csv"
        meta_path = unit_dir / "plasticity.meta.json"
        signature = stable_signature(
            {**signature_payload, "stage": "plasticity"}
        )
        if not args.overwrite and _cache_valid(
            meta_path, signature, [phase_path, shap_path]
        ):
            print("  [cache] utility-plasticity")
        else:
            print("  [fit] SHAP plasticity from reconstructed Frozen endpoints")
            phase, shap_features = compute_plasticity(
                context, utility=endpoint_utility["utility"]
            )
            phase = assign_update_states(
                phase,
                plasticity_threshold=args.plasticity_threshold,
                regression_tolerance=args.regression_utility_tolerance,
                classification_tolerance=args.classification_utility_tolerance,
            )
            _write_csv(phase_path, phase)
            _write_csv(shap_path, shap_features)
            _write_stage_meta(
                meta_path,
                signature,
                "utility_plasticity",
                [phase_path, shap_path],
            )
        stage_outputs["plasticity"] = [str(phase_path), str(shap_path)]

    if args.include_extra_trees:
        sensitivity_path = unit_dir / "model_family_sensitivity.csv"
        meta_path = unit_dir / "model_sensitivity.meta.json"
        signature = stable_signature(
            {**signature_payload, "stage": "model_sensitivity", "family": "ExtraTrees"}
        )
        if not args.overwrite and _cache_valid(
            meta_path, signature, [sensitivity_path]
        ):
            print("  [cache] ExtraTrees model-family sensitivity")
        else:
            print("  [fit] ExtraTrees sensitivity (fixed rule; no HPO)")
            sensitivity = compute_extra_trees_sensitivity(
                context, n_jobs=args.n_jobs
            )
            sensitivity = assign_update_states(
                sensitivity,
                plasticity_threshold=args.plasticity_threshold,
                regression_tolerance=args.regression_utility_tolerance,
                classification_tolerance=args.classification_utility_tolerance,
            )
            _write_csv(sensitivity_path, sensitivity)
            _write_stage_meta(
                meta_path,
                signature,
                "model_family_sensitivity",
                [sensitivity_path],
            )
        stage_outputs["model_sensitivity"] = [str(sensitivity_path)]

    unit_manifest = {
        "completed_utc": _utc_now(),
        "target": target,
        "requested_seed": int(unit["requested_seed"]),
        "effective_seed": seed,
        "used_seed42_fallback": bool(unit["used_seed42_fallback"]),
        "source_config": str(unit["config_path"]),
        "source_predictions": str(unit["prediction_path"]),
        "configuration_mode": "frozen_no_hpo",
        "stages": stage_outputs,
    }
    _atomic_write_json(unit_dir / "unit_manifest.json", unit_manifest)
    return unit_manifest


def _collect_csv(
    unit_manifests: list[dict[str, Any]],
    stage: str,
    file_index: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for manifest in unit_manifests:
        paths = manifest.get("stages", {}).get(stage, [])
        if len(paths) <= file_index:
            continue
        path = Path(paths[file_index])
        if path.is_file():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def consolidate_outputs(
    output_dir: Path,
    unit_manifests: list[dict[str, Any]],
    validation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, str]:
    """Build paper-facing aggregate tables from completed work-unit caches."""
    destinations: dict[str, str] = {}
    mapping = [
        ("audit", 0, "Audits", "endpoint_reproduction_audit.csv"),
        ("dose", 0, "Dose", "assimilation_curve_raw.csv"),
        ("dose", 1, "Dose", "assimilation_curve_summary.csv"),
        ("data_value", 0, "DataValue", "experimental_data_value.csv"),
        ("data_value", 1, "DataValue", "data_value_correlations.csv"),
        ("propagation", 0, "Propagation", "propagation_samples.csv"),
        ("propagation", 1, "Propagation", "propagation_regions.csv"),
        ("plasticity", 0, "Plasticity", "utility_plasticity_phase_map.csv"),
        ("plasticity", 1, "Plasticity", "shap_feature_values.csv"),
        (
            "model_sensitivity",
            0,
            "ModelSensitivity",
            "model_family_sensitivity.csv",
        ),
    ]
    for stage, file_index, subdir, name in mapping:
        frame = _collect_csv(unit_manifests, stage, file_index)
        if frame.empty and stage != "audit":
            continue
        path = output_dir / subdir / name
        _write_csv(path, frame)
        destinations[f"{subdir}/{name}"] = str(path)

    manifest = {
        "completed_utc": _utc_now(),
        "analysis": "Sparse Experimental Knowledge Assimilation",
        "source_results_mode": "read_only",
        "source_results": validation["source_results_read_only"],
        "output_dir": str(output_dir),
        "default_seed_policy": "seed_42; optional completed seeds via --seeds available",
        "configuration_mode": "Frozen configuration reuse; no HPO",
        "configuration_selection_calls": 0,
        "optuna_calls": 0,
        "dose_repeats": args.dose_repeats,
        "dose_fractions": list(parse_dose_fractions(args.dose_fractions)),
        "plasticity_threshold": args.plasticity_threshold,
        "regression_utility_tolerance": args.regression_utility_tolerance,
        "classification_utility_tolerance": args.classification_utility_tolerance,
        "include_extra_trees": args.include_extra_trees,
        "python": sys.version,
        "platform": platform.platform(),
        "unit_manifests": unit_manifests,
        "aggregate_files": destinations,
    }
    manifest_path = output_dir / "analysis_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    destinations["analysis_manifest.json"] = str(manifest_path)
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent, no-HPO knowledge-assimilation analyses using "
            "completed Frozen artifacts."
        )
    )
    parser.add_argument(
        "--source-results",
        type=Path,
        default=DEFAULT_SOURCE_RESULTS,
        help="Read-only Results directory of the running/completed original pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Isolated knowledge-assimilation output directory.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="'42' (default), comma-separated seeds, 'available', or 'all'.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS),
        help="Targets to analyze.",
    )
    parser.add_argument(
        "--dose-repeats",
        type=int,
        default=20,
        help="Random nested experiment orders for intermediate doses.",
    )
    parser.add_argument(
        "--dose-fractions",
        default=",".join(str(value) for value in DEFAULT_DOSE_FRACTIONS),
        help="Comma-separated fractions; must include 0 and 1.",
    )
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=N_CV_FOLDS)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="CPU threads per low-cost XGBoost fit.",
    )
    parser.add_argument(
        "--plasticity-threshold",
        type=float,
        default=DEFAULT_PLASTICITY_THRESHOLD,
    )
    parser.add_argument(
        "--regression-utility-tolerance",
        type=float,
        default=0.01,
        help="Near-zero band for R2 utility.",
    )
    parser.add_argument(
        "--classification-utility-tolerance",
        type=float,
        default=0.02,
        help="Near-zero band for balanced-accuracy utility.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read/validate inputs and estimate cost; fit no model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute matching stage caches (does not delete other results).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-dose", action="store_true")
    parser.add_argument("--skip-loeo", action="store_true")
    parser.add_argument("--skip-value-attributes", action="store_true")
    parser.add_argument("--skip-propagation", action="store_true")
    parser.add_argument("--skip-plasticity", action="store_true")
    parser.add_argument(
        "--include-extra-trees",
        action="store_true",
        help=(
            "Optional fixed-rule ExtraTrees utility/plasticity sensitivity; "
            "adds two non-HPO fits per target."
        ),
    )
    parser.add_argument(
        "--no-seed42-fallback",
        action="store_true",
        help="Fail instead of falling back to completed seed 42 artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dose_repeats < 1:
        parser.error("--dose-repeats must be >= 1")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1")
    if args.knn_neighbors < 1:
        parser.error("--knn-neighbors must be >= 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be >= 2")

    try:
        validation, units = validate_inputs(args)
    except Exception as exc:
        print(f"[FATAL] validation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    output_dir = Path(validation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "Audits" / "validation_report.json"
    validation_table_path = output_dir / "Audits" / "validation_report.csv"
    _atomic_write_json(validation_path, validation)
    _write_csv(validation_table_path, pd.DataFrame(validation["rows"]))

    total_estimated = sum(
        row["total_frozen_fits_estimate"] for row in validation["rows"]
    )
    print("\nValidation passed.")
    print(f"  Source (read-only): {validation['source_results_read_only']}")
    print(f"  Output:             {validation['output_dir']}")
    print(f"  Effective units:    {len(units)}")
    print(f"  Estimated fits:     {total_estimated} Frozen fits")
    print("  HPO / Optuna fits:  0")
    if args.validate_only:
        print("  Validation-only mode: no model was fitted.")
        return 0

    status_path = output_dir / "run_status.json"
    _atomic_write_json(
        status_path,
        {
            "status": "running",
            "started_utc": _utc_now(),
            "process_id": os.getpid(),
            "seeds": args.seeds,
            "targets": list(args.targets),
            "dose_repeats": args.dose_repeats,
            "hpo_or_optuna_calls": 0,
        },
    )

    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for unit in units:
        try:
            manifests.append(run_unit(unit, args, output_dir))
        except Exception as exc:
            failures.append(
                {
                    "target": unit["target"],
                    "seed": unit["effective_seed"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[ERROR] {unit['target']} seed={unit['effective_seed']}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

    destinations = consolidate_outputs(
        output_dir, manifests, validation=validation, args=args
    )
    if failures:
        _atomic_write_json(output_dir / "failed_units.json", failures)

    if not args.no_plots and manifests:
        try:
            from plot_knowledge_assimilation import build_all_figures

            build_all_figures(output_dir)
        except Exception as exc:
            failures.append(
                {
                    "stage": "plotting",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            _atomic_write_json(output_dir / "failed_units.json", failures)
            print(f"[WARN] plotting failed: {exc}", file=sys.stderr)
            traceback.print_exc()

    print("\nPost-hoc analysis finished.")
    print(f"  Aggregate files: {len(destinations)}")
    print(f"  Failed units:    {len(failures)}")
    print(f"  Output:          {output_dir}")
    _atomic_write_json(
        status_path,
        {
            "status": "failed" if failures else "complete",
            "finished_utc": _utc_now(),
            "process_id": os.getpid(),
            "seeds": args.seeds,
            "targets": list(args.targets),
            "dose_repeats": args.dose_repeats,
            "failed_units": len(failures),
            "hpo_or_optuna_calls": 0,
            "analysis_manifest": str(
                output_dir / "analysis_manifest.json"
            ),
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())