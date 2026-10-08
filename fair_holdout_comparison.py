"""
================================================================================
fair_holdout_comparison.py — Controlled data-increment ablation study
================================================================================

Refactored into two explicit protocols:

  Frozen protocol (primary analysis):
    Features, K, hyperparameters and preprocessing are all determined and
    frozen on the literature training set. The only systematic difference
    between Before/After is whether the experimental rows enter training.

  Adaptive protocol (secondary analysis):
    Before reuses the literature-only configuration selected for Frozen.
    After reselects features, K and hyperparameters on the augmented training
    set under the same search rules and budget. Each track fits preprocessing
    on its own training set; the literature holdout is unchanged.

Targets (5): LOI, UL94_Rating, THR, TSP, Flexural_Strength (pHRR removed)
================================================================================
"""

import pandas as pd
import numpy as np
import os, sys, json, joblib, warnings, traceback, chardet
BASE = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, f1_score,
)

from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# ── Import shared configuration ──
from config import (
    BASE_DIR, PROJECT_DIR, WITH_DATA_DIR, WITHOUT_DATA_DIR, OUTPUT_DIR,
    TARGETS, DISPLAY_NAMES, SEEDS, DEFAULT_SEED, TEST_SIZE,
    N_TRIALS, N_CV_FOLDS, MIN_DELTA, PATIENCE,
    HPO_SPACE, MAX_K, SCAN_BASE_PARAMS,
    N_TOP_CANDIDATE_K, CV_TOLERANCE, K_NEIGHBOR_RADIUS,
    PATIENCE_K, MIN_DELTA_K, TOP_K_CANDIDATES,
    JOINT_HPO_TRIALS, HPO_PATIENCE,
    MAX_ALTERNATING_ROUNDS, SENTINEL_TOLERANCE,
    FORCED_FEATURES,
)
from evaluation import (
    paired_bootstrap_regression,
    regression_metrics_all,
    classification_statistics,
)
from shap_metrics import (
    calculate_shap_structure_metrics,
    ranking_similarity,
    export_shap_stability,
)
from distribution import export_distribution_data


# ==========================================================================
# Data loading (consistent with the original project)
# ==========================================================================
def detect_encoding(fp):
    with open(fp, "rb") as f:
        return chardet.detect(f.read())["encoding"]

def read_csv_enc(fp):
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding=detect_encoding(fp))

def clean_numeric(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            for ch in ["?", "*", "#", " "]:
                df[c] = df[c].astype(str).str.replace(ch, "", regex=False)
            # Strip invisible Unicode whitespace (zero-width, narrow, BOM, NBSP, etc.) before numeric conversion
            df[c] = df[c].astype(str).str.replace(
                "[​         ﻿ ]",
                "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_features(data_dir):
    files = ["Curing_Des.csv", "Curing_Strategy.csv", "EP_Des.csv",
             "FR_Des.csv", "Other_Material_1_Des.csv",
             "Other_Material_2_Des.csv", "Other_Material_3_Des.csv"]
    dfs = []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            dfs.append(clean_numeric(read_csv_enc(fp)))
    if not dfs:
        raise FileNotFoundError(f"No feature files in {data_dir}")
    X = pd.concat(dfs, axis=1)

    # ── Forced features: extract formulation amount columns from Dataset_with_SMILES ──
    # Flame-retardant / curing-agent / Other-Material amounts (curing T/t/p already in Curing_Strategy.csv)
    # NOTE: CSV version is 1 row short (2355 vs 2356); the XLSX version (2356 rows, aligned with descriptor matrices) takes priority
    smiles_fp = os.path.join(data_dir, "Dataset_with_SMILES.xlsx")
    use_xlsx = os.path.exists(smiles_fp)
    if not use_xlsx:
        smiles_fp = os.path.join(data_dir, "Dataset_with_SMILES.csv")
    if os.path.exists(smiles_fp):
        if use_xlsx:
            raw = pd.read_excel(smiles_fp)
        else:
            for enc in ["gbk", "gb18030", "latin1", "cp1252", "utf-8"]:
                try:
                    raw = pd.read_csv(smiles_fp, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raw = pd.read_csv(smiles_fp, encoding=detect_encoding(smiles_fp))
        amount_cols = [c for c in FORCED_FEATURES
                       if c in raw.columns and "AdditionAmount" in c]
        if amount_cols:
            amounts = clean_numeric(raw[amount_cols])
            # ── NaN semantics for amounts ──
            # Rule: material-name column empty -> amount treated as 0 (not added);
            #       material name present but amount NaN -> flagged as incomplete (NaN kept, handled by imputer)
            material_name_map = {
                "Flame_retardant_AdditionAmount(wt%)": "Flame_retardant",
                "Curing_agent_AdditionAmount(wt%)": "Curing_agent ",
                "Other_Material_AdditionAmount_1 (wt%)": "Other_Material_1",
                "Other_Material_AdditionAmount_2 (wt%)": "Other_Material_2",
                "Other_Material_AdditionAmount_3 (wt%)": "Other_Material_3",
            }
            for col in amounts.columns:
                if col in material_name_map:
                    name_col = material_name_map[col]
                    if name_col in raw.columns:
                        name_empty = raw[name_col].isna() | (raw[name_col].astype(str).str.strip() == "")
                        amount_nan = amounts[col].isna()
                        # Empty material name + NaN amount -> fill 0 (component not added)
                        amounts.loc[name_empty & amount_nan, col] = 0.0
                        # Material name present + NaN amount -> keep NaN (incomplete-data marker)
                        n_incomplete = int((~name_empty & amount_nan).sum())
                        if n_incomplete > 0:
                            print(f"  [WARN] {col}: {n_incomplete} rows have material "
                                  f"name but NaN amount (kept as missing)")
            # Align row counts with existing X (DataFrame concat aligns by position)
            if len(amounts) == len(X):
                X = pd.concat([X, amounts], axis=1)
            else:
                print(f"  [WARN] Dataset_with_SMILES rows={len(amounts)} "
                      f"!= feature rows={len(X)}; skip amount columns")
    return X.loc[:, ~X.columns.duplicated()]

def load_targets(data_dir):
    fp = os.path.join(data_dir, "Target.csv")
    return clean_numeric(read_csv_enc(fp))

def process_ul94(y):
    """UL-94: raw value 3 -> positive class 1 (V-0), all others -> 0 (Non-V-0)."""
    y = pd.to_numeric(y, errors="coerce").fillna(-1).astype(int)
    return (y == 3).astype(int)


# ==========================================================================
# Fixed data split
# ==========================================================================
def prepare_fixed_split():
    """Load data and determine the literature/experiment split."""
    print("=" * 60)
    print("  Loading data & determining literature/experiment split")
    print("=" * 60)

    X_all = load_features(WITH_DATA_DIR)
    y_all = load_targets(WITH_DATA_DIR)
    X_wo = load_features(WITHOUT_DATA_DIR)

    N_LIT = len(X_wo)
    N_ALL = len(X_all)
    is_literature = np.arange(N_ALL) < N_LIT

    # Data identity assertion
    try:
        pd.testing.assert_frame_equal(
            X_all.iloc[:N_LIT].reset_index(drop=True),
            X_wo.reset_index(drop=True),
            check_dtype=False,
        )
        print("  [OK] Literature data identity verified")
    except AssertionError as exc:
        raise RuntimeError(
            "FATAL: Literature rows in WITH_DATA do not match WITHOUT_DATA. "
            "Check data sources."
        ) from exc

    print(f"  Literature: {N_LIT}  |  Experiment: {N_ALL - N_LIT}  |  Total: {N_ALL}")
    return X_all, y_all, N_LIT, is_literature


def get_train_test_for_target(X_all, y_all, is_literature, target_col, seed=DEFAULT_SEED):
    """Build fixed Before/After training sets + fixed test set for a target column. Seed-aware."""
    valid = y_all[target_col].notna().values
    X_v = X_all.loc[valid].copy()
    y_v = y_all.loc[valid, target_col].copy()
    source_v = is_literature[valid]
    sample_ids = X_v.index.to_numpy()

    is_cls = "UL94" in target_col
    if is_cls:
        y_v = process_ul94(y_v)

    X_lit = X_v[source_v].reset_index(drop=True)
    y_lit = y_v[source_v].reset_index(drop=True)
    id_lit = sample_ids[source_v]

    X_exp = X_v[~source_v].reset_index(drop=True)
    y_exp = y_v[~source_v].reset_index(drop=True)
    id_exp = sample_ids[~source_v]

    if len(X_lit) < 10:
        return None

    try:
        strat = y_lit if (is_cls and y_lit.nunique() >= 2) else None
        (X_train_lit, X_test, y_train_lit, y_test,
         id_train_lit, id_test) = train_test_split(
            X_lit, y_lit, id_lit,
            test_size=TEST_SIZE, random_state=seed, stratify=strat)
    except Exception:
        return None

    return {
        "target": target_col,
        "is_classification": is_cls,
        "n_lit": len(X_lit),
        "n_exp": len(X_exp),
        "n_train_before": len(X_train_lit),
        "n_train_after": len(X_train_lit) + len(X_exp),
        "n_test": len(X_test),

        "X_train_before": X_train_lit.reset_index(drop=True),
        "y_train_before": y_train_lit.reset_index(drop=True),

        "X_experiment": X_exp,
        "y_experiment": y_exp,

        "X_train_after": pd.concat([X_train_lit, X_exp], ignore_index=True),
        "y_train_after": pd.concat([y_train_lit, y_exp], ignore_index=True),

        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "test_sample_ids": np.asarray(id_test),
        "experiment_sample_ids": np.asarray(id_exp),
    }


# ==========================================================================
# Configuration selection (literature training set only; test set fully excluded)
# ==========================================================================
def build_splitter(is_cls, seed, y=None):
    """CV splitter: StratifiedKFold for classification, with dynamic fold adjustment."""
    if is_cls:
        if y is None:
            raise ValueError("y is required for classification CV")
        minority_n = int(pd.Series(y).value_counts().min())
        if minority_n < 2:
            raise ValueError("Too few minority-class samples for stratified CV")
        n_splits = min(N_CV_FOLDS, minority_n)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)


def build_fold_rankings(X, y, is_cls, seed):
    """Compute per-CV-fold feature rankings and cache preprocessed numpy arrays.
    Key: keep the full-feature preprocessed X_train_p/X_valid_p, sliced by index later."""
    splitter = build_splitter(is_cls, seed, y if is_cls else None)

    split_iter = splitter.split(X, y) if is_cls else splitter.split(X)
    rankings = []

    for train_idx, valid_idx in split_iter:
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])

        X_train_p = preprocessor.fit_transform(X_train)
        X_valid_p = preprocessor.transform(X_valid)

        ranker = (XGBClassifier(random_state=seed, eval_metric="logloss")
                  if is_cls else XGBRegressor(random_state=seed))
        ranker.fit(X_train_p, y_train)

        ranking = np.argsort(ranker.feature_importances_)[::-1]
        rankings.append({
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "ranking": ranking,
            "X_train_p": X_train_p,
            "X_valid_p": X_valid_p,
        })

    return rankings


def cv_objective_at_k(trial, X, y, k, is_cls, seed, fold_rankings,
                        forced_idx=None, nonmandatory_idx=None):
    """Given top-K non-mandatory features (+mandatory), run HPO within CV folds and return the mean CV score.
    K = number of extra non-mandatory features selected."""
    import optuna

    param = {
        "n_estimators": trial.suggest_int("n_estimators", *HPO_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *HPO_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *HPO_SPACE["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *HPO_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *HPO_SPACE["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *HPO_SPACE["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *HPO_SPACE["reg_lambda"]),
        "random_state": seed,
    }

    scores = []
    for fr in fold_rankings:
        # Slice pre-saved numpy arrays by index to avoid sklearn feature-name mismatches
        # K = number of non-mandatory features: take top-K non-mandatory from fold ranking, then merge mandatory
        if nonmandatory_idx is not None:
            nonmandatory_set = set(nonmandatory_idx)
            top_nonmandatory = [i for i in fr["ranking"] if i in nonmandatory_set][:k]
        else:
            top_nonmandatory = list(fr["ranking"][:k])
        top_idx = np.asarray(top_nonmandatory)
        if forced_idx is not None and len(forced_idx) > 0:
            top_idx = np.unique(np.concatenate([top_idx, np.asarray(forced_idx)]))

        X_train_k_p = fr["X_train_p"][:, top_idx]
        X_valid_k_p = fr["X_valid_p"][:, top_idx]

        model = (XGBClassifier(**param, eval_metric="logloss")
                 if is_cls else XGBRegressor(**param))
        model.fit(X_train_k_p, y.iloc[fr["train_idx"]])

        preds = model.predict(X_valid_k_p)
        if is_cls:
            scores.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
        else:
            scores.append(r2_score(y.iloc[fr["valid_idx"]], preds))

    return np.mean(scores)


def scan_cv_score_at_k(X, y, k, is_cls, seed, fold_rankings,
                         forced_idx=None, nonmandatory_idx=None):
    """Fast evaluation for the K-scan stage: default XGBoost params, no Optuna.
    K = number of extra non-mandatory features selected."""
    scores = []
    for fr in fold_rankings:
        if nonmandatory_idx is not None:
            nonmandatory_set = set(nonmandatory_idx)
            top_nonmandatory = [i for i in fr["ranking"] if i in nonmandatory_set][:k]
        else:
            top_nonmandatory = list(fr["ranking"][:k])
        top_idx = np.asarray(top_nonmandatory)
        if forced_idx is not None and len(forced_idx) > 0:
            top_idx = np.unique(np.concatenate([top_idx, np.asarray(forced_idx)]))

        X_train_k_p = fr["X_train_p"][:, top_idx]
        X_valid_k_p = fr["X_valid_p"][:, top_idx]

        model = (XGBClassifier(random_state=seed, eval_metric="logloss")
                 if is_cls else XGBRegressor(random_state=seed))
        model.fit(X_train_k_p, y.iloc[fr["train_idx"]])

        preds = model.predict(X_valid_k_p)
        if is_cls:
            scores.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
        else:
            scores.append(r2_score(y.iloc[fr["valid_idx"]], preds))

    return np.mean(scores)


def fit_final_ranking(X, y, is_cls, seed):
    """Fit the final feature ranking on the full data (no CV)."""
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    X_p = preprocessor.fit_transform(X)
    ranker = (XGBClassifier(random_state=seed, eval_metric="logloss")
              if is_cls else XGBRegressor(random_state=seed))
    ranker.fit(X_p, y)

    ranking = np.argsort(ranker.feature_importances_)[::-1]
    feature_names = list(X.columns)
    return [feature_names[i] for i in ranking]


def _scan_full_k(X, y, is_cls, seed, fold_rankings,
                 nonmandatory_idx, mandatory_idx, max_k=None, params=None,
                 nonmandatory_rankings=None, patience=None, min_delta=None):
    """Incremental K scan with fixed base hyperparameters and 5-fold CV.
    patience: stop after N consecutive K without improvement (above min_delta); no HPO.
    When patience is None, PATIENCE_K is used; when min_delta is None,
    MIN_DELTA_K is used. The scan can stop before reaching max_k.
    Returns (scores dict, fold_scores dict)."""
    if max_k is None:
        max_k = MAX_K
    if params is None:
        params = SCAN_BASE_PARAMS
    if patience is None:
        patience = PATIENCE_K
    if min_delta is None:
        min_delta = MIN_DELTA_K
    # Precompute per-fold non-mandatory ranking indices (avoid rebuilding sets/arrays per K)
    if nonmandatory_rankings is None:
        nonmandatory_set = set(nonmandatory_idx)
        nonmandatory_rankings = [
            np.asarray([i for i in fr["ranking"] if i in nonmandatory_set])
            for fr in fold_rankings
        ]
    mandatory_arr = np.asarray(mandatory_idx, dtype=int)
    scores = {}
    fold_scores = {}
    best_score = -np.inf
    no_improve = 0
    for k in range(1, max_k + 1):
        per_fold = []
        for fi, fr in enumerate(fold_rankings):
            top_nonmandatory = nonmandatory_rankings[fi][:k]
            top_idx = top_nonmandatory
            if mandatory_arr.size > 0:
                top_idx = np.unique(np.concatenate([top_idx, mandatory_arr]))
            X_tr = fr["X_train_p"][:, top_idx]
            X_va = fr["X_valid_p"][:, top_idx]
            model = (XGBClassifier(**params, eval_metric="logloss")
                     if is_cls else XGBRegressor(**params))
            model.fit(X_tr, y.iloc[fr["train_idx"]])
            preds = model.predict(X_va)
            if is_cls:
                per_fold.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
            else:
                per_fold.append(r2_score(y.iloc[fr["valid_idx"]], preds))
        scores[k] = float(np.mean(per_fold))
        fold_scores[k] = per_fold
        if k % 50 == 0:
            print(f"      [scan] K={k}, CV={scores[k]:.4f}", flush=True)

        # patience early stop: update best only when enabled and genuinely improved
        if patience is not None:
            if scores[k] > best_score + min_delta:
                best_score = scores[k]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"      [scan] early stop at K={k} (no improvement for "
                      f"{patience} K steps), best CV={best_score:.4f}", flush=True)
                break
    print(f"      [scan] done: scanned K=1..{max(scores)}", flush=True)
    return scores, fold_scores


def _build_candidate_k(scores, fold_scores):
    """Legacy helper; not called by the formal single-K workflow.
    Candidate K set: only fully duplicate K removed; adjacent K not merged.
    Keeps: 1. top-N_TOP_CANDIDATE_K by CV; 2. all K with CV_max-CV_k<=CV_TOLERANCE;
    3. +/-K_NEIGHBOR_RADIUS integer points around the above candidates."""
    cv_max = max(scores.values())
    cand = set()
    top_k = sorted(scores, key=lambda k: scores[k], reverse=True)[:N_TOP_CANDIDATE_K]
    cand.update(top_k)
    cand.update(k for k, s in scores.items() if cv_max - s <= CV_TOLERANCE)
    for k in list(cand):
        cand.update(k2 for k2 in range(max(1, k - K_NEIGHBOR_RADIUS), k + K_NEIGHBOR_RADIUS + 1)
                    if k2 in scores)
    return sorted(cand)


def _joint_hpo(X, y, is_cls, seed, fold_rankings, candidate_k,
               nonmandatory_idx, mandatory_idx, n_trials=JOINT_HPO_TRIALS,
               nonmandatory_rankings=None):
    """Legacy joint K/HPO helper; not called by the formal single-K workflow."""
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if nonmandatory_rankings is None:
        nonmandatory_set = set(nonmandatory_idx)
        nonmandatory_rankings = [
            np.asarray([i for i in fr["ranking"] if i in nonmandatory_set])
            for fr in fold_rankings
        ]
    mandatory_arr = np.asarray(mandatory_idx, dtype=int)

    def objective(trial):
        k = trial.suggest_categorical("k_nonmandatory", candidate_k)
        param = {
            "n_estimators": trial.suggest_int("n_estimators", *HPO_SPACE["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *HPO_SPACE["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *HPO_SPACE["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *HPO_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *HPO_SPACE["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *HPO_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *HPO_SPACE["reg_lambda"]),
            "random_state": seed,
        }
        scores = []
        for fi, fr in enumerate(fold_rankings):
            top_idx = nonmandatory_rankings[fi][:k]
            if mandatory_arr.size > 0:
                top_idx = np.unique(np.concatenate([top_idx, mandatory_arr]))
            X_tr = fr["X_train_p"][:, top_idx]
            X_va = fr["X_valid_p"][:, top_idx]
            model = (XGBClassifier(**param, eval_metric="logloss")
                     if is_cls else XGBRegressor(**param))
            model.fit(X_tr, y.iloc[fr["train_idx"]])
            preds = model.predict(X_va)
            if is_cls:
                scores.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
            else:
                scores.append(r2_score(y.iloc[fr["valid_idx"]], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_early_stop])
    if study.trials:
        print(f"      [hpo-earlystop] ran {len(study.trials)} trials (patience={HPO_PATIENCE})")
    return study


def _check_upper_bound(X, y, is_cls, seed, fold_rankings, nonmandatory_idx,
                       mandatory_idx, params):
    """Legacy high-K check; not called by the formal single-K workflow.
    Evaluates K=MAX_K+500, MAX_K+1000 and all non-mandatory features.
    """
    n_features = len(nonmandatory_idx)
    checks = sorted({min(MAX_K + 500, n_features), min(MAX_K + 1000, n_features), n_features})
    nonmandatory_set = set(nonmandatory_idx)
    nonmandatory_rankings = [
        np.asarray([i for i in fr["ranking"] if i in nonmandatory_set])
        for fr in fold_rankings
    ]
    mandatory_arr = np.asarray(mandatory_idx, dtype=int)
    results = {}
    for k in checks:
        scores = []
        for fi, fr in enumerate(fold_rankings):
            top_idx = nonmandatory_rankings[fi][:k]
            if mandatory_arr.size > 0:
                top_idx = np.unique(np.concatenate([top_idx, mandatory_arr]))
            X_tr = fr["X_train_p"][:, top_idx]
            X_va = fr["X_valid_p"][:, top_idx]
            model = (XGBClassifier(**params, eval_metric="logloss")
                     if is_cls else XGBRegressor(**params))
            model.fit(X_tr, y.iloc[fr["train_idx"]])
            preds = model.predict(X_va)
            if is_cls:
                scores.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
            else:
                scores.append(r2_score(y.iloc[fr["valid_idx"]], preds))
        results[k] = float(np.mean(scores))
        print(f"      [bound] K={k} (n_nonmandatory={n_features}), CV={results[k]:.4f}")
    return results


def select_configuration(X, y, is_cls, seed):
    """Select the configuration using the formal single-K workflow.

    1. Scan integer K values from 1 to MAX_K with SCAN_BASE_PARAMS and
       internal N_CV_FOLDS-fold CV. Stop after PATIENCE_K consecutive
       steps without a score exceeding the tracked best by MIN_DELTA_K.
    2. Select the single K with the highest mean score among scanned K.
    3. Optimise XGBoost hyperparameters only at that K using Optuna,
       with at most JOINT_HPO_TRIALS trials and HPO_PATIENCE early stopping.
    4. Recompute the feature ranking on the full supplied training set.
    5. Retain present mandatory features and the top K non-mandatory features.

    K counts non-mandatory retained features. No multiple-candidate K HPO,
    joint K/hyperparameter search, alternating search or high-K sentinel
    check is executed. The holdout is not used for configuration selection.
    """
    feat_names = list(X.columns)
    mandatory_idx = [i for i, c in enumerate(feat_names) if c in FORCED_FEATURES]
    nonmandatory_idx = [i for i in range(len(feat_names)) if i not in set(mandatory_idx)]

    fold_rankings = build_fold_rankings(X, y, is_cls, seed)

    _nm_set = set(nonmandatory_idx)
    nonmandatory_rankings = [
        np.asarray([i for i in fr["ranking"] if i in _nm_set])
        for fr in fold_rankings
    ]

    # ── 1. Incremental K scan (patience early stop, no HPO) ──
    print(f"    [scan] K incremental scan (fixed base params, {N_CV_FOLDS}-fold CV, "
          f"patience={PATIENCE_K})...")
    scan_scores, fold_scores = _scan_full_k(
        X, y, is_cls, seed, fold_rankings, nonmandatory_idx, mandatory_idx,
        nonmandatory_rankings=nonmandatory_rankings)
    print(f"    [scan] scanned {len(scan_scores)} K values (1..{max(scan_scores)})")

    # ── 2. Take only the single highest-scan-CV K ──
    final_k = max(scan_scores, key=lambda k: scan_scores[k])
    scan_best_cv = scan_scores[final_k]
    print(f"    [best-k-scan] K={final_k}, scan CV={scan_best_cv:.4f}")

    # ── 3. Run HPO once on that K ──
    study = _hpo_for_fixed_k(X, y, is_cls, seed, fold_rankings, final_k,
                             nonmandatory_idx, mandatory_idx,
                             nonmandatory_rankings=nonmandatory_rankings)
    best_params = dict(study.best_params)
    best_params["random_state"] = seed
    final_cv = float(study.best_value)
    print(f"      [hpo K={final_k}] CV={final_cv:.4f}, "
          f"n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"lr={best_params['learning_rate']:.4f}")
    print(f"    [best] K={final_k}, HPO_CV={final_cv:.4f}")

    # The formal single-K workflow does not execute a high-K sentinel check.

    # Final features: mandatory (active) union Top-K non-mandatory
    forced_present = [f for f in FORCED_FEATURES if f in feat_names]
    n_mandatory_active = len(forced_present)
    final_ranking = fit_final_ranking(X, y, is_cls, seed)
    top_nonmandatory = [f for f in final_ranking if f not in set(forced_present)][:final_k]
    best_features = list(dict.fromkeys(forced_present + top_nonmandatory))

    print(f"    [select] K_selected_nonmandatory={final_k}, n_mandatory_active={n_mandatory_active}, "
          f"n_features_final={len(best_features)}, CV={final_cv:.4f}")

    traj_scan = pd.DataFrame({
        "k": list(scan_scores.keys()),
        "k_selected_nonmandatory": list(scan_scores.keys()),
        "cv_score": list(scan_scores.values()),
        "phase": "scan",
    })
    trajectory = traj_scan.copy()

    return {
        "k": final_k,
        "features": best_features,
        "params": best_params,
        "cv_score": final_cv,
        "trajectory": trajectory,
        "k_selected_nonmandatory": final_k,
        "n_mandatory_configured": len(FORCED_FEATURES),
        "n_mandatory_active": n_mandatory_active,
        "n_features_final": len(best_features),
        "top_k": final_k,
        "converged": True,
    }


def _hpo_for_fixed_k(X, y, is_cls, seed, fold_rankings, k,
                     nonmandatory_idx, mandatory_idx,
                     nonmandatory_rankings=None, n_trials=JOINT_HPO_TRIALS):
    """Optimise XGBoost hyperparameters for a fixed K (TPE, Optuna). Returns the study."""
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if nonmandatory_rankings is None:
        _nm_set = set(nonmandatory_idx)
        nonmandatory_rankings = [
            np.asarray([i for i in fr["ranking"] if i in _nm_set])
            for fr in fold_rankings
        ]
    mandatory_arr = np.asarray(mandatory_idx, dtype=int)

    _hpo_state = {"last_best": None, "no_improve": 0}

    def _early_stop(study, trial):
        """Stop after HPO_PATIENCE consecutive new trials fail to refresh best_value."""
        v = trial.value
        if v is None:
            return
        lb = _hpo_state["last_best"]
        if lb is None or v > lb:
            _hpo_state["last_best"] = v
            _hpo_state["no_improve"] = 0
        else:
            _hpo_state["no_improve"] += 1
            if _hpo_state["no_improve"] >= HPO_PATIENCE:
                study.stop()

    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", *HPO_SPACE["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *HPO_SPACE["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *HPO_SPACE["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *HPO_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *HPO_SPACE["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *HPO_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *HPO_SPACE["reg_lambda"]),
            "random_state": seed,
        }
        scores = []
        for fi, fr in enumerate(fold_rankings):
            top_idx = nonmandatory_rankings[fi][:k]
            if mandatory_arr.size > 0:
                top_idx = np.unique(np.concatenate([top_idx, mandatory_arr]))
            X_tr = fr["X_train_p"][:, top_idx]
            X_va = fr["X_valid_p"][:, top_idx]
            model = (XGBClassifier(**param, eval_metric="logloss")
                     if is_cls else XGBRegressor(**param))
            model.fit(X_tr, y.iloc[fr["train_idx"]])
            preds = model.predict(X_va)
            if is_cls:
                scores.append(balanced_accuracy_score(y.iloc[fr["valid_idx"]], preds))
            else:
                scores.append(r2_score(y.iloc[fr["valid_idx"]], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_early_stop])
    if study.trials:
        print(f"      [hpo-earlystop] ran {len(study.trials)} trials (patience={HPO_PATIENCE})")
    return study


# ==========================================================================
def make_model(params, is_cls, seed):
    """Create an XGBoost model from the given parameters."""
    base = {
        "n_estimators": params.get("n_estimators", 100),
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "reg_alpha": params.get("reg_alpha", 0),
        "reg_lambda": params.get("reg_lambda", 1),
        "random_state": seed,
    }
    if is_cls:
        return XGBClassifier(**base, eval_metric="logloss")
    return XGBRegressor(**base)


def _configuration_payload(config):
    """JSON-safe configuration state required to resume without HPO."""
    payload = {
        "k": int(config["k"]),
        "features": list(config["features"]),
        "params": dict(config["params"]),
        "cv_score": float(config["cv_score"]),
    }
    for key in [
        "k_selected_nonmandatory", "n_mandatory_configured",
        "n_mandatory_active", "n_features_final", "candidate_k",
        "boundary_cv", "converged",
    ]:
        if key in config:
            payload[key] = config[key]
    return payload


def run_frozen_comparison(data, config, seed):
    """Frozen protocol: shared features/K/hyperparameters/preprocessor.
    The preprocessor is fitted only on the literature training set."""
    features = config["features"]
    params = config["params"]
    is_cls = data["is_classification"]

    # Shared preprocessor (fitted only on Before)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    X_before = preprocessor.fit_transform(data["X_train_before"][features])
    X_after = preprocessor.transform(data["X_train_after"][features])
    X_test = preprocessor.transform(data["X_test"][features])

    model_before = make_model(params, is_cls, seed)
    model_after = make_model(params, is_cls, seed)

    model_before.fit(X_before, data["y_train_before"])
    model_after.fit(X_after, data["y_train_after"])

    pred_before = model_before.predict(X_test)
    pred_after = model_after.predict(X_test)

    # ── Save ──
    frozen_dir = os.path.join(OUTPUT_DIR, "Frozen")
    target = data["target"]

    # Predictions
    pred_df = pd.DataFrame({
        "sample_id": data["test_sample_ids"],
        "target": target,
        "seed": seed,
        "protocol": "frozen",
        "y_true": data["y_test"].to_numpy(),
        "y_pred_before": pred_before,
        "y_pred_after": pred_after,
    })
    if not is_cls:
        pred_df["error_before"] = pred_df["y_true"] - pred_df["y_pred_before"]
        pred_df["error_after"] = pred_df["y_true"] - pred_df["y_pred_after"]
        pred_df["abs_error_before"] = pred_df["error_before"].abs()
        pred_df["abs_error_after"] = pred_df["error_after"].abs()

    pred_df.to_csv(
        os.path.join(frozen_dir, f"predictions_{target}_seed_{seed}.csv"),
        index=False, encoding="utf-8-sig")

    # Config + trajectory
    with open(os.path.join(frozen_dir, f"config_{target}_seed_{seed}.json"), "w") as f:
        json.dump(_configuration_payload(config), f,
                  ensure_ascii=False, indent=2)
    config["trajectory"].to_csv(
        os.path.join(frozen_dir, f"trajectory_{target}_seed_{seed}.csv"),
        index=False, encoding="utf-8-sig")

    # Metrics
    if is_cls:
        metrics = classification_statistics(
            data["y_test"].to_numpy(), pred_before, pred_after)
    else:
        metrics = regression_metrics_all(
            data["y_test"].to_numpy(), pred_before, pred_after, seed=seed)

    metrics["target"] = target
    metrics["seed"] = seed
    metrics["protocol"] = "frozen"
    metrics["n_train_before"] = data["n_train_before"]
    metrics["n_train_after"] = data["n_train_after"]
    metrics["n_test"] = data["n_test"]
    metrics["k"] = config["k"]

    # Unified primary-metric interface
    if is_cls:
        metrics["primary_metric"] = "balanced_accuracy"
        metrics["before_primary"] = metrics["before_balanced_accuracy"]
        metrics["after_primary"] = metrics["after_balanced_accuracy"]
    else:
        metrics["primary_metric"] = "r2"
        metrics["before_primary"] = metrics["before_r2"]
        metrics["after_primary"] = metrics["after_r2"]
    metrics["improvement_delta"] = metrics["after_primary"] - metrics["before_primary"]

    # ── Frozen SHAP metrics ──
    try:
        import shap
        explainer_b = shap.TreeExplainer(model_before)
        shap_b_vals = explainer_b(X_test).values
        if is_cls and isinstance(shap_b_vals, list):
            shap_b_vals = shap_b_vals[1]

        explainer_a = shap.TreeExplainer(model_after)
        shap_a_vals = explainer_a(X_test).values
        if is_cls and isinstance(shap_a_vals, list):
            shap_a_vals = shap_a_vals[1]

        sm_b = calculate_shap_structure_metrics(shap_b_vals, features)
        sm_a = calculate_shap_structure_metrics(shap_a_vals, features)

        # Ranking similarity
        rs = ranking_similarity(sm_b["normalized_shap"], sm_a["normalized_shap"], features)

        metrics["shap_before"] = sm_b
        metrics["shap_after"] = sm_a
        metrics["shap_similarity"] = rs
        # Structure metrics (one row each for Before + After)
        metrics["shap_entries"] = [
            {"target": target, "seed": seed, "protocol": "frozen",
             "track": "before", "metrics": sm_b},
            {"target": target, "seed": seed, "protocol": "frozen",
             "track": "after", "metrics": sm_a},
        ]
        # Ranking similarity stored separately
        metrics["shap_similarity_entry"] = {
            "target": target, "seed": seed, "protocol": "frozen",
            "top10_jaccard": rs["top10_jaccard"],
            "spearman_rho": rs["spearman_rho"],
            "spearman_p": rs["spearman_p"],
        }
    except Exception as exc:
        print(f"  [WARN] SHAP failed for {target} seed={seed}: {exc}")
        traceback.print_exc()

    return {
        "metrics": metrics,
        "predictions": [pred_df],
    }


# ==========================================================================
# Adaptive protocol (secondary analysis)
# ==========================================================================
def fit_and_evaluate_adaptive_track(X_train, y_train, X_test, y_test,
                                     config, track, target, seed):
    """Train and evaluate under the given configuration."""
    features = config["features"]
    params = config["params"]
    is_cls = "UL94" in target

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    X_tr = preprocessor.fit_transform(X_train[features])
    X_te = preprocessor.transform(X_test[features])

    model = make_model(params, is_cls, seed)
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    if is_cls:
        m = {
            "track": track,
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "mcc": float(matthews_corrcoef(y_test, preds)),
            "v0_recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
            "non_v0_recall": float(recall_score(y_test, preds, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_test, preds, average="binary", zero_division=0)),
        }
    else:
        m = {
            "track": track,
            "r2": float(r2_score(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
        }

    m["k"] = config["k"]
    m["cv_score"] = config["cv_score"]
    return m


def run_adaptive_comparison(data, seed, config_before=None, config_after=None):
    """Evaluate paired Adaptive tracks with training-specific preprocessing.

    The formal main workflow supplies the Frozen literature-only configuration
    as config_before and reselects config_after on augmented training data.
    A supplied configuration is reused; an omitted one is selected on that
    track's training data. Saved configurations may be supplied when resuming.
    """
    target = data["target"]
    is_cls = data["is_classification"]

    if config_before is None:
        config_before = select_configuration(
            X=data["X_train_before"], y=data["y_train_before"],
            is_cls=is_cls, seed=seed)

    if config_after is None:
        config_after = select_configuration(
            X=data["X_train_after"], y=data["y_train_after"],
            is_cls=is_cls, seed=seed)

    before_row = fit_and_evaluate_adaptive_track(
        X_train=data["X_train_before"], y_train=data["y_train_before"],
        X_test=data["X_test"], y_test=data["y_test"],
        config=config_before, track="Before", target=target, seed=seed)

    after_row = fit_and_evaluate_adaptive_track(
        X_train=data["X_train_after"], y_train=data["y_train_after"],
        X_test=data["X_test"], y_test=data["y_test"],
        config=config_after, track="After", target=target, seed=seed)

    # Save features and trajectory
    adaptive_dir = os.path.join(OUTPUT_DIR, "Adaptive")
    for track, cfg in [("Before", config_before), ("After", config_after)]:
        prefix = f"{track.lower()}_{target}_seed_{seed}"
        with open(os.path.join(adaptive_dir, f"{prefix}_features.json"), "w") as f:
            json.dump(cfg["features"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(adaptive_dir, f"{prefix}_config.json"), "w") as f:
            json.dump(_configuration_payload(cfg), f,
                      ensure_ascii=False, indent=2)
        cfg["trajectory"].to_csv(
            os.path.join(adaptive_dir, f"trajectory_{prefix}.csv"),
            index=False, encoding="utf-8-sig")

    # Paired results
    row = {
        "target": target, "seed": seed, "protocol": "adaptive",
        "before_k": config_before["k"], "after_k": config_after["k"],
        "before_cv_score": config_before["cv_score"],
        "after_cv_score": config_after["cv_score"],
    }

    if is_cls:
        row.update({
            "before_primary": before_row["balanced_accuracy"],
            "after_primary": after_row["balanced_accuracy"],
            "improvement_delta": after_row["balanced_accuracy"] - before_row["balanced_accuracy"],
            "primary_metric": "balanced_accuracy",
            "before_accuracy": before_row["accuracy"],
            "after_accuracy": after_row["accuracy"],
            "before_mcc": before_row["mcc"], "after_mcc": after_row["mcc"],
        })
    else:
        row.update({
            "before_primary": before_row["r2"],
            "after_primary": after_row["r2"],
            "improvement_delta": after_row["r2"] - before_row["r2"],
            "primary_metric": "r2",
            "before_rmse": before_row["rmse"], "after_rmse": after_row["rmse"],
        })

    return row


# ==========================================================================
# Sample sizes
# ==========================================================================
def build_sample_size_row(data, seed):
    return {
        "target": data["target"],
        "seed": seed,
        "n_literature_total": data["n_lit"],
        "n_literature_train": data["n_train_before"],
        "n_experiment": data["n_exp"],
        "n_after_train": data["n_train_after"],
        "n_test": data["n_test"],
        "augmentation_ratio": (
            data["n_exp"] / data["n_train_before"]
            if data["n_train_before"] > 0 else np.nan),
        "augmentation_percent": (
            100.0 * data["n_exp"] / data["n_train_before"]
            if data["n_train_before"] > 0 else np.nan),
    }


# ==========================================================================
# Per-unit checkpoints and interruption-safe resume
# ==========================================================================
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "Checkpoints")
PIPELINE_STATUS_PATH = os.path.join(OUTPUT_DIR, "pipeline_status.json")
CHECKPOINT_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(temp, path)


def _checkpoint_path(target, seed):
    return os.path.join(CHECKPOINT_DIR, f"unit_{target}_seed_{seed}.joblib")


def _atomic_checkpoint_dump(target, seed, payload):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    destination = _checkpoint_path(target, seed)
    temp = destination + ".tmp"
    joblib.dump(payload, temp)
    os.replace(temp, destination)


def _load_checkpoint(target, seed):
    path = _checkpoint_path(target, seed)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    try:
        payload = joblib.load(path)
        if (payload.get("version") != CHECKPOINT_VERSION or
                payload.get("target") != target or
                int(payload.get("seed")) != int(seed)):
            raise ValueError("checkpoint identity/version mismatch")
        return payload
    except Exception as exc:
        print(f"  [WARN] Ignore unreadable checkpoint {path}: {exc}")
        return None


def _write_pipeline_status(status, current=None, error=None):
    completed_frozen = 0
    completed_adaptive = 0
    if os.path.isdir(CHECKPOINT_DIR):
        for name in os.listdir(CHECKPOINT_DIR):
            if not name.endswith(".joblib"):
                continue
            try:
                payload = joblib.load(os.path.join(CHECKPOINT_DIR, name))
                completed_frozen += int("frozen_metrics" in payload)
                completed_adaptive += int("adaptive_metrics" in payload)
            except Exception:
                continue
    _atomic_json(PIPELINE_STATUS_PATH, {
        "status": status,
        "updated_at": _utc_now(),
        "current": current,
        "completed_frozen_units": completed_frozen,
        "completed_adaptive_units": completed_adaptive,
        "expected_units": len(SEEDS) * len(TARGETS),
        "error": error,
    })


def _load_frozen_config_from_artifacts(target, seed):
    frozen_dir = os.path.join(OUTPUT_DIR, "Frozen")
    config_path = os.path.join(frozen_dir, f"config_{target}_seed_{seed}.json")
    prediction_path = os.path.join(frozen_dir, f"predictions_{target}_seed_{seed}.csv")
    trajectory_path = os.path.join(frozen_dir, f"trajectory_{target}_seed_{seed}.csv")
    required = [config_path, prediction_path, trajectory_path]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in required):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for key in ["k", "features", "params", "cv_score"]:
            if key not in config:
                raise KeyError(key)
        config["trajectory"] = pd.read_csv(trajectory_path)
        return config
    except Exception as exc:
        print(f"  [WARN] Frozen artifacts cannot be resumed for {target} seed={seed}: {exc}")
        return None


def _load_adaptive_configs_from_artifacts(target, seed):
    adaptive_dir = os.path.join(OUTPUT_DIR, "Adaptive")
    configs = []
    for track in ["before", "after"]:
        prefix = f"{track}_{target}_seed_{seed}"
        config_path = os.path.join(adaptive_dir, f"{prefix}_config.json")
        trajectory_path = os.path.join(adaptive_dir, f"trajectory_{prefix}.csv")
        if not (os.path.isfile(config_path) and os.path.getsize(config_path) > 0 and
                os.path.isfile(trajectory_path) and os.path.getsize(trajectory_path) > 0):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for key in ["k", "features", "params", "cv_score"]:
                if key not in config:
                    raise KeyError(key)
            config["trajectory"] = pd.read_csv(trajectory_path)
            configs.append(config)
        except Exception as exc:
            print(f"  [WARN] Adaptive artifacts cannot be resumed for {target} seed={seed}: {exc}")
            return None
    return tuple(configs)


def _run_or_resume_unit(data, seed):
    """Return complete unit results, skipping every safely checkpointed stage."""
    target = data["target"]
    payload = _load_checkpoint(target, seed) or {
        "version": CHECKPOINT_VERSION,
        "target": target,
        "seed": int(seed),
        "sample_size_row": build_sample_size_row(data, seed),
    }

    frozen_config = payload.get("frozen_config")
    frozen_metrics = payload.get("frozen_metrics")
    if frozen_config is not None and frozen_metrics is not None:
        print(f"  [RESUME] {target} seed={seed}: Frozen checkpoint found; skip selection/refit")
    else:
        frozen_config = _load_frozen_config_from_artifacts(target, seed)
        if frozen_config is not None:
            print(f"  [RESUME] {target} seed={seed}: recover Frozen config; skip HPO, refit endpoint only")
        else:
            frozen_config = select_configuration(
                X=data["X_train_before"], y=data["y_train_before"],
                is_cls=data["is_classification"], seed=seed)
        frozen_result = run_frozen_comparison(data, frozen_config, seed)
        frozen_metrics = frozen_result["metrics"]
        payload["frozen_config"] = frozen_config
        payload["frozen_metrics"] = frozen_metrics
        payload["frozen_completed_at"] = _utc_now()
        _atomic_checkpoint_dump(target, seed, payload)
        _write_pipeline_status("running", current={
            "seed": int(seed), "target": target, "stage": "frozen_complete",
        })

    adaptive_metrics = payload.get("adaptive_metrics")
    if adaptive_metrics is not None:
        print(f"  [RESUME] {target} seed={seed}: Adaptive checkpoint found; skip selection/refit")
    else:
        saved_adaptive = _load_adaptive_configs_from_artifacts(target, seed)
        if saved_adaptive is not None:
            print(f"  [RESUME] {target} seed={seed}: recover Adaptive configs; skip HPO, refit endpoints only")
            config_before, config_after = saved_adaptive
            adaptive_metrics = run_adaptive_comparison(
                data, seed, config_before=config_before, config_after=config_after)
        else:
            adaptive_metrics = run_adaptive_comparison(
                data, seed, config_before=frozen_config)
        payload["adaptive_metrics"] = adaptive_metrics
        payload["adaptive_completed_at"] = _utc_now()
        _atomic_checkpoint_dump(target, seed, payload)
        _write_pipeline_status("running", current={
            "seed": int(seed), "target": target, "stage": "adaptive_complete",
        })

    return payload["sample_size_row"], frozen_metrics, adaptive_metrics


# ==========================================================================
# Main entry
# ==========================================================================
def main():
    global OUTPUT_DIR, CHECKPOINT_DIR, PIPELINE_STATUS_PATH
    import argparse
    parser = argparse.ArgumentParser(description="Frozen/Adaptive comparison")
    parser.add_argument("--tag", default="", help="output tag for shard CSVs")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default: config.SEEDS)")
    parser.add_argument("--targets", default=None, help="comma-separated targets (default: config.TARGETS)")
    parser.add_argument("--outdir-suffix", default="", help="append suffix to output dir, e.g. '42' -> Results_42")
    args = parser.parse_args()
    # Redirect every derived state path before the first status/checkpoint write.
    # This keeps auxiliary shards from modifying the formal Results status file.
    if args.outdir_suffix:
        OUTPUT_DIR = OUTPUT_DIR.rstrip("/\\") + "_" + args.outdir_suffix
        CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "Checkpoints")
        PIPELINE_STATUS_PATH = os.path.join(OUTPUT_DIR, "pipeline_status.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for sub in ["SampleSizes", "Frozen", "Adaptive", "Robustness",
                    "SHAP_Stability", "Distribution", "Logs", "Checkpoints"]:
            os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
        print(f"  [outdir] redirected to {OUTPUT_DIR}")
    # 1. Load & fixed split
    X_all, y_all, N_LIT, is_literature = prepare_fixed_split()
    _write_pipeline_status("running", current={"stage": "initializing"})

    sample_size_rows = []
    frozen_rows = []
    adaptive_rows = []
    shap_export_entries = []
    shap_similarity_entries = []

    # 2. Multi-seed loop
    seeds = SEEDS if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    targets = TARGETS if args.targets is None else [t.strip() for t in args.targets.split(",")]
    tag = args.tag
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")

        for target in targets:
            data = get_train_test_for_target(
                X_all, y_all, is_literature, target, seed=seed)
            if data is None:
                print(f"  [WARN] {target}: insufficient samples, skipping")
                continue

            print(f"\n  --- {target} ---")

            _write_pipeline_status("running", current={
                "seed": int(seed), "target": target,
                "stage": "starting_or_resuming",
            })
            try:
                sample_row, frozen_metrics, adaptive_metrics = _run_or_resume_unit(
                    data, seed)
            except Exception as exc:
                _write_pipeline_status("failed", current={
                    "seed": int(seed), "target": target, "stage": "failed",
                }, error=repr(exc))
                raise

            sample_size_rows.append(sample_row)
            frozen_rows.append(frozen_metrics)
            adaptive_rows.append(adaptive_metrics)

            if "shap_entries" in frozen_metrics:
                shap_export_entries.extend(frozen_metrics["shap_entries"])
            if "shap_similarity_entry" in frozen_metrics:
                shap_similarity_entries.append(
                    frozen_metrics["shap_similarity_entry"])

    # 3. Save
    print(f"\n{'='*60}")
    print("  Saving results")
    print(f"{'='*60}")

    pd.DataFrame(sample_size_rows).to_csv(
        os.path.join(OUTPUT_DIR, "SampleSizes", ("target_sample_sizes.csv" if not tag else f"target_sample_sizes_{tag}.csv")),
        index=False, encoding="utf-8-sig")

    frozen_out = pd.DataFrame(frozen_rows)
    frozen_out.drop(columns=["shap_before", "shap_after", "shap_similarity",
        "shap_entries", "shap_similarity_entry"], errors="ignore").to_csv(
        os.path.join(OUTPUT_DIR, "Frozen", ("frozen_metrics.csv" if not tag else f"frozen_metrics_{tag}.csv")),
        index=False, encoding="utf-8-sig")

    pd.DataFrame(adaptive_rows).to_csv(
        os.path.join(OUTPUT_DIR, "Adaptive", ("adaptive_metrics.csv" if not tag else f"adaptive_metrics_{tag}.csv")),
        index=False, encoding="utf-8-sig")

    # Multi-seed summary (unified interface: every row has primary_metric/before_primary/after_primary/improvement_delta)
    robust_dir = os.path.join(OUTPUT_DIR, "Robustness")
    all_rows = frozen_rows + adaptive_rows

    robustness = []
    for row in all_rows:
        robustness.append({
            "target": row["target"],
            "seed": row["seed"],
            "protocol": row["protocol"],
            "primary_metric": row.get("primary_metric", ""),
            "before_primary": row.get("before_primary"),
            "after_primary": row.get("after_primary"),
            "improvement_delta": row.get("improvement_delta"),
            "before_k": row.get("before_k", row.get("k")),
            "after_k": row.get("after_k", row.get("k")),
            "direction_positive": int(row.get("improvement_delta", 0) > 0) if row.get("improvement_delta") is not None else 0,
        })

    pd.DataFrame(robustness).to_csv(
        os.path.join(robust_dir, "multiseed_metrics.csv"),
        index=False, encoding="utf-8-sig")

    # SHAP structure metrics (top1/top5/eff_n/entropy for Before+After)
    if shap_export_entries:
        export_shap_stability(shap_export_entries,
                              os.path.join(OUTPUT_DIR, "SHAP_Stability"))

    # SHAP ranking similarity (top10_jaccard/spearman_rho)
    if shap_similarity_entries:
        pd.DataFrame(shap_similarity_entries).to_csv(
            os.path.join(OUTPUT_DIR, "SHAP_Stability", "shap_ranking_similarity.csv"),
            index=False, encoding="utf-8-sig")

    # Distribution data
    export_distribution_data(X_all, is_literature)
    _write_pipeline_status("complete", current={"stage": "complete"})

    print(f"\n{'='*60}")
    print("  All done.")
    print(f"{'='*60}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"    SampleSizes/target_sample_sizes.csv")
    print(f"    Frozen/frozen_metrics.csv")
    print(f"    Adaptive/adaptive_metrics.csv")
    print(f"    Robustness/multiseed_metrics.csv")
    print(f"    SHAP_Stability/shap_structure_metrics.csv")
    print(f"    SHAP_Stability/shap_feature_stability.csv")
    print(f"    SHAP_Stability/shap_ranking_similarity.csv")
    print(f"    Distribution/umap_coordinates.csv")
    print(f"    Distribution/experimental_knn_density.csv")


if __name__ == "__main__":
    main()
