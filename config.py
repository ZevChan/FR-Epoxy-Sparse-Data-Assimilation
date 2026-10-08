"""
config.py — Shared configuration for the controlled data-increment ablation study
==================================================================================
Frozen protocol:  configuration determined and frozen on the literature training set
Adaptive protocol: Before reuses the Frozen literature-only configuration;
                   After reselects its configuration on augmented training data.
                   Each track fits preprocessing on its own training set.
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECT_DIR = DATA_DIR  # kept for backward compatibility
WITH_DATA_DIR = os.path.join(DATA_DIR, "BasicData")
WITHOUT_DATA_DIR = os.path.join(DATA_DIR, "BasicData_wo")

OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# ===== Target variables =====
# NOTE: pHRR was removed and is no longer a prediction target
TARGETS = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
    "UL94_Rating": "UL-94",
}

UNITS = {
    "LOI": "%",
    "THR": "MJ/m²",
    "TSP": "m²",
    "Flexural_Strength": "MPa",
}

# ===== Experimental design =====
# Seed 42 is fixed as the method-development seed: its fixed literature hold-out
# set participated in method development and is excluded from the formal
# statistics. The formal paper uses the fresh EVALUATION_SEEDS.
DEVELOPMENT_SEED = 42
PRIMARY_SEED = DEVELOPMENT_SEED
DEFAULT_SEED = PRIMARY_SEED
TEST_SIZE = 0.20

# Formal evaluation seeds (10 fresh seeds, excluding 42)
EVALUATION_SEEDS = [7, 13, 19, 29, 37, 43, 53, 61, 71, 79]  # all 10 seeds

# ===== Configuration-selection parameters =====
N_TRIALS = 50
N_CV_FOLDS = 5
MIN_DELTA = 0.001
PATIENCE = 10
N_BOOTSTRAP = 5000

# ===== K scan (formal method: integer steps with early stopping) =====
# Scan K=1..MAX_K with SCAN_BASE_PARAMS and N_CV_FOLDS-fold internal CV.
# Early stopping uses PATIENCE_K and MIN_DELTA_K; MAX_K is an upper bound.
# Only the single K with the highest scanned mean CV score is passed to HPO.
MAX_K = 1000
# Scan uses fixed base hyperparameters (no tuning)
SCAN_BASE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0,
    "reg_lambda": 1,
}
# Legacy options retained for compatibility; not consulted by the formal
# select_configuration -> _scan_full_k -> _hpo_for_fixed_k workflow.
TOP_K_CANDIDATES = 1        # historical setting; single-K selection is implemented directly
N_TOP_CANDIDATE_K = 10      # used only by the legacy candidate-set helper
CV_TOLERANCE = 0.002        # used only by the legacy candidate-set helper
K_NEIGHBOR_RADIUS = 2       # used only by the legacy candidate-set helper
# Active parameters in the formal workflow:
PATIENCE_K = 50             # K scan: stop after N consecutive K without improvement (pre-HPO early stop)
MIN_DELTA_K = 0.0005        # minimum CV increment treated as "improvement" in the K scan
JOINT_HPO_TRIALS = 50       # Optuna HPO trials per single K
HPO_PATIENCE = 10           # HPO early stop: stop after N consecutive trials without improvement
# Legacy options; no alternating search or sentinel gate is executed:
MAX_ALTERNATING_ROUNDS = 4  # retained for compatibility; unused by the formal workflow
SENTINEL_TOLERANCE = 0.002  # retained for compatibility; unused by the formal workflow

# ── Experiment seeds ──
# Formal runs use EVALUATION_SEEDS (10 fresh seeds)
# For development/debugging, use SEEDS=[DEVELOPMENT_SEED]
SEEDS = EVALUATION_SEEDS

# ===== Forced (always-retained) features =====
# Flame-retardant amount, curing-agent amount, Other-Material amounts,
# curing temperature/time/pressure
FORCED_FEATURE_GROUPS = {
    "flame_retardant_amount": ["Flame_retardant_AdditionAmount(wt%)"],
    "curing_agent_amount": ["Curing_agent_AdditionAmount(wt%)"],
    "other_material_amount": [
        "Other_Material_AdditionAmount_1 (wt%)",
        "Other_Material_AdditionAmount_2 (wt%)",
        "Other_Material_AdditionAmount_3 (wt%)",
    ],
    "curing_strategy": [
        "Curing_Tem1", "Curing_Time1", "Curing_Tem2", "Curing_Time2",
        "Curing_Tem3", "Curing_Time3", "Curing_Tem4", "Curing_Time4",
        "Curing_Tem5", "Curing_Time5", "Curing_Tem6", "Curing_Time6",
        "Curing_Tem7", "Curing_Time8", "Curing_Tem9", "Curing_Time9",
        "Curing_Tem10", "Curing_Time10", "Curing_Pressure",
    ],
}
# Flattened list (fixed order for reproducibility)
FORCED_FEATURES = [c for grp in FORCED_FEATURE_GROUPS.values() for c in grp]

# ===== HPO search space =====
HPO_SPACE = {
    "n_estimators": (50, 500),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0, 10),
    "reg_lambda": (0, 10),
}

# ===== Colors =====
COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER = "#C91511"
COLOR_LINE = "#8C92AC"

# ===== UMAP =====
SMILES_COLS = [
    "EPOXY STRUCTURE",
    "Flame_retardant",
    "Curing_agent ",
    "Other_Material_1",
    "Other_Material_2",
]

# ===== Output subdirectories =====
for sub in ["SampleSizes", "Frozen", "Adaptive", "Robustness",
            "SHAP_Stability", "Distribution"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
