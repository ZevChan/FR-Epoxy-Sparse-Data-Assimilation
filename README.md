# FR-Epoxy-Sparse-Data-Assimilation — Frozen / Adaptive Evaluation for Flame-Retardant Epoxy

Machine-learning-guided evaluation of **sparse experimental data value** in flame-retardant epoxy (FREP) property prediction.

This repository implements a controlled, reproducible evaluation framework that quantifies how a small set of new experimental samples improves XGBoost models trained on literature data. It supports two explicit protocols:

- **Frozen protocol (primary):** features, K, hyperparameters, and preprocessing are all determined on the literature training set and *frozen*. The only systematic difference between Before (literature only) and After (literature + experiment) is whether the experimental rows are included in training.
- **Adaptive protocol (secondary):** Before reuses the literature-only configuration selected for Frozen. After reselects features, K and hyperparameters on the augmented training set under the same search rules and budget. Preprocessing is fitted separately on each track's training set. Both protocols evaluate the same literature holdout for each target and seed.

A low-cost **knowledge-assimilation** post-processing layer (`run_knowledge_assimilation.py`) additionally measures per-experiment marginal value (LOEO), dose-response curves, knowledge propagation, and utility–plasticity phase maps — without re-running Optuna or feature selection.

## Formal configuration-selection procedure

The formal workflow in `select_configuration()` uses a single selected K:

1. Scan integer K values from 1 to `MAX_K=1000` with fixed `SCAN_BASE_PARAMS` and five-fold internal CV. K counts non-mandatory features; present mandatory features are retained in addition.
2. Stop after `PATIENCE_K=50` consecutive steps without a score exceeding the tracked best by `MIN_DELTA_K=0.0005`, or at the upper bound. The scan need not reach K=1000.
3. Select the single K with the highest mean CV score among all scanned K values.
4. Optimise XGBoost hyperparameters only at this K using Optuna, with at most `JOINT_HPO_TRIALS=50` trials. HPO stops early after `HPO_PATIENCE=10` consecutive trials without a new best score.
5. Recompute the feature ranking on the full supplied training set and retain present mandatory features plus the top K non-mandatory features.

No top-three-K optimisation, joint K/hyperparameter search, alternating search or high-K sentinel check is executed. Legacy helpers and configuration options remain for compatibility and are identified in their documentation. The fixed literature holdout is not used for feature selection or HPO.

## Project structure

```
FR-Epoxy-Sparse-Data-Assimilation/
├── fair_holdout_comparison.py    # Core pipeline: data loading → Frozen/Adaptive evaluation
├── config.py                     # Shared paths, targets, seeds, HPO space, forced features
├── evaluation.py                 # Paired bootstrap + classification statistics (McNemar)
├── distribution.py               # Unsupervised UMAP + high-dim kNN density analysis
├── shap_metrics.py               # SHAP decision-structure stability metrics
├── knowledge_assimilation.py     # LOEO value, dose curves, propagation, plasticity (library)
├── run_knowledge_assimilation.py # CLI entry for the knowledge-assimilation analyses
├── compute_signed_shap.py        # Sample-level signed SHAP for the 10-seed Frozen analysis
├── data_prep.py                  # Generate data/BasicData_wo (literature subset) from data/BasicData
├── plot_*.py                     # Figure scripts (paper figures, SI, UMAP, SHAP, etc.)
├── data/
│   └── BasicData/                # alvaDesc descriptor matrices (WITH: literature + experiment)
├── experimental_data.csv / literature_data.csv  # SMILES-level experiment/literature tables (UMAP scripts input)
├── tests/                        # pytest tests for knowledge assimilation / checkpoint resume
└── requirements.txt
```

## Targets (6 → 5 in the current revision)

| Target | Type | Note |
|---|---|---|
| `LOI` | regression | Limiting oxygen index |
| `UL94_Rating` | binary classification | V-0 vs non-V-0 |
| `THR` | regression | Total heat release |
| `TSP` | regression | Total smoke production |
| `Flexural_Strength` | regression | Flexural strength |

`pHRR` was removed from the target list in the current revision (see `config.py`).

## Data

- `data/BasicData/` contains the alvaDesc descriptor matrices (~317 MB, 7 descriptor groups + curing strategy + targets + `Dataset_with_SMILES.xlsx`). Rows 0–2331 are literature samples; rows 2332–2355 are the 24 experimental samples.
- `data/BasicData_wo/` (literature-only subset) is **not stored** in git; generate it once after cloning:

```bash
python data_prep.py
```

> If you package the repository as a ZIP (rather than git), exclude `Results/`, `Graphs/`, and `data/BasicData_wo/` — they are regenerable.

> The descriptor columns were computed with [alvaDesc](https://www.alvascience.com/alvadesc/) (molecular descriptors). To reproduce from raw SMILES, use the alvaDesc CLI with `--descriptors=ALL` and rename columns with the per-role suffixes (`_EP`, `_FR`, `_Curing`, `_Other_Material_1/2/3`) as in the shipped CSVs.

## Dependencies

Python 3.10+ (tested in Conda env `DFT_FR_GNN_transformer`):

```bash
conda activate DFT_FR_GNN_transformer
pip install -r requirements.txt
```

Main packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `matplotlib`, `seaborn`, `umap-learn`, `rdkit`, `chardet`, `joblib`.

## Usage

### 1. Core Frozen/Adaptive comparison (10 formal seeds)

```bash
python fair_holdout_comparison.py --seeds 7,13,19,29,37,43,53,61,71,79
```

Options:
- `--targets LOI,THR` — restrict targets
- `--outdir-suffix 42` — redirect outputs to `Results_42` (keeps the formal Results untouched)
- `--tag mytag` — tag shard CSVs

The pipeline writes to `Results/` (`Frozen/`, `Adaptive/`, `Robustness/`, `SHAP_Stability/`, `Distribution/`, `SampleSizes/`) with per-unit checkpoints for crash-safe resume.

### 2. Knowledge assimilation (post-processing, no HPO)

```bash
# validate inputs only (no model fitting)
python run_knowledge_assimilation.py --validate-only

# formal analysis on the ten evaluation seeds (requires completed core outputs)
python run_knowledge_assimilation.py --seeds 7,13,19,29,37,43,53,61,71,79 --dose-repeats 20
```

Seed 42 is a development/example run and is excluded from formal inference. An explicitly requested example run is `python run_knowledge_assimilation.py --seeds 42 --dose-repeats 20`; it requires the corresponding completed core outputs. Use the explicit ten-seed command above for the formal analysis rather than relying on the CLI default of 42.

Outputs land in `Results/Knowledge_Assimilation/` (`Dose/`, `DataValue/`, `Propagation/`, `Plasticity/`, `ModelSensitivity/`, `SHAP_Relationships/`, `Audits/`).

### 3. Figures

After the pipeline has produced `Results/`, regenerate paper figures, e.g.:

```bash
python plot_new_figure2_v2.py      # Figure 2
python plot_figure3.py             # Figure 3 (dose assimilation)
python plot_figure4_5_combined.py  # combined Figure 4/5
python plot_knowledge_assimilation.py
```

### 4. Tests

```bash
python -m pytest tests/ -q
```

## Formal seed policy

- Inferential panels use seeds `7, 13, 19, 29, 37, 43, 53, 61, 71, 79`.
- Seed 42 is used for method development and single-trajectory illustrations only; it is excluded from the formal statistics.
- Dose / LOEO / propagation / utility–plasticity / model-family sensitivity treat seed as the independent replication unit.

## Outputs

`Results/` and `Graphs/` are regenerable and ignored by git. The main result artifacts are:

- `Results/Frozen/frozen_metrics.csv`, `predictions_<target>_seed_<seed>.csv`, `config_*.json`, `trajectory_*.csv`
- `Results/Robustness/multiseed_metrics.csv`
- `Results/Knowledge_Assimilation/**`

## License

For research use. If this repository is used in published work, please cite the associated manuscript (details to be added).

## Contact

For access to the full descriptor matrices or questions, please open an issue in this repository.
