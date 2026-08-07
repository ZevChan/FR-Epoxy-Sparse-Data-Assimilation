# FAEFREP — Frozen / Adaptive Evaluation for Flame-Retardant Epoxy (FREP)

Machine-learning-guided evaluation of **sparse experimental data value** in flame-retardant epoxy (FREP) property prediction.

This repository implements a controlled, reproducible evaluation framework that quantifies how a small set of new experimental samples improves XGBoost models trained on literature data. It supports two explicit protocols:

- **Frozen protocol (primary):** features, K, hyperparameters, and preprocessing are all determined on the literature training set and *frozen*. The only systematic difference between Before (literature only) and After (literature + experiment) is whether the experimental rows are included in training.
- **Adaptive protocol (secondary):** Before/After each select K and hyperparameters inside their own training set, under identical search rules and budget.

A low-cost **knowledge-assimilation** post-processing layer (`run_knowledge_assimilation.py`) additionally measures per-experiment marginal value (LOEO), dose-response curves, knowledge propagation, and utility–plasticity phase maps — without re-running Optuna or feature selection.

## Project structure

```
FAEFREP_Frozen_Adaptive_Evaluation_FREP/
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
├── 实验数据.csv / 文献数据.csv    # SMILES-level experiment/literature tables (UMAP scripts input)
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

# main analysis on seed 42 (or any completed seed)
python run_knowledge_assimilation.py --seeds 42 --dose-repeats 20
```

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
