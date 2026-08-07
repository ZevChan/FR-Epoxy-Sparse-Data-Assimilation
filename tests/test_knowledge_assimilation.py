from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import knowledge_assimilation as ka
import plot_knowledge_assimilation as kaplot


class KnowledgeAssimilationTests(unittest.TestCase):
    def test_directory_isolation_allows_only_named_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Results"
            source.mkdir()
            allowed = source / "Knowledge_Assimilation"
            self.assertEqual(
                ka.ensure_separate_directories(source, allowed),
                (source.resolve(), allowed.resolve()),
            )
            with self.assertRaises(ValueError):
                ka.ensure_separate_directories(source, source / "Other")
            with self.assertRaises(ValueError):
                ka.ensure_separate_directories(source, source)

    def test_runner_has_no_hpo_or_configuration_selection_calls(self):
        """The independent runner must never invoke the expensive selector."""
        for name in ["knowledge_assimilation.py", "run_knowledge_assimilation.py"]:
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
            self.assertNotIn("select_configuration", calls)
            self.assertNotIn("optimize", calls)

    def test_assimilation_curve_has_one_copy_of_identical_endpoints(self):
        context = {
            "target": "demo",
            "seed": 42,
            "metric": "r2",
            "before_score": 0.10,
            "after_score": 0.40,
            "X_exp": np.zeros((4, 1)),
            "experiment_sample_ids": np.array([10, 11, 12, 13]),
        }

        def fake_score(ctx, indices, n_jobs):
            return 0.10 + 0.05 * len(indices), np.zeros(2)

        with patch.object(ka, "_fit_subset_and_score", side_effect=fake_score):
            raw, summary = ka.compute_assimilation_curve(
                context,
                repeats=3,
                fractions=(0.0, 0.5, 1.0),
                n_jobs=1,
            )
        counts = raw.groupby("dose_fraction").size().to_dict()
        self.assertEqual(counts, {0.0: 1, 0.5: 3, 1.0: 1})
        utility = summary.loc[
            summary["dose_fraction"] == 1.0, "utility_mean"
        ].iloc[0]
        self.assertAlmostEqual(utility, 0.30)

    def test_assign_update_states_uses_predeclared_thresholds(self):
        phase = pd.DataFrame(
            {
                "target": ["a", "b", "c", "d"],
                "seed": [42] * 4,
                "primary_metric": ["r2"] * 4,
                "utility": [0.05, 0.05, 0.005, -0.03],
                "plasticity": [0.2, 0.8, 0.2, 0.7],
            }
        )
        result = ka.assign_update_states(
            phase,
            plasticity_threshold=0.5,
            regression_tolerance=0.01,
        )
        self.assertEqual(
            result["phase_state"].tolist(),
            [
                "Stable assimilation",
                "Plastic assimilation",
                "Inert update",
                "Destructive interference",
            ],
        )
        self.assertEqual(
            result["update_outcome"].tolist(),
            ["assimilation", "assimilation", "inertness", "interference"],
        )

    def test_propagation_positive_loss_improvement_means_after_is_better(self):
        context = {
            "target": "demo",
            "seed": 42,
            "is_classification": False,
            "metric": "r2",
            "X_exp": np.array([[0.0], [10.0]]),
            "X_test": np.array([[0.1], [1.0], [9.0], [9.9], [5.0], [6.0]]),
            "test_sample_ids": np.arange(6),
        }
        predictions = pd.DataFrame(
            {
                "sample_id": np.arange(6),
                "target": "demo",
                "seed": 42,
                "protocol": "frozen",
                "y_true": [0, 1, 2, 3, 4, 5],
                "y_pred_before": [1, 2, 3, 4, 5, 6],
                "y_pred_after": [0, 1, 2, 3, 4, 5],
            }
        )
        samples, regions = ka.compute_propagation(context, predictions)
        self.assertTrue((samples["loss_improvement"] == 1.0).all())
        self.assertTrue((samples["knowledge_effect"] == "improved").all())
        self.assertEqual(
            set(regions["neighborhood"]), {"near", "middle", "far"}
        )

    def test_fit_estimate_contains_zero_hpo_calls(self):
        estimate = ka.estimate_fit_count(
            n_experiments=17,
            dose_repeats=20,
            dose_fractions=(0, 0.25, 0.5, 0.75, 1),
        )
        self.assertEqual(estimate["hpo_or_optuna_models"], 0)
        self.assertEqual(estimate["dose_models"], 60)
        self.assertEqual(estimate["loeo_models"], 17)

    def test_paper_figures_render_from_cached_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for subdir in ["Dose", "DataValue", "Propagation", "Plasticity"]:
                (output / subdir).mkdir()

            pd.DataFrame(
                {
                    "target": ["LOI"] * 3,
                    "seed": [42] * 3,
                    "dose_percent": [0, 50, 100],
                    "utility_mean": [0.0, 0.02, 0.03],
                    "utility_ci_low": [0.0, 0.01, 0.03],
                    "utility_ci_high": [0.0, 0.03, 0.03],
                }
            ).to_csv(output / "Dose" / "assimilation_curve_summary.csv", index=False)
            pd.DataFrame(
                {
                    "target": ["LOI"] * 3,
                    "seed": [42] * 3,
                    "experiment_sample_id": [1, 2, 3],
                    "marginal_value": [-0.01, 0.00, 0.02],
                    "descriptor_novelty_percentile": [0.2, 0.5, 0.8],
                    "label_surprise": [0.4, 1.0, 1.8],
                    "local_compatibility": [0.8, 0.6, 0.3],
                }
            ).to_csv(
                output / "DataValue" / "experimental_data_value.csv", index=False
            )
            pd.DataFrame(
                {
                    "target": ["LOI"] * 9,
                    "seed": [42] * 9,
                    "distance_percentile": np.linspace(0.1, 0.9, 9),
                    "loss_improvement": np.linspace(-1, 1, 9),
                }
            ).to_csv(
                output / "Propagation" / "propagation_samples.csv", index=False
            )
            pd.DataFrame(
                {
                    "target": ["LOI"] * 3,
                    "seed": [42] * 3,
                    "neighborhood": ["near", "middle", "far"],
                    "mean_loss_improvement": [0.2, 0.1, -0.1],
                }
            ).to_csv(
                output / "Propagation" / "propagation_regions.csv", index=False
            )
            pd.DataFrame(
                {
                    "target": ["LOI"],
                    "seed": [42],
                    "utility": [0.03],
                    "plasticity": [0.4],
                    "plasticity_threshold": [0.5],
                }
            ).to_csv(
                output / "Plasticity" / "utility_plasticity_phase_map.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "target": ["LOI"] * 6,
                    "seed": [42] * 6,
                    "track": ["before"] * 3 + ["after"] * 3,
                    "feature": ["a", "b", "c"] * 2,
                    "rank": [1, 2, 3, 2, 1, 3],
                    "normalized_mean_abs_shap": [0.5, 0.3, 0.2, 0.3, 0.5, 0.2],
                }
            ).to_csv(
                output / "Plasticity" / "shap_feature_values.csv", index=False
            )

            paths = kaplot.build_all_figures(output)
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
