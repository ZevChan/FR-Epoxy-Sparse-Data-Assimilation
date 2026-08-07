import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fair_holdout_comparison as pipeline


class ResumeCheckpointTests(unittest.TestCase):
    def test_complete_unit_checkpoint_skips_all_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "Checkpoints"
            status_path = Path(temp_dir) / "pipeline_status.json"
            payload = {
                "version": pipeline.CHECKPOINT_VERSION,
                "target": "LOI",
                "seed": 7,
                "sample_size_row": {"target": "LOI", "seed": 7},
                "frozen_config": {"k": 1},
                "frozen_metrics": {"target": "LOI", "seed": 7},
                "adaptive_metrics": {"target": "LOI", "seed": 7},
            }
            with mock.patch.object(pipeline, "CHECKPOINT_DIR", str(checkpoint_dir)), \
                    mock.patch.object(pipeline, "PIPELINE_STATUS_PATH", str(status_path)):
                pipeline._atomic_checkpoint_dump("LOI", 7, payload)
                with mock.patch.object(
                    pipeline, "select_configuration",
                    side_effect=AssertionError("HPO must be skipped"),
                ), mock.patch.object(
                    pipeline, "run_frozen_comparison",
                    side_effect=AssertionError("Frozen refit must be skipped"),
                ), mock.patch.object(
                    pipeline, "run_adaptive_comparison",
                    side_effect=AssertionError("Adaptive refit must be skipped"),
                ):
                    sample, frozen, adaptive = pipeline._run_or_resume_unit(
                        {"target": "LOI"}, 7)
            self.assertEqual(sample["target"], "LOI")
            self.assertEqual(frozen["seed"], 7)
            self.assertEqual(adaptive["seed"], 7)

    def test_configuration_payload_preserves_resume_state(self):
        config = {
            "k": 12,
            "features": ["a", "b"],
            "params": {"max_depth": 4},
            "cv_score": 0.25,
            "k_selected_nonmandatory": 12,
            "n_mandatory_active": 2,
            "converged": True,
        }
        payload = pipeline._configuration_payload(config)
        self.assertEqual(payload["params"], {"max_depth": 4})
        self.assertEqual(payload["features"], ["a", "b"])
        self.assertTrue(payload["converged"])


if __name__ == "__main__":
    unittest.main()
