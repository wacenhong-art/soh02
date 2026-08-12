import unittest

import numpy as np

from soh_pipeline.evaluation import evaluate_regression, make_lifecycle_split
from soh_pipeline.protocol import PAPER_FIXED, build_stage


class ExperimentContractTests(unittest.TestCase):
    def test_lifecycle_split_is_ordered_and_disjoint(self):
        split = make_lifecycle_split(100, .6, .2)
        self.assertEqual((split.train_end, split.validation_end), (60, 80))
        self.assertEqual(list(range(100))[split.test][0], 80)

    def test_r2_matches_mean_baseline_definition(self):
        metrics = evaluate_regression([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(metrics["R2"], 0.0)
        self.assertAlmostEqual(metrics["RMSE"], metrics["MeanBaselineRMSE"])

    def test_negative_r2_is_preserved(self):
        self.assertLess(evaluate_regression([1, 2, 3], [10, 10, 10])["R2"], 0)

    def test_protocol_forbids_reference_project_leakage(self):
        forbidden = " ".join(PAPER_FIXED["forbidden"])
        self.assertIn("test-label offset calibration", forbidden)
        self.assertIn("fit scaler on target validation/test", forbidden)

    def test_grid_is_deterministic_and_hashed(self):
        first = build_stage("A_representation")
        second = build_stage("A_representation")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 144)
        self.assertEqual(len({row["config_hash"] for row in first}), 144)


if __name__ == "__main__":
    unittest.main()
