import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


physics = load("physics", ROOT / "scripts" / "08_physics_trend_ablation.py")
fusion = load("fusion_test", ROOT / "scripts" / "04_attention_fusion.py")


class LastLag(torch.nn.Module):
    def forward(self, x):
        return x[:, -1, -1:]


class LargeConstant(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1), 1000.0, device=x.device)


class PaperBaselineTests(unittest.TestCase):
    def test_physics_loss_definitions(self):
        pred = torch.tensor([[3.0], [2.0], [4.0]])
        mono = physics.physics_loss(pred, "monotonic")
        smooth = physics.physics_loss(pred, "smooth")
        self.assertAlmostEqual(float(mono), 1.0, places=6)
        self.assertAlmostEqual(float(smooth), 9.0, places=6)
        self.assertEqual(float(physics.physics_loss(pred, "none")), 0.0)

    def test_discontinuous_trajectory_is_rejected(self):
        aug = pd.DataFrame({
            "DatasetID": [0, 0, 0], "Cycle": [0, 2, 3],
            "HF1": [1, 1, 1], "HF2": [1, 1, 1], "HF3": [1, 1, 1],
        })
        state = np.ones((4, 1), np.float32)
        values = np.ones(4, np.float32)
        with self.assertRaises(ValueError):
            physics.make_trajectories(
                aug, lambda frame: frame[["HF1", "HF2", "HF3"]].to_numpy(np.float32),
                state, values, 0.0, 1.0,
            )

    def test_recursive_prediction_ignores_future_component_labels(self):
        n, known_end = 30, 15
        hf = np.zeros((n, 3), np.float32)
        values_a = np.arange(n, dtype=np.float32)
        values_b = values_a.copy(); values_b[known_end:] = 10000
        model = LastLag().to(physics.DEVICE)
        a = physics.causal_component_predict(model, hf, values_a, 0.0, 1.0,
                                             known_end, n, known_end)
        b = physics.causal_component_predict(model, hf, values_b, 0.0, 1.0,
                                             known_end, n, known_end)
        np.testing.assert_allclose(a, b)

    def test_recursive_state_clip_uses_only_known_range(self):
        n, known_end = 30, 15
        values = np.arange(n, dtype=np.float32)
        prediction = physics.causal_component_predict(
            LargeConstant().to(physics.DEVICE), np.zeros((n, 3), np.float32),
            values, 0.0, 1.0, known_end, n, known_end,
            use_component_lag=True, clip_recursive_state=True,
        )
        expected_upper = values[:known_end].max() + .1 * np.ptp(values[:known_end])
        self.assertTrue(np.all(prediction <= expected_upper + 1e-6))

    def test_fusion_target_accepts_explicit_true_soh(self):
        frame = pd.DataFrame({
            "True_SOH": [90.0, 89.0],
            "Pred_Trend": [89.0, 88.0],
            "Pred_Seasonal": [1.0, 1.0],
            "Pred_Residual": [0.0, 0.0],
        })
        np.testing.assert_allclose(fusion.true_soh_target(frame), [90.0, 89.0])

    def test_prepare_split_standardizes_soh_target(self):
        rows = fusion.SEQ_LEN + 2
        frame = pd.DataFrame({
            "True_SOH": np.linspace(80.0, 82.0, rows),
            "Pred_Trend": np.linspace(79.0, 81.0, rows),
            "Pred_Seasonal": np.ones(rows),
            "Pred_Residual": np.zeros(rows),
        })
        arrays, raw_target = fusion.prepare_split(
            frame, np.zeros(3), np.ones(3), np.zeros(9), np.ones(9), 80.0, 2.0
        )
        expected = (raw_target[fusion.SEQ_LEN:] - 80.0) / 2.0
        np.testing.assert_allclose(arrays[2], expected, atol=1e-6)

    def test_attention_weights_sum_to_one(self):
        model = fusion.AttentionBiLSTMFusion(
            np.zeros(3), np.ones(3), 80.0, 5.0, fusion_mode="paper"
        ).eval()
        x = torch.zeros(4, fusion.SEQ_LEN, 9)
        comp = torch.zeros(4, 3)
        with torch.no_grad():
            _, weights = model(x, comp)
        np.testing.assert_allclose(weights.sum(1).numpy(), np.ones(4), atol=1e-6)

    def test_fusion_output_is_in_standardized_target_space(self):
        model = fusion.AttentionBiLSTMFusion(
            np.zeros(3), np.ones(3), 80.0, 5.0, fusion_mode="paper"
        ).eval()
        for parameter in model.predict_head.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            prediction_z, _ = model(
                torch.zeros(2, fusion.SEQ_LEN, 9), torch.zeros(2, 3)
            )
        np.testing.assert_allclose(prediction_z.numpy(), 0.0, atol=1e-6)

    def test_stable_fusion_initialization_reconstructs_components(self):
        model = fusion.AttentionBiLSTMFusion(
            np.array([70.0, 1.0, 0.0]), np.array([5.0, 2.0, 1.0]),
            71.0, 8.0, fusion_mode="paper", stable_init=True,
        ).eval()
        component = torch.tensor([[75.0, 3.0, 1.0]])
        with torch.no_grad():
            prediction_z, weights = model(
                torch.zeros(1, fusion.SEQ_LEN, 9), component
            )
        np.testing.assert_allclose(weights.numpy(), np.full((1, 3), 1 / 3), atol=1e-6)
        np.testing.assert_allclose(prediction_z.numpy(), [[1.0]], atol=1e-6)

    def test_source_batteries_exclude_targets(self):
        source = (ROOT / "scripts" / "02_scf_gan.py").read_text(encoding="utf-8")
        self.assertIn('source_bats = ["B0005", "B0006"]', source)

    def test_stl_reconstructs_training_prefix(self):
        for battery in ("B0007", "B0018"):
            raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
            train_end = int(len(raw) * .6)
            soh = raw["SOH"].to_numpy(float)[:train_end] * 100
            dec = STL(soh, period=physics.ts.STL_PERIOD, robust=True).fit()
            np.testing.assert_allclose(soh, dec.trend + dec.seasonal + dec.resid,
                                       atol=1e-8)

    def test_hf_only_component_model_shapes(self):
        x = torch.zeros(2, physics.ts.SEQ_LEN, 3)
        with torch.no_grad():
            timesnet_out = physics.ts.TimesNet(input_dim=3)(x)
            itransformer_out = physics.ts.iTransformer()(x)
        self.assertEqual(tuple(timesnet_out.shape), (2, 1))
        self.assertEqual(tuple(itransformer_out.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
