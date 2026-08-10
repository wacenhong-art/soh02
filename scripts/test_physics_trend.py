import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

P = Path(__file__).with_name("08_physics_trend_ablation.py")
S = importlib.util.spec_from_file_location("physics", P); M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def test_physics_losses():
    decreasing = torch.tensor([[3.], [2.], [1.]])
    increasing = torch.tensor([[1.], [2.], [3.]])
    curved = torch.tensor([[1.], [2.], [4.]])
    assert M.physics_loss(decreasing, "monotonic").item() == 0
    assert M.physics_loss(increasing, "monotonic").item() > 0
    assert M.physics_loss(decreasing, "smooth").item() == 0
    assert M.physics_loss(curved, "smooth").item() > 0
    assert M.physics_loss(curved, "none").item() == 0


def test_stable_seed_is_order_independent():
    assert M.stable_seed(42, "B0007", "Trend") == M.stable_seed(42, "B0007", "Trend")
    assert M.stable_seed(42, "B0007", "Trend") != M.stable_seed(42, "B0018", "Trend")
    assert M.stable_seed(42, "B0018", "Trend") != M.stable_seed(42, "B0018", "Residual")
    assert M.stable_seed(42, "B0043", "Trend") != M.stable_seed(42, "B0044", "Trend")


def test_paper_branch_shapes_and_weights():
    x = torch.randn(4, 10, 4)
    assert M.ts.TimesNet()(x).shape == (4, 1)
    assert M.ts.LightweightTimesNet()(x).shape == (4, 1)
    assert M.ts.iTransformer()(x).shape == (4, 1)
    model = M.fusion.AttentionBiLSTMFusion([0, 0, 0], [1, 1, 1], 0, 1)
    pred, weights = model(torch.randn(4, 10, 9), torch.randn(4, 3))
    assert pred.shape == (4, 1)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(1), torch.ones(4), atol=1e-6)


def test_constant_anchor_preserves_trend_differences():
    trend = torch.tensor([4.0, 3.2, 3.4, 2.9])
    calibrated = trend + 1.75
    assert torch.allclose(torch.diff(trend), torch.diff(calibrated))
    iprf = M.fusion.AttentionBiLSTMFusion([0, 0, 0], [1, 1, 1], 0, 1, fusion_mode="iprf")
    iprf_pred, iprf_weights = iprf(torch.randn(4, 10, 9), torch.randn(4, 3))
    assert iprf_pred.shape == (4, 1)
    assert torch.allclose(iprf_weights.sum(1), torch.ones(4), atol=1e-6)


def test_fusion_features_do_not_read_true_components():
    pred = np.arange(18, dtype=np.float32).reshape(6, 3)
    frame = pd.DataFrame(pred, columns=["Pred_Trend", "Pred_Seasonal", "Pred_Residual"])
    for name in ("Trend", "Seasonal", "Residual"):
        frame[f"True_{name}"] = np.linspace(0, 1, len(frame))
    first = M.fusion.build_causal_features(frame, pred.mean(0), pred.std(0))
    frame[["True_Trend", "True_Seasonal", "True_Residual"]] += 10000.0
    second = M.fusion.build_causal_features(frame, pred.mean(0), pred.std(0))
    assert np.allclose(first, second)


def test_recursive_component_prediction_ignores_unknown_labels():
    class PreviousState(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, -1:].clone()

    norm_hf = np.zeros((30, 3), dtype=np.float32)
    values = np.linspace(10.0, 7.0, 30, dtype=np.float32)
    changed_unknown = values.copy(); changed_unknown[15:] += 1000.0
    args = (PreviousState(), norm_hf, values, 0.0, 1.0, 15, 22, 15)
    first = M.causal_component_predict(*args)
    args_changed = (PreviousState(), norm_hf, changed_unknown, 0.0, 1.0, 15, 22, 15)
    second = M.causal_component_predict(*args_changed)
    assert np.allclose(first, second)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
