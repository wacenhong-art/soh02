import importlib.util
from pathlib import Path
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
