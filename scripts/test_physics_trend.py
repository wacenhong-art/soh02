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
