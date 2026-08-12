"""Shared metrics and lifecycle split contracts.

The functions here are deliberately model-agnostic.  They make it difficult
for an experiment script to silently change the R2 definition or data boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LifecycleSplit:
    rows: int
    train_end: int
    validation_end: int

    @property
    def train(self) -> slice:
        return slice(0, self.train_end)

    @property
    def validation(self) -> slice:
        return slice(self.train_end, self.validation_end)

    @property
    def test(self) -> slice:
        return slice(self.validation_end, self.rows)

    def manifest(self) -> dict[str, int]:
        return asdict(self)


def make_lifecycle_split(rows: int, train_fraction: float, validation_fraction: float) -> LifecycleSplit:
    if rows < 3:
        raise ValueError("至少需要3个按生命周期排序的样本")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("训练和验证比例必须位于(0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("必须保留非空测试段")
    train_end = int(rows * train_fraction)
    validation_end = int(rows * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= rows:
        raise ValueError("按当前样本数无法形成非空训练/验证/测试段")
    return LifecycleSplit(rows, train_end, validation_end)


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return one canonical metric set in SOH percentage-point space."""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape or true.size < 2:
        raise ValueError("真实值和预测值必须是同长度且至少含2点")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("指标输入包含NaN或无穷值")
    error = true - pred
    sse = float(np.sum(error ** 2))
    centered = true - float(np.mean(true))
    sst = float(np.sum(centered ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.sum(np.abs(error)) / max(np.sum(np.abs(true)), 1e-12) * 100),
        "R2": r2,
        "MeanBaselineRMSE": float(np.sqrt(np.mean(centered ** 2))),
    }
