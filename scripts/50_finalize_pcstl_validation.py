"""Finalize the preregistered PC-STL development validation without test access."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "physics_consistent_trend_v1"
SEEDS = [42, 52, 62]


def main() -> None:
    frames = []
    for seed in SEEDS:
        path = OUT / f"validation_mu1_seed{seed}" / f"validation_metrics_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing preregistered validation artifact: {path}")
        frames.append(pd.read_csv(path))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUT / "validation_multiseed_metrics.csv", index=False, encoding="utf-8-sig")
    pivot = metrics.pivot(index=["BatteryID", "Seed"], columns="Branch", values="RMSE").reset_index()
    pivot["PairedRMSEDelta_PCSTL_minus_P0"] = pivot.pcstl - pivot.p0
    pivot["PCSTLImproved"] = pivot["PairedRMSEDelta_PCSTL_minus_P0"] < 0
    pivot.to_csv(OUT / "paired_validation_rmse.csv", index=False, encoding="utf-8-sig")
    rows = []
    for battery, group in pivot.groupby("BatteryID"):
        base = group.p0.to_numpy(float); proposal = group.pcstl.to_numpy(float)
        source = metrics[metrics.BatteryID.eq(battery)].pivot(index="Seed", columns="Branch")
        violation_delta = (source["TrendViolationRate", "pcstl"] - source["TrendViolationRate", "p0"]).to_numpy(float)
        rows.append({"BatteryID": battery, "P0MedianRMSE": float(np.median(base)),
                     "PCSTLMedianRMSE": float(np.median(proposal)),
                     "MedianPairedRMSEDelta": float(np.median(proposal - base)),
                     "ImprovedSeeds": int((proposal < base).sum()), "TotalSeeds": len(group),
                     "MedianTrendViolationDelta": float(np.median(violation_delta)),
                     "PassThreeSeedGate": bool((proposal < base).sum() >= 2 and np.median(proposal - base) < 0 and np.median(violation_delta) <= 0)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "validation_summary.csv", index=False, encoding="utf-8-sig")
    passed = bool(summary.PassThreeSeedGate.all())
    lock = {"schema": "pcstl_validation_lock/1", "created_at": datetime.now().isoformat(),
            "phase": "development_validation", "seeds": SEEDS, "mu": 1.0,
            "test_results_accessed": False, "configuration_locked_for_confirmation": False,
            "three_seed_gate_passed": passed,
            "reason": "B0018 did not improve in at least two of three seeds; five-seed and confirmation tests are prohibited." if not passed else "Eligible for five-seed validation.",
            "metrics_sha256": hashlib.sha256((OUT / "validation_multiseed_metrics.csv").read_bytes()).hexdigest()}
    (OUT / "validation_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger = pd.DataFrame([{"AttemptID": "PCSTL-D03", "Timestamp": datetime.now().isoformat(),
                            "Change": "training-only monotonic smooth STL Trend reconstruction, mu=1.0",
                            "Reason": "isolated alternative after soft physics-loss failure",
                            "Stage": "three-seed development validation",
                            "TestResultsViewed": False,
                            "FailedGate": "three_seed_cross_battery_consistency" if not passed else ""}])
    ledger.to_csv(OUT / "attempt_ledger.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot = metrics.pivot(index="Seed", columns=["BatteryID", "Branch"], values="RMSE")
    plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Validation RMSE (percentage points)"); ax.set_xlabel("Random seed")
    ax.set_title("PC-STL development validation (no test access)")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(ROOT / "output" / "figures" / "pcstl_v1_validation_rmse.png", dpi=200)
    plt.close(fig)
    text = f"""# PC-STL 三随机种子开发验证报告

## 结论

`three_seed_gate_passed={passed}`。因此禁止继续进入五随机种子、配置锁定和任何确认测试。

PC-STL只使用训练前缀的STL Trend建立单调平滑监督目标，差额回写Residual；P0与PC-STL均使用相同P1 Attention初始化。验证集用于选择mu=1.0；没有读取测试标签。

| Battery | P0 median RMSE | PC-STL median RMSE | Improved seeds | Median violation delta | Gate |
"""
    for _, row in summary.iterrows():
        text += f"| {row.BatteryID} | {row.P0MedianRMSE:.4f} | {row.PCSTLMedianRMSE:.4f} | {row.ImprovedSeeds}/{row.TotalSeeds} | {row.MedianTrendViolationDelta:+.4f} | {'PASS' if row.PassThreeSeedGate else 'FAIL'} |\n"
    text += "\nB0018未满足至少2/3种子RMSE改善；继续在同一开发电池调参会产生选择偏差，故该创新路线以负结果停止。\n"
    (OUT / "PCSTL_VALIDATION_REPORT.md").write_text(text, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
