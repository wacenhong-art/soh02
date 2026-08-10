"""Create the stability-rescue attempt ledger, metrics table and audit figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "paper_direct_gmtc"
FIGURES = ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

attempts = [
    ("E00", "B0034", "guard_none/fusion_paper", "Original direct G-MTC", "development", False),
    ("E01", "B0034", "guard_seasonal_residual/fusion_paper", "Seasonal/Residual guard", "development", False),
    ("E02", "B0034", "guard_none/fusion_iprf", "Bounded additive fusion", "development", False),
    ("E03", "B0034", "guard_none/anchor_ridge_residual/fusion_paper", "Trend-target Ridge anchor", "development", False),
    ("E04", "B0034", "guard_none/anchor_soh_ridge_residual/fusion_paper", "SOH-scale Trend anchor", "development", False),
    ("E05", "B0034", "guard_none/anchor_soh_ridge_residual/fusion_iprf", "Anchor + bounded fusion", "development selected", False),
    ("C01", "B0036", "guard_none/anchor_soh_ridge_residual/fusion_iprf", "Frozen E05 confirmation", "untouched confirmation", False),
    ("T01", "B0018", "guard_none/anchor_soh_ridge_residual/fusion_iprf", "Externally frozen E05", "contacted target evaluation", True),
    ("T02", "B0018", "guard_seasonal_residual/anchor_soh_ridge_residual/fusion_iprf", "Amplitude safety audit", "post-hoc development audit", True),
]

rows = []
for attempt, battery, suffix, change, role, target_seen in attempts:
    path = BASE / battery / "seed_42" / "width_16" / suffix / "metrics.csv"
    metrics = pd.read_csv(path).iloc[0]
    rows.append({
        "Attempt": attempt,
        "BatteryID": battery,
        "SingleChange": change,
        "EvaluationRole": role,
        "TargetTestPreviouslyContacted": target_seen,
        "TestUsedToSelectNextChange": attempt in {"E00", "E01", "E02", "E03", "E04"},
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "MAPE": float(metrics["MAPE"]),
        "R2": float(metrics["R2"]),
        "Passed": attempt == "E05",
        "ResultPath": str(path.relative_to(ROOT)),
    })

ledger = pd.DataFrame(rows)
ledger.to_csv(BASE / "stability_attempt_ledger.csv", index=False, encoding="utf-8-sig")
ledger[["Attempt", "BatteryID", "EvaluationRole", "RMSE", "MAE", "MAPE", "R2", "Passed"]].to_csv(
    BASE / "stability_metrics.csv", index=False, encoding="utf-8-sig"
)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
dev = ledger[ledger["BatteryID"] == "B0034"]
colors = ["#8c8c8c", "#8c8c8c", "#c44e52", "#8172b3", "#4c72b0", "#2a9d8f"]
bars = axes[0].bar(dev["Attempt"], dev["RMSE"], color=colors)
for bar, value in zip(bars, dev["RMSE"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", fontsize=9)
axes[0].set_title("B0034 outer-development attempts")
axes[0].set_ylabel("RMSE (percentage points)")
axes[0].grid(axis="y", alpha=0.25)

prediction_path = (
    BASE / "B0018" / "seed_42" / "width_16" / "guard_seasonal_residual"
    / "anchor_soh_ridge_residual" / "fusion_iprf" / "per_cycle_predictions.csv"
)
prediction = pd.read_csv(prediction_path)
axes[1].plot(prediction["Cycle"], prediction["TrueSOH"], color="black", linewidth=1.8, label="Measured")
axes[1].plot(prediction["Cycle"], prediction["PredSOH"], color="#2a9d8f", linewidth=1.5,
             label="Stability audit (RMSE 2.733)")
axes[1].plot(prediction["Cycle"], prediction["PredTrend"], color="#4c72b0", linewidth=1.1,
             label="Anchored Trend (RMSE 2.330)")
axes[1].axvline(25, color="#666666", linestyle=":", label="First-20% boundary")
axes[1].set_title("B0018 contacted development audit")
axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("SOH / Trend (%)")
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False, fontsize=8)

fig.suptitle("G-MTC stability rescue: improvement is real but not confirmatory", y=1.02)
fig.tight_layout()
figure_path = FIGURES / "gmtc_stability_rescue_v2.png"
fig.savefig(figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)
print(ledger.to_string(index=False))
print(figure_path)
