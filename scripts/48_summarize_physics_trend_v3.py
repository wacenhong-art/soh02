"""Summarize trajectory-safe Physics-guided Trend experiments."""

from pathlib import Path
import hashlib
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "paper_direct_gmtc"
OUT = ROOT / "output" / "physics_trend_v3"
FIGURES = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
CONFIGS = OUT / "configs"
LOGS = OUT / "training_logs"
CONFIGS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def run_path(battery: str, mode: str, lam: float) -> Path:
    return (
        BASE / battery / "seed_42" / "width_16" / "guard_seasonal_residual"
        / "anchor_soh_ridge_trainable_residual" / "fusion_iprf"
        / f"physics_{mode}_lambda_{lam:g}"
    )


def read_metrics(battery: str, mode: str, lam: float) -> dict:
    row = pd.read_csv(run_path(battery, mode, lam) / "metrics.csv").iloc[0].to_dict()
    row.update({"BatteryID": battery, "PhysicsMode": mode, "LambdaPhy": lam, "Seed": 42})
    return row


ablation_rows = [read_metrics("B0034", "none", 0.0)]
ablation_rows += [read_metrics("B0034", mode, 0.001) for mode in ["monotonic", "smooth", "both"]]
ablation = pd.DataFrame(ablation_rows)
ablation.to_csv(OUT / "ablation_metrics.csv", index=False, encoding="utf-8-sig")

lambda_rows = [read_metrics("B0034", "none", 0.0)]
lambda_rows += [read_metrics("B0034", "both", lam) for lam in [0.001, 0.01, 0.1]]
lambda_table = pd.DataFrame(lambda_rows)
lambda_table.to_csv(OUT / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")

pair_rows, prediction_frames = [], []
for battery in ["B0034", "B0036", "B0018"]:
    for mode, lam in [("none", 0.0), ("both", 0.001)]:
        metric = read_metrics(battery, mode, lam)
        pair_rows.append(metric)
        prediction = pd.read_csv(run_path(battery, mode, lam) / "per_cycle_predictions.csv")
        prediction.insert(0, "LambdaPhy", lam)
        prediction.insert(0, "PhysicsMode", mode)
        prediction_frames.append(prediction)

paired_metrics = pd.DataFrame(pair_rows)
paired_metrics.to_csv(OUT / "paired_battery_metrics.csv", index=False, encoding="utf-8-sig")
predictions = pd.concat(prediction_frames, ignore_index=True)
predictions.to_csv(OUT / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
predictions[[
    "BatteryID", "DatasetID", "Cycle", "PhysicsMode", "LambdaPhy",
    "TrueSOH", "PredSOH", "PredTrend", "PredSeasonal", "PredResidual",
]].to_csv(OUT / "trend_predictions.csv", index=False, encoding="utf-8-sig")


def block_bootstrap_delta(original: pd.DataFrame, proposed: pd.DataFrame, seed: int = 20260810):
    joined = original[["Cycle", "TrueSOH", "PredSOH"]].merge(
        proposed[["Cycle", "PredSOH"]], on="Cycle", suffixes=("_original", "_physics")
    )
    truth = joined["TrueSOH"].to_numpy(float)
    pred_original = joined["PredSOH_original"].to_numpy(float)
    pred_physics = joined["PredSOH_physics"].to_numpy(float)
    n = len(joined); block = max(3, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(2000):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend((np.arange(start, start + block) % n).tolist())
        index = np.asarray(sampled[:n], dtype=int)
        rmse_original = np.sqrt(np.mean((truth[index] - pred_original[index]) ** 2))
        rmse_physics = np.sqrt(np.mean((truth[index] - pred_physics[index]) ** 2))
        deltas.append(rmse_physics - rmse_original)
    observed = np.sqrt(np.mean((truth - pred_physics) ** 2)) - np.sqrt(
        np.mean((truth - pred_original) ** 2)
    )
    return {
        "Rows": n,
        "BlockLength": block,
        "BootstrapReplicates": 2000,
        "ObservedDeltaRMSE_PhysicsMinusOriginal": observed,
        "CI95Lower": float(np.percentile(deltas, 2.5)),
        "CI95Upper": float(np.percentile(deltas, 97.5)),
        "CIExcludesZero": bool(np.percentile(deltas, 97.5) < 0 or np.percentile(deltas, 2.5) > 0),
    }


bootstrap_rows = []
for battery in ["B0034", "B0036", "B0018"]:
    original = predictions[(predictions.BatteryID == battery) & (predictions.PhysicsMode == "none")]
    physics = predictions[(predictions.BatteryID == battery) & (predictions.PhysicsMode == "both")]
    bootstrap_rows.append({"BatteryID": battery, **block_bootstrap_delta(original, physics)})
pd.DataFrame(bootstrap_rows).to_csv(
    OUT / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig"
)

projection_rows = []
projection_base = (
    BASE / "B0034" / "seed_42" / "width_16" / "guard_seasonal_residual"
    / "anchor_soh_ridge_residual" / "fusion_iprf" / "physics_none_lambda_0"
)
for projection in ["cummin", "soft_cummin_025"]:
    row = pd.read_csv(projection_base / f"projection_{projection}" / "metrics.csv").iloc[0].to_dict()
    row["TrendProjection"] = projection
    projection_rows.append(row)
pd.DataFrame(projection_rows).to_csv(
    OUT / "projection_diagnostics.csv", index=False, encoding="utf-8-sig"
)

artifact_index = []
selected_runs = [
    ("B0034", "none", 0.0),
    ("B0034", "monotonic", 0.001),
    ("B0034", "smooth", 0.001),
    ("B0034", "both", 0.001),
    ("B0034", "both", 0.01),
    ("B0034", "both", 0.1),
    ("B0036", "none", 0.0),
    ("B0036", "both", 0.001),
    ("B0018", "none", 0.0),
    ("B0018", "both", 0.001),
]
for battery, mode, lam in selected_runs:
    source = run_path(battery, mode, lam)
    prefix = f"{battery}_{mode}_lambda_{lam:g}"
    for filename in [
        "config.json", "trend_training_log.csv", "seasonal_training_log.csv",
        "residual_training_log.csv", "component_diagnostics.csv",
    ]:
        source_file = source / filename
        destination_dir = CONFIGS if filename == "config.json" else LOGS
        destination = destination_dir / f"{prefix}_{filename}"
        shutil.copy2(source_file, destination)
        artifact_index.append({
            "BatteryID": battery, "PhysicsMode": mode, "LambdaPhy": lam,
            "Artifact": filename, "CopiedPath": str(destination.relative_to(ROOT)),
            "SourcePath": str(source_file.relative_to(ROOT)),
        })
pd.DataFrame(artifact_index).to_csv(LOGS / "ARTIFACT_INDEX.csv", index=False, encoding="utf-8-sig")

lock = {
    "selected_on": "B0034 development evaluation",
    "selected_physics_mode": "both",
    "selected_lambda_phy": 0.001,
    "architecture": "SOH-scale trainable Ridge anchor + original TimesNet residual; Seasonal lightweight TimesNet; Residual iTransformer; bounded Attention Bi-LSTM fusion",
    "component_guard": "Seasonal/Residual first-prefix range plus 5%",
    "physics_scope": "STL Trend only",
    "trajectory_rule": "same BatteryID/DatasetID with continuous Cycle only",
    "test_tuning": False,
    "confirmation_status": "failed on contacted B0018; B0034/B0036 bootstrap intervals cross zero",
}
canonical = json.dumps(lock, ensure_ascii=False, sort_keys=True).encode("utf-8")
lock["config_sha256"] = hashlib.sha256(canonical).hexdigest()
(CONFIGS / "physics_v3_lock.json").write_text(
    json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axes[0].bar(ablation.PhysicsMode, ablation.RMSE, color=["#888888", "#4c72b0", "#dd8452", "#2a9d8f"])
axes[0].set_title("B0034 ablation (lambda=0.001)")
axes[0].set_ylabel("RMSE")
axes[0].grid(axis="y", alpha=0.25)
for index, value in enumerate(ablation.RMSE):
    axes[0].text(index, value + 0.04, f"{value:.3f}", ha="center", fontsize=8)

axes[1].plot(lambda_table.LambdaPhy, lambda_table.RMSE, "o-", label="RMSE")
axes[1].set_xscale("symlog", linthresh=0.0005)
axes[1].set_title("B0034 both-mode sensitivity")
axes[1].set_xlabel("lambda_phy")
axes[1].set_ylabel("RMSE")
axes[1].grid(alpha=0.25)

x = np.arange(3); width = 0.34
ordered = ["B0034", "B0036", "B0018"]
original_values = [paired_metrics[(paired_metrics.BatteryID == b) & (paired_metrics.PhysicsMode == "none")].RMSE.iloc[0] for b in ordered]
physics_values = [paired_metrics[(paired_metrics.BatteryID == b) & (paired_metrics.PhysicsMode == "both")].RMSE.iloc[0] for b in ordered]
axes[2].bar(x - width / 2, original_values, width, label="Original", color="#888888")
axes[2].bar(x + width / 2, physics_values, width, label="both, lambda=0.001", color="#2a9d8f")
axes[2].set_xticks(x, ordered)
axes[2].set_title("Frozen paired evaluation")
axes[2].set_ylabel("RMSE")
axes[2].legend(frameon=False, fontsize=8)
axes[2].grid(axis="y", alpha=0.25)

fig.suptitle("Physics-guided Trend v3: small development gains do not generalize", y=1.02)
fig.tight_layout()
figure_path = FIGURES / "physics_trend_v3_summary.png"
fig.savefig(figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)
print(ablation[["PhysicsMode", "LambdaPhy", "RMSE", "R2", "TrendViolationRate"]].to_string(index=False))
print(pd.DataFrame(bootstrap_rows).to_string(index=False))
print(figure_path)
