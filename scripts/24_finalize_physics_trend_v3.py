"""Consolidate physics_trend_v3, bootstrap paired deltas, and render reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "physics_trend_v3"
FIG = ROOT / "output" / "figures"
SEEDS = [42, 52, 62, 72, 82]
MODES = ["none", "monotonic", "smooth", "both"]


def block_indices(length, block, rng):
    selected = []
    while len(selected) < length:
        start = int(rng.integers(0, max(length - block + 1, 1)))
        selected.extend(range(start, min(start + block, length)))
    return np.asarray(selected[:length], int)


def bootstrap(predictions, battery, mode, iterations=2000, block=5):
    rng = np.random.default_rng(20260810)
    samples = []
    source = predictions[predictions.BatteryID.eq(battery)]
    for _ in range(iterations):
        seed_deltas = []
        for seed in SEEDS:
            base = source[source.Seed.eq(seed) & source.PhysicsMode.eq("none")].sort_values("Cycle")
            proposed = source[source.Seed.eq(seed) & source.PhysicsMode.eq(mode)].sort_values("Cycle")
            if not np.array_equal(base.Cycle.to_numpy(), proposed.Cycle.to_numpy()):
                raise RuntimeError("bootstrap配对Cycle不一致")
            indices = block_indices(len(base), block, rng)
            truth = base.TrueSOH.to_numpy()[indices]
            original_rmse = np.sqrt(np.mean((truth - base.PredSOH.to_numpy()[indices]) ** 2))
            proposed_rmse = np.sqrt(np.mean((truth - proposed.PredSOH.to_numpy()[indices]) ** 2))
            seed_deltas.append(proposed_rmse - original_rmse)
        samples.append(np.mean(seed_deltas))
    values = np.asarray(samples)
    return float(values.mean()), float(np.quantile(values, .025)), float(np.quantile(values, .975)), float(np.mean(values < 0))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    metric_frames, prediction_frames = [], []
    log_rows = []
    for seed in SEEDS:
        seed_dir = OUT / "formal_test" / f"seed_{seed}"
        metrics = pd.read_csv(seed_dir / "physics_ablation_metrics.csv"); metrics["Seed"] = seed
        predictions = pd.read_csv(seed_dir / "physics_ablation_predictions.csv"); predictions["Seed"] = seed
        metric_frames.append(metrics); prediction_frames.append(predictions)
        for file in seed_dir.rglob("*_train_log.csv"):
            log_rows.append({"Seed": seed, "Path": str(file.relative_to(OUT)).replace("\\", "/")})
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["DatasetID"] = "raw"
    metrics.to_csv(OUT / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    predictions[["BatteryID", "DatasetID", "Seed", "PhysicsMode", "Cycle", "TrueTrend", "PredTrend"]].to_csv(
        OUT / "trend_predictions.csv", index=False, encoding="utf-8-sig"
    )
    logs = OUT / "training_logs"; logs.mkdir(exist_ok=True)
    pd.DataFrame(log_rows).to_csv(logs / "log_index.csv", index=False, encoding="utf-8-sig")

    bootstrap_rows, criteria_rows = [], []
    for battery in ["B0007", "B0018"]:
        base = metrics[metrics.BatteryID.eq(battery) & metrics.PhysicsMode.eq("none")].set_index("Seed")
        for mode in MODES[1:]:
            proposed = metrics[metrics.BatteryID.eq(battery) & metrics.PhysicsMode.eq(mode)].set_index("Seed")
            mean_delta, low, high, probability = bootstrap(predictions, battery, mode)
            improves = int((proposed.RMSE < base.RMSE).sum())
            median_delta = float((proposed.RMSE - base.RMSE).median())
            violation_delta = float(proposed.TrendViolationRate.mean() - base.TrendViolationRate.mean())
            bootstrap_rows.append({"BatteryID": battery, "PhysicsMode": mode, "PairedMeanRMSEDelta": mean_delta,
                                   "CI95Low": low, "CI95High": high, "ProbabilityImprovement": probability,
                                   "ImprovedSeeds": improves, "TotalSeeds": 5, "PairedMedianRMSEDelta": median_delta,
                                   "MeanTrendViolationRateDelta": violation_delta})
            criteria_rows.append({"BatteryID": battery, "PhysicsMode": mode,
                                  "MedianRMSELower": proposed.RMSE.median() < base.RMSE.median(),
                                  "AtLeast3of5SeedsImprove": improves >= 3,
                                  "TrendViolationLower": proposed.TrendViolationRate.mean() < base.TrendViolationRate.mean(),
                                  "BootstrapCIEntirelyBelowZero": high < 0,
                                  "MAENotWorse5Percent": proposed.MAE.mean() <= 1.05 * base.MAE.mean(),
                                  "R2NotMateriallyWorse": proposed.R2.mean() >= base.R2.mean() - .03})
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    criteria = pd.DataFrame(criteria_rows)
    criteria.to_csv(OUT / "success_criteria.csv", index=False, encoding="utf-8-sig")
    supported_modes = []
    for mode in MODES[1:]:
        q = criteria[criteria.PhysicsMode.eq(mode)]
        if len(q) == 2 and q.drop(columns=["BatteryID", "PhysicsMode"]).all(axis=None):
            supported_modes.append(mode)
    supported = bool(supported_modes)

    summary = metrics.groupby(["BatteryID", "PhysicsMode"])[["RMSE", "MAE", "MAPE", "R2", "TrendViolationRate"]].agg(["mean", "std", "median"])
    summary.to_csv(OUT / "ablation_summary.csv", encoding="utf-8-sig")

    # Ablation RMSE figure.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        q = metrics[metrics.BatteryID.eq(battery)]
        values = [q[q.PhysicsMode.eq(mode)].RMSE.to_numpy() for mode in MODES]
        ax.boxplot(values, tick_labels=["Original", "mono", "smooth", "both"], showmeans=True)
        ax.set_title(battery); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Test RMSE"); fig.tight_layout()
    fig.savefig(FIG / "physics_ablation_rmse_physics_v3.png", dpi=180); plt.close(fig)

    # Representative Trend and violation markers, using seed nearest mode median.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        for mode in MODES:
            q = metrics[metrics.BatteryID.eq(battery) & metrics.PhysicsMode.eq(mode)]
            seed = int(q.iloc[np.argmin(np.abs(q.RMSE - q.RMSE.median()))].Seed)
            curve = predictions[predictions.BatteryID.eq(battery) & predictions.Seed.eq(seed) & predictions.PhysicsMode.eq(mode)].sort_values("Cycle")
            ax.plot(curve.Cycle, curve.PredTrend, label=f"{mode}, seed {seed}")
            if mode == "both":
                delta = np.diff(curve.PredTrend.to_numpy()); bad = np.where(delta > 0)[0] + 1
                ax.scatter(curve.Cycle.to_numpy()[bad], curve.PredTrend.to_numpy()[bad], c="red", s=18, marker="x")
        ax.set_title(f"{battery} Trend predictions (red x: positive jump for both)"); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Cycle"); fig.tight_layout()
    fig.savefig(FIG / "trend_violation_physics_v3.png", dpi=180); plt.close(fig)

    sensitivity = pd.read_csv(OUT / "lambda_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        q = sensitivity[sensitivity.BatteryID.eq(battery) & sensitivity.PhysicsMode.ne("none")]
        for mode in MODES[1:]:
            z = q[q.PhysicsMode.eq(mode)].sort_values("LambdaPhy")
            ax.plot(z.LambdaPhy, z.RMSE, marker="o", label=mode)
        ax.set_xscale("log"); ax.set_title(battery); ax.set_xlabel("lambda_phy"); ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel("Validation RMSE"); fig.tight_layout()
    fig.savefig(FIG / "lambda_sensitivity_physics_v3.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5)); labels, centers, lows, highs = [], [], [], []
    for _, row in bootstrap_frame.iterrows():
        labels.append(f"{row.BatteryID}-{row.PhysicsMode}"); centers.append(row.PairedMeanRMSEDelta)
        lows.append(row.PairedMeanRMSEDelta-row.CI95Low); highs.append(row.CI95High-row.PairedMeanRMSEDelta)
    y = np.arange(len(labels)); ax.errorbar(centers, y, xerr=[lows, highs], fmt="o", capsize=4)
    ax.axvline(0, color="red", linestyle="--"); ax.set_yticks(y, labels); ax.set_xlabel("Paired RMSE delta (physics - Original)")
    ax.set_title("Time-block bootstrap 95% CI"); ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "bootstrap_ci_physics_v3.png", dpi=180); plt.close(fig)

    lines = ["# Physics-guided Trend constraint报告（physics_trend_v3）", "",
             f"- PhysicsInnovationSupported: **{str(supported).lower()}**", "- λ=0.1仅由验证集选择；正式测试后未调参。",
             "- 物理损失仅作用于STL Trend；未约束最终SOH。", "",
             "## 五随机种子正式结果", "",
             "| Battery | Mode | RMSE mean±std | RMSE median | Trend violation mean | improved seeds |", "|---|---|---:|---:|---:|---:|"]
    for battery in ["B0007", "B0018"]:
        base = metrics[metrics.BatteryID.eq(battery) & metrics.PhysicsMode.eq("none")].set_index("Seed")
        for mode in MODES:
            q = metrics[metrics.BatteryID.eq(battery) & metrics.PhysicsMode.eq(mode)]
            improved = "-" if mode == "none" else f"{int((q.set_index('Seed').RMSE < base.RMSE).sum())}/5"
            lines.append(f"| {battery} | {mode} | {q.RMSE.mean():.4f}±{q.RMSE.std():.4f} | {q.RMSE.median():.4f} | {q.TrendViolationRate.mean():.4f} | {improved} |")
    lines += ["", "## 结论", "",
              "没有任何单一physics_mode同时满足两电池中位RMSE下降、各至少3/5种子改善、Trend违规率下降且bootstrap区间完全低于0。",
              "因此不能声称物理约束稳定降低RMSE。部分模式在B0007或个别种子显著改善，但B0018改善次数不足，monotonic还出现B0007种子82严重失败。",
              "该结果可作为探索性/负结果写入论文，但不足以支持强创新有效性结论；确认性结论还需要未接触外部电池。"]
    (OUT / "PHYSICS_TREND_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    state = {"PhysicsInnovationSupported": supported, "supported_modes": supported_modes,
             "selected_lambda_phy": .1, "test_used_for_selection": False,
             "formal_test_completed": True, "baseline_config_hash": "c9b526cab0f080aaf480c10798ca8c234d022b34e566d9ba2b6ba3e0748b670a"}
    (OUT / "physics_v3_status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
