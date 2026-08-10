"""Recompute and audit the frozen five-seed CILA outer test from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 52, 62, 72, 82]
MODES = {"original": "Original", "cila": "CILA"}


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "MAE": float(mean_absolute_error(true, pred)),
        "MAPE": float(np.abs(true - pred).sum() / np.maximum(np.abs(true).sum(), 1e-12) * 100),
        "R2": float(r2_score(true, pred)),
    }


def circular_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    pieces: list[np.ndarray] = []
    while sum(len(item) for item in pieces) < n:
        start = int(rng.integers(0, n))
        pieces.append((start + np.arange(block)) % n)
    return np.concatenate(pieces)[:n]


def bootstrap(joined: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260810)
    rows = []
    for battery, battery_frame in joined.groupby("BatteryID", sort=True):
        groups = [g.sort_values("Cycle").reset_index(drop=True) for _, g in battery_frame.groupby("Seed")]
        observed = []
        for group in groups:
            observed.append(metrics(group.True_SOH.to_numpy(), group.Pred_SOH_CILA.to_numpy())["RMSE"]
                            - metrics(group.True_SOH.to_numpy(), group.Pred_SOH_Original.to_numpy())["RMSE"])
        draws = []
        for _ in range(replicates):
            seed_deltas = []
            for group in groups:
                idx = circular_indices(len(group), max(3, round(np.sqrt(len(group)))), rng)
                sample = group.iloc[idx]
                seed_deltas.append(metrics(sample.True_SOH.to_numpy(), sample.Pred_SOH_CILA.to_numpy())["RMSE"]
                                   - metrics(sample.True_SOH.to_numpy(), sample.Pred_SOH_Original.to_numpy())["RMSE"])
            draws.append(float(np.median(seed_deltas)))
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "BatteryID": battery,
            "Statistic": "median_seed_paired_RMSE_delta_CILA_minus_Original",
            "Observed": float(np.median(observed)),
            "CI95Lower": float(low),
            "CI95Upper": float(high),
            "Replicates": replicates,
            "BlockRule": "circular; max(3,round(sqrt(n_cycles)))",
            "CIExcludesZeroInImprovementDirection": bool(high < 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "output" / "ieee_innovation_v1" / "phase4" / "formal_outer_test"))
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    base = Path(args.root)
    metric_rows = []
    prediction_frames = []
    recomputation_rows = []
    for seed in SEEDS:
        for directory, label in MODES.items():
            run = base / f"seed_{seed}" / directory
            reported = pd.read_csv(run / "metrics.csv")
            if set(reported.EvaluatedSplit) != {"validation", "test"}:
                raise RuntimeError(f"unexpected split set in {run}")
            prediction = pd.read_csv(run / "per_cycle_predictions_test.csv")
            prediction.insert(0, "Mode", label)
            prediction.insert(0, "Seed", seed)
            for battery, frame in prediction.groupby("BatteryID"):
                recalculated = metrics(frame.True_SOH.to_numpy(), frame.Pred_SOH.to_numpy())
                reported_row = reported[(reported.BatteryID == battery) & (reported.EvaluatedSplit == "test")].iloc[0]
                for name, value in recalculated.items():
                    reported_value = float(reported_row[name])
                    difference = abs(value - reported_value)
                    passed = bool(np.isclose(value, reported_value, atol=1e-4, rtol=1e-6))
                    recomputation_rows.append({"Seed": seed, "Mode": label, "BatteryID": battery,
                                               "Metric": name, "Reported": reported_value,
                                               "Recomputed": value, "AbsoluteDifference": difference,
                                               "Tolerance": "atol=1e-4,rtol=1e-6", "Pass": passed})
                    if not passed:
                        raise RuntimeError(f"{seed}/{label}/{battery}/{name} recomputation mismatch")
                metric_rows.append({"Seed": seed, "Mode": label, "BatteryID": battery, **recalculated,
                                    "ConfigHash": str(reported_row.ConfigHash)})
            prediction_frames.append(prediction)
    all_metrics = pd.DataFrame(metric_rows).sort_values(["BatteryID", "Seed", "Mode"])
    all_predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["BatteryID", "Seed", "Mode", "Cycle"])
    pd.DataFrame(recomputation_rows).to_csv(base / "metric_recomputation_audit.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(base / "formal_outer_metrics.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(base / "formal_outer_per_cycle_predictions.csv", index=False, encoding="utf-8-sig")

    pivot = all_metrics.pivot(index=["Seed", "BatteryID"], columns="Mode", values=["RMSE", "MAE", "MAPE", "R2"])
    rows = []
    for (seed, battery), values in pivot.iterrows():
        rows.append({"Seed": seed, "BatteryID": battery,
                     "OriginalRMSE": values[("RMSE", "Original")], "CILARMSE": values[("RMSE", "CILA")],
                     "DeltaRMSE": values[("RMSE", "CILA")] - values[("RMSE", "Original")],
                     "DeltaMAE": values[("MAE", "CILA")] - values[("MAE", "Original")],
                     "DeltaMAPE": values[("MAPE", "CILA")] - values[("MAPE", "Original")],
                     "DeltaR2": values[("R2", "CILA")] - values[("R2", "Original")]})
    paired = pd.DataFrame(rows).sort_values(["BatteryID", "Seed"])
    paired.to_csv(base / "formal_outer_paired_deltas.csv", index=False, encoding="utf-8-sig")

    original = all_predictions[all_predictions.Mode == "Original"].rename(columns={"Pred_SOH": "Pred_SOH_Original"})
    cila = all_predictions[all_predictions.Mode == "CILA"].rename(columns={"Pred_SOH": "Pred_SOH_CILA"})
    joined = original[["Seed", "BatteryID", "Cycle", "True_SOH", "Pred_SOH_Original"]].merge(
        cila[["Seed", "BatteryID", "Cycle", "True_SOH", "Pred_SOH_CILA"]],
        on=["Seed", "BatteryID", "Cycle"], suffixes=("_Original", "_CILA"), validate="one_to_one")
    if not np.allclose(joined.True_SOH_Original, joined.True_SOH_CILA, atol=0, rtol=0):
        raise RuntimeError("paired truth labels differ")
    joined = joined.rename(columns={"True_SOH_Original": "True_SOH"}).drop(columns="True_SOH_CILA")
    ci = bootstrap(joined, args.replicates)
    ci.to_csv(base / "formal_outer_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    summary = paired.groupby("BatteryID").agg(
        OriginalMedianRMSE=("OriginalRMSE", "median"), CILAMedianRMSE=("CILARMSE", "median"),
        MedianDeltaRMSE=("DeltaRMSE", "median"), MeanDeltaRMSE=("DeltaRMSE", "mean"),
        ImprovedSeeds=("DeltaRMSE", lambda x: int((x < 0).sum())), SeedCount=("Seed", "count"),
        WorstDeltaRMSE=("DeltaRMSE", "max"), BestDeltaRMSE=("DeltaRMSE", "min"),
    ).reset_index().merge(ci[["BatteryID", "CI95Lower", "CI95Upper", "CIExcludesZeroInImprovementDirection"]], on="BatteryID")
    summary["BatteryGatePass"] = ((summary.CILAMedianRMSE < summary.OriginalMedianRMSE)
                                   & (summary.ImprovedSeeds >= 3)
                                   & summary.CIExcludesZeroInImprovementDirection)
    summary.to_csv(base / "formal_outer_summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "config_hash": "797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10",
        "seeds": SEEDS,
        "metrics_recomputed_from_per_cycle": True,
        "paired_truth_identical": True,
        "test_used_for_tuning": False,
        "all_battery_gates_pass": bool(summary.BatteryGatePass.all()),
        "PhysicsInnovationSupported": False,
        "CILAInnovationSupported": bool(summary.BatteryGatePass.all()),
        "claim_if_false": "outer confirmation failed; preserve as negative generalization evidence",
    }
    (base / "formal_outer_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
