"""Aggregate CILA development-validation runs without reading test outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cila_validation_module", ROOT / "scripts" / "30_cila_fusion.py")
cila = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cila
SPEC.loader.exec_module(cila)


def collect_run_dirs(phase4: Path) -> list[Path]:
    paths = []
    for parent in [phase4 / "three_seed_validation", phase4 / "five_seed_validation"]:
        if parent.exists():
            paths.extend(path.parent for path in parent.glob("seed_*/*/metrics.csv"))
    unique = sorted(set(paths))
    if len(unique) != 10:
        raise RuntimeError(f"expected 10 five-seed mode runs, found {len(unique)}")
    return unique


def recompute_rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(mean_squared_error(frame["True_SOH"], frame["Pred_SOH"])))


def extract_context_diagnostic(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    domains = {name: Path(path) for name, path in config["domains"].items()}
    cila.fusion.LAMBDA_WEIGHT = float(config["lambda_weight"])
    data = cila.build_pooled_data(domains, include_test=False)
    model = cila.fusion.AttentionBiLSTMFusion(
        data.comp_mean, data.comp_std, data.y_mean, data.y_std,
        fusion_mode="paper", stable_init=bool(config["stable_init"]),
    ).to(cila.fusion.device)
    model.load_state_dict(torch.load(run_dir / "fusion_att_bilstm_cila.pth", map_location=cila.fusion.device))
    model.eval()

    def contexts(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        values, domains_out = [], []
        with torch.no_grad():
            for x_seq, _, _, domain_ids, _ in loader:
                values.append(model.encode_context(x_seq.to(cila.fusion.device)).cpu().numpy())
                domains_out.append(domain_ids.numpy())
        return np.concatenate(values), np.concatenate(domains_out)

    x_train, y_train = contexts(data.train)
    x_val, y_val = contexts(data.validation)
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(max_iter=1000, random_state=int(config["seed"])).fit(scaler.transform(x_train), y_train)
    prediction = classifier.predict(scaler.transform(x_val))
    return {
        "Seed": int(config["seed"]),
        "CILAMode": config["mode"],
        "LambdaAlign": float(config["lambda_align"]),
        "ValidationDomainBalancedAccuracy": float(balanced_accuracy_score(y_val, prediction)),
        "ChanceLevel": 0.5,
        "InterpretationBoundary": "BatteryID separability diagnostic; not a causal condition metric",
    }


def circular_block_sample(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    while sum(len(piece) for piece in pieces) < length:
        start = int(rng.integers(0, length))
        pieces.append((start + np.arange(block_length)) % length)
    return np.concatenate(pieces)[:length]


def block_bootstrap(paired_predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for battery, battery_frame in paired_predictions.groupby("BatteryID"):
        groups = [group.sort_values("Cycle").reset_index(drop=True) for _, group in battery_frame.groupby("Seed")]
        observed_seed_deltas = []
        for group in groups:
            observed_seed_deltas.append(
                np.sqrt(np.mean((group.True_SOH - group.Pred_SOH_coral) ** 2))
                - np.sqrt(np.mean((group.True_SOH - group.Pred_SOH_none) ** 2))
            )
        draws = []
        for _ in range(replicates):
            seed_deltas = []
            for group in groups:
                block_length = max(3, int(round(np.sqrt(len(group)))))
                indices = circular_block_sample(len(group), block_length, rng)
                sample = group.iloc[indices]
                seed_deltas.append(
                    np.sqrt(np.mean((sample.True_SOH - sample.Pred_SOH_coral) ** 2))
                    - np.sqrt(np.mean((sample.True_SOH - sample.Pred_SOH_none) ** 2))
                )
            draws.append(float(np.median(seed_deltas)))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "BatteryID": battery,
            "Statistic": "median_seed_paired_RMSE_delta_CILA_minus_Original",
            "Observed": float(np.median(observed_seed_deltas)),
            "CI95Lower": float(lower),
            "CI95Upper": float(upper),
            "Replicates": replicates,
            "BlockRule": "circular blocks; length=max(3,round(sqrt(n_cycles)))",
            "CIExcludesZeroInImprovementDirection": bool(upper < 0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-dir", default=str(ROOT / "output" / "ieee_innovation_v1" / "phase4"))
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    phase4 = Path(args.phase4_dir)
    run_dirs = collect_run_dirs(phase4)

    metric_frames = []
    prediction_frames = []
    diagnostics = []
    for run_dir in run_dirs:
        metrics = pd.read_csv(run_dir / "metrics.csv")
        if set(metrics["EvaluatedSplit"]) != {"validation"}:
            raise RuntimeError(f"non-validation metric detected in {run_dir}")
        predictions = pd.read_csv(run_dir / "per_cycle_predictions_validation.csv")
        mode = str(metrics.iloc[0].CILAMode)
        seed = int(metrics.iloc[0].Seed)
        predictions.insert(0, "CILAMode", mode)
        predictions.insert(0, "Seed", seed)
        for _, row in metrics.iterrows():
            frame = predictions[predictions.BatteryID == row.BatteryID]
            difference = abs(recompute_rmse(frame) - float(row.RMSE))
            if difference > 1e-5:
                raise RuntimeError(f"RMSE recomputation mismatch {difference} in {run_dir}")
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        diagnostics.append(extract_context_diagnostic(run_dir))

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["Seed", "BatteryID", "CILAMode"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(phase4 / "five_seed_validation_metrics.csv", index=False)
    pd.DataFrame(diagnostics).sort_values(["Seed", "CILAMode"]).to_csv(phase4 / "domain_separability.csv", index=False)

    pivot = metrics.pivot(index=["Seed", "BatteryID"], columns="CILAMode", values=["RMSE", "MAE", "MAPE", "R2"])
    paired_rows = []
    for (seed, battery), row in pivot.iterrows():
        paired_rows.append({"Seed": seed, "BatteryID": battery,
                            "OriginalRMSE": row[("RMSE", "none")], "CILARMSE": row[("RMSE", "coral")],
                            "DeltaRMSE": row[("RMSE", "coral")] - row[("RMSE", "none")],
                            "DeltaMAE": row[("MAE", "coral")] - row[("MAE", "none")],
                            "DeltaMAPE": row[("MAPE", "coral")] - row[("MAPE", "none")],
                            "DeltaR2": row[("R2", "coral")] - row[("R2", "none")]})
    paired = pd.DataFrame(paired_rows).sort_values(["BatteryID", "Seed"])
    paired.to_csv(phase4 / "paired_validation_deltas.csv", index=False)

    none_pred = predictions[predictions.CILAMode == "none"].rename(columns={"Pred_SOH": "Pred_SOH_none"})
    coral_pred = predictions[predictions.CILAMode == "coral"].rename(columns={"Pred_SOH": "Pred_SOH_coral"})
    joined = none_pred[["Seed", "BatteryID", "Cycle", "True_SOH", "Pred_SOH_none"]].merge(
        coral_pred[["Seed", "BatteryID", "Cycle", "True_SOH", "Pred_SOH_coral"]],
        on=["Seed", "BatteryID", "Cycle"], suffixes=("_none", "_coral"), validate="one_to_one")
    if not np.allclose(joined.True_SOH_none, joined.True_SOH_coral):
        raise RuntimeError("paired true labels differ")
    joined = joined.rename(columns={"True_SOH_none": "True_SOH"}).drop(columns=["True_SOH_coral"])
    bootstrap = block_bootstrap(joined, args.replicates, 20260810)
    bootstrap.to_csv(phase4 / "bootstrap_confidence_intervals.csv", index=False)

    summary = paired.groupby("BatteryID").agg(
        OriginalMedianRMSE=("OriginalRMSE", "median"), CILAMedianRMSE=("CILARMSE", "median"),
        MedianDeltaRMSE=("DeltaRMSE", "median"), MeanDeltaRMSE=("DeltaRMSE", "mean"),
        ImprovedSeeds=("DeltaRMSE", lambda values: int((values < 0).sum())), SeedCount=("Seed", "count"),
        WorstDeltaRMSE=("DeltaRMSE", "max"),
    ).reset_index()
    summary = summary.merge(bootstrap[["BatteryID", "CI95Lower", "CI95Upper", "CIExcludesZeroInImprovementDirection"]], on="BatteryID")
    summary.to_csv(phase4 / "five_seed_validation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
