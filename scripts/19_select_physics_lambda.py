"""Select lambda from validation-only multi-seed physics experiments."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir", default=str(ROOT / "output" / "physics_trend_v2" / "selection")
    )
    parser.add_argument("--rmse-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    root = Path(args.selection_dir)
    files = sorted(root.glob("lambda_*/seed_*/physics_validation_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"未找到验证指标: {root}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if not (frame["EvaluatedSplit"] == "validation").all():
            raise RuntimeError(f"选择目录混入非验证指标: {path}")
        if "test" in str(path).lower():
            raise RuntimeError(f"选择路径疑似包含测试结果: {path}")
        frame = frame[frame["PhysicsMode"] == "monotonic"].copy()
        frame["Seed"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(root / "lambda_validation_raw.csv", index=False, encoding="utf-8-sig")

    summary = raw.groupby("LambdaPhy", as_index=False).agg(
        ValidationRMSEMean=("RMSE", "mean"),
        ValidationRMSEStd=("RMSE", "std"),
        StrictViolationRateMean=("TrendViolationRate", "mean"),
        ToleranceViolationRateMean=("TrendToleranceViolationRate", "mean"),
        PositiveVariationMean=("TrendTotalPositiveVariation", "mean"),
        MaxPositiveJumpMean=("TrendMaxPositiveJump", "mean"),
        Runs=("RMSE", "size"),
    )
    minimum_rmse = float(summary["ValidationRMSEMean"].min())
    summary["RMSEEligible"] = (
        summary["ValidationRMSEMean"] <= minimum_rmse * (1.0 + args.rmse_tolerance)
    )
    candidates = summary[summary["RMSEEligible"]].sort_values(
        [
            "StrictViolationRateMean",
            "ToleranceViolationRateMean",
            "PositiveVariationMean",
            "ValidationRMSEMean",
            "LambdaPhy",
        ]
    )
    selected = float(candidates.iloc[0]["LambdaPhy"])
    summary["Selected"] = summary["LambdaPhy"].eq(selected)
    summary.to_csv(root / "lambda_validation_summary.csv", index=False, encoding="utf-8-sig")

    record = {
        "selected_lambda": selected,
        "selection_split": "validation-only",
        "selection_seeds": sorted(raw["Seed"].unique().astype(int).tolist()),
        "batteries": sorted(raw["BatteryID"].unique().tolist()),
        "rmse_eligibility_tolerance": args.rmse_tolerance,
        "selection_order": [
            "strict_violation_rate",
            "tolerance_violation_rate",
            "positive_variation",
            "validation_rmse",
        ],
        "test_results_read": False,
    }
    (root / "selected_lambda.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
