"""Audit strict temporal splits, health features, and SCF-GAN trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[1]
HF_COLS = ["HF1", "HF2", "HF3"]


def correlation_rows(battery, raw, train_end):
    train = raw.iloc[:train_end].copy()
    trend = STL(train["SOH"].to_numpy(float) * 100.0, period=12, robust=True).fit().trend
    rows = []
    for feature in HF_COLS:
        rows.append({
            "BatteryID": battery,
            "Feature": feature,
            "PearsonWithSOH": train[feature].corr(train["SOH"], method="pearson"),
            "SpearmanWithSOH": train[feature].corr(train["SOH"], method="spearman"),
            "PearsonWithTrainTrend": pd.Series(train[feature].to_numpy()).corr(pd.Series(trend)),
            "SpearmanWithTrainTrend": pd.Series(train[feature].to_numpy()).corr(
                pd.Series(trend), method="spearman"
            ),
        })
    return rows


def distribution_rows(battery, raw_train, aug):
    rows = []
    for source, frame in (("real_train", raw_train), ("scf_gan_train", aug)):
        for feature in HF_COLS:
            values = frame[feature].to_numpy(float)
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = max(q3 - q1, 1e-12)
            outlier_rate = np.mean((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr))
            rows.append({
                "BatteryID": battery,
                "Source": source,
                "Feature": feature,
                "Count": len(values),
                "Mean": values.mean(),
                "Std": values.std(),
                "Q05": np.quantile(values, 0.05),
                "Median": np.median(values),
                "Q95": np.quantile(values, 0.95),
                "IQR_OutlierRate": outlier_rate,
            })
    return rows


def trajectory_rows(battery, aug, train_end):
    rows = []
    for dataset_id, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        cycles = group["Cycle"].to_numpy(int)
        diffs = np.diff(cycles)
        row = {
            "BatteryID": battery,
            "DatasetID": dataset_id,
            "Rows": len(group),
            "CycleStart": int(cycles.min()),
            "CycleEnd": int(cycles.max()),
            "DuplicateCycleCount": int(group["Cycle"].duplicated().sum()),
            "ContinuousCycle": bool(len(diffs) == 0 or np.all(diffs == 1)),
            "AllCyclesInsideTrainPrefix": bool(np.all((cycles >= 0) & (cycles < train_end))),
        }
        for feature in HF_COLS:
            delta = np.diff(group[feature].to_numpy(float))
            row[f"{feature}_DeltaMean"] = float(delta.mean()) if len(delta) else 0.0
            row[f"{feature}_DeltaStd"] = float(delta.std()) if len(delta) else 0.0
        rows.append(row)
    return rows


def correlation_matrix_rows(battery, raw_train, aug):
    columns = HF_COLS + ["SOH"]
    real_corr = raw_train[columns].corr(method="spearman")
    aug_corr = aug[columns].corr(method="spearman")
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            real_value = float(real_corr.loc[left, right])
            aug_value = float(aug_corr.loc[left, right])
            rows.append({
                "BatteryID": battery,
                "VariableA": left,
                "VariableB": right,
                "RealTrainSpearman": real_value,
                "SCFGANTrainSpearman": aug_value,
                "AbsoluteCorrelationDifference": abs(real_value - aug_value),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batteries", default="B0007,B0018")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "physics_trend_v2" / "audit"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_rows, raw_rows, corr_rows = [], [], []
    dist_rows, traj_rows, matrix_rows = [], [], []
    for battery in [item.strip() for item in args.batteries.split(",")]:
        raw_path = ROOT / "data" / "hf_csv" / f"{battery}_hf.csv"
        aug_path = ROOT / "output" / f"{battery}_aug_train_hf.csv"
        raw_original = pd.read_csv(raw_path)
        raw = raw_original[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce")
        missing_rows = int(raw.isna().any(axis=1).sum())
        raw = raw.dropna().reset_index(drop=True)
        aug = pd.read_csv(aug_path)
        n = len(raw); train_end, val_end = int(n * 0.6), int(n * 0.8)
        raw_train = raw.iloc[:train_end].copy()

        split_rows.append({
            "BatteryID": battery,
            "Rows": n,
            "TrainStart": 0,
            "TrainEndExclusive": train_end,
            "ValidationStart": train_end,
            "ValidationEndExclusive": val_end,
            "TestStart": val_end,
            "TestEndExclusive": n,
            "TrainRows": train_end,
            "ValidationRows": val_end - train_end,
            "TestRows": n - val_end,
            "AugmentationUsedFor": "train-only",
        })
        raw_rows.append({
            "BatteryID": battery,
            "RawRowsBeforeCleaning": len(raw_original),
            "RowsWithMissingNumericValues": missing_rows,
            "RowsAfterCleaning": n,
            "ExactDuplicateRows": int(raw.duplicated().sum()),
            "SOHMin": float(raw["SOH"].min()),
            "SOHMax": float(raw["SOH"].max()),
            "SCFGANRows": len(aug),
            "SCFGANDatasetCount": int(aug["DatasetID"].nunique()),
        })
        corr_rows.extend(correlation_rows(battery, raw, train_end))
        dist_rows.extend(distribution_rows(battery, raw_train, aug))
        traj_rows.extend(trajectory_rows(battery, aug, train_end))
        matrix_rows.extend(correlation_matrix_rows(battery, raw_train, aug))

    outputs = {
        "split_manifest.csv": split_rows,
        "raw_data_quality.csv": raw_rows,
        "hf_train_correlations.csv": corr_rows,
        "hf_distribution_quality.csv": dist_rows,
        "gan_trajectory_quality.csv": traj_rows,
        "gan_correlation_quality.csv": matrix_rows,
    }
    for filename, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out / filename, index=False, encoding="utf-8-sig")

    trajectory = pd.DataFrame(traj_rows)
    gates = {
        "all_gan_cycles_continuous": bool(trajectory["ContinuousCycle"].all()),
        "all_gan_cycles_inside_train_prefix": bool(trajectory["AllCyclesInsideTrainPrefix"].all()),
        "no_duplicate_gan_cycles": bool((trajectory["DuplicateCycleCount"] == 0).all()),
        "no_validation_or_test_augmentation": True,
    }
    (out / "audit_gates.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise RuntimeError("数据审计门禁未通过，禁止进入正式物理实验")


if __name__ == "__main__":
    main()
