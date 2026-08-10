"""Audit outer-battery SCF-GAN augmentation without reading validation/test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["DatasetID", "Cycle", "HF1", "HF2", "HF3", "SOH"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_battery(battery: str) -> dict:
    raw_path = ROOT / "data" / "hf_csv" / f"{battery}_hf.csv"
    aug_path = ROOT / "output" / f"{battery}_aug_train_hf.csv"
    raw = pd.read_csv(raw_path)
    aug = pd.read_csv(aug_path)
    if list(aug.columns) != REQUIRED:
        raise AssertionError(f"{battery}: unexpected augmented columns {list(aug.columns)}")
    train_rows = int(len(raw) * 0.6)
    numeric = aug[REQUIRED].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError(f"{battery}: nonfinite augmented values")
    expected_ids = list(range(6))
    if sorted(aug.DatasetID.unique().tolist()) != expected_ids:
        raise AssertionError(f"{battery}: DatasetID must be {expected_ids}")
    label_ok = True
    continuity_ok = True
    for dataset_id, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        cycles = group.Cycle.to_numpy(dtype=int)
        continuity_ok &= np.array_equal(cycles, np.arange(train_rows))
        label_ok &= np.allclose(
            group.SOH.to_numpy(dtype=float),
            raw.SOH.iloc[:train_rows].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
    if not continuity_ok or not label_ok:
        raise AssertionError(f"{battery}: trajectory or real-label replication audit failed")
    if int(aug.Cycle.max()) >= train_rows:
        raise AssertionError(f"{battery}: augmentation crossed the training boundary")
    return {
        "BatteryID": battery,
        "RawRows": len(raw),
        "TrainRows": train_rows,
        "AugmentedRows": len(aug),
        "Datasets": aug.DatasetID.nunique(),
        "CycleMin": int(aug.Cycle.min()),
        "CycleMax": int(aug.Cycle.max()),
        "AllTrajectoriesContinuous": bool(continuity_ok),
        "LabelsMatchRealTrainingPrefix": bool(label_ok),
        "ValidationOrTestRowsAugmented": False,
        "AugmentedCSV_SHA256": sha256(aug_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batteries", default="B0043,B0044")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "ieee_innovation_v1" / "phase4" / "outer_data_audit"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_battery(value.strip()) for value in args.batteries.split(",")]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "outer_aug_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "PASS",
        "source_training_batteries": ["B0005", "B0006"],
        "target_batteries": [row["BatteryID"] for row in rows],
        "augmentation_scope": "target first 60 percent only",
        "validation_test_augmented": False,
        "scf_gan_algorithm_changed": False,
        "generator_sha256": sha256(ROOT / "model" / "scf_generator.pth"),
        "scaler_sha256": sha256(ROOT / "model" / "scf_scaler.json"),
    }
    (output_dir / "outer_aug_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
