"""Read-only NASA ZIP extraction and HF audit for the provisional CILA outer group."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hf_extractor_outer", ROOT / "scripts" / "01_extract_hf.py")
hf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hf
SPEC.loader.exec_module(hf)

BATTERIES = ["B0043", "B0044"]
ARCHIVE_NAME = "3. BatteryAgingARC_25-44.zip"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def locate_archive(dataset_root: Path) -> Path:
    matches = list(dataset_root.rglob(ARCHIVE_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {ARCHIVE_NAME} below {dataset_root}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=r"C:\Users\ROG\Documents\5.+Battery+Data+Set")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "hf_csv"))
    parser.add_argument("--audit-dir", default=str(ROOT / "output" / "ieee_innovation_v1" / "phase4" / "outer_data_audit"))
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    archive = locate_archive(dataset_root)
    manifest = pd.read_csv(ROOT / "output" / "ieee_innovation_v1" / "datasets" / "nasa_dataset_manifest.csv")
    rows = []
    with zipfile.ZipFile(archive, "r") as bundle:
        for battery in BATTERIES:
            entry = f"{battery}.mat"
            raw = bundle.read(entry)
            expected = manifest.loc[(manifest.BatteryID == battery) & (manifest.MatEntry == entry), "MatSHA256"]
            if len(expected) != 1:
                raise RuntimeError(f"manifest entry mismatch for {battery}")
            actual_hash = sha256_bytes(raw)
            if expected.iloc[0] != f"sha256:{actual_hash}":
                raise RuntimeError(f"SHA256 mismatch for {battery}")
            frame = hf.extract_battery_frame(loadmat(io.BytesIO(raw)), battery, verbose=False)
            if frame.empty:
                raise RuntimeError(f"{battery} has zero cycles compatible with the frozen HF definition")
            frame = frame.apply(pd.to_numeric, errors="coerce")
            if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                raise RuntimeError(f"non-finite HF data for {battery}")
            frame.insert(0, "Cycle", np.arange(len(frame), dtype=np.int64))
            frame.insert(0, "DatasetID", battery)
            frame.insert(0, "BatteryID", battery)
            frame.to_csv(output_dir / f"{battery}_hf.csv", index=False, encoding="utf-8-sig")
            count = len(frame); train_end = int(count * 0.6); val_end = int(count * 0.8)
            fusion_train_windows = max(train_end - 2 * 10, 0)
            rows.append({"BatteryID": battery, "Archive": str(archive), "MatEntry": entry,
                         "MatSHA256Verified": True, "HFRowCount": count, "TrainEnd": train_end,
                         "ValidationEnd": val_end, "ValidationRows": val_end-train_end,
                         "TestRows": count-val_end, "ExpectedFusionTrainWindows": fusion_train_windows,
                         "OuterProtocolFeasible": bool(fusion_train_windows >= 2),
                         "SOHMetricsInspected": False, "SourceArchiveModified": False})
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_dir / "outer_hf_audit.csv", index=False)
    summary = {"archive": str(archive), "archive_read_only": True, "batteries": BATTERIES,
               "all_hashes_verified": bool(audit.MatSHA256Verified.all()),
               "all_protocols_feasible": bool(audit.OuterProtocolFeasible.all()),
               "soh_metrics_inspected": False}
    (audit_dir / "outer_hf_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(audit.to_string(index=False))
    if not summary["all_protocols_feasible"]:
        raise RuntimeError("provisional outer group failed post-extraction length gate")


if __name__ == "__main__":
    main()
