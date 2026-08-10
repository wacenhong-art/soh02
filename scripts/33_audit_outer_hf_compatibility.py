"""Count frozen-paper HF-compatible rows for untouched NASA candidates only."""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hf_compatibility", ROOT / "scripts" / "01_extract_hf.py")
hf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hf
SPEC.loader.exec_module(hf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "output" / "ieee_innovation_v1" / "phase4" / "outer_hf_compatibility.csv"))
    args = parser.parse_args()
    manifest = pd.read_csv(ROOT / "output" / "ieee_innovation_v1" / "datasets" / "nasa_dataset_manifest.csv")
    registry = pd.read_csv(ROOT / "output" / "ieee_innovation_v1" / "datasets" / "nasa_split_registry.csv")
    excluded_roles = {"existing_source_train_contacted", "existing_target_dev_contacted"}
    candidates = registry.loc[~registry.CurrentRole.isin(excluded_roles), "BatteryID"].tolist()
    rows = []
    archive_cache: dict[str, zipfile.ZipFile] = {}
    try:
        for battery in candidates:
            records = manifest[(manifest.BatteryID == battery) & (manifest.ParseStatus == "ok")].copy()
            if records.empty:
                rows.append({"BatteryID": battery, "HFCompatibleRows": 0, "AuditStatus": "manifest_missing"})
                continue
            record = records.iloc[0]
            archive_path = str(record.ArchivePath)
            if archive_path not in archive_cache:
                archive_cache[archive_path] = zipfile.ZipFile(archive_path, "r")
            raw = archive_cache[archive_path].read(str(record.MatEntry))
            frame = hf.extract_battery_frame(loadmat(io.BytesIO(raw)), battery, verbose=False)
            count = len(frame); train_end = int(count * 0.6); val_end = int(count * 0.8)
            rows.append({"BatteryID": battery, "HFCompatibleRows": count, "TrainRows": train_end,
                         "ValidationRows": val_end-train_end, "TestRows": count-val_end,
                         "ExpectedFusionTrainWindows": max(train_end-20, 0),
                         "ProtocolFeasible": bool(train_end-20 >= 2 and val_end-train_end > 10 and count-val_end > 10),
                         "DeclaredConditionSummary": record.DeclaredConditionSummary,
                         "AuditStatus": "count_only_no_model_metrics", "SOHMetricsInspected": False})
    finally:
        for bundle in archive_cache.values():
            bundle.close()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows).sort_values(["ProtocolFeasible", "HFCompatibleRows"], ascending=[False, False])
    audit.to_csv(output, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
