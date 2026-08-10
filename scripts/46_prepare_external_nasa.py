"""Extract selected untouched NASA batteries directly from the official ZIP bundle.

The source archive is read-only.  This script writes only the derived HF CSVs
needed for outer-development/confirmation experiments and records provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path

from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    Path.home()
    / "Documents"
    / "5.+Battery+Data+Set"
    / "5. Battery Data Set"
    / "3. BatteryAgingARC_25-44.zip"
)


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "paper_hf_extractor", ROOT / "scripts" / "01_extract_hf.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--batteries", default="B0034,B0036")
    args = parser.parse_args()
    batteries = [item.strip() for item in args.batteries.split(",") if item.strip()]
    extractor = load_extractor()
    output_dir = ROOT / "data" / "hf_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for battery in batteries:
            filename = f"{battery}.mat"
            if filename not in names:
                raise FileNotFoundError(f"{filename} not present in {args.archive}")
            payload = archive.read(names[filename])
            mat = loadmat(io.BytesIO(payload))
            frame = extractor.extract_battery_frame(mat, battery, verbose=False)
            if frame.empty or frame.isna().any().any():
                raise ValueError(f"{battery}: empty or incomplete extracted HF frame")
            destination = output_dir / f"{battery}_hf.csv"
            frame.to_csv(destination, index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "BatteryID": battery,
                    "Rows": len(frame),
                    "FirstSOH": float(frame["SOH"].iloc[0]),
                    "LastSOH": float(frame["SOH"].iloc[-1]),
                    "DerivedCSV": str(destination.relative_to(ROOT)),
                    "MatEntry": names[filename],
                    "MatEntrySHA256": sha256(payload).hexdigest(),
                }
            )
    provenance = {
        "source_archive": str(args.archive),
        "source_archive_sha256": file_sha256(args.archive),
        "source_access": "read-only",
        "extractor": "scripts/01_extract_hf.py",
        "batteries": rows,
    }
    provenance_path = ROOT / "output" / "paper_protocol_rescue" / "external_nasa_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
