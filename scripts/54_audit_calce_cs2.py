"""Audit CALCE CS2 raw Excel archives before any model integration.

This reader deliberately uses only the Python standard library: it reads the
XML stored inside XLSX files without introducing an unpinned Excel dependency.
The audit is read-only with respect to raw data and emits a provenance table
only.  It does not train a model or inspect the reserved NASA confirmation
battery.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REQUIRED_SUMMARY = {"Cycle_Index", "Discharge_Capacity(Ah)", "Vmax_On_Cycle(V)"}


def shared_strings(book: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in node.iter(NS + "t")) for node in root.findall(NS + "si")]


def sheet_rows(book: zipfile.ZipFile, sheet: str) -> list[list[str]]:
    strings = shared_strings(book)
    root = ET.fromstring(book.read(sheet))
    rows: list[list[str]] = []
    for row in root.findall(".//" + NS + "sheetData/" + NS + "row"):
        values: list[str] = []
        for cell in row.findall(NS + "c"):
            value = cell.find(NS + "v")
            text = "" if value is None else (value.text or "")
            if cell.get("t") == "s" and text:
                text = strings[int(text)]
            values.append(text)
        rows.append(values)
    return rows


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_one(outer: zipfile.ZipFile, entry: str) -> dict:
    with zipfile.ZipFile(io.BytesIO(outer.read(entry))) as book:
        # CALCE CS2 files conventionally place per-cycle statistics on sheet4.
        if "xl/worksheets/sheet4.xml" not in book.namelist():
            return {"File": entry, "AuditStatus": "missing_statistics_sheet"}
        rows = sheet_rows(book, "xl/worksheets/sheet4.xml")
    if not rows:
        return {"File": entry, "AuditStatus": "empty_statistics_sheet"}
    header, body = rows[0], rows[1:]
    missing = sorted(REQUIRED_SUMMARY - set(header))
    if missing:
        return {"File": entry, "AuditStatus": "missing_required_columns", "MissingColumns": ";".join(missing)}
    columns = {name: idx for idx, name in enumerate(header)}
    cycle = [numeric(r[columns["Cycle_Index"]]) for r in body if len(r) > columns["Cycle_Index"]]
    cap = [numeric(r[columns["Discharge_Capacity(Ah)"]]) for r in body if len(r) > columns["Discharge_Capacity(Ah)"]]
    vmax = [numeric(r[columns["Vmax_On_Cycle(V)"]]) for r in body if len(r) > columns["Vmax_On_Cycle(V)"]]
    cycle = [x for x in cycle if x is not None]
    cap = [x for x in cap if x is not None]
    vmax = [x for x in vmax if x is not None]
    return {
        "File": entry, "AuditStatus": "ok", "SummaryRows": len(body),
        "CycleMin": min(cycle) if cycle else None, "CycleMax": max(cycle) if cycle else None,
        "UniqueCycleCount": len(set(cycle)), "CapacityNonNull": len(cap),
        "CapacityMinAh": min(cap) if cap else None, "CapacityMaxAh": max(cap) if cap else None,
        "VmaxNear4p2Count": sum(4.15 <= x <= 4.25 for x in vmax),
        "VmaxNonNull": len(vmax), "RequiredColumnsPresent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(ROOT / "data" / "external" / "calce" / "CS2_3.zip"))
    parser.add_argument("--output", default=str(ROOT / "output" / "p2_multisource_trend_v1" / "external_data"))
    args = parser.parse_args()
    archive, out = Path(args.archive), Path(args.output)
    if not archive.exists():
        raise FileNotFoundError(f"CALCE archive not found: {archive}")
    with zipfile.ZipFile(archive) as outer:
        entries = sorted(x for x in outer.namelist() if x.lower().endswith(".xlsx"))
        records = [audit_one(outer, entry) for entry in entries]
    table = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calce_cs2_file_audit.csv", index=False, encoding="utf-8-sig")
    valid = table[table.AuditStatus.eq("ok")]
    provenance = {
        "schema": "external_source_audit/1", "source": "CALCE CS2_3 official archive",
        "archive": str(archive), "archive_bytes": archive.stat().st_size,
        "files_seen": len(table), "files_with_cycle_statistics": len(valid),
        "raw_data_policy": "read-only; ignored by Git; never used for confirmation/test selection",
        "eligibility": "not yet approved for P2 pretraining; requires raw charge-curve HF1/HF2/HF3 extraction and cross-file trajectory stitching audit",
        "confirmation_holdout_untouched": ["B0033"],
    }
    (out / "calce_cs2_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
