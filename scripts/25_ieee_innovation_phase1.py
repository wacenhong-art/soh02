"""Create the ARS Phase-1 research and read-only NASA dataset audit artifacts.

This script never extracts or modifies the source archives.  MAT files are
opened from in-memory ZIP entries only to inspect structure and cycle counts;
capacity values are not exported or used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio


DEFAULT_DATASET_ROOT = Path(r"C:\Users\ROG\Documents\5.+Battery+Data+Set\5. Battery Data Set")
DEFAULT_OUTPUT_ROOT = Path("output/ieee_innovation_v1")

CHARGE_REQUIRED = {
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_charge",
    "Voltage_charge",
    "Time",
}
DISCHARGE_REQUIRED = {
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Time",
    "Capacity",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def flatten_cycles(value) -> list:
    if isinstance(value, np.ndarray):
        return list(value.flat)
    return [value]


def decode_readme(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def battery_number(battery_id: str) -> int:
    return int(re.sub(r"\D", "", battery_id))


def declared_condition(battery_id: str) -> str:
    n = battery_number(battery_id)
    if n in {5, 6, 7, 18}:
        cutoffs = {5: 2.7, 6: 2.5, 7: 2.2, 18: 2.5}
        return f"room temperature; CC discharge 2A; cutoff {cutoffs[n]}V"
    if 25 <= n <= 28:
        cutoffs = {25: 2.0, 26: 2.2, 27: 2.5, 28: 2.7}
        return f"24C; 0.05Hz square-wave 4A, 50% duty; cutoff {cutoffs[n]}V"
    if 29 <= n <= 32:
        cutoffs = {29: 2.0, 30: 2.2, 31: 2.5, 32: 2.7}
        return f"43C; CC discharge 4A; cutoff {cutoffs[n]}V"
    if n in {33, 34, 36}:
        current = 2 if n == 36 else 4
        cutoff = {33: 2.0, 34: 2.2, 36: 2.7}[n]
        return f"24C; CC discharge {current}A; cutoff {cutoff}V"
    if 38 <= n <= 40:
        cutoff = {38: 2.2, 39: 2.5, 40: 2.7}[n]
        return f"24C and 44C; discharge levels 1/2/4A; cutoff {cutoff}V"
    if 41 <= n <= 44:
        cutoff = {41: 2.0, 42: 2.2, 43: 2.5, 44: 2.7}[n]
        return f"4C; discharge levels 4A and 1A; cutoff {cutoff}V"
    if 45 <= n <= 48:
        cutoff = {45: 2.0, 46: 2.2, 47: 2.5, 48: 2.7}[n]
        return f"4C; CC discharge 1A; cutoff {cutoff}V"
    if 49 <= n <= 56:
        cutoff = {1: 2.0, 2: 2.2, 3: 2.5, 0: 2.7}[n % 4]
        return f"4C; CC discharge 2A; cutoff {cutoff}V"
    return "not documented by parsed README mapping"


def find_readme_for_battery(readmes: dict[str, str], battery_id: str) -> str:
    number = str(battery_number(battery_id))
    for name, text in readmes.items():
        if re.search(rf"(?:Battery\s*#?|B)\s*0*{number}\b", text, flags=re.IGNORECASE):
            return name
        if re.search(rf"\b{number}\b", name):
            return name
    if len(readmes) == 1:
        return next(iter(readmes))
    return ""


def inspect_mat(raw: bytes, expected_id: str) -> dict:
    parsed = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    keys = [key for key in parsed if not key.startswith("__")]
    if not keys:
        raise ValueError("MAT file contains no public top-level variable")
    key = expected_id if expected_id in parsed else keys[0]
    root = parsed[key]
    cycles = flatten_cycles(root.cycle)
    type_counts = Counter(str(getattr(cycle, "type", "unknown")) for cycle in cycles)
    temperatures: set[float] = set()
    fields_by_type: dict[str, set[str]] = defaultdict(set)
    for cycle in cycles:
        try:
            temperatures.add(float(cycle.ambient_temperature))
        except (TypeError, ValueError, AttributeError):
            pass
        cycle_type = str(getattr(cycle, "type", "unknown"))
        data = getattr(cycle, "data", None)
        fields_by_type[cycle_type].update(getattr(data, "_fieldnames", []) or [])
    charge_fields = fields_by_type.get("charge", set())
    discharge_fields = fields_by_type.get("discharge", set())
    impedance_fields = fields_by_type.get("impedance", set())
    return {
        "TopLevelVariable": key,
        "TotalOperations": len(cycles),
        "ChargeCycles": type_counts.get("charge", 0),
        "DischargeCycles": type_counts.get("discharge", 0),
        "ImpedanceCycles": type_counts.get("impedance", 0),
        "OtherOperations": sum(v for k, v in type_counts.items() if k not in {"charge", "discharge", "impedance"}),
        "ObservedAmbientTemperaturesC": "|".join(f"{value:g}" for value in sorted(temperatures)),
        "ChargeFields": "|".join(sorted(charge_fields)),
        "DischargeFields": "|".join(sorted(discharge_fields)),
        "ImpedanceFields": "|".join(sorted(impedance_fields)),
        "ChargeRequiredFieldsPresent": CHARGE_REQUIRED.issubset(charge_fields),
        "DischargeRequiredFieldsPresent": DISCHARGE_REQUIRED.issubset(discharge_fields),
        "HasCapacityLabel": "Capacity" in discharge_fields,
        "HFInputFieldsPresent": CHARGE_REQUIRED.issubset(charge_fields),
    }


def rq_brief() -> str:
    return """# Research Question Brief

## Topic Area

面向真实BMS应用约束的锂离子电池SOH估计，并在冻结的G-MTC架构上实施单一、可关闭、可证伪的创新。

## Primary Research Question

在不改变G-MTC核心架构、只增加一个可关闭机制的条件下，IEEE于2024-08-10至2026-08-10发表的SOH研究所揭示的哪些现实问题导向干预，能够在无数据泄漏的NASA及预注册外部数据评估中，为冻结G-MTC基线带来最可信、可重复的改善？

## FINER Assessment

| Criterion | Score | Justification |
|---|---:|---|
| Feasible | 4/5 | 当前代码、可信基线和NASA完整数据可用；外部数据兼容性尚待核验。 |
| Interesting | 5/5 | 直接针对实验室SOH模型的真实部署障碍。 |
| Novel | 4/5 | 新颖性由近两年IEEE空缺和单机制隔离共同决定，而非网络堆叠。 |
| Ethical | 5/5 | 使用公开电池数据，不涉及人类受试者。 |
| Relevant | 5/5 | 直接服务BMS泛化、可靠性与可复现实证。 |
| **Average** | **4.6/5** | 达到ARS阈值。 |

## Scope Boundaries

**In scope:** 2024-08-10至2026-08-10 IEEE论文；锂离子电池SOH；小样本、跨工况、噪声/缺失、部分充电、在线可靠性等现实痛点；一个新增机制；公开且可审计数据。

**Out of scope:** 完整PINN；同时替换TimesNet/iTransformer/SCF-GAN；多个创新模块堆叠；测试集调参；依赖不可获得的专用传感器；把B0007/B0018当作未接触确认集。

**Key assumptions:** 冻结基线用于相对比较而非宣称复现论文数值；最终方向由文献证据和数据可验证性共同决定；正式测试划分在标签结果评估前冻结。

## Sub-questions

1. 近两年IEEE SOH论文集中暴露了哪些尚未解决的真实应用痛点？
2. 哪些痛点能够由现有NASA数据或合法公开数据严格验证？
3. 哪一个单一机制在现实价值、G-MTC契合度、可实现性和可证伪性方面综合最优？

## Phase Status

等待用户确认后进入Sider Scholar文献检索阶段。
"""


def methodology_blueprint() -> str:
    return """# Methodology Blueprint

## Research Paradigm

**Selected:** Pragmatist-positivist applied research.

**Justification:** 研究先用结构化文献证据识别工程问题，再用可重复计算实验检验一个明确机制；结论由可观察指标和预注册门槛决定。

## Method

**Type:** 结构化证据扫描 + 多准则决策分析 + 配对计算实验。

1. 使用Sider Scholar检索近两年IEEE论文，保留完整检索式、时间和筛选理由。
2. 至少20篇候选、8篇深入核验，提取WHY/HOW/WHAT、数据和现实局限。
3. 最多形成3个候选创新，以预先固定的100分量表选择唯一方向。
4. 用户确认后实现；Original与Proposed保持相同数据、种子、预算和架构，仅一个机制不同。
5. 先验证后锁定配置，正式测试只运行一次；保存逐循环预测和配对bootstrap。

## Data Strategy

- 文献：IEEE Xplore收录且发布日期位于2024-08-10至2026-08-10。
- 主数据：NASA完整归档，原始ZIP只读。
- 已接触开发评估：B0007、B0018。
- 未接触候选池：B0025-B0056；本阶段不指定正式测试角色，不比较其模型误差。
- 外部数据：仅当NASA不能回答最终研究问题时，从官方或带许可证研究存储库下载。

## Analytical Framework

- 文献：去重、来源核验、证据等级、现实痛点主题归纳和反证搜索。
- 方向选择：现实价值25、架构契合25、数据可行20、研究空缺15、隔离消融10、成本5。
- 实验：RMSE、MAE、MAPE、R2、专用痛点指标、5种子分布、配对时间块bootstrap、复杂度和推理时间。

## Validity Criteria

| Criterion | Strategy |
|---|---|
| Internal validity | 单因素改动；训练/验证/测试边界单元测试；配置哈希锁定。 |
| External validity | 预注册未接触电池或外部数据；分开报告跨电池、跨温度和跨数据集结果。 |
| Reliability | 固定五种子；逐循环预测可重算；训练日志和失败尝试保留。 |
| Construct validity | 每个候选必须有与现实痛点对应的专用指标，不能只看RMSE。 |

## Limitations by Design

- 冻结G-MTC基线未匹配论文精度，因此只能用于公平相对比较。
- NASA是实验室数据；即便跨温度/负载测试通过，也不能自动外推到真实车队。
- B0007/B0018已被接触，只能作为开发评估集。

## Ethics and Registration

- IRB：不适用，使用公开非个人数据。
- Reporting：PRISMA-inspired检索日志；不在未达到完整系统综述标准时声称PRISMA系统综述。
- Preregistration：推荐；至少在仓库内冻结protocol.json、数据角色、成功门槛和配置哈希。

## Phase Status

等待用户确认研究问题和方法蓝图。
"""


def devils_advocate_report() -> str:
    return """# Devil's Advocate Report - Checkpoint 1

## Verdict

**PASS WITH MANDATORY SAFEGUARDS**

## Critical Issues

No critical issue remains after adding the safeguards below.

## Major Issues

1. **Weak-baseline attribution risk**
   - 当前G-MTC基线方法可审计，但没有复现论文数值。
   - 新模块的相对改善不得表述为原论文复现成功，且必须与简单模型公平比较。

2. **Previously exposed test batteries**
   - B0007/B0018已被多次查看，不能承担强确认性结论。
   - 最终必须从未接触候选池或合法外部数据中预注册外层测试。

3. **Feasibility-driven novelty bias**
   - 容易因为NASA现有字段方便而选择研究价值较弱的问题。
   - 现实价值与数据可行性必须分别评分，且检索必须主动包含反证和难以实现的方向。

## Strongest Counter-Argument

即使某个模块改善了当前指标，也可能只是在修补一个工程近似且性能较弱的复现，而不是推进SOH研究。回应这一批评需要：同一可信基线、简单模型对照、未接触工况验证、专用痛点指标及不过度结论。

## Stress Test

| Test | Result |
|---|---|
| Remove strongest supporting paper | 必须依赖多篇收敛证据，不能依赖单篇。 |
| Flip the expected result | 允许创新失败并完整保留负结果。 |
| Apply to a different condition | 必须预注册跨温度/负载或外部数据测试。 |
| So what? | 只有解决可测现实问题，才具有论文意义。 |

## Phase Status

Checkpoint 1通过；ARS仍要求用户确认后才能进入Phase 2。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"NASA dataset directory not found: {dataset_root}")

    archives = sorted(dataset_root.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found under: {dataset_root}")

    manifest_rows: list[dict] = []
    field_rows: list[dict] = []
    archive_hashes = {archive.name: sha256_file(archive) for archive in archives}

    for archive in archives:
        with zipfile.ZipFile(archive) as zip_file:
            readmes = {
                entry.filename: decode_readme(zip_file.read(entry))
                for entry in zip_file.infolist()
                if "README" in entry.filename.upper()
            }
            for entry in zip_file.infolist():
                if not entry.filename.lower().endswith(".mat"):
                    continue
                battery_id = Path(entry.filename).stem.upper()
                raw = zip_file.read(entry)
                row = {
                    "ArchiveName": archive.name,
                    "ArchivePath": str(archive),
                    "ArchiveBytes": archive.stat().st_size,
                    "ArchiveSHA256": f"sha256:{archive_hashes[archive.name]}",
                    "MatEntry": entry.filename,
                    "MatUncompressedBytes": entry.file_size,
                    "MatCompressedBytes": entry.compress_size,
                    "MatCRC32": f"crc32:{entry.CRC:08x}",
                    "MatSHA256": f"sha256:{sha256_bytes(raw)}",
                    "BatteryID": battery_id,
                    "READMEFile": find_readme_for_battery(readmes, battery_id),
                    "DeclaredConditionSummary": declared_condition(battery_id),
                    "SourceAccessMode": "read-only/in-memory",
                    "CapacityValuesExported": False,
                    "ParseStatus": "ok",
                    "ParseError": "",
                }
                try:
                    inspection = inspect_mat(raw, battery_id)
                    row.update(inspection)
                except Exception as exc:  # Preserve failed batteries in the audit.
                    row["ParseStatus"] = "error"
                    row["ParseError"] = f"{type(exc).__name__}: {exc}"
                    for key in (
                        "TopLevelVariable", "TotalOperations", "ChargeCycles",
                        "DischargeCycles", "ImpedanceCycles", "OtherOperations",
                        "ObservedAmbientTemperaturesC", "ChargeFields",
                        "DischargeFields", "ImpedanceFields",
                        "ChargeRequiredFieldsPresent", "DischargeRequiredFieldsPresent",
                        "HasCapacityLabel", "HFInputFieldsPresent",
                    ):
                        row[key] = ""
                manifest_rows.append(row)

    counts = Counter(row["BatteryID"] for row in manifest_rows)
    hashes_by_battery: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        hashes_by_battery[row["BatteryID"]].add(row["MatSHA256"])
    for row in manifest_rows:
        row["BatteryOccurrenceCount"] = counts[row["BatteryID"]]
        row["DuplicateBatteryID"] = counts[row["BatteryID"]] > 1
        row["DuplicateContentIdentical"] = len(hashes_by_battery[row["BatteryID"]]) == 1
        field_rows.append({
            "BatteryID": row["BatteryID"],
            "ArchiveName": row["ArchiveName"],
            "ParseStatus": row["ParseStatus"],
            "ChargeCycles": row.get("ChargeCycles", ""),
            "DischargeCycles": row.get("DischargeCycles", ""),
            "ImpedanceCycles": row.get("ImpedanceCycles", ""),
            "ObservedAmbientTemperaturesC": row.get("ObservedAmbientTemperaturesC", ""),
            "ChargeRequiredFieldsPresent": row.get("ChargeRequiredFieldsPresent", ""),
            "DischargeRequiredFieldsPresent": row.get("DischargeRequiredFieldsPresent", ""),
            "HFInputFieldsPresent": row.get("HFInputFieldsPresent", ""),
            "HasCapacityLabel": row.get("HasCapacityLabel", ""),
            "CycleContinuityAudit": "not_assessed_structure_only",
            "CapacityValuesInspectedForSelection": False,
            "Notes": row.get("ParseError", ""),
        })

    manifest_fields = [
        "ArchiveName", "ArchivePath", "ArchiveBytes", "ArchiveSHA256",
        "MatEntry", "MatUncompressedBytes", "MatCompressedBytes", "MatCRC32",
        "MatSHA256", "BatteryID", "BatteryOccurrenceCount", "DuplicateBatteryID",
        "DuplicateContentIdentical", "READMEFile", "DeclaredConditionSummary",
        "TopLevelVariable", "TotalOperations", "ChargeCycles", "DischargeCycles",
        "ImpedanceCycles", "OtherOperations", "ObservedAmbientTemperaturesC",
        "ChargeFields", "DischargeFields", "ImpedanceFields",
        "ChargeRequiredFieldsPresent", "DischargeRequiredFieldsPresent",
        "HasCapacityLabel", "HFInputFieldsPresent", "SourceAccessMode",
        "CapacityValuesExported", "ParseStatus", "ParseError",
    ]
    write_csv(output_root / "datasets/nasa_dataset_manifest.csv", manifest_fields, manifest_rows)
    write_csv(output_root / "datasets/nasa_field_audit.csv", list(field_rows[0]), field_rows)

    split_rows = []
    for battery_id in sorted(counts, key=battery_number):
        if battery_id in {"B0005", "B0006"}:
            role = "existing_source_train_contacted"
            eligible = False
            reason = "Used by the frozen SCF-GAN source training protocol."
        elif battery_id in {"B0007", "B0018"}:
            role = "existing_target_dev_contacted"
            eligible = False
            reason = "Previously inspected development evaluation battery."
        else:
            role = "untouched_candidate_unassigned"
            eligible = True
            reason = "No model prediction or capacity error evaluated in this audit."
        split_rows.append({
            "BatteryID": battery_id,
            "CurrentRole": role,
            "FormalOuterTestEligible": eligible,
            "FinalSplitAssigned": False,
            "SOHResultInspectedThisPhase": False,
            "DeclaredConditionSummary": declared_condition(battery_id),
            "Reason": reason,
        })
    write_csv(output_root / "datasets/nasa_split_registry.csv", list(split_rows[0]), split_rows)

    preprocessing_rows = [{
        "Timestamp": datetime.now().astimezone().isoformat(),
        "Step": "phase1_structure_audit",
        "Input": str(dataset_root),
        "Operation": "ZIP metadata/hash and in-memory MAT structure inspection",
        "Output": str(output_root / "datasets"),
        "OriginalFilesModified": False,
        "ArchivesExtracted": False,
        "CapacityValuesUsedForSelection": False,
        "Status": "complete" if all(row["ParseStatus"] == "ok" for row in manifest_rows) else "complete_with_parse_errors",
    }]
    write_csv(
        output_root / "datasets/nasa_preprocessing_log.csv",
        list(preprocessing_rows[0]),
        preprocessing_rows,
    )

    write_text(output_root / "phase1/RQ_BRIEF.md", rq_brief())
    write_text(output_root / "phase1/METHODOLOGY_BLUEPRINT.md", methodology_blueprint())
    write_text(output_root / "phase1/DEVILS_ADVOCATE_CHECKPOINT1.md", devils_advocate_report())

    protocol = {
        "schema": "ieee_innovation_phase1/1",
        "status": "waiting_user_confirmation",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_baseline": "3d55f12",
        "literature_window": {"from": "2024-08-10", "to": "2026-08-10"},
        "literature_provider_required": "Sider Scholar",
        "literature_search_started": False,
        "source_archives_read_only": True,
        "formal_outer_test_selected": False,
        "B0007_B0018_status": "previously_contacted_development_evaluation",
        "untouched_candidate_batteries": [
            row["BatteryID"] for row in split_rows if row["FormalOuterTestEligible"]
        ],
        "model_code_modified": False,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "archives": len(archives),
        "mat_entries": len(manifest_rows),
        "unique_batteries": len(counts),
        "duplicate_battery_ids": sorted([battery for battery, count in counts.items() if count > 1]),
        "parse_errors": sum(row["ParseStatus"] != "ok" for row in manifest_rows),
        "untouched_candidates": sum(row["FormalOuterTestEligible"] for row in split_rows),
        "output_root": str(output_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
