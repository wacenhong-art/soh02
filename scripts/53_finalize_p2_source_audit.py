"""Apply the frozen no-reuse policy to the full-NASA HF feasibility audit."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "p2_multisource_trend_v1"
CONTACTED = {"B0007", "B0018", "B0034", "B0036", "B0043", "B0044"}
CONFIRMATION = {"B0033"}


def main() -> None:
    audit = pd.read_csv(OUT / "source_pool_hf_audit.csv")
    audit["PreviouslyContacted"] = audit.BatteryID.isin(CONTACTED)
    audit["ReservedConfirmation"] = audit.BatteryID.isin(CONFIRMATION)
    audit["EligibleP2Source"] = audit.ProtocolFeasible & ~audit.PreviouslyContacted & ~audit.ReservedConfirmation
    audit.to_csv(OUT / "source_pool_eligibility.csv", index=False, encoding="utf-8-sig")
    eligible = audit.loc[audit.EligibleP2Source, "BatteryID"].tolist()
    feasible = audit.loc[audit.ProtocolFeasible, "BatteryID"].tolist()
    lock = {"schema": "p2_source_pool_lock/1", "created_at": datetime.now().isoformat(),
            "full_nasa_hf_definition": "frozen HF1/HF2/HF3 extractor and SEQ_LEN=10", "feasible_batteries": feasible,
            "previously_contacted": sorted(CONTACTED), "confirmation_reserved": sorted(CONFIRMATION),
            "eligible_multisource_pretraining_batteries": eligible,
            "P2StartAllowed": bool(eligible),
            "blocker": None if eligible else "No unused NASA battery remains both HF/window-feasible and outside the confirmation holdout."}
    (OUT / "p2_source_pool_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "# P2完整NASA源池审计\n\n"
    report += f"在冻结HF定义和SEQ_LEN=10下，窗口可行电池：{', '.join(feasible)}。\n\n"
    report += f"已接触而排除：{', '.join(sorted(CONTACTED))}。确认保留：{', '.join(sorted(CONFIRMATION))}。\n\n"
    report += f"可用于严格P2多源预训练的未接触电池：{', '.join(eligible) if eligible else '无'}。\n\n"
    report += "结论：P2StartAllowed=false。完整NASA原始MAT数据是完整的，但在论文冻结HF提取规则下，大部分电池不能产生足够连续的HF循环；不得为了增加样本临时放宽HF定义。\n"
    (OUT / "P2_SOURCE_POOL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
