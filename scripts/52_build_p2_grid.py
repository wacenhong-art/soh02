"""Create an auditable, validation-only grid registry for P2 multi-source Trend pretraining.

This script creates configurations only.  It intentionally does not train or
inspect any confirmation/test battery, so the registry can be frozen before a
P2 runner exists.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "p2_multisource_trend_v1" / "grid_search"

FROZEN = {
    "p1_commit": "18c71e0",
    "scf_gan_sources": ["B0005", "B0006"],
    "development_targets": ["B0007", "B0018"],
    "confirmation_holdout": ["B0033"],
    "excluded_contacted": ["B0007", "B0018", "B0034", "B0036", "B0043", "B0044"],
    "target_adaptation_split": "first60/train,next20/validation,last20/not-accessed",
    "architecture": "STL -> original TimesNet Trend + original TimesNet Seasonal + iTransformer Residual -> paper Attention Bi-LSTM",
    "forbidden_changes": ["SCF-GAN algorithm", "target/test labels as features", "test-set parameter selection", "P1 architecture replacement", "B0033 pretraining"],
}

# All tunable factors that affect the P2 Trend training itself.  The first
# stage is deliberately coarse; ceilings/early stopping are fixed controls,
# not selectable sources of test-set optimisation.
GRID = {
    "trend_width": [32, 64],
    "pretrain_learning_rate": [0.0003, 0.001],
    "finetune_learning_rate": [0.0001, 0.0003],
    "weight_decay": [0.00001, 0.0001],
    "batch_size": [16, 32],
    "source_sampling": ["uniform_battery", "condition_balanced"],
}

FIXED_TRAINING_CONTROLS = {
    "pretrain_epoch_cap": 150,
    "finetune_epoch_cap": 100,
    "early_stopping_patience": 20,
    "sequence_length": 10,
    "optimizer": "AdamW",
    "gradient_clip_norm": 1.0,
    "fusion_stable_init": True,
    "seed_for_coarse_screen": 42,
    "selection_metric": "mean validation RMSE across B0007 and B0018",
    "selection_rule": "both batteries must improve versus fixed P1; rank by mean paired delta",
}


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys, values = list(GRID), list(GRID.values())
    rows = []
    for index, choice in enumerate(itertools.product(*values), start=1):
        params = dict(zip(keys, choice))
        record = {"schema": "p2_grid_config/1", "frozen": FROZEN, "tunable": params,
                  "fixed_training_controls": FIXED_TRAINING_CONTROLS}
        config_hash = digest(record)
        path = OUT / "configs" / f"P2G{index:03d}_{config_hash[:10]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**record, "config_hash": config_hash}, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"AttemptID": f"P2G{index:03d}", **params, "ConfigHash": config_hash,
                     "Status": "pending", "ValidationOnly": True, "TestResultsViewed": False,
                     "Rank": None, "FailureGate": None})
    registry = pd.DataFrame(rows)
    registry.to_csv(OUT / "grid_registry.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "created_at": datetime.now().isoformat(), "stage": "coarse_screen",
        "candidate_count": len(rows), "frozen": FROZEN, "tunable_grid": GRID,
        "fixed_training_controls": FIXED_TRAINING_CONTROLS,
        "promotion": "only configurations improving both development batteries proceed to three fixed seeds [42,52,62]",
        "lock": "after five-seed validation, configuration hash is locked before B0033 inference",
    }
    protocol["protocol_hash"] = digest(protocol)
    (OUT / "grid_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# P2 多源Trend预训练网格搜索\n\n"
        "本目录只包含验证集搜索配置，尚未训练。B0033被列为确认保留集，任何训练运行器都必须拒绝读取其标签。\n"
        f"\n粗筛组合数：{len(rows)}。通过双电池验证门槛后才允许进入三种子验证。\n", encoding="utf-8")
    print(f"Created {len(rows)} validation-only configurations at {OUT}")


if __name__ == "__main__":
    main()
