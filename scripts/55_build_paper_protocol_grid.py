"""Register a paper-protocol, validation-only hyperparameter search.

The registry intentionally separates parameters explicitly specified by the
paper (frozen) from reproducibility choices the paper does not disclose
(eligible for validation-only search).  It creates no model outputs and never
opens a target test split.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_protocol_grid_v1"

PAPER_FIXED = {
    "scf_gan_source_batteries": ["B0005", "B0006"],
    "target_batteries": ["B0007", "B0018"],
    "target_labeled_prefix_fraction": 0.20,
    "synthetic_trajectories_per_target": 5,
    "scf_gan_noise_dim": 8,
    "scf_gan_condition_dim": 2,
    "scf_gan_feature_dim": 3,
    "scf_gan_hidden_width": 256,
    "scf_gan_feature_matching_lambda": 0.1,
    "scf_gan_learning_rate": 0.001,
    "scf_gan_batch_size": 32,
    "scf_gan_epochs": 800,
    "trend_model": "original TimesNet",
    "seasonal_model": "original TimesNet",
    "residual_model": "iTransformer",
    "fusion_model": "Attention Bi-LSTM",
    "physics_mode": "none",
    "lambda_phy": 0.0,
    "forbidden": ["lightweight TimesNet", "OSELM", "causal multirate", "transfer learning", "PINN", "final SOH constraint"],
}

# These factors are not fully disclosed by the paper or are implementation
# controls.  They are split into stages so the full Cartesian product is never
# used to mine development results: one stage locks before the next begins.
STAGES = {
    "A_representation": {
        "sequence_length": [8, 10, 12],
        "stl_period": [8, 12, 16],
        "timesnet_width": [32, 64],
        "timesnet_top_k": [2, 3],
        "itransformer_heads": [2, 4],
        "itransformer_layers": [1, 2],
    },
    "B_optimizer": {
        "component_learning_rate": [0.0003, 0.001, 0.003],
        "component_batch_size": [16, 32],
        "component_weight_decay": [0.00001, 0.0001],
        "component_epoch_cap": [400, 800],
        "component_patience": [50, 100],
        "gradient_clip_norm": [0.5, 1.0],
    },
    "C_fusion": {
        "fusion_hidden_size": [8, 16, 32],
        "fusion_learning_rate": [0.0003, 0.001, 0.003],
        "fusion_weight_decay": [0.00001, 0.0001],
        "fusion_weight_regularization": [0.001, 0.01, 0.1],
        "fusion_epoch_cap": [400, 800],
        "fusion_patience": [50, 100],
    },
}

REPEAT_CONTROL = {
    "coarse_seed": 42,
    "promotion_seeds": [42, 52, 62],
    "final_validation_seeds": [42, 52, 62, 72, 82],
    "selection_metric": "mean validation RMSE across B0007 and B0018",
    "selection_rule": "same configuration must improve the validation RMSE of both targets; rank only after this gate",
}


def sha256(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_stage(name: str, tunable: dict) -> list[dict]:
    keys = list(tunable)
    records = []
    for number, values in enumerate(itertools.product(*(tunable[k] for k in keys)), start=1):
        choice = dict(zip(keys, values))
        record = {
            "schema": "paper_protocol_grid_config/1", "stage": name,
            "paper_fixed": PAPER_FIXED, "tunable": choice,
            "repeat_control": REPEAT_CONTROL,
            "data_boundary": "B0007/B0018: prefix 20% train, validation split must be predefined inside remaining development portion; target test is never loaded during selection",
            "execution_gate": "requires parameterized paper-protocol runner; current legacy scripts hard-code B0018 and 60/20/20",
        }
        record["config_hash"] = sha256(record)
        record["attempt_id"] = f"{name[:1]}{number:03d}"
        records.append(record)
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for stage, params in STAGES.items():
        records = build_stage(stage, params)
        stage_dir = OUT / "configs" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            (stage_dir / f"{record['attempt_id']}_{record['config_hash'][:10]}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            all_rows.append({
                "AttemptID": record["attempt_id"], "Stage": stage,
                **record["tunable"], "ConfigHash": record["config_hash"],
                "Status": "pending", "ValidationOnly": True, "TestResultsViewed": False,
                "PromotionGate": "both B0007 and B0018 validation RMSE improve",
            })
    registry = pd.DataFrame(all_rows)
    registry.to_csv(OUT / "grid_registry.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "schema": "paper_protocol_grid/1", "created_at": datetime.now().isoformat(),
        "paper_fixed": PAPER_FIXED, "stages": STAGES, "repeat_control": REPEAT_CONTROL,
        "stage_order": ["A_representation", "B_optimizer", "C_fusion"],
        "lock_rule": "lock the winning hash after each stage before beginning the next; do not form a cross-stage Cartesian product",
        "test_rule": "after five-seed validation lock, a formal test is run once; its result cannot create a new configuration",
        "protocol_hash": None,
    }
    protocol["protocol_hash"] = sha256(protocol)
    (OUT / "grid_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 论文协议参数网格\n\n"
        "该网格只搜索论文未完整披露、但会影响训练稳定性和拟合的工程参数；论文已指定的SCF-GAN结构、"
        "损失、5套增强、B0005/B0006源训练、B0007/B0018目标角色均锁定。\n\n"
        "阶段A先锁定表示参数，阶段B只在A的锁定值下搜索优化参数，阶段C只在A+B锁定后搜索融合参数。"
        "全部选择只看B0007/B0018验证段；目标测试段不得加载。\n",
        encoding="utf-8",
    )
    print(registry.groupby("Stage").size().to_string())
    print(f"Created {len(registry)} validation-only configurations in {OUT}")


if __name__ == "__main__":
    main()
