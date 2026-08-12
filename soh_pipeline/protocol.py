"""Pre-registered paper protocol and validation-only search space."""
from __future__ import annotations

import hashlib
import itertools
import json


PAPER_FIXED = {
    "scf_gan_source_batteries": ["B0005", "B0006"],
    "target_batteries": ["B0007", "B0018"],
    "target_labeled_prefix_fraction": 0.20,
    "synthetic_trajectories_per_target": 5,
    "trend_model": "original TimesNet",
    "seasonal_model": "original TimesNet",
    "residual_model": "iTransformer",
    "fusion_model": "Attention Bi-LSTM",
    "physics_mode": "none",
    "forbidden": [
        "fit scaler on target validation/test", "test-label offset calibration",
        "test-set hyperparameter selection", "OSELM", "causal multirate",
        "transfer learning", "final SOH monotonic constraint",
    ],
}

STAGES = {
    "A_representation": {
        "sequence_length": [8, 10, 12], "stl_period": [8, 12, 16],
        "timesnet_width": [32, 64], "timesnet_top_k": [2, 3],
        "itransformer_heads": [2, 4], "itransformer_layers": [1, 2],
    },
}

REPEAT_CONTROL = {
    "coarse_seed": 42,
    "promotion_seeds": [42, 52, 62],
    "final_validation_seeds": [42, 52, 62, 72, 82],
    "selection_metric": "mean validation RMSE across B0007 and B0018",
    "selection_rule": "both target validation RMSE values must improve",
}


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_stage(stage: str) -> list[dict]:
    tunable = STAGES[stage]
    keys = list(tunable)
    result = []
    for number, values in enumerate(itertools.product(*(tunable[k] for k in keys)), 1):
        record = {
            "schema": "paper_protocol_grid_config/2", "stage": stage,
            "paper_fixed": PAPER_FIXED, "tunable": dict(zip(keys, values)),
            "repeat_control": REPEAT_CONTROL,
            "data_boundary": "target prefix 20% train; validation selects; test remains unopened",
        }
        record["attempt_id"] = f"{stage[0]}{number:03d}"
        record["config_hash"] = digest(record)
        result.append(record)
    return result
