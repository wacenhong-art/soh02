"""Create the immutable outer-test manifest before any outer test inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "output" / "ieee_innovation_v1" / "phase4"
FORMAL_OUTPUT = PHASE4 / "formal_outer_test"
LOCK_PATH = PHASE4 / "outer_test_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(f"outer test is already locked: {LOCK_PATH}")
    if FORMAL_OUTPUT.exists():
        raise FileExistsError(f"formal output exists before lock: {FORMAL_OUTPUT}")
    artifacts = [
        "scripts/01_extract_hf.py",
        "scripts/02_scf_gan.py",
        "scripts/03_time_series.py",
        "scripts/04_attention_fusion.py",
        "scripts/08_physics_trend_ablation.py",
        "scripts/30_cila_fusion.py",
        "data/hf_csv/B0043_hf.csv",
        "data/hf_csv/B0044_hf.csv",
        "output/B0043_aug_train_hf.csv",
        "output/B0044_aug_train_hf.csv",
        "model/scf_generator.pth",
        "model/scf_scaler.json",
    ]
    payload = {
        "schema": "outer_test_lock/1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_parent": "35e4c78",
        "outer_domains": ["B0043", "B0044"],
        "protocol": "first 60% labeled adaptation; next 20% validation; last 20% strict test",
        "zero_label_cross_battery": False,
        "seeds": [42, 52, 62, 72, 82],
        "branch_configuration": {
            "physics_mode": "none",
            "lambda_phy": 0.0,
            "lambda_weight": 0.1,
            "epochs": 150,
            "patience": 30,
            "fusion_stable_init": True,
            "component_input": "HF1/HF2/HF3 plus lagged component",
            "recursive_state_clipping": False,
            "timesnet": "original",
            "residual": "iTransformer",
            "fusion": "Attention Bi-LSTM",
        },
        "paired_fusion_configurations": {
            "Original": {"mode": "none", "lambda_align": 0.0},
            "CILA": {"mode": "coral", "lambda_align": 0.1},
            "common": {
                "lambda_weight": 0.1,
                "epochs": 150,
                "patience": 30,
                "stable_init": True,
            },
        },
        "selection_source": "B0007/B0018 validation only",
        "outer_selection_used_model_metrics": False,
        "outer_test_results_inspected": False,
        "test_reuse_for_tuning_prohibited": True,
        "artifact_sha256": {name: sha256(ROOT / name) for name in artifacts},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
