"""Run one paper-protocol grid configuration on validation only.

This runner is deliberately limited to smoke/validation. It generates five
SCF-GAN trajectories from B0005/B0006, then evaluates the selected single
configuration on B0007 and B0018 without loading their test partitions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_protocol_grid_v1" / "runs"


def load(path: Path, module: str):
    spec = importlib.util.spec_from_file_location(module, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tunable = config["tunable"]
    if config["stage"] != "A_representation":
        raise ValueError("首轮仅允许运行A_representation配置；B/C必须等待前一阶段锁定")
    if config["paper_fixed"]["target_labeled_prefix_fraction"] != .2:
        raise ValueError("拒绝非论文20%小样本配置")
    os.environ["TARGET_TRAIN_FRACTION"] = "0.2"
    os.environ["SCF_GAN_SEED"] = str(args.seed)
    gan = load(ROOT / "scripts" / "02_scf_gan.py", "paper_grid_gan")
    gan.train_scf_gan()
    for battery in config["paper_fixed"]["target_batteries"]:
        gan.TARGET_BATTERY = battery
        gan.generate_aug_data()
    engine = load(ROOT / "scripts" / "08_physics_trend_ablation.py", "paper_grid_engine")
    engine.ts.SEQ_LEN = int(tunable["sequence_length"])
    engine.ts.STL_PERIOD = int(tunable["stl_period"])
    engine.ts.LR = .001
    engine.ts.WEIGHT_DECAY = 1e-5
    engine.ts.BATCH_SIZE = 32
    run_dir = OUT / f"{config['attempt_id']}_{config['config_hash'][:10]}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, manifests = [], []
    for battery in config["paper_fixed"]["target_batteries"]:
        metrics, _, manifest = engine.run_battery(
            battery, ["none"], 0.0, args.epochs, args.patience, run_dir, args.seed,
            "validation", False, True, False, train_fraction=.2, validation_fraction=.2,
            model_config=tunable,
        )
        rows.extend(metrics); manifests.append(manifest)
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "ConfigHash", config["config_hash"])
    metrics.insert(1, "AttemptID", config["attempt_id"])
    metrics.insert(2, "Seed", args.seed)
    metrics.to_csv(run_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    (run_dir / "manifest.json").write_text(json.dumps({"config": config, "runs": manifests,
        "test_labels_loaded": False, "stage": "smoke_validation_only"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["BatteryID", "RMSE", "MAE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
