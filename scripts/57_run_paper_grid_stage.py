"""Validation-only batch runner for one pre-registered paper-grid stage."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "output" / "paper_protocol_grid_v1"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A_representation"], default="A_representation")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-configs", type=int, default=None,
                   help="仅用于分批运行；省略则运行该阶段全部配置")
    args = p.parse_args()
    rows = pd.read_csv(GRID / "grid_registry.csv")
    stage_rows = rows[rows.Stage.eq(args.stage)].dropna(subset=["AttemptID"]).copy()
    if args.max_configs is not None:
        stage_rows = stage_rows.head(args.max_configs)
    configs = []
    for _, row in stage_rows.iterrows():
        matches = list((GRID / "configs" / args.stage).glob(f"{row.AttemptID}_{str(row.ConfigHash)[:10]}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"配置文件无法唯一定位：{row.AttemptID}")
        configs.append(json.loads(matches[0].read_text(encoding="utf-8")))
    if not configs:
        raise RuntimeError("没有待运行配置")
    fixed = configs[0]["paper_fixed"]
    if fixed["target_labeled_prefix_fraction"] != .2:
        raise RuntimeError("拒绝非论文20%小样本配置")
    run_id = f"{args.stage}_all_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = GRID / "batch_runs" / run_id; out.mkdir(parents=True)
    os.environ["TARGET_TRAIN_FRACTION"] = "0.2"; os.environ["SCF_GAN_SEED"] = str(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    gan = load(ROOT / "scripts" / "02_scf_gan.py", "batch_gan")
    gan.train_scf_gan()
    for battery in fixed["target_batteries"]:
        gan.TARGET_BATTERY = battery; gan.generate_aug_data()
    engine = load(ROOT / "scripts" / "08_physics_trend_ablation.py", "batch_engine")
    recorded = []
    for number, config in enumerate(configs, 1):
        t = config["tunable"]
        engine.ts.SEQ_LEN = int(t["sequence_length"])
        engine.ts.STL_PERIOD = int(t["stl_period"])
        engine.ts.LR = .001; engine.ts.WEIGHT_DECAY = 1e-5; engine.ts.BATCH_SIZE = 32
        config_dir = out / f"{config['attempt_id']}_{config['config_hash'][:10]}"
        try:
            metrics = []
            for battery in fixed["target_batteries"]:
                rows_out, _, manifest = engine.run_battery(
                    battery, ["none"], 0.0, args.epochs, args.patience, config_dir, args.seed,
                    "validation", False, True, False, train_fraction=.2, validation_fraction=.2,
                    model_config=t,
                )
                for item in rows_out:
                    item.update({"AttemptID": config["attempt_id"], "ConfigHash": config["config_hash"],
                                 "Seed": args.seed, "TestResultsViewed": False, "Manifest": json.dumps(manifest)})
                    metrics.append(item)
            recorded.extend(metrics)
            pd.DataFrame(recorded).to_csv(out / "stage_validation_metrics.csv", index=False, encoding="utf-8-sig")
            print(f"[{number}/{len(configs)}] {config['attempt_id']} complete")
        except Exception as exc:
            recorded.append({"AttemptID": config["attempt_id"], "ConfigHash": config["config_hash"],
                             "Seed": args.seed, "Status": "failed", "Error": repr(exc), "TestResultsViewed": False})
            pd.DataFrame(recorded).to_csv(out / "stage_validation_metrics.csv", index=False, encoding="utf-8-sig")
            print(f"[{number}/{len(configs)}] {config['attempt_id']} failed: {exc}")
    (out / "batch_manifest.json").write_text(json.dumps({"stage": args.stage, "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience, "config_count": len(configs),
        "test_labels_loaded": False, "status": "completed"}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
