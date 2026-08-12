"""Build and run the validation-only paper protocol grid."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soh_pipeline.protocol import PAPER_FIXED, REPEAT_CONTROL, STAGES, build_stage, digest


GRID = ROOT / "output" / "paper_protocol_grid_v2"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(path)
    spec.loader.exec_module(module)
    return module


def build() -> None:
    import pandas as pd

    GRID.mkdir(parents=True, exist_ok=True)
    registry = []
    for stage in STAGES:
        directory = GRID / "configs" / stage
        directory.mkdir(parents=True, exist_ok=True)
        for record in build_stage(stage):
            path = directory / f"{record['attempt_id']}_{record['config_hash'][:10]}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            registry.append({
                "AttemptID": record["attempt_id"], "Stage": stage,
                **record["tunable"], "ConfigHash": record["config_hash"],
                "Status": "pending", "ValidationOnly": True,
                "TestResultsViewed": False,
            })
    pd.DataFrame(registry).to_csv(GRID / "grid_registry.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "schema": "paper_protocol_grid/2", "created_at": datetime.now().isoformat(),
        "paper_fixed": PAPER_FIXED, "stages": STAGES, "repeat_control": REPEAT_CONTROL,
        "test_rule": "lock the five-seed validation winner, then test exactly once",
    }
    protocol["protocol_hash"] = digest(protocol)
    (GRID / "grid_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已创建 {len(registry)} 个只用于验证的配置：{GRID}")


def prepare_engine(seed: int):
    import numpy as np
    import torch

    os.environ["TARGET_TRAIN_FRACTION"] = "0.2"
    os.environ["SCF_GAN_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gan = load(ROOT / "scripts" / "02_scf_gan.py", f"paper_grid_gan_{seed}")
    gan.train_scf_gan()
    for battery in PAPER_FIXED["target_batteries"]:
        gan.TARGET_BATTERY = battery
        gan.generate_aug_data()
    return load(ROOT / "scripts" / "08_physics_trend_ablation.py", f"paper_grid_engine_{seed}")


def validate_config(config: dict) -> None:
    if config.get("stage") != "A_representation":
        raise ValueError("当前只允许运行已预注册的 A_representation 阶段")
    if config.get("paper_fixed") != PAPER_FIXED:
        raise ValueError("配置的论文固定项与当前协议不一致")
    expected = dict(config)
    claimed_hash = expected.pop("config_hash", None)
    if claimed_hash != digest(expected):
        raise ValueError("配置哈希校验失败")


def run_config(config: dict, engine, seed: int, epochs: int, patience: int, out: Path) -> list[dict]:
    validate_config(config)
    tunable = config["tunable"]
    engine.ts.SEQ_LEN = int(tunable["sequence_length"])
    engine.ts.STL_PERIOD = int(tunable["stl_period"])
    engine.ts.LR = .001
    engine.ts.WEIGHT_DECAY = 1e-5
    engine.ts.BATCH_SIZE = 32
    out.mkdir(parents=True, exist_ok=True)
    recorded = []
    for battery in PAPER_FIXED["target_batteries"]:
        rows, _, manifest = engine.run_battery(
            battery, ["none"], 0.0, epochs, patience, out, seed,
            "validation", False, True, False,
            train_fraction=.2, validation_fraction=.2, model_config=tunable,
        )
        for row in rows:
            row.update({
                "AttemptID": config["attempt_id"], "ConfigHash": config["config_hash"],
                "Seed": seed, "TestResultsViewed": False,
                "Manifest": json.dumps(manifest, ensure_ascii=False),
            })
        recorded.extend(rows)
    return recorded


def load_registered_configs(stage: str) -> list[dict]:
    import pandas as pd

    registry = pd.read_csv(GRID / "grid_registry.csv")
    rows = registry[registry.Stage.eq(stage)].dropna(subset=["AttemptID"])
    configs = []
    for row in rows.itertuples():
        matches = list((GRID / "configs" / stage).glob(
            f"{row.AttemptID}_{str(row.ConfigHash)[:10]}.json"
        ))
        if len(matches) != 1:
            raise RuntimeError(f"配置无法唯一定位：{row.AttemptID}")
        configs.append(json.loads(matches[0].read_text(encoding="utf-8")))
    return configs


def run_one(args: argparse.Namespace) -> None:
    import pandas as pd

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    engine = prepare_engine(args.seed)
    out = GRID / "runs" / f"{config['attempt_id']}_{config['config_hash'][:10]}_seed{args.seed}"
    rows = run_config(config, engine, args.seed, args.epochs, args.patience, out)
    pd.DataFrame(rows).to_csv(out / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "manifest.json").write_text(json.dumps({
        "config": config, "seed": args.seed, "test_labels_loaded": False,
        "status": "validation_only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["BatteryID", "RMSE", "MAE", "MAPE", "R2"]].to_string(index=False))


def run_stage(args: argparse.Namespace) -> None:
    import pandas as pd

    configs = load_registered_configs(args.stage)
    if args.max_configs is not None:
        configs = configs[:args.max_configs]
    if not configs:
        raise RuntimeError("没有待运行配置")
    engine = prepare_engine(args.seed)
    run_id = f"{args.stage}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = GRID / "batch_runs" / run_id
    out.mkdir(parents=True)
    recorded = []
    for number, config in enumerate(configs, 1):
        config_out = out / f"{config['attempt_id']}_{config['config_hash'][:10]}"
        try:
            recorded.extend(run_config(config, engine, args.seed, args.epochs, args.patience, config_out))
            print(f"[{number}/{len(configs)}] {config['attempt_id']} complete")
        except Exception as exc:
            recorded.append({
                "AttemptID": config["attempt_id"], "ConfigHash": config["config_hash"],
                "Seed": args.seed, "Status": "failed", "Error": repr(exc),
                "TestResultsViewed": False,
            })
            print(f"[{number}/{len(configs)}] {config['attempt_id']} failed: {exc}")
        pd.DataFrame(recorded).to_csv(out / "stage_validation_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "batch_manifest.json").write_text(json.dumps({
        "stage": args.stage, "seed": args.seed, "epochs": args.epochs,
        "patience": args.patience, "config_count": len(configs),
        "test_labels_loaded": False, "status": "completed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    for command in ("one", "stage"):
        q = sub.add_parser(command)
        q.add_argument("--stage", default="A_representation", choices=list(STAGES))
        q.add_argument("--epochs", type=int, default=20)
        q.add_argument("--patience", type=int, default=8)
        q.add_argument("--seed", type=int, default=42)
        if command == "one":
            q.add_argument("--config", required=True)
        else:
            q.add_argument("--max-configs", type=int)
    args = p.parse_args()
    {"build": lambda _: build(), "one": run_one, "stage": run_stage}[args.command](args)


if __name__ == "__main__":
    main()
