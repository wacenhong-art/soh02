"""Single command-line entry point for the strict G-MTC experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script: str, *args: object) -> None:
    command = [sys.executable, "-X", "utf8", str(ROOT / script), *map(str, args)]
    subprocess.run(command, cwd=ROOT, check=True)


def run_paper(args: argparse.Namespace) -> None:
    """Run only the paper-core chain; engineering add-ons are excluded."""
    for title, script in [
        ("健康特征提取", "scripts/01_extract_hf.py"),
        ("SCF-GAN训练增强", "scripts/02_scf_gan.py"),
        ("STL分解与分量预测", "scripts/03_time_series.py"),
        ("注意力Bi-LSTM融合", "scripts/04_attention_fusion.py"),
    ]:
        print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
        run_script(script)
    if not args.skip_baselines:
        run_script("baseline_compare.py")
    run_script("scripts/05_paper_compliance.py")


def run_physics(args: argparse.Namespace) -> None:
    command = [
        "--batteries", args.batteries,
        "--modes", args.modes,
        "--lambda-phy", args.lambda_phy,
        "--epochs", args.epochs,
        "--patience", args.patience,
        "--seed", args.seed,
        "--evaluation-split", args.evaluation_split,
        "--output-dir", args.output_dir,
    ]
    run_script("scripts/08_physics_trend_ablation.py", *command)


def run_grid(args: argparse.Namespace) -> None:
    command: list[object] = [args.grid_command]
    if args.grid_command != "build":
        command += ["--stage", args.stage, "--epochs", args.epochs,
                    "--patience", args.patience, "--seed", args.seed]
        if args.grid_command == "one":
            command += ["--config", args.config]
        elif args.max_configs is not None:
            command += ["--max-configs", args.max_configs]
    run_script("scripts/09_paper_grid.py", *command)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="严格、无测试泄漏的 G-MTC 实验入口")
    sub = p.add_subparsers(dest="command", required=True)

    paper = sub.add_parser("paper", help="运行论文核心链")
    paper.add_argument("--skip-baselines", action="store_true")
    paper.set_defaults(func=run_paper)

    physics = sub.add_parser("physics", help="运行 Trend-only 物理约束消融")
    physics.add_argument("--batteries", default="B0007,B0018")
    physics.add_argument("--modes", default="none,monotonic,smooth,both")
    physics.add_argument("--lambda-phy", type=float, default=.01, choices=[0, .001, .01, .1])
    physics.add_argument("--epochs", type=int, default=250)
    physics.add_argument("--patience", type=int, default=30)
    physics.add_argument("--seed", type=int, default=42)
    physics.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    physics.add_argument("--output-dir", default=str(ROOT / "output" / "physics_trend"))
    physics.set_defaults(func=run_physics)

    grid = sub.add_parser("grid", help="构建或运行预注册论文参数网格")
    grid.add_argument("grid_command", choices=["build", "one", "stage"])
    grid.add_argument("--stage", default="A_representation", choices=["A_representation"])
    grid.add_argument("--config")
    grid.add_argument("--epochs", type=int, default=20)
    grid.add_argument("--patience", type=int, default=8)
    grid.add_argument("--seed", type=int, default=42)
    grid.add_argument("--max-configs", type=int)
    grid.set_defaults(func=run_grid)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "grid" and args.grid_command == "one" and not args.config:
        raise SystemExit("grid one 必须提供 --config")
    args.func(args)


if __name__ == "__main__":
    main()
