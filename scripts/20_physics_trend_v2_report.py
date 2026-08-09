"""汇总严格的 Physics-guided Trend 五随机种子正式测试。

本脚本只读取已经完成的正式测试输出，不训练、不选参，也不改变测试结果。
它生成逐种子配对表、时间块 bootstrap、论文核验报告和图片。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = ["none", "monotonic", "smooth", "both"]
MODE_LABEL = {
    "none": "Original",
    "monotonic": "Monotonic",
    "smooth": "Smooth",
    "both": "Both",
}
COLORS = {
    "none": "#4C78A8",
    "monotonic": "#F58518",
    "smooth": "#54A24B",
    "both": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="output/physics_trend_v2/formal_test"
    )
    parser.add_argument(
        "--selection-file",
        default="output/physics_trend_v2/selection/selected_lambda.json",
    )
    parser.add_argument("--output-dir", default="output/physics_trend_v2")
    parser.add_argument("--figure-dir", default="output/figures")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", path.parent.name)
    if not match:
        raise ValueError(f"无法从路径识别随机种子：{path}")
    return int(match.group(1))


def load_formal_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_files = sorted(input_dir.glob("seed_*/physics_ablation_metrics.csv"))
    prediction_files = sorted(
        input_dir.glob("seed_*/physics_ablation_predictions.csv")
    )
    if not metric_files or len(metric_files) != len(prediction_files):
        raise FileNotFoundError("正式测试指标或逐循环预测文件不完整")

    metrics = []
    predictions = []
    for path in metric_files:
        frame = pd.read_csv(path)
        frame.insert(0, "Seed", seed_from_path(path))
        metrics.append(frame)
    for path in prediction_files:
        frame = pd.read_csv(path)
        frame.insert(0, "Seed", seed_from_path(path))
        predictions.append(frame)

    metrics_df = pd.concat(metrics, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    expected = set(MODE_ORDER)
    for battery, group in metrics_df.groupby("BatteryID"):
        for seed, seed_group in group.groupby("Seed"):
            actual = set(seed_group["PhysicsMode"])
            if actual != expected:
                raise ValueError(
                    f"{battery}/seed={seed} 模式不完整：{sorted(actual)}"
                )
    return metrics_df, predictions_df


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "TrendViolationRate",
        "TrendToleranceViolationRate",
        "TrendTotalPositiveVariation",
        "TrendMeanAbsSecondDifference",
    ]
    summary = (
        metrics.groupby(["BatteryID", "PhysicsMode"])[numeric]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def build_paired(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (battery, seed), group in metrics.groupby(["BatteryID", "Seed"]):
        indexed = group.set_index("PhysicsMode")
        original = indexed.loc["none"]
        mono = indexed.loc["monotonic"]
        rows.append(
            {
                "BatteryID": battery,
                "Seed": seed,
                "OriginalRMSE": original["RMSE"],
                "MonotonicRMSE": mono["RMSE"],
                "RMSEDelta_MonotonicMinusOriginal": mono["RMSE"]
                - original["RMSE"],
                "RMSEImproved": bool(mono["RMSE"] < original["RMSE"]),
                "OriginalTrendViolationRate": original["TrendViolationRate"],
                "MonotonicTrendViolationRate": mono["TrendViolationRate"],
                "TrendViolationRateDelta": mono["TrendViolationRate"]
                - original["TrendViolationRate"],
                "OriginalToleranceViolationRate": original[
                    "TrendToleranceViolationRate"
                ],
                "MonotonicToleranceViolationRate": mono[
                    "TrendToleranceViolationRate"
                ],
                "ToleranceViolationRateDelta": mono[
                    "TrendToleranceViolationRate"
                ]
                - original["TrendToleranceViolationRate"],
                "OriginalPositiveVariation": original[
                    "TrendTotalPositiveVariation"
                ],
                "MonotonicPositiveVariation": mono[
                    "TrendTotalPositiveVariation"
                ],
                "PositiveVariationDelta": mono["TrendTotalPositiveVariation"]
                - original["TrendTotalPositiveVariation"],
            }
        )
    return pd.DataFrame(rows).sort_values(["BatteryID", "Seed"])


def circular_block_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block_length))
    blocks = [
        (start + np.arange(block_length, dtype=int)) % n for start in starts
    ]
    return np.concatenate(blocks)[:n]


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    paired: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery, battery_predictions in predictions.groupby("BatteryID"):
        per_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for seed, seed_predictions in battery_predictions.groupby("Seed"):
            original = (
                seed_predictions[seed_predictions["PhysicsMode"] == "none"]
                .sort_values("Cycle")
                .set_index("Cycle")
            )
            mono = (
                seed_predictions[
                    seed_predictions["PhysicsMode"] == "monotonic"
                ]
                .sort_values("Cycle")
                .set_index("Cycle")
            )
            common = original.index.intersection(mono.index)
            if len(common) < 3:
                raise ValueError(f"{battery}/seed={seed} 可配对周期不足")
            true_original = original.loc[common, "TrueSOH"].to_numpy(float)
            true_mono = mono.loc[common, "TrueSOH"].to_numpy(float)
            if not np.allclose(true_original, true_mono, atol=1e-8):
                raise ValueError(f"{battery}/seed={seed} 的真实标签不一致")
            error_original = (
                original.loc[common, "PredSOH"].to_numpy(float) - true_original
            )
            error_mono = mono.loc[common, "PredSOH"].to_numpy(float) - true_mono
            per_seed[int(seed)] = (error_original, error_mono)

        seeds = np.array(sorted(per_seed))
        n_cycles = len(per_seed[int(seeds[0])][0])
        block_length = max(2, int(math.ceil(math.sqrt(n_cycles))))
        deltas = np.empty(reps, dtype=float)
        for rep in range(reps):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_deltas = []
            for sampled_seed in sampled_seeds:
                err_original, err_mono = per_seed[int(sampled_seed)]
                indices = circular_block_indices(
                    len(err_original), block_length, rng
                )
                rmse_original = float(
                    np.sqrt(np.mean(np.square(err_original[indices])))
                )
                rmse_mono = float(np.sqrt(np.mean(np.square(err_mono[indices]))))
                seed_deltas.append(rmse_mono - rmse_original)
            deltas[rep] = float(np.mean(seed_deltas))

        observed = paired.loc[
            paired["BatteryID"] == battery,
            "RMSEDelta_MonotonicMinusOriginal",
        ].to_numpy(float)
        rows.append(
            {
                "BatteryID": battery,
                "Seeds": ",".join(map(str, seeds)),
                "CyclesPerSeed": n_cycles,
                "BlockLength": block_length,
                "BootstrapRepetitions": reps,
                "ObservedMeanRMSEDelta": float(np.mean(observed)),
                "ObservedMedianRMSEDelta": float(np.median(observed)),
                "CI95Lower": float(np.quantile(deltas, 0.025)),
                "CI95Upper": float(np.quantile(deltas, 0.975)),
                "ProbabilityRMSEDeltaBelowZero": float(np.mean(deltas < 0)),
                "CIExcludesZeroInImprovementDirection": bool(
                    np.quantile(deltas, 0.975) < 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_success_gates(
    metrics: pd.DataFrame, paired: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery in sorted(metrics["BatteryID"].unique()):
        battery_metrics = metrics[metrics["BatteryID"] == battery]
        battery_pairs = paired[paired["BatteryID"] == battery]
        original = battery_metrics[battery_metrics["PhysicsMode"] == "none"]
        mono = battery_metrics[battery_metrics["PhysicsMode"] == "monotonic"]
        boot = bootstrap[bootstrap["BatteryID"] == battery].iloc[0]
        improvements = int(battery_pairs["RMSEImproved"].sum())
        rows.extend(
            [
                {
                    "BatteryID": battery,
                    "Criterion": "Monotonic mean RMSE lower than Original",
                    "Value": float(mono["RMSE"].mean() - original["RMSE"].mean()),
                    "Passed": bool(mono["RMSE"].mean() < original["RMSE"].mean()),
                },
                {
                    "BatteryID": battery,
                    "Criterion": "Monotonic improves RMSE in majority of 5 seeds",
                    "Value": f"{improvements}/5",
                    "Passed": bool(improvements >= 3),
                },
                {
                    "BatteryID": battery,
                    "Criterion": "Mean strict Trend violation rate lower",
                    "Value": float(
                        mono["TrendViolationRate"].mean()
                        - original["TrendViolationRate"].mean()
                    ),
                    "Passed": bool(
                        mono["TrendViolationRate"].mean()
                        < original["TrendViolationRate"].mean()
                    ),
                },
                {
                    "BatteryID": battery,
                    "Criterion": "Mean positive Trend variation lower",
                    "Value": float(
                        mono["TrendTotalPositiveVariation"].mean()
                        - original["TrendTotalPositiveVariation"].mean()
                    ),
                    "Passed": bool(
                        mono["TrendTotalPositiveVariation"].mean()
                        < original["TrendTotalPositiveVariation"].mean()
                    ),
                },
                {
                    "BatteryID": battery,
                    "Criterion": "Paired block-bootstrap 95% CI below zero",
                    "Value": f"[{boot['CI95Lower']:.6f}, {boot['CI95Upper']:.6f}]",
                    "Passed": bool(boot["CIExcludesZeroInImprovementDirection"]),
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_rmse(metrics: pd.DataFrame, output: Path) -> None:
    batteries = sorted(metrics["BatteryID"].unique())
    fig, axes = plt.subplots(1, len(batteries), figsize=(12, 4.8), squeeze=False)
    for ax, battery in zip(axes[0], batteries):
        subset = metrics[metrics["BatteryID"] == battery]
        for x, mode in enumerate(MODE_ORDER):
            values = subset.loc[subset["PhysicsMode"] == mode, "RMSE"].to_numpy()
            ax.scatter(
                np.full(len(values), x), values, s=35, alpha=0.8,
                color=COLORS[mode], edgecolor="white", linewidth=0.5,
            )
            ax.plot(
                [x - 0.22, x + 0.22], [np.median(values)] * 2,
                color="black", linewidth=2,
            )
        ax.set_xticks(range(len(MODE_ORDER)), [MODE_LABEL[m] for m in MODE_ORDER])
        ax.set_yscale("log")
        ax.set_title(f"{battery} (points = 5 seeds)")
        ax.set_ylabel("Test RMSE (percentage points, log scale)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Strict recursive test: RMSE distribution")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta(paired: pd.DataFrame, output: Path) -> None:
    batteries = sorted(paired["BatteryID"].unique())
    fig, axes = plt.subplots(1, len(batteries), figsize=(12, 4.8), squeeze=False)
    for ax, battery in zip(axes[0], batteries):
        subset = paired[paired["BatteryID"] == battery]
        colors = np.where(subset["RMSEImproved"], "#2A9D8F", "#E76F51")
        ax.bar(subset["Seed"].astype(str), subset["RMSEDelta_MonotonicMinusOriginal"], color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(battery)
        ax.set_xlabel("Seed")
        ax.set_ylabel("RMSE delta: Monotonic - Original")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Paired test RMSE change (negative = improvement)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_physics(metrics: pd.DataFrame, output: Path) -> None:
    batteries = sorted(metrics["BatteryID"].unique())
    fig, axes = plt.subplots(2, len(batteries), figsize=(12, 8), squeeze=False)
    for col, battery in enumerate(batteries):
        subset = metrics[metrics["BatteryID"] == battery]
        for row, (metric, ylabel) in enumerate(
            [
                ("TrendViolationRate", "Strict violation rate"),
                ("TrendTotalPositiveVariation", "Total positive variation"),
            ]
        ):
            ax = axes[row, col]
            data = [
                subset.loc[subset["PhysicsMode"] == mode, metric].to_numpy()
                for mode in MODE_ORDER
            ]
            box = ax.boxplot(data, patch_artist=True, showfliers=True)
            for patch, mode in zip(box["boxes"], MODE_ORDER):
                patch.set_facecolor(COLORS[mode])
                patch.set_alpha(0.65)
            ax.set_xticks(range(1, len(MODE_ORDER) + 1), [MODE_LABEL[m] for m in MODE_ORDER])
            ax.set_title(f"{battery}: {ylabel}")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Trend physical-consistency metrics across seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_median_predictions(predictions: pd.DataFrame, output: Path) -> None:
    batteries = sorted(predictions["BatteryID"].unique())
    fig, axes = plt.subplots(len(batteries), 1, figsize=(11, 8), squeeze=False)
    for ax, battery in zip(axes[:, 0], batteries):
        subset = predictions[predictions["BatteryID"] == battery]
        truth = subset.groupby("Cycle", as_index=False)["TrueSOH"].first()
        ax.plot(truth["Cycle"], truth["TrueSOH"], color="black", linewidth=2, label="True SOH")
        for mode in ["none", "monotonic"]:
            mode_data = subset[subset["PhysicsMode"] == mode]
            grouped = mode_data.groupby("Cycle")["PredSOH"]
            cycles = np.array(sorted(mode_data["Cycle"].unique()))
            median = grouped.median().reindex(cycles).to_numpy()
            low = grouped.quantile(0.25).reindex(cycles).to_numpy()
            high = grouped.quantile(0.75).reindex(cycles).to_numpy()
            ax.plot(cycles, median, color=COLORS[mode], linewidth=1.8, label=f"{MODE_LABEL[mode]} median")
            ax.fill_between(cycles, low, high, color=COLORS[mode], alpha=0.18, label=f"{MODE_LABEL[mode]} IQR")
        ax.set_title(f"{battery}: strict test prediction, median and IQR of 5 seeds")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(
    selection: dict[str, object],
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    selected_lambda = selection.get("selected_lambda", "unknown")
    seed_text = ", ".join(map(str, sorted(metrics["Seed"].unique())))
    lines = [
        "# Physics-guided Trend constraint：严格正式测试报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 协议锁定",
        "",
        "- 模型结构保持：Trend/Seasonal 使用 original TimesNet，Residual 使用 iTransformer，之后由 Attention Bi-LSTM 融合。",
        "- 物理损失只作用于 STL Trend 预测，不约束最终 SOH；未修改 SCF-GAN，未加入完整 PINN，也未使用 Frozen ELM。",
        "- B0005/B0006 仅作为 SCF-GAN 源训练；B0007/B0018 分别按前 60% 训练适配、20% 验证、后 20% 严格测试。",
        "- 这不是零标签纯跨电池协议：目标电池前 60% 的标签用于小样本适配，20% 标签用于模型/λ选择，最后 20% 仅用于一次锁定测试。",
        f"- λ={selected_lambda} 仅由验证集选择；正式测试种子为 {seed_text}，测试后未调参、未按测试结果重跑。",
        "- 测试预测采用递归方式，未知测试标签不作为下一时刻输入；因此比 teacher-forcing 结果更严格。",
        "",
        "## 五随机种子结果",
        "",
        "| Battery | Mode | RMSE mean±std | RMSE median | MAE mean | MAPE mean | R² mean | Trend violation mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for battery in sorted(metrics["BatteryID"].unique()):
        for mode in MODE_ORDER:
            subset = metrics[(metrics["BatteryID"] == battery) & (metrics["PhysicsMode"] == mode)]
            lines.append(
                f"| {battery} | {MODE_LABEL[mode]} | {subset['RMSE'].mean():.4f}±{subset['RMSE'].std(ddof=1):.4f} | "
                f"{subset['RMSE'].median():.4f} | {subset['MAE'].mean():.4f} | {subset['MAPE'].mean():.4f}% | "
                f"{subset['R2'].mean():.4f} | {subset['TrendViolationRate'].mean():.4f} |"
            )
    nontrend_weight_max = float(
        (
            metrics["FusionMeanWeightSeasonal"]
            + metrics["FusionMeanWeightResidual"]
        ).max()
    )
    lines.extend(
        [
            "",
            "## 融合权重诊断",
            "",
            f"所有正式测试组合中，Seasonal 与 Residual 平均权重之和的最大值仅为 {nontrend_weight_max:.6f}；融合层几乎全部选择 Trend（权重约为 1）。这属于注意力塌缩，说明当前严格结果不能用来证明多分支 Transformer 融合优于其他模型，也提示复现框架本身仍需先解决训练稳定性。",
            "",
            "## Monotonic 与 Original 配对检验",
            "",
        ]
    )
    for battery in sorted(paired["BatteryID"].unique()):
        subset = paired[paired["BatteryID"] == battery]
        improved = int(subset["RMSEImproved"].sum())
        boot = bootstrap[bootstrap["BatteryID"] == battery].iloc[0]
        lines.append(
            f"- {battery}：RMSE 改善 {improved}/5 个种子；平均配对差值 "
            f"{subset['RMSEDelta_MonotonicMinusOriginal'].mean():.4f}，中位配对差值 "
            f"{subset['RMSEDelta_MonotonicMinusOriginal'].median():.4f}。时间块 bootstrap 95% CI "
            f"[{boot['CI95Lower']:.4f}, {boot['CI95Upper']:.4f}]，改善概率 "
            f"{boot['ProbabilityRMSEDeltaBelowZero']:.3f}。"
        )
    lines.extend(["", "## 预注册成功标准", ""])
    for battery in sorted(gates["BatteryID"].unique()):
        lines.append(f"### {battery}")
        lines.append("")
        for _, row in gates[gates["BatteryID"] == battery].iterrows():
            symbol = "PASS" if bool(row["Passed"]) else "FAIL"
            lines.append(f"- [{symbol}] {row['Criterion']}；值={row['Value']}")
        lines.append("")
    all_passed = bool(gates["Passed"].all())
    lines.extend(
        [
            "## 可用于论文的结论",
            "",
            (
                "全部预注册标准通过，可以将结果表述为严格协议下的稳定改善。"
                if all_passed
                else "本轮严格正式测试未通过全部预注册标准，不能声称该约束已经稳定降低 RMSE 或稳定改善物理一致性。"
            ),
            "",
            "可以客观报告：验证集选择的 λ 在部分种子/电池上有效，但对初始化敏感；严格递归预测暴露了发散和跨种子不稳定。异常大 RMSE 使均值具有误导性，因此必须同时报告中位数、逐种子配对结果和 bootstrap 区间。",
            "",
            "不能使用本次测试结果继续选择 λ、种子或架构；若要继续改进，应建立新的训练稳定化方案，并在新的未触碰外部测试电池/数据集上预注册后验证。",
            "",
            "## 文件核验",
            "",
            "- `final_multiseed_metrics.csv`：全部五种子、四模式原始指标。",
            "- `paired_monotonic_vs_original.csv`：逐种子配对差值。",
            "- `paired_block_bootstrap.csv`：时间块 bootstrap 置信区间。",
            "- `success_criteria.csv`：预注册标准逐项 PASS/FAIL。",
            "- `formal_test_predictions.csv`：全部逐循环真实值、预测值、Trend 与融合权重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics, predictions = load_formal_results(input_dir)
    selection_path = Path(args.selection_file)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = build_summary(metrics)
    paired = build_paired(metrics)
    bootstrap = paired_block_bootstrap(
        predictions,
        paired,
        args.bootstrap_reps,
        np.random.default_rng(args.bootstrap_seed),
    )
    gates = build_success_gates(metrics, paired, bootstrap)

    metrics.to_csv(output_dir / "final_multiseed_metrics.csv", index=False)
    predictions.to_csv(output_dir / "formal_test_predictions.csv", index=False)
    summary.to_csv(output_dir / "final_multiseed_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_monotonic_vs_original.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_block_bootstrap.csv", index=False)
    gates.to_csv(output_dir / "success_criteria.csv", index=False)

    protocol = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_lambda": selection.get("selected_lambda"),
        "lambda_selected_from_validation_only": True,
        "test_results_read_during_selection": False,
        "formal_test_seeds": sorted(int(v) for v in metrics["Seed"].unique()),
        "physics_modes": MODE_ORDER,
        "target_split": "60% train adaptation / 20% validation / 20% strict test",
        "pure_zero_label_cross_battery": False,
        "test_prediction": "recursive; no unknown target labels as lag inputs",
        "test_tuning_or_result_based_rerun": False,
        "frozen_elm_used": False,
        "bootstrap_repetitions": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "formal_test_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "STRICT_FORMAL_TEST_REPORT.md").write_text(
        markdown_report(selection, metrics, paired, bootstrap, gates),
        encoding="utf-8",
    )

    figures = [
        ("fig21_physics_v2_multiseed_rmse.png", plot_rmse, metrics),
        ("fig22_physics_v2_paired_rmse_delta.png", plot_paired_delta, paired),
        ("fig23_physics_v2_trend_consistency.png", plot_physics, metrics),
        ("fig24_physics_v2_prediction_iqr.png", plot_median_predictions, predictions),
    ]
    for filename, function, data in figures:
        function(data, figure_dir / filename)
    pd.DataFrame(
        [
            {
                "Figure": filename,
                "Purpose": purpose,
                "Source": "strict formal test, seeds 42/52/62/72/82",
            }
            for filename, purpose in [
                (figures[0][0], "four-mode RMSE distribution"),
                (figures[1][0], "paired monotonic-original RMSE delta"),
                (figures[2][0], "Trend physical consistency"),
                (figures[3][0], "median prediction and interquartile range"),
            ]
        ]
    ).to_csv(figure_dir / "physics_v2_figures_manifest.csv", index=False)

    print("\nPaired monotonic vs original:\n")
    print(paired.to_string(index=False))
    print("\nBlock bootstrap:\n")
    print(bootstrap.to_string(index=False))
    print("\nSuccess criteria:\n")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
