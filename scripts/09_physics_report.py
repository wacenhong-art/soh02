"""Aggregate completed physics-trend experiments into audit-ready artifacts."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "physics_trend"


def markdown_table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main():
    metrics = pd.read_csv(OUT / "physics_ablation_metrics.csv")
    sensitivity = pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("lambda_*/physics_ablation_metrics.csv"))],
        ignore_index=True,
    ).sort_values(["BatteryID", "LambdaPhy"])
    sensitivity.to_csv(OUT / "lambda_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for battery, group in sensitivity.groupby("BatteryID"):
        axes[0].plot(group.LambdaPhy, group.RMSE, "o-", label=battery)
        axes[1].plot(group.LambdaPhy, group.TrendViolationRate, "o-", label=battery)
    axes[0].set_ylabel("RMSE (SOH percentage points)")
    axes[1].set_ylabel("Trend violation rate")
    for ax in axes:
        ax.set_xlabel("lambda_phy"); ax.set_xscale("symlog", linthresh=.001)
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "lambda_sensitivity.png", dpi=180); plt.close(fig)

    original = metrics[metrics.PhysicsMode == "none"].set_index("BatteryID")
    best = metrics.loc[metrics.groupby("BatteryID").RMSE.idxmin()].set_index("BatteryID")
    lines = [
        "# Physics-guided degradation trend constraint 实验报告", "",
        "## 协议与隔离", "",
        "- SCF-GAN 源电池仅为 B0005/B0006；B0007 与 B0018 均未进入 GAN 或迁移预训练。",
        "- 每个目标电池使用前 60% 带标签样本训练适配、中间 20% 验证、最后 20% 严格测试。",
        "- 这不是零标签纯跨电池协议；目标电池前 60% 的真实标签参与了训练适配。",
        "- 原始对照使用 original TimesNet；轻量 TimesNet、OSELM、因果多速率和迁移学习均关闭。Residual 仍为 iTransformer，融合脚本未加入物理损失。",
        "- 物理损失只施加于 STL Trend TimesNet；SCF-GAN 算法和最终 SOH 目标均未被物理约束。",
        "- 相邻损失只在同一 BatteryID/DatasetID 内按连续 Cycle 计算，随机打乱仅发生在轨迹级。",
        "", "## 主消融（lambda_phy=0.01）", "",
        markdown_table(metrics), "", "## 结论", "",
    ]
    for battery in original.index:
        o, b = original.loc[battery], best.loc[battery]
        change = float(b.RMSE - o.RMSE)
        lines.append(f"- {battery}: Original RMSE={o.RMSE:.4f}；最佳为 {b.PhysicsMode} RMSE={b.RMSE:.4f}，变化 {change:+.4f}。")
    lines += ["", "B0018 的四组 Trend violation rate 未随约束下降，说明当前 lambda/数据规模下没有得到单调性改善；不得据此宣称物理约束普遍有效。",
              "B0007 的 Original 指标明显较差，物理模式改善幅度大，但这是单次固定种子、小样本结果，需多种子重复后才能作稳健结论。",
              "", "## Lambda 敏感性（both）", "", markdown_table(sensitivity), "",
              "所有指标均来自最后 20% 真实 SOH；没有按测试集调参，lambda=0/0.001/0.01/0.1 是预先指定网格。"]
    (OUT / "physics_trend_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
