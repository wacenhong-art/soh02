"""Create an honest paper-reference and physics-ablation compliance report."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_aligned_physics_final"
PAPER = {"BatteryID": "B0018", "PhysicsMode": "Paper reference",
         "MAE": .8946, "MAPE": .9012, "RMSE": .9123, "R2": .8765}


def markdown_table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda value: f"{value:.6g}")
    return "\n".join([
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)],
    ])


def main():
    result = pd.read_csv(OUT / "paper_fusion_ablation_metrics.csv")
    comparison = pd.concat([result, pd.DataFrame([PAPER])], ignore_index=True)
    comparison.to_csv(OUT / "paper_reference_comparison.csv", index=False, encoding="utf-8-sig")
    b18 = result[result.BatteryID == "B0018"].copy()
    baseline = float(b18.loc[b18.PhysicsMode == "none", "RMSE"].iloc[0])
    b18["RMSEDeltaVsOriginal"] = b18.RMSE - baseline
    b18.to_csv(OUT / "B0018_physics_effect.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for battery, group in result.groupby("BatteryID"):
        axes[0].plot(group.PhysicsMode, group.RMSE, "o-", label=battery)
        axes[1].plot(group.PhysicsMode, group.TrendViolationRate, "o-", label=battery)
    axes[0].axhline(PAPER["RMSE"], color="black", ls="--", label="paper B0018")
    axes[0].set_ylabel("RMSE (SOH percentage points)")
    axes[1].set_ylabel("Trend violation rate")
    for ax in axes: ax.grid(alpha=.3); ax.legend(); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(OUT / "paper_aligned_physics_comparison.png", dpi=180); plt.close(fig)
    config = {"paper_reference": PAPER, "published_logic": {
        "trend": "deep TimesNet top-k multi-period", "seasonal": "lightweight TimesNet",
        "residual": "iTransformer", "fusion": "9D dual-attention Bi-LSTM with 3-16-1 head",
        "split": "60/20/20", "source_batteries": ["B0005", "B0006"],
        "target_batteries": ["B0007", "B0018"]},
        "unpublished_assumption": "L_weight approximated by L2 concentration penalty, lambda_weight=0.01",
        "physics": "Trend only; lambda_phy=0.01; final SOH unconstrained"}
    (OUT / "final_experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 论文对齐重构与Physics-guided Trend最终报告", "", "## 结论", "",
        "公开方法逻辑已对齐到深Trend TimesNet、轻量Seasonal TimesNet、Residual iTransformer及Attention Bi-LSTM完整融合；物理损失仅作用于Trend。实验结果未达到论文参考值，且物理约束未同时提高两个目标电池的精度与一致性。", "",
        "## 完整融合消融", "", markdown_table(result), "", "## 论文B0018参考", "",
        "论文：MAE=0.8946, MAPE=0.9012, RMSE=0.9123, R2=0.8765。当前Original：RMSE=1.8031, R2=-0.2996，未达标。", "",
        "## 物理创新判断", "",
        "B0018 violation rate从0.56下降至both的0.44，但RMSE升至2.7804；B0007 both RMSE从3.6545小幅降至3.5977，但violation rate未下降。当前证据不能证明约束同时改善精度和物理一致性。", "",
        "## 论文复现边界", "",
        "论文未公开完整TimesNet/iTransformer源码及L_weight定义；L_weight采用L2集中度惩罚是显式近似。SCF-GAN在创新阶段保持冻结。STL仍是完整序列分解，属于转导式处理。这些限制必须写入论文，不能声称源码级完全复现。"]
    (OUT / "final_compliance_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
