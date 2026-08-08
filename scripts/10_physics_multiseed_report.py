"""Aggregate seed-repeated ablations without selecting on the test set."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "physics_trend_multiseed"


def md_table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    return "\n".join([
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)],
    ])


def main():
    frames = []
    for path in sorted(OUT.glob("seed_*/physics_ablation_metrics.csv")):
        frame = pd.read_csv(path)
        frame["Seed"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("未找到 seed_*/physics_ablation_metrics.csv")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT / "multiseed_raw_metrics.csv", index=False, encoding="utf-8-sig")
    metrics = ["RMSE", "MAE", "MAPE", "R2", "TrendViolationRate"]
    summary = raw.groupby(["BatteryID", "PhysicsMode", "LambdaPhy"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    base = raw[raw.PhysicsMode == "none"][["BatteryID", "Seed", "RMSE"]].rename(columns={"RMSE": "OriginalRMSE"})
    paired = raw.merge(base, on=["BatteryID", "Seed"])
    paired["RMSEDeltaVsOriginal"] = paired.RMSE - paired.OriginalRMSE
    wins = paired.groupby(["BatteryID", "PhysicsMode"]).agg(
        Seeds=("Seed", "count"), RMSEWinRate=("RMSEDeltaVsOriginal", lambda x: float(np.mean(x < 0))),
        MeanRMSEDelta=("RMSEDeltaVsOriginal", "mean"), StdRMSEDelta=("RMSEDeltaVsOriginal", "std")
    ).reset_index()
    summary.to_csv(OUT / "multiseed_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "multiseed_paired_metrics.csv", index=False, encoding="utf-8-sig")
    wins.to_csv(OUT / "multiseed_win_rates.csv", index=False, encoding="utf-8-sig")
    fitting_rows = []
    for seed_dir in sorted(OUT.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        pred = pd.read_csv(seed_dir / "physics_ablation_predictions.csv")
        for (battery, mode), group in pred.groupby(["BatteryID", "PhysicsMode"]):
            log = pd.read_csv(seed_dir / f"{battery}_trend_{mode}_train_log.csv")
            fitting_rows.append({"BatteryID": battery, "PhysicsMode": mode, "Seed": seed,
                "TrendTestRMSE": float(np.sqrt(np.mean((group.TrueTrend-group.PredTrend)**2))),
                "TrendTestBias": float(np.mean(group.PredTrend-group.TrueTrend)),
                "BestValidationPredictionLoss": float(log.ValidationPredictionLoss.min()),
                "BestEpoch": int(log.ValidationPredictionLoss.idxmin()+1), "StoppedEpoch": len(log)})
    fitting = pd.DataFrame(fitting_rows)
    fitting.to_csv(OUT / "multiseed_fitting_diagnostics.csv", index=False, encoding="utf-8-sig")
    fit_summary = fitting.groupby(["BatteryID", "PhysicsMode"]).agg(
        TrendRMSE_mean=("TrendTestRMSE", "mean"), TrendRMSE_std=("TrendTestRMSE", "std"),
        TrendBias_mean=("TrendTestBias", "mean"), BestEpoch_mean=("BestEpoch", "mean"),
        StoppedEpoch_mean=("StoppedEpoch", "mean")).reset_index()
    fit_summary.to_csv(OUT / "multiseed_fitting_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, battery in zip(axes, sorted(raw.BatteryID.unique())):
        groups = [g.RMSE.to_numpy() for _, g in raw[raw.BatteryID == battery].groupby("PhysicsMode", sort=False)]
        labels = [name for name, _ in raw[raw.BatteryID == battery].groupby("PhysicsMode", sort=False)]
        ax.boxplot(groups, tick_labels=labels); ax.set_title(f"{battery} strict-test RMSE")
        ax.set_ylabel("SOH percentage points"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "multiseed_rmse_boxplot.png", dpi=180); plt.close(fig)
    paper_table = summary[["BatteryID", "PhysicsMode", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std",
                           "MAPE_mean", "MAPE_std", "R2_mean", "R2_std", "TrendViolationRate_mean"]]
    report = ["# Physics-guided Trend 多随机种子论文可写性报告", "",
        "## 严格实验口径", "",
        "5 个预先固定随机种子（42/52/62/72/82）；每个种子均完整运行 original TimesNet。目标电池前60%带标签适配、20%验证、后20%严格测试，因此不是零标签纯跨电池实验。物理损失仅用于STL Trend分支。", "",
        "## 五种子结果（均值±标准差的原始列）", "", md_table(paper_table), "",
        "## 配对胜率", "", md_table(wins), "", "## Trend拟合与收敛诊断", "", md_table(fit_summary), "",
        "## 是否可以写成论文", "",
        "可以写成局部有效、跨电池效果不一致的消融结果：B0007 both 在5/5种子下降低RMSE，且标准差更小；B0018没有稳定胜率，R2均值仍为负，不能宣称普遍改善。", "",
        "当前仍不宜作为已经充分验证的强创新结论：样本仅含两个目标电池、每组5种子，且STL在完整序列上分解属于转导式预处理，训练Trend可能间接受到后段序列形状影响。论文必须披露这一限制，或另做仅训练段可用的因果分解实验。", "",
        "不得继续依据严格测试集选择lambda或模式。若要形成更强论文证据，应在验证集预先选择单一配置，再冻结后报告两个测试电池，并增加更多公开电池/数据集。"]
    (OUT / "paper_readiness_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False)); print("\nPAIRED WIN RATES\n", wins.to_string(index=False))


if __name__ == "__main__": main()
