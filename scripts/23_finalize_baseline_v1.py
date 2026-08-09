"""Finalize auditable reports and figures after the locked formal test."""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_baseline_v1"
FIG = ROOT / "output" / "figures"

SPEC = importlib.util.spec_from_file_location("baseline_runner", ROOT / "scripts" / "21_paper_baseline_v1.py")
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)


def metric_values(true, pred):
    return {
        "RMSE": mean_squared_error(true, pred) ** .5,
        "MAE": mean_absolute_error(true, pred),
        "MAPE": np.abs(true - pred).sum() / np.abs(true).sum() * 100,
        "R2": r2_score(true, pred),
    }


def fmt(value):
    return f"{value:.4f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT / "multiseed_metrics.csv")
    pred = pd.read_csv(OUT / "per_cycle_predictions.csv")
    compare = pd.read_csv(OUT / "baseline_comparison.csv")
    comparison_detail = pd.read_csv(OUT / "comparison_multiseed_metrics.csv")
    lock_path = OUT / "baseline_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    recomputed, attention = [], []
    for (battery, seed), group in pred.groupby(["BatteryID", "Seed"]):
        values = metric_values(group["TrueSOH"].to_numpy(), group["PredSOH"].to_numpy())
        recomputed.append({"BatteryID": battery, "Seed": seed, **values})
        weight = group[["w_Trend", "w_Seasonal", "w_Residual"]]
        attention.append({
            "BatteryID": battery, "Seed": seed,
            "MeanWeightTrend": weight.w_Trend.mean(), "MeanWeightSeasonal": weight.w_Seasonal.mean(),
            "MeanWeightResidual": weight.w_Residual.mean(), "StdWeightTrend": weight.w_Trend.std(),
            "StdWeightSeasonal": weight.w_Seasonal.std(), "StdWeightResidual": weight.w_Residual.std(),
            "WeightSumMaxAbsError": np.abs(weight.sum(axis=1) - 1).max(),
            "TrendCollapse": bool(weight.w_Trend.mean() > .95), "NearStatic": bool(weight.std().max() < .005),
        })
    recomputed = pd.DataFrame(recomputed); attention = pd.DataFrame(attention)
    attention.to_csv(OUT / "attention_diagnostics.csv", index=False, encoding="utf-8-sig")
    recomputed.to_csv(OUT / "metric_recomputation_test.csv", index=False, encoding="utf-8-sig")
    check = metrics.merge(recomputed, on=["BatteryID", "Seed"], suffixes=("", "_recomputed"))
    max_metric_difference = max(float(np.max(np.abs(check[name] - check[f"{name}_recomputed"])))
                                for name in ["RMSE", "MAE", "MAPE", "R2"])

    summary = metrics.groupby("BatteryID")[["RMSE", "MAE", "MAPE", "R2"]].agg(["mean", "std", "median"])
    transformer_best = True
    for battery in ["B0007", "B0018"]:
        battery_compare = compare[compare.BatteryID.eq(battery)].set_index("Model")
        transformer_best &= battery_compare.loc["Transformer", "RMSE_median"] == battery_compare["RMSE_median"].min()
    b18_rmse = float(summary.loc["B0018", ("RMSE", "median")])
    b18_r2 = float(summary.loc["B0018", ("R2", "median")])
    paper_numeric = abs(b18_rmse - .9123) <= .05 * .9123 and abs(b18_r2 - .8765) <= .03
    complete = bool(lock.get("formal_test_completed") and lock.get("configuration_locked")
                    and len(metrics) == 10 and np.isfinite(metrics[["RMSE", "MAE", "MAPE", "R2"]]).all().all()
                    and max_metric_difference < 1e-4 and all(lock["data_gates"].values()))
    lock.update({"BaselineComplete": complete, "PaperNumericMatched": bool(paper_numeric),
                 "TransformerBestObserved": bool(transformer_best),
                 "metric_recomputation_max_abs_difference": max_metric_difference,
                 "metric_recomputation_tolerance": 1e-4,
                 "attention_near_static_combinations": int(attention.NearStatic.sum())})
    if complete:
        lock["baseline_complete_pending"] = []
    runner.write_json(lock_path, lock)

    runner.append_attempt("B06", "固定比较模型正式评估", "判断Transformer真实排名", "formal-comparison",
                          comparison_detail.to_json(orient="records"), "" if transformer_best else "transformer_not_best", True)

    # Data split figure.
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        n = len(pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")); a, b = int(.6*n), int(.8*n)
        ax.barh([0], [a], color="#4C78A8", label="Train adaptation 60%")
        ax.barh([0], [b-a], left=[a], color="#F2CF5B", label="Validation 20%")
        ax.barh([0], [n-b], left=[b], color="#E45756", label="Strict test 20%")
        ax.set_title(f"{battery}: n={n}"); ax.set_yticks([]); ax.set_xlabel("Cycle index")
    axes[0].legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "data_split_baseline_v1.png", dpi=180); plt.close(fig)

    # Median-seed prediction curve (seed closest to battery median RMSE).
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        q = metrics[metrics.BatteryID.eq(battery)].copy(); median = q.RMSE.median()
        seed = int(q.iloc[np.argmin(np.abs(q.RMSE - median))].Seed)
        curve = pred[pred.BatteryID.eq(battery) & pred.Seed.eq(seed)]
        ax.plot(curve.Cycle, curve.TrueSOH, "k-", label="Measured SOH")
        ax.plot(curve.Cycle, curve.PredSOH, color="#4C78A8", label=f"G-MTC seed {seed}")
        ax.set_title(f"{battery} locked test, median-representative seed"); ax.set_ylabel("SOH (%)")
        ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel("Cycle"); fig.tight_layout()
    fig.savefig(FIG / "soh_prediction_baseline_v1.png", dpi=180); plt.close(fig)

    # RMSE model comparison using medians.
    gmtc = metrics.groupby("BatteryID").RMSE.median().rename("G-MTC")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        q = compare[compare.BatteryID.eq(battery)].set_index("Model")["RMSE_median"]
        values = [q["CNN"], q["LSTM"], q["Transformer"], gmtc[battery]]
        ax.bar(["CNN", "LSTM", "Transformer", "G-MTC"], values, color=["#59A14F", "#F28E2B", "#B279A2", "#4C78A8"])
        ax.set_title(battery); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Test RMSE (percentage points)"); fig.tight_layout()
    fig.savefig(FIG / "model_rmse_comparison_baseline_v1.png", dpi=180); plt.close(fig)

    # Five-seed error distribution.
    fig, ax = plt.subplots(figsize=(8, 4)); data = [metrics[metrics.BatteryID.eq(b)].RMSE for b in ["B0007", "B0018"]]
    ax.boxplot(data, tick_labels=["B0007", "B0018"], showmeans=True); ax.set_ylabel("Test RMSE")
    ax.set_title("Locked G-MTC five-seed distribution"); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "five_seed_rmse_baseline_v1.png", dpi=180); plt.close(fig)

    # Attention curves from the same representative seeds.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        q = metrics[metrics.BatteryID.eq(battery)]; seed = int(q.iloc[np.argmin(np.abs(q.RMSE-q.RMSE.median()))].Seed)
        curve = pred[pred.BatteryID.eq(battery) & pred.Seed.eq(seed)]
        for col, label in [("w_Trend", "Trend"), ("w_Seasonal", "Seasonal"), ("w_Residual", "Residual")]:
            ax.plot(curve.Cycle, curve[col], label=label)
        ax.set_title(f"{battery} attention weights, seed {seed}"); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(ncol=3)
    axes[-1].set_xlabel("Cycle"); fig.tight_layout()
    fig.savefig(FIG / "attention_weights_baseline_v1.png", dpi=180); plt.close(fig)

    report = f"""# G-MTC可信基线报告（paper_baseline_v1）

## 三个独立结论

- BaselineComplete: **{str(complete).lower()}**。数据门禁、五种子训练、逐循环指标重算和锁定测试证据完整。
- PaperNumericMatched: **{str(bool(paper_numeric)).lower()}**。项目预注册门槛为RMSE相对论文值5%且R²绝对差0.03。
- TransformerBestObserved: **{str(bool(transformer_best)).lower()}**。实际比较中CNN在B0007/B0018均为最低RMSE。

## 锁定正式测试

| Battery | RMSE mean±std | RMSE median | MAE mean | MAPE mean | R² mean |
|---|---:|---:|---:|---:|---:|
| B0007 | {summary.loc['B0007',('RMSE','mean')]:.4f}±{summary.loc['B0007',('RMSE','std')]:.4f} | {summary.loc['B0007',('RMSE','median')]:.4f} | {summary.loc['B0007',('MAE','mean')]:.4f} | {summary.loc['B0007',('MAPE','mean')]:.4f}% | {summary.loc['B0007',('R2','mean')]:.4f} |
| B0018 | {summary.loc['B0018',('RMSE','mean')]:.4f}±{summary.loc['B0018',('RMSE','std')]:.4f} | {summary.loc['B0018',('RMSE','median')]:.4f} | {summary.loc['B0018',('MAE','mean')]:.4f} | {summary.loc['B0018',('MAPE','mean')]:.4f}% | {summary.loc['B0018',('R2','mean')]:.4f} |

论文B0018报告RMSE=0.9123、R²=0.8765；当前中位RMSE={b18_rmse:.4f}、中位R²={b18_r2:.4f}，数值未复现。不能在同一测试集继续调参。

## 方法与近似

已运行SCF-GAN训练增强、STL、两支original TimesNet、Residual iTransformer、Attention Bi-LSTM；physics关闭。PDF实际写Seasonal为lightweight TimesNet，而任务预注册要求两支original TimesNet，因此这里是用户指定隔离基线，不是逐字PDF Seasonal配置。SCF-GAN层宽、iTransformer细节、9维融合误差特征和L_weight公式存在论文未披露/工程近似，详见method_compliance_matrix.csv。

P1稳定化包括训练前缀标准化、梯度裁剪、早停、融合加性重构初始化、L_weight=0.1和训练范围递归状态裁剪。这些均不是论文原始方法。Attention无Trend塌缩，但{int(attention.NearStatic.sum())}/10组合近乎静态，论文动态权重行为未充分复现。

逐循环指标重算最大差异={max_metric_difference:.2e}；锁定配置哈希=`{lock['config_hash']}`。
"""
    (OUT / "BASELINE_REPORT.md").write_text(report, encoding="utf-8")
    assessment = f"""# 论文可写性评估（基线阶段）

当前结果可以写成“严格协议下的可重复复现与负结果审计”，但不能写成“成功复现论文精度”或“Transformer最好”。

- 可写证据：无硬性泄漏门禁失败；五种子结果和逐循环预测完整；测试配置预先锁定；失败尝试保留。
- 不足：B0018 RMSE与论文差距大，R²为负；Attention大多近乎静态；CNN在当前协议明显优于Transformer/G-MTC；B0007/B0018是此前已接触开发评估集。
- 物理创新可继续作为隔离消融，但若没有未接触外部电池，只能形成探索性结论，不能作为强确认性证据。
"""
    (OUT / "PAPER_READY_ASSESSMENT.md").write_text(assessment, encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
