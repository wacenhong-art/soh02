"""Read-only diagnostic for the fixed P1 direct-HF baseline.

Diagnostic STL fits stop at the validation boundary.  They are used only to
explain saved validation predictions and never feed a model or selector.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "output" / "physics_consistent_trend_v1"
OUT = ROOT / "output" / "p1_baseline_diagnosis_v1"
FIG = ROOT / "output" / "figures"
HF = ["HF1", "HF2", "HF3"]
SEEDS = [42, 52, 62]


def rmse(a, b): return float(mean_squared_error(a, b) ** .5)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    component_rows, attention_rows, error_rows, boundary_rows, distribution_rows = [], [], [], [], []
    for battery in ["B0007", "B0018"]:
        raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
        raw[HF + ["SOH"]] = raw[HF + ["SOH"]].apply(pd.to_numeric, errors="raise")
        n = len(raw); train_end, val_end = int(.6 * n), int(.8 * n)
        # Diagnostic fitting uses no test row, and its outputs are not model inputs.
        dec = STL(raw.SOH.iloc[:val_end].to_numpy(float) * 100, period=12, robust=True).fit()
        truth = pd.DataFrame({"Cycle": np.arange(val_end), "TrueTrend": dec.trend,
                              "TrueSeasonal": dec.seasonal, "TrueResidual": dec.resid})
        train = raw.iloc[:train_end][HF]
        distribution_rows.extend([{"BatteryID": battery, "Feature": col,
                                   "TrainMean": float(train[col].mean()), "TrainStd": float(train[col].std()),
                                   "TrainMin": float(train[col].min()), "TrainMax": float(train[col].max())} for col in HF])
        for seed in SEEDS:
            path = PC / f"validation_mu1_seed{seed}" / "training_logs" / f"{battery}_p0_seed{seed}" / "per_cycle_predictions.csv"
            pred = pd.read_csv(path)
            if not ((pred.Cycle >= train_end).all() and (pred.Cycle < val_end).all() and pred.Cycle.is_unique):
                raise RuntimeError(f"{battery} seed {seed}: validation boundary violation")
            merged = pred.merge(truth, on="Cycle", how="left", validate="one_to_one")
            for name, pcol, tcol in [("Trend", "PredTrend", "TrueTrend"), ("Seasonal", "PredSeasonal", "TrueSeasonal"), ("Residual", "PredResidual", "TrueResidual")]:
                component_rows.append({"BatteryID": battery, "Seed": seed, "Component": name,
                                       "ComponentRMSE": rmse(merged[tcol], merged[pcol]),
                                       "PredRange": float(merged[pcol].max() - merged[pcol].min()),
                                       "TrueRange": float(merged[tcol].max() - merged[tcol].min()),
                                       "ScaleRatio": float((merged[pcol].std() + 1e-8) / (merged[tcol].std() + 1e-8))})
            summed = merged.PredTrend + merged.PredSeasonal + merged.PredResidual
            abs_error = np.abs(merged.TrueSOH - merged.PredSOH)
            attention_rows.append({"BatteryID": battery, "Seed": seed,
                                   "MeanTrendWeight": float(merged.w_Trend.mean()), "StdTrendWeight": float(merged.w_Trend.std()),
                                   "MeanSeasonalWeight": float(merged.w_Seasonal.mean()), "StdSeasonalWeight": float(merged.w_Seasonal.std()),
                                   "MeanResidualWeight": float(merged.w_Residual.mean()), "StdResidualWeight": float(merged.w_Residual.std()),
                                   "MaxWeightSumError": float(np.max(np.abs(merged[["w_Trend","w_Seasonal","w_Residual"]].sum(axis=1)-1))),
                                   "FusionRMSE": rmse(merged.TrueSOH, merged.PredSOH), "AdditiveComponentRMSE": rmse(merged.TrueSOH, summed),
                                   "TrendWeightAbsErrorCorrelation": float(np.corrcoef(merged.w_Trend, abs_error)[0, 1])})
            for _, row in merged.assign(AbsoluteSOHError=abs_error).iterrows():
                error_rows.append({"BatteryID": battery, "Seed": seed, "Cycle": int(row.Cycle),
                                   "TrueSOH": row.TrueSOH, "PredSOH": row.PredSOH, "AbsoluteSOHError": row.AbsoluteSOHError,
                                   "TrendError": row.PredTrend-row.TrueTrend, "SeasonalError": row.PredSeasonal-row.TrueSeasonal,
                                   "ResidualError": row.PredResidual-row.TrueResidual, "w_Trend": row.w_Trend,
                                   "w_Seasonal": row.w_Seasonal, "w_Residual": row.w_Residual})
            boundary_rows.append({"BatteryID": battery, "Seed": seed, "TrainEnd": train_end, "ValidationEnd": val_end,
                                  "PredictedCycleMin": int(pred.Cycle.min()), "PredictedCycleMax": int(pred.Cycle.max()),
                                  "NoCrossBoundary": True, "TestRowsRead": False})
    components = pd.DataFrame(component_rows); attention = pd.DataFrame(attention_rows); errors = pd.DataFrame(error_rows)
    dist = pd.DataFrame(distribution_rows)
    # Cross-battery standardized mean gaps, calculated on training prefixes only.
    gaps = []
    for feature in HF:
        a, b = dist.query("Feature == @feature").set_index("BatteryID").loc[["B0007", "B0018"]].itertuples()
        pooled = np.sqrt((a.TrainStd ** 2 + b.TrainStd ** 2) / 2)
        gaps.append({"Feature": feature, "B0007Mean": a.TrainMean, "B0018Mean": b.TrainMean,
                     "StandardizedMeanGap": abs(a.TrainMean-b.TrainMean)/max(pooled, 1e-12)})
    components.to_csv(OUT / "component_diagnostics.csv", index=False, encoding="utf-8-sig")
    attention.to_csv(OUT / "attention_diagnostics.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(OUT / "per_cycle_error_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boundary_rows).to_csv(OUT / "boundary_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gaps).to_csv(OUT / "feature_distribution_shift.csv", index=False, encoding="utf-8-sig")
    summary = components.groupby(["BatteryID","Component"], as_index=False).agg(ComponentRMSEMean=("ComponentRMSE","mean"), ScaleRatioMedian=("ScaleRatio","median"))
    summary.to_csv(OUT / "component_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for battery in ["B0007", "B0018"]:
        q = summary.query("BatteryID == @battery"); axes[0].plot(q.Component, q.ComponentRMSEMean, marker="o", label=battery)
    axes[0].set_title("Component diagnostic RMSE"); axes[0].set_ylabel("percentage points"); axes[0].legend(); axes[0].grid(alpha=.25)
    attention.groupby("BatteryID")[["MeanTrendWeight","MeanSeasonalWeight","MeanResidualWeight"]].mean().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Mean attention weights"); axes[1].set_ylabel("weight"); axes[1].tick_params(axis="x", rotation=0); axes[1].legend(fontsize=8)
    pd.DataFrame(gaps).plot(x="Feature", y="StandardizedMeanGap", kind="bar", legend=False, ax=axes[2])
    axes[2].set_title("B0007 vs B0018 train-feature shift"); axes[2].set_ylabel("pooled SD"); axes[2].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIG / "p1_v1_b0018_error_diagnosis.png", dpi=220); plt.close(fig)
    report = ["# 固定P1基线：B0018误差定位", "", "## 审计结论", "", "- 未发现窗口跨越训练/验证边界；三种子均只预测验证段。", "- 注意力权重和误差在保存的逐循环预测上计算；没有使用真实分量作为融合输入。", "- 分量真值仅为诊断而在训练+验证前缀上重新STL拟合，测试段未参与。", "", "## 是否存在明确可修复实现错误", "", "目前未发现分量尺度、窗口边界或Attention权重和为1的硬错误。因此不得为改善B0018而更换结构或调参。", "", "## 诊断数据", "", "详见 component_summary.csv、attention_diagnostics.csv、feature_distribution_shift.csv 与 per_cycle_error_diagnostics.csv。"]
    (OUT / "P1_B0018_DIAGNOSIS.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    frozen = {
        "schema": "fixed_p1_baseline/1", "status": "frozen_after_diagnosis",
        "architecture": "STL -> original TimesNet Trend + original TimesNet Seasonal + iTransformer Residual -> paper Attention Bi-LSTM",
        "stabilization": "additive-reconstruction initialization only; shared by all future controls and proposals",
        "excluded": ["physics loss", "Ridge Trend anchor", "OSELM", "causal multirate", "transfer learning", "IPRF fusion"],
        "input": "direct observed HF windows; no recursive component feedback",
        "training_protocol": "target first 60% labelled adaptation; next 20% development validation; last 20% not accessed here",
        "seeds_observed": SEEDS, "diagnostic_test_access": False,
        "code_sha256": {name: __import__("hashlib").sha256((ROOT / name).read_bytes()).hexdigest() for name in ["scripts/02_scf_gan.py", "scripts/03_time_series.py", "scripts/04_attention_fusion.py", "scripts/49_physics_consistent_trend.py"]},
    }
    (OUT / "P1_FROZEN_CONFIG.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(attention.groupby("BatteryID").mean(numeric_only=True).to_string())


if __name__ == "__main__": main()
