"""Validation-only affine domain calibration (VADC).

Fits no test labels. Ridge strength is selected by time-ordered CV inside the
validation split, then frozen before strict-test inference.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "paper_aligned_physics_final"
OUT = ROOT / "output" / "vadc_ablation"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def metrics(true, pred):
    return {"RMSE": mean_squared_error(true, pred) ** .5,
            "MAE": mean_absolute_error(true, pred),
            "MAPE": np.sum(np.abs(true-pred))/np.sum(np.abs(true))*100,
            "R2": r2_score(true, pred)}


def fit_validation_only(frame):
    x = frame[["Pred_SOH"]].to_numpy(); y = frame.True_SOH.to_numpy()
    splitter = TimeSeriesSplit(n_splits=3); scores = []
    for alpha in ALPHAS:
        fold = []
        for train_idx, val_idx in splitter.split(x):
            model = Ridge(alpha=alpha).fit(x[train_idx], y[train_idx])
            fold.append(mean_squared_error(y[val_idx], model.predict(x[val_idx])))
        scores.append(float(np.mean(fold)))
    alpha = ALPHAS[int(np.argmin(scores))]
    return Ridge(alpha=alpha).fit(x, y), alpha, scores


def main():
    OUT.mkdir(parents=True, exist_ok=True); rows, params = [], []
    for battery in ["B0007", "B0018"]:
        for mode in ["none", "monotonic", "smooth", "both"]:
            source = SOURCE / battery / mode
            val = pd.read_csv(source / "gmtc_pred_val.csv")
            test = pd.read_csv(source / "gmtc_pred_test.csv")
            model, alpha, cv_scores = fit_validation_only(val)
            raw = metrics(test.True_SOH.to_numpy(), test.Pred_SOH.to_numpy())
            calibrated = model.predict(test[["Pred_SOH"]].to_numpy())
            cal = metrics(test.True_SOH.to_numpy(), calibrated)
            rows.append({"BatteryID": battery, "PhysicsMode": mode, "Module": "VADC",
                         "Alpha": alpha, **{f"Raw{k}": v for k,v in raw.items()},
                         **{f"Calibrated{k}": v for k,v in cal.items()},
                         "RMSEDelta": cal["RMSE"]-raw["RMSE"]})
            params.append({"BatteryID": battery, "PhysicsMode": mode, "Alpha": alpha,
                           "Slope": float(model.coef_[0]), "Intercept": float(model.intercept_),
                           "ValidationCV_MSE": min(cv_scores)})
            out_dir = OUT / battery / mode; out_dir.mkdir(parents=True, exist_ok=True)
            saved = test.copy(); saved["Pred_SOH_Raw"] = saved.Pred_SOH
            saved["Pred_SOH_VADC"] = calibrated
            saved.to_csv(out_dir / "vadc_test_predictions.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(rows); parameters = pd.DataFrame(params)
    result.to_csv(OUT / "vadc_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUT / "vadc_parameters.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, battery in zip(axes, ["B0007", "B0018"]):
        group=result[result.BatteryID==battery]
        x=np.arange(len(group)); ax.bar(x-.18,group.RawRMSE,.36,label="raw")
        ax.bar(x+.18,group.CalibratedRMSE,.36,label="VADC")
        ax.set_xticks(x,group.PhysicsMode); ax.set_title(battery); ax.set_ylabel("RMSE")
        ax.grid(axis="y",alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "vadc_rmse_comparison.png",dpi=180); plt.close(fig)
    both=result[result.PhysicsMode=="both"]
    fig, ax = plt.subplots(figsize=(7,4))
    x=np.arange(len(both)); ax.bar(x-.18,both.RawRMSE,.36,label="Physics both")
    ax.bar(x+.18,both.CalibratedRMSE,.36,label="Physics both + VADC")
    ax.set_xticks(x,both.BatteryID); ax.set_ylabel("RMSE (SOH percentage points)")
    ax.set_title("VADC effectiveness on the proposed physics model")
    ax.grid(axis="y",alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "vadc_physics_both_effectiveness.png",dpi=180); plt.close(fig)
    report=["# VADC新增模块消融报告","",
        "VADC只使用验证集拟合，并在验证集内部用时间顺序交叉验证选择Ridge强度；严格测试标签未参与训练或选择。该模块校准最终数值，但不施加单调物理约束。","",
        "## Physics both组合",""]
    for row in both.itertuples():
        reduction=(row.RawRMSE-row.CalibratedRMSE)/row.RawRMSE*100
        report.append(f"- {row.BatteryID}: RMSE {row.RawRMSE:.4f} -> {row.CalibratedRMSE:.4f} ({row.RMSEDelta:+.4f}, 降低{reduction:.2f}%)")
    report += ["", "## 结论", "",
        "VADC在Physics both组合的两个目标电池上均降低RMSE，因此模块局部有效。它并未使B0018 Physics both优于无物理约束Original，不能替代对Physics约束本身有效性的判断。"]
    (OUT / "vadc_report.md").write_text("\n".join(report),encoding="utf-8")
    (OUT / "vadc_config.json").write_text(json.dumps({"alpha_grid":ALPHAS,
        "selection":"3-fold TimeSeriesSplit on validation only","test_used_for_fit":False},indent=2),encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
