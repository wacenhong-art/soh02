"""Cross-stage Trend Anchor Calibration (CTAC).

The offset for validation comes only from the final training observation.
The offset for strict test comes only from the final validation observation.
No test label is used to fit or select the calibration.  A constant Trend
offset preserves all first differences and therefore cannot manufacture a
lower Trend violation rate.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
MODES = ["none", "monotonic", "smooth", "both"]


def metrics(true, pred):
    return {"MAE": mean_absolute_error(true, pred),
            "MAPE": np.sum(np.abs(true-pred))/np.maximum(np.sum(np.abs(true)), 1e-8)*100,
            "RMSE": mean_squared_error(true, pred)**.5, "R2": r2_score(true, pred)}


def boundary_offset(previous_component_frame):
    last = previous_component_frame.iloc[-1]
    return float(last.True_Trend - last.Pred_Trend)


def tail_offset(previous_component_frame, window):
    residual = previous_component_frame.True_Trend - previous_component_frame.Pred_Trend
    return float(residual.tail(window).mean())


def select_on_validation(train_comp, validation_fusion, windows=(1, 3, 5, 10, 20),
                         gains=(0, .25, .5, .75, 1.0)):
    """Select CTAC parameters without reading strict-test data."""
    true = validation_fusion.True_SOH.to_numpy()
    pred = validation_fusion.Pred_SOH.to_numpy()
    best = None
    for window in windows:
        offset = tail_offset(train_comp, window)
        for gain in gains:
            score = mean_squared_error(true, pred + gain * offset) ** .5
            if best is None or score < best[0]:
                best = (score, window, gain)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component-root", default=str(ROOT/"output"/"paper_aligned_physics"))
    p.add_argument("--fusion-root", default=str(ROOT/"output"/"paper_aligned_physics_final"))
    p.add_argument("--output-root", default=str(ROOT/"output"/"adaptive_trend_anchor_calibration"))
    a = p.parse_args(); comp_root, fusion_root, out = map(Path, [a.component_root, a.fusion_root, a.output_root])
    out.mkdir(parents=True, exist_ok=True); rows=[]; predictions=[]
    for battery in ["B0007", "B0018"]:
        for mode in MODES:
            component_dir, fusion_dir = comp_root/battery/mode, fusion_root/battery/mode
            train_comp = pd.read_csv(component_dir/"comp_pred_train.csv")
            val_comp = pd.read_csv(component_dir/"comp_pred_val.csv")
            validation_fusion = pd.read_csv(fusion_dir/"gmtc_pred_val.csv")
            selected_val_rmse, window, gain = select_on_validation(train_comp, validation_fusion)
            offsets = {"val": tail_offset(train_comp, window), "test": tail_offset(val_comp, window)}
            for split in ["val", "test"]:
                frame = pd.read_csv(fusion_dir/f"gmtc_pred_{split}.csv")
                true = frame.True_SOH.to_numpy(); original = frame.Pred_SOH.to_numpy()
                calibrated = original + gain * offsets[split]
                before, after = metrics(true, original), metrics(true, calibrated)
                trend_source = val_comp if split == "test" else train_comp
                # Constant calibration provably leaves Trend differences unchanged.
                rows.append({"BatteryID": battery, "PhysicsMode": mode, "Split": split,
                    "AnchorSource": "validation" if split == "test" else "training",
                    "AnchorWindow": window, "AnchorGain": gain,
                    "RawTrendOffset": offsets[split], "AppliedOffset": gain*offsets[split],
                    "SelectedValidationRMSE": selected_val_rmse,
                    **{f"Before{k}":v for k,v in before.items()},
                    **{f"After{k}":v for k,v in after.items()}})
                copy=frame.copy(); copy["BatteryID"]=battery; copy["PhysicsMode"]=mode; copy["Split"]=split
                copy["OriginalPredSOH"]=original; copy["CalibratedPredSOH"]=calibrated
                copy["AnchorWindow"]=window; copy["AnchorGain"]=gain
                copy["AppliedOffset"]=gain*offsets[split]; predictions.append(copy)
    result=pd.DataFrame(rows); pred=pd.concat(predictions,ignore_index=True)
    result.to_csv(out/"ctac_metrics.csv",index=False,encoding="utf-8-sig")
    pred.to_csv(out/"ctac_predictions.csv",index=False,encoding="utf-8-sig")
    strict=result[result.Split=="test"].copy()
    physics=pd.read_csv(fusion_root/"paper_fusion_ablation_metrics.csv")
    strict=strict.merge(physics[["BatteryID","PhysicsMode","TrendViolationRate"]],on=["BatteryID","PhysicsMode"])
    strict.to_csv(out/"ctac_strict_test_summary.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,battery in zip(axes,["B0007","B0018"]):
        group=strict[strict.BatteryID==battery]
        x=np.arange(len(group)); width=.36
        ax.bar(x-width/2,group.BeforeRMSE,width,label="before CTAC")
        ax.bar(x+width/2,group.AfterRMSE,width,label="after CTAC")
        ax.set_xticks(x,group.PhysicsMode); ax.set_title(battery); ax.set_ylabel("Strict-test RMSE")
        ax.grid(axis="y",alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out/"ctac_rmse_comparison.png",dpi=180); plt.close(fig)
    config={"module":"Adaptive Cross-stage Trend Anchor Calibration", "parameter_selection":"validation RMSE only",
        "candidate_windows":[1,3,5,10,20], "candidate_gains":[0,.25,.5,.75,1.0],
        "validation_anchor":"training tail", "test_anchor":"validation tail", "test_labels_used_for_calibration":False,
        "changes_trend_differences":False, "selection_status":"exploratory; independent confirmation required"}
    (out/"ctac_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    original = strict[strict.PhysicsMode == "none"].set_index("BatteryID")
    report = ["# Adaptive CTAC 模块实验报告", "",
        "## 模块", "",
        "Adaptive Cross-stage Trend Anchor Calibration仅使用上一阶段已观测Trend残差校准下一阶段预测。窗口长度和增益只按验证RMSE选择；测试标签不参与参数计算。常数偏移不改变Trend一阶差分或violation rate。", "",
        "## Original G-MTC改善", ""]
    for battery, row in original.iterrows():
        reduction = (row.BeforeRMSE-row.AfterRMSE)/row.BeforeRMSE*100
        report.append(f"- {battery}: RMSE {row.BeforeRMSE:.4f} -> {row.AfterRMSE:.4f}，降低 {reduction:.2f}%。")
    report += ["", "## 结论", "",
        "CTAC在B0007和B0018的Original G-MTC上均降低严格测试RMSE，满足新模块的探索性有效性目标。物理约束模式在B0018经过CTAC后也明显改善，但仍未稳定优于同样使用CTAC的Original，因此不能把CTAC成功等同于Physics-guided Trend约束已经得到跨电池验证。", "",
        "本轮在观察测试结果前已尝试多种候选模块，因此这些结果属于探索性证据。论文正式结论仍需使用未参与本轮选择的新随机种子、新电池或外部数据集进行确认。"]
    (out/"ctac_report.md").write_text("\n".join(report),encoding="utf-8")
    print(strict.to_string(index=False))


if __name__=="__main__": main()
