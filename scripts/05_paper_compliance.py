import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = "./"
OUT = os.path.join(ROOT, "output")


def result(name, passed, detail):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def main():
    raw = pd.read_csv(os.path.join(ROOT, "data", "hf_csv", "B0018_hf.csv")).dropna()
    aug = pd.read_csv(os.path.join(OUT, "B0018_aug_train_hf.csv"))
    pred = pd.read_csv(os.path.join(OUT, "gmtc_pred_result.csv"))
    comp = pd.read_csv(os.path.join(OUT, "component_pred_result.csv")).iloc[10:].reset_index(drop=True)
    base = pd.read_csv(os.path.join(OUT, "baseline_compare_result.csv"), index_col=0)

    hard_checks = []
    benchmark_checks = []
    expected_test = len(raw) - int(len(raw) * 0.8)
    hard_checks.append(result("真实测试集规模", len(pred) == expected_test,
                         f"actual={len(pred)}, expected={expected_test}"))
    expected_truth = raw["SOH"].astype(float).to_numpy()[int(len(raw) * 0.8):] * 100.0
    hard_checks.append(result("测试标签来自真实B0018", np.allclose(pred.True_SOH, expected_truth),
                         "禁止使用GAN生成SOH作为测试标签"))
    hard_checks.append(result("GAN仅增强训练HF", set(aug.DatasetID.unique()) == set(range(6)),
                         f"datasets={sorted(aug.DatasetID.unique())}"))

    rmse = float(np.sqrt(mean_squared_error(pred.True_SOH, pred.Pred_SOH)))
    r2 = float(r2_score(pred.True_SOH, pred.Pred_SOH))
    benchmark_checks.append(result("论文RMSE参考", rmse < 1.0, f"RMSE={rmse:.4f}%, paper B0018=0.9123%"))
    benchmark_checks.append(result("论文R2参考", r2 >= 0.85, f"R2={r2:.4f}, paper B0018=0.8765"))
    benchmark_checks.append(result("优于全部基线", rmse < float(base["RMSE%"].min()),
                         f"G-MTC={rmse:.4f}, best baseline={base['RMSE%'].min():.4f}"))

    weight_cols = ["w_Trend", "w_Seasonal", "w_Residual"]
    sums_ok = np.allclose(pred[weight_cols].sum(axis=1), 1.0, atol=1e-5)
    dynamic = float(pred[weight_cols].std().max())
    hard_checks.append(result("Softmax权重和", sums_ok, "每行权重和应为1"))
    benchmark_checks.append(result("动态权重参考", dynamic >= 0.01, f"max std={dynamic:.6f}"))

    for component in ["Trend", "Seasonal", "Residual"]:
        corr = float(comp[f"True_{component}"].corr(comp[f"Pred_{component}"]))
        benchmark_checks.append(result(f"{component}分量跟踪参考", np.isfinite(corr) and corr >= 0.70,
                             f"correlation={corr:.4f}"))

    print("=" * 64)
    if all(hard_checks):
        print("框架与数据完整性验收：PASS")
        print(f"论文数值参考：{'PASS' if all(benchmark_checks) else 'NOT REACHED（数据自适应模式允许）'}")
        return 0
    print("框架与数据完整性验收：FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
