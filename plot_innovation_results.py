import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT = "./output"
FIGURES = os.path.join(OUTPUT, "figures")
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0


def save(name):
    path = os.path.join(FIGURES, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def load_data():
    base = pd.read_csv(os.path.join(OUTPUT, "gmtc_pred_result.csv"))
    best = pd.read_csv(os.path.join(OUTPUT, "gmtc_oselm_pred_result.csv"))
    reliability = pd.read_csv(os.path.join(OUTPUT, "causal_reliability_predictions.csv"))
    ablation = pd.read_csv(os.path.join(OUTPUT, "innovation_complete_ablation.csv"))
    efficiency = pd.read_csv(os.path.join(OUTPUT, "timesnet_efficiency.csv"))
    baseline = pd.read_csv(os.path.join(OUTPUT, "baseline_compare_result.csv"), index_col=0)
    return base, best, reliability, ablation, efficiency, baseline


def plot_scatter(best):
    true = best["True_SOH"].to_numpy()
    pred = best["Pred_OSELM_Online"].to_numpy()
    low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=28, alpha=0.8, color="#1565c0", label="Selective-transfer + OSELM")
    plt.plot([low, high], [low, high], "--", color="#d32f2f", label="Ideal")
    plt.xlabel("Reference SOH (%)")
    plt.ylabel("Estimated SOH (%)")
    plt.title("Innovation Model SOH Estimation")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    save("fig7_soh_scatter.png")


def plot_timeseries(best):
    cycle = best["Cycle"].to_numpy()
    plt.figure(figsize=(9, 5))
    plt.plot(cycle, best["True_SOH"], "o-", color="#212121", label="Reference SOH")
    plt.plot(cycle, best["Pred_Light_GMTC"], "--", color="#ef6c00", label="Light G-MTC")
    plt.plot(cycle, best["Pred_OSELM_Online"], "s-", color="#1565c0", markersize=3.5,
             label="Selective-transfer + robust OSELM")
    plt.xlabel("Cycle")
    plt.ylabel("SOH (%)")
    plt.title("SOH Prediction Before and After Online Correction")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    save("fig8_innovation_soh_timeseries.png")


def plot_rmse(baseline, ablation):
    base_models = [m for m in ("CNN", "LSTM", "Transformer") if m in baseline.index]
    names = base_models + ["Light-GMTC", "A2 MultiRate", "A4 Best"]
    values = [float(baseline.loc[m, "RMSE%"]) for m in base_models]
    values += [
        float(ablation.loc[ablation["Experiment"].str.startswith("A0"), "RMSE%"].iloc[0]),
        float(ablation.loc[ablation["Experiment"].str.startswith("A2"), "RMSE%"].iloc[0]),
        float(ablation.loc[ablation["Experiment"].str.startswith("A4"), "RMSE%"].iloc[0]),
    ]
    colors = ["#b0bec5"] * len(base_models) + ["#ffb74d", "#66bb6a", "#1565c0"]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values, color=colors, edgecolor="#424242", linewidth=0.7)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("RMSE (percentage points)")
    plt.title("RMSE Comparison with Innovation Modules")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    save("fig11_rmse_bar.png")


def plot_ablation(ablation):
    labels = ablation["Experiment"].str.extract(r"^(A\d)")[0].tolist()
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(x - 0.18, ablation["RMSE%"], width=0.36, color="#42a5f5", label="RMSE")
    ax1.set_ylabel("RMSE")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, ablation["R2"], width=0.36, color="#66bb6a", label="R²")
    ax2.set_ylabel("R²")
    ax1.set_title("Innovation Module Ablation")
    for bar, value in zip(bars, ablation["RMSE%"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                 ha="center", fontsize=8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save("fig12_innovation_ablation.png")


def plot_reliability(reliability):
    cycle = reliability["Cycle"].to_numpy()
    plt.figure(figsize=(9, 5))
    plt.fill_between(cycle, reliability["Lower90"], reliability["Upper90"],
                     color="#90caf9", alpha=0.45, label="Validation-calibrated 90% interval")
    plt.plot(cycle, reliability["True_SOH"], "o-", color="#212121", label="Reference SOH")
    plt.plot(cycle, reliability["Pred_SOH"], "--", color="#1565c0", label="Reliable prediction")
    plt.xlabel("Cycle")
    plt.ylabel("SOH (%)")
    plt.title("Predictive Reliability and Uncertainty Interval")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    save("fig13_reliability_interval.png")


def plot_efficiency(efficiency):
    values = efficiency["Parameters"].astype(float).to_numpy()
    labels = ["Original TimesNet", "Lightweight TimesNet"]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#78909c", "#26a69a"], edgecolor="#424242")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + values.max() * 0.02,
                 f"{int(value):,}", ha="center")
    reduction = 100.0 * (1.0 - values[1] / values[0])
    plt.text(0.5, values.max() * 0.72, f"Parameter reduction: {reduction:.2f}%",
             ha="center", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    plt.ylabel("Trainable parameters")
    plt.title("TimesNet Lightweight Efficiency")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    save("fig14_timesnet_efficiency.png")


def plot_errors(best):
    cycle = best["Cycle"].to_numpy()
    base_error = np.abs(best["True_SOH"] - best["Pred_Light_GMTC"])
    final_error = np.abs(best["True_SOH"] - best["Pred_OSELM_Online"])
    plt.figure(figsize=(9, 5))
    plt.plot(cycle, base_error, "o--", color="#ef6c00", label="Light G-MTC absolute error")
    plt.plot(cycle, final_error, "s-", color="#1565c0", label="Final absolute error")
    plt.xlabel("Cycle")
    plt.ylabel("Absolute error (percentage points)")
    plt.title("Per-cycle Error Audit")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    save("fig15_per_cycle_error.png")


def verification_outputs(base, best, reliability, ablation, efficiency):
    a0_rmse = float(ablation.loc[ablation["Experiment"].str.startswith("A0"), "RMSE%"].iloc[0])
    summary = ablation.copy()
    summary["RMSE_Improvement_vs_A0_%"] = (a0_rmse - summary["RMSE%"]) / a0_rmse * 100.0
    summary["Best_RMSE"] = summary["RMSE%"] == summary["RMSE%"].min()
    summary.to_csv(os.path.join(OUTPUT, "manual_verification_summary.csv"), index=False)

    detail = best.copy()
    detail["AbsError_Light_GMTC"] = np.abs(detail["True_SOH"] - detail["Pred_Light_GMTC"])
    detail["AbsError_Final"] = np.abs(detail["True_SOH"] - detail["Pred_OSELM_Online"])
    rel_cols = reliability[["Cycle", "Lower90", "Upper90", "EnsembleStd", "ReliabilityGate"]]
    detail = detail.merge(rel_cols, on="Cycle", how="left")
    detail["CoveredBy90Interval"] = (
        (detail["True_SOH"] >= detail["Lower90"])
        & (detail["True_SOH"] <= detail["Upper90"])
    )
    detail.to_csv(os.path.join(OUTPUT, "manual_verification_predictions.csv"), index=False)

    true = detail["True_SOH"].to_numpy()
    pred = detail["Pred_OSELM_Online"].to_numpy()
    best_row = summary.loc[summary["RMSE%"].idxmin()]
    reduction = 100.0 * (1.0 - efficiency.iloc[1]["Parameters"] / efficiency.iloc[0]["Parameters"])
    report = f"""# Innovation result verification report

- Test samples: {len(detail)} real B0018 cycles
- Best experiment: {best_row['Experiment']}
- Final MAE: {mean_absolute_error(true, pred):.4f}
- Final RMSE: {np.sqrt(mean_squared_error(true, pred)):.4f}
- Final R2: {r2_score(true, pred):.4f}
- TimesNet parameter reduction: {reduction:.2f}%
- Reliability interval empirical coverage: {detail['CoveredBy90Interval'].mean() * 100:.2f}%
- Future features used: No
- Online protocol: predict first, then update with the observed label

The causal multi-rate branch improves the no-transfer configuration, but it is not
additive with selective transfer. The final accuracy model therefore uses selective
Trend transfer plus robust OSELM, while the reliability branch is reported separately.
"""
    with open(os.path.join(OUTPUT, "manual_verification_report.md"), "w", encoding="utf-8") as file:
        file.write(report)


def main():
    base, best, reliability, ablation, efficiency, baseline = load_data()
    plot_scatter(best)
    plot_timeseries(best)
    plot_rmse(baseline, ablation)
    plot_ablation(ablation)
    plot_reliability(reliability)
    plot_efficiency(efficiency)
    plot_errors(best)
    verification_outputs(base, best, reliability, ablation, efficiency)
    print("[OK] 人工核验表和报告已写入原 output 目录")


if __name__ == "__main__":
    main()
