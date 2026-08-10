"""Generate versioned figures for the frozen CILA formal outer test."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "ieee_innovation_v1" / "phase4" / "formal_outer_test"
FIG = ROOT / "output" / "figures"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def rmse_log(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (battery, frame) in zip(axes, metrics.groupby("BatteryID", sort=True)):
        pivot = frame.pivot(index="Seed", columns="Mode", values="RMSE").sort_index()
        x = np.arange(len(pivot)); width = 0.36
        axis.bar(x - width/2, pivot.Original, width, label="Original")
        axis.bar(x + width/2, pivot.CILA, width, label="CILA")
        axis.set_yscale("log")
        axis.set_xticks(x, pivot.index.astype(str))
        axis.set_title(battery)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Test RMSE (log scale)")
        axis.grid(axis="y", alpha=.25)
    axes[1].legend()
    fig.suptitle("Frozen outer test: severe seed-dependent extrapolation")
    save(fig, "cila_outer_rmse_multiseed_log_v1.png")


def deltas(paired):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (battery, frame) in zip(axes, paired.groupby("BatteryID", sort=True)):
        colors = np.where(frame.DeltaRMSE < 0, "#2ca02c", "#d62728")
        axis.bar(frame.Seed.astype(str), frame.DeltaRMSE, color=colors)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(battery)
        axis.set_xlabel("Seed")
        axis.set_ylabel("CILA - Original RMSE")
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Paired outer-test RMSE delta (green favors CILA)")
    save(fig, "cila_outer_paired_delta_v1.png")


def prediction_iqr(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (battery, frame) in zip(axes, predictions.groupby("BatteryID", sort=True)):
        truth = frame.groupby("Cycle", as_index=False).True_SOH.first()
        axis.plot(truth.Cycle, truth.True_SOH, "k--", linewidth=2, label="True SOH")
        for mode, color in [("Original", "#4472C4"), ("CILA", "#ED7D31")]:
            pivot = frame[frame.Mode == mode].pivot(index="Cycle", columns="Seed", values="Pred_SOH")
            median = pivot.median(axis=1); low = pivot.quantile(.25, axis=1); high = pivot.quantile(.75, axis=1)
            axis.plot(pivot.index, median, color=color, label=f"{mode} median")
            axis.fill_between(pivot.index, low, high, color=color, alpha=.18, label=f"{mode} IQR")
        axis.set_yscale("symlog", linthresh=10)
        axis.set_title(battery)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("SOH prediction (symlog)")
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Outer-test prediction dispersion across five seeds")
    save(fig, "cila_outer_prediction_iqr_v1.png")


def bootstrap_ci(ci):
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ci)); observed = ci.Observed.to_numpy(float)
    low = ci.CI95Lower.to_numpy(float); high = ci.CI95Upper.to_numpy(float)
    axis.errorbar(x, observed, yerr=[observed-low, high-observed], fmt="o", capsize=6)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, ci.BatteryID)
    axis.set_ylabel("Median paired RMSE delta")
    axis.set_title("Outer-test circular time-block bootstrap 95% CI")
    axis.grid(alpha=.25)
    save(fig, "cila_outer_bootstrap_ci_v1.png")


def main():
    metrics = pd.read_csv(BASE / "formal_outer_metrics.csv")
    paired = pd.read_csv(BASE / "formal_outer_paired_deltas.csv")
    predictions = pd.read_csv(BASE / "formal_outer_per_cycle_predictions.csv")
    ci = pd.read_csv(BASE / "formal_outer_bootstrap_ci.csv")
    rmse_log(metrics); deltas(paired); prediction_iqr(predictions); bootstrap_ci(ci)
    names = ["cila_outer_rmse_multiseed_log_v1.png", "cila_outer_paired_delta_v1.png",
             "cila_outer_prediction_iqr_v1.png", "cila_outer_bootstrap_ci_v1.png"]
    pd.DataFrame({"Figure": names, "Stage": "formal_outer_test", "TestTuning": False}).to_csv(
        FIG / "cila_outer_figures_manifest.csv", index=False, encoding="utf-8-sig")
    print("\n".join(str(FIG / name) for name in names))


if __name__ == "__main__":
    main()
