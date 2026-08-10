"""Create versioned, evidence-only figures for the frozen CILA validation stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "output" / "ieee_innovation_v1" / "phase4"
FIGURES = ROOT / "output" / "figures"


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def paired_rmse() -> None:
    frame = pd.read_csv(PHASE4 / "five_seed_validation_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, (battery, group) in zip(axes, frame.groupby("BatteryID", sort=True)):
        pivot = group.pivot(index="Seed", columns="CILAMode", values="RMSE").sort_index()
        for seed, row in pivot.iterrows():
            axis.plot([0, 1], [row["none"], row["coral"]], marker="o", alpha=0.75, label=f"seed {seed}")
        axis.set_xticks([0, 1], ["Original", "CILA"])
        axis.set_ylabel("Validation RMSE (percentage points)")
        axis.set_title(battery)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Frozen five-seed paired validation (not outer test)")
    save(fig, "cila_paired_rmse_validation_v1.png")


def delta_and_ci() -> None:
    deltas = pd.read_csv(PHASE4 / "paired_validation_deltas.csv")
    ci = pd.read_csv(PHASE4 / "bootstrap_confidence_intervals.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"B0007": "#4472C4", "B0018": "#ED7D31"}
    for battery, group in deltas.groupby("BatteryID", sort=True):
        axes[0].plot(group.Seed, group.DeltaRMSE, marker="o", label=battery, color=colors[battery])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("CILA - Original RMSE")
    axes[0].set_title("Paired seed deltas")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    x = np.arange(len(ci))
    observed = ci.Observed.to_numpy(float)
    lower = ci.CI95Lower.to_numpy(float)
    upper = ci.CI95Upper.to_numpy(float)
    axes[1].errorbar(x, observed, yerr=[observed - lower, upper - observed], fmt="o", capsize=5)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, ci.BatteryID)
    axes[1].set_ylabel("Median paired RMSE delta")
    axes[1].set_title("Circular time-block bootstrap 95% CI")
    axes[1].grid(alpha=0.25)
    save(fig, "cila_delta_bootstrap_validation_v1.png")


def domain_separability() -> None:
    frame = pd.read_csv(PHASE4 / "domain_separability.csv")
    pivot = frame.pivot(index="Seed", columns="CILAMode", values="ValidationDomainBalancedAccuracy").sort_index()
    x = np.arange(len(pivot))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - width / 2, pivot["none"], width, label="Original")
    axis.bar(x + width / 2, pivot["coral"], width, label="CILA")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_xticks(x, pivot.index.astype(str))
    axis.set_xlabel("Seed")
    axis.set_ylabel("Domain balanced accuracy")
    axis.set_title("Validation domain separability diagnostic")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save(fig, "cila_domain_separability_validation_v1.png")


def outer_split() -> None:
    frame = pd.read_csv(PHASE4 / "outer_data_audit" / "outer_aug_audit.csv")
    train = frame.TrainRows.to_numpy(int)
    total = frame.RawRows.to_numpy(int)
    val_end = (total * 0.8).astype(int)
    validation = val_end - train
    test = total - val_end
    x = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x, train, label="Adaptation train 60%")
    axis.bar(x, validation, bottom=train, label="Validation 20%")
    axis.bar(x, test, bottom=train + validation, label="Strict test 20%")
    axis.set_xticks(x, frame.BatteryID)
    axis.set_ylabel("Lifecycle rows")
    axis.set_title("Locked outer protocol (test not yet executed)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save(fig, "cila_outer_split_lock_v1.png")


def main() -> None:
    paired_rmse()
    delta_and_ci()
    domain_separability()
    outer_split()
    names = [
        "cila_paired_rmse_validation_v1.png",
        "cila_delta_bootstrap_validation_v1.png",
        "cila_domain_separability_validation_v1.png",
        "cila_outer_split_lock_v1.png",
    ]
    pd.DataFrame({"Figure": names, "Stage": ["validation", "validation", "validation", "outer-lock"]}).to_csv(
        FIGURES / "cila_validation_figures_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print("\n".join(str(FIGURES / name) for name in names))


if __name__ == "__main__":
    main()
