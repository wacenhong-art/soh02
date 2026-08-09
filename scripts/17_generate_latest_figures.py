"""Generate paper-ready figures from the latest strict-test experiment outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
PAPER_B0018_RMSE = 0.9123


def _save(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {path}")


def _annotate_bars(ax, bars, decimals=3, pad_ratio=0.015):
    ymax = max((bar.get_height() for bar in bars), default=1.0)
    pad = max(ymax * pad_ratio, 0.015)
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + pad,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def load_results():
    baseline = pd.read_csv(OUTPUT / "baseline_compare_result.csv", index_col=0)
    physics = pd.read_csv(
        OUTPUT / "candidate_stable_physics" / "physics_ablation_metrics.csv"
    )
    residual = {
        battery: pd.read_csv(
            OUTPUT
            / "candidate_residual_module"
            / battery
            / "residual_module_ablation.csv"
        )
        for battery in ("B0007", "B0018")
    }
    predictions = {
        battery: pd.read_csv(
            OUTPUT
            / "candidate_residual_module"
            / battery
            / "residual_module_predictions.csv"
        )
        for battery in ("B0007", "B0018")
    }
    multiseed = pd.read_csv(
        OUTPUT / "candidate_residual_module" / "five_seed_raw.csv"
    )
    return baseline, physics, residual, predictions, multiseed


def plot_method_rmse(baseline, physics, residual):
    b18 = physics[physics["BatteryID"] == "B0018"].set_index("PhysicsMode")
    frozen = residual["B0018"].set_index("Model")
    labels = [
        "CNN",
        "LSTM",
        "Transformer",
        "Original\nG-MTC",
        "Physics-Guided\nG-MTC",
        "Physics +\nFrozen ELM",
    ]
    values = [
        float(baseline.loc["CNN", "RMSE%"]),
        float(baseline.loc["LSTM", "RMSE%"]),
        float(baseline.loc["Transformer", "RMSE%"]),
        float(b18.loc["none", "RMSE"]),
        float(b18.loc["monotonic", "RMSE"]),
        float(frozen.loc["G-MTC+monotonic+ELM-frozen", "RMSE%"]),
    ]
    colors = ["#b0bec5", "#b0bec5", "#78909c", "#ffb74d", "#66bb6a", "#1565c0"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="#37474f", linewidth=0.8)
    _annotate_bars(ax, bars)
    ax.axhline(
        PAPER_B0018_RMSE,
        color="#c62828",
        linestyle="--",
        linewidth=1.5,
        label=f"Paper B0018 G-MTC reference: {PAPER_B0018_RMSE:.4f}%",
    )
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_title("NASA B0018 RMSE: Baselines vs Current G-MTC Pipeline")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.5,
        -0.16,
        "Target protocol: first 60% train adaptation, 20% validation, final 20% strict test; lower is better.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#455a64",
    )
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig11_rmse_bar.png")


def plot_physics_ablation(physics):
    modes = ["none", "monotonic", "smooth", "both"]
    display = ["Original", "Monotonic", "Smooth", "Both"]
    colors = ["#b0bec5", "#43a047", "#ffb74d", "#1e88e5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    for ax, battery in zip(axes, ("B0007", "B0018")):
        part = physics[physics["BatteryID"] == battery].set_index("PhysicsMode")
        values = [float(part.loc[mode, "RMSE"]) for mode in modes]
        bars = ax.bar(display, values, color=colors, edgecolor="#37474f", linewidth=0.7)
        _annotate_bars(ax, bars)
        ax.set_title(f"{battery} Trend-Physics Ablation")
        ax.set_ylabel("RMSE (percentage points)")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if battery == "B0018":
            ax.axhline(
                PAPER_B0018_RMSE,
                color="#c62828",
                linestyle="--",
                linewidth=1.3,
                label="Paper reference",
            )
            ax.legend(frameon=False)

    fig.suptitle("Physics-Guided STL Trend Constraint Ablation (lambda = 0.01)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig16_physics_ablation_rmse.png")


def plot_trend_violation(physics):
    modes = ["none", "monotonic", "smooth", "both"]
    display = ["Original", "Monotonic", "Smooth", "Both"]
    x = np.arange(len(modes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.3))

    all_bars = []
    for offset, battery, color in (
        (-width / 2, "B0007", "#64b5f6"),
        (width / 2, "B0018", "#ef5350"),
    ):
        part = physics[physics["BatteryID"] == battery].set_index("PhysicsMode")
        values = [100.0 * float(part.loc[mode, "TrendViolationRate"]) for mode in modes]
        bars = ax.bar(x + offset, values, width, label=battery, color=color, edgecolor="#37474f")
        all_bars.extend(bars)

    _annotate_bars(ax, all_bars, decimals=1, pad_ratio=0.02)
    ax.set_xticks(x, display)
    ax.set_ylabel("Trend violation rate (%)")
    ax.set_title("Predicted STL Trend Physical-Consistency Diagnostic")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    ax.text(
        0.5,
        -0.14,
        "The B0018 monotonic loss lowers RMSE but does not lower the measured violation rate in this run.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#b71c1c",
    )
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig17_trend_violation_rate.png")


def plot_residual_module(residual):
    batteries = ["B0007", "B0018"]
    x = np.arange(len(batteries))
    width = 0.34
    base_values = []
    frozen_values = []
    for battery in batteries:
        part = residual[battery].set_index("Model")
        base_values.append(float(part.loc["G-MTC+monotonic", "RMSE%"]))
        frozen_values.append(float(part.loc["G-MTC+monotonic+ELM-frozen", "RMSE%"]))

    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    base_bars = ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Physics-Guided G-MTC",
        color="#90a4ae",
        edgecolor="#37474f",
    )
    frozen_bars = ax.bar(
        x + width / 2,
        frozen_values,
        width,
        label="+ Frozen ELM residual module",
        color="#1565c0",
        edgecolor="#37474f",
    )
    _annotate_bars(ax, list(base_bars) + list(frozen_bars))
    ax.set_xticks(x, batteries)
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_title("Strict-Test Effect of the Frozen Residual Module")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    ax.text(
        0.5,
        -0.14,
        "The correction model is frozen during test inference and receives no true test SOH labels.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#455a64",
    )
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig18_frozen_residual_module_rmse.png")


def plot_cycle_predictions(predictions, residual):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=False)
    for ax, battery in zip(axes, ("B0007", "B0018")):
        pred = predictions[battery]
        metrics = residual[battery].set_index("Model")
        base_rmse = float(metrics.loc["G-MTC+monotonic", "RMSE%"])
        frozen_rmse = float(metrics.loc["G-MTC+monotonic+ELM-frozen", "RMSE%"])
        ax.plot(pred["Cycle"], pred["True_SOH"], color="#212121", linewidth=2.0, label="True SOH")
        ax.plot(
            pred["Cycle"],
            pred["Pred_Light_GMTC"],
            color="#ef6c00",
            linewidth=1.5,
            linestyle="--",
            label=f"Physics G-MTC (RMSE {base_rmse:.3f}%)",
        )
        ax.plot(
            pred["Cycle"],
            pred["Pred_ELM_Frozen"],
            color="#1565c0",
            linewidth=1.7,
            label=f"+ Frozen ELM (RMSE {frozen_rmse:.3f}%)",
        )
        ax.set_title(f"{battery} Final 20% Strict-Test Cycles")
        ax.set_ylabel("SOH (%)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False, ncol=3, fontsize=8.5)
    axes[-1].set_xlabel("Cycle")
    fig.suptitle("Per-Cycle SOH Prediction of the Latest Pipeline", y=0.995)
    fig.tight_layout()
    _save(fig, "fig19_latest_cycle_predictions.png")


def plot_multiseed(multiseed):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    width = 0.36
    for ax, battery in zip(axes, ("B0007", "B0018")):
        part = multiseed[multiseed["BatteryID"] == battery].sort_values("Seed")
        x = np.arange(len(part))
        base = ax.bar(
            x - width / 2,
            part["BaseRMSE"],
            width,
            label="Physics G-MTC",
            color="#90a4ae",
            edgecolor="#37474f",
        )
        frozen = ax.bar(
            x + width / 2,
            part["FrozenELMRMSE"],
            width,
            label="+ Frozen ELM",
            color="#1565c0",
            edgecolor="#37474f",
        )
        _annotate_bars(ax, list(base) + list(frozen), decimals=2, pad_ratio=0.02)
        ax.set_xticks(x, [str(seed) for seed in part["Seed"]])
        ax.set_xlabel("Random seed")
        ax.set_ylabel("RMSE (percentage points)")
        ax.set_title(f"{battery}: 4/5 Seeds Improved")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if battery == "B0018":
            ax.axhline(
                PAPER_B0018_RMSE,
                color="#c62828",
                linestyle="--",
                linewidth=1.3,
                label="Paper reference",
            )
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Five-Seed Robustness of the Frozen Residual Module", y=1.02)
    fig.tight_layout()
    _save(fig, "fig20_multiseed_robustness.png")


def write_manifest():
    rows = [
        ("fig11_rmse_bar.png", "B0018 baselines, original G-MTC, physics G-MTC, and final module"),
        ("fig16_physics_ablation_rmse.png", "Original/monotonic/smooth/both Trend-physics ablation"),
        ("fig17_trend_violation_rate.png", "Trend physical-consistency diagnostic"),
        ("fig18_frozen_residual_module_rmse.png", "Frozen residual module RMSE effect"),
        ("fig19_latest_cycle_predictions.png", "Per-cycle strict-test SOH fitting"),
        ("fig20_multiseed_robustness.png", "Five-seed robustness diagnostic"),
    ]
    pd.DataFrame(rows, columns=["Figure", "Description"]).to_csv(
        FIGURES / "latest_figures_manifest.csv", index=False
    )


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 15,
        }
    )
    baseline, physics, residual, predictions, multiseed = load_results()
    plot_method_rmse(baseline, physics, residual)
    plot_physics_ablation(physics)
    plot_trend_violation(physics)
    plot_residual_module(residual)
    plot_cycle_predictions(predictions, residual)
    plot_multiseed(multiseed)
    write_manifest()


if __name__ == "__main__":
    main()
