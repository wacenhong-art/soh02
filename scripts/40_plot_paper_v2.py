from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "paper_v2"
BASE = ROOT / "output" / "paper_baseline_v1"
PHYS = ROOT / "output" / "physics_trend_v3"
OUTER = ROOT / "output" / "ieee_innovation_v1" / "phase4" / "formal_outer_test"

FIGURE_WIDTH_MM = 183
WIDTH = FIGURE_WIDTH_MM / 25.4
COLORS = {
    "neutral": "#666666",
    "light": "#B9C2C9",
    "blue": "#2F6B9A",
    "cyan": "#47A8BD",
    "orange": "#D9822B",
    "red": "#B84A4A",
    "green": "#4F8A5B",
    "purple": "#7A6FA8",
    "black": "#202020",
}
MODE_COLORS = {
    "none": COLORS["neutral"],
    "monotonic": COLORS["blue"],
    "smooth": COLORS["orange"],
    "both": COLORS["green"],
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def panel(ax, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def box(ax, xy, width, height, text, fc, ec="#6E6E6E", fontsize=6.6, lw=0.8, ls="-"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#555555", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=8, lw=lw, color=color))


def figure_framework() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.35), gridspec_kw={"height_ratios": [1.2, 1.0]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, "a")
    x = [0.01, 0.15, 0.29, 0.43, 0.82]
    labels = ["Battery cycles", "HF1/HF2/HF3", "SCF-GAN\n(training only)", "STL decomposition", "Attention Bi-LSTM\nfusion"]
    colors = ["#EEF1F3", "#E6F0F6", "#F7E7D3", "#E8F0DD", "#EAE5F3"]
    widths = [0.11, 0.11, 0.11, 0.13, 0.15]
    for xi, label, fc, width in zip(x, labels, colors, widths):
        box(ax, (xi, 0.60), width, 0.20, label, fc)
    for i in range(3):
        arrow(ax, (x[i] + widths[i], 0.70), (x[i + 1], 0.70))
    branch_x = 0.60
    branch_w = 0.17
    branch_y = [0.74, 0.46, 0.18]
    branch_names = ["Trend -> TimesNet", "Seasonal -> TimesNet", "Residual -> iTransformer"]
    branch_fc = ["#DCEAF5", "#E4F0EA", "#F1E4EF"]
    for yi, name, fc in zip(branch_y, branch_names, branch_fc):
        box(ax, (branch_x, yi), branch_w, 0.15, name, fc)
        arrow(ax, (0.56, 0.70), (branch_x, yi + 0.075))
        arrow(ax, (branch_x + branch_w, yi + 0.075), (0.82, 0.70))
    box(ax, (0.84, 0.24), 0.11, 0.15, "SOH prediction", "#E6F0F6")
    arrow(ax, (0.895, 0.60), (0.895, 0.39))
    ax.text(0.01, 0.96, "Frozen G-MTC forecasting backbone", fontsize=8, fontweight="bold", va="top")

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, "b")
    box(ax, (0.03, 0.58), 0.30, 0.25, "Trend-only physical constraint\nL = Lpred + lambda_phy Lphysics", "#E5F0F7", ec=COLORS["blue"], lw=1.1)
    box(ax, (0.37, 0.58), 0.25, 0.25, "Monotonic penalty\nReLU(T[t+1] - T[t])", "#F3F7FA", ec=COLORS["blue"])
    box(ax, (0.67, 0.58), 0.27, 0.25, "Smoothness penalty\nsecond difference of Trend", "#F3F7FA", ec=COLORS["blue"])
    arrow(ax, (0.33, 0.705), (0.37, 0.705), color=COLORS["blue"])
    arrow(ax, (0.62, 0.705), (0.67, 0.705), color=COLORS["blue"])
    box(ax, (0.03, 0.16), 0.30, 0.25, "CILA fusion-context alignment\nsource-battery covariance penalty", "#F5EBDD", ec=COLORS["orange"], lw=1.1)
    box(ax, (0.40, 0.16), 0.24, 0.25, "Same branch forecasts\nand same fusion model", "#FBF7F1", ec=COLORS["orange"])
    box(ax, (0.70, 0.16), 0.24, 0.25, "No final-SOH monotonicity\nand no test-set tuning", "#FBF7F1", ec=COLORS["orange"])
    arrow(ax, (0.33, 0.285), (0.40, 0.285), color=COLORS["orange"])
    arrow(ax, (0.64, 0.285), (0.70, 0.285), color=COLORS["orange"])
    ax.text(0.03, 0.94, "Two isolated, removable interventions", fontsize=8, fontweight="bold", va="top")
    save_bundle(fig, "fig1_framework_energy_v2")


def model_rows() -> pd.DataFrame:
    comparison = pd.read_csv(BASE / "comparison_multiseed_metrics.csv")[["BatteryID", "Seed", "Model", "RMSE"]]
    gmtc = pd.read_csv(PHYS / "ablation_metrics.csv")
    gmtc = gmtc[gmtc["PhysicsMode"].eq("none")][["BatteryID", "Seed", "RMSE"]].assign(Model="G-MTC")
    return pd.concat([comparison, gmtc], ignore_index=True)


def nearest_median_seed(df: pd.DataFrame, metric: str, group_filter: pd.Series) -> int:
    subset = df[group_filter].copy()
    median = subset[metric].median()
    return int(subset.loc[(subset[metric] - median).abs().idxmin(), "Seed"])


def figure_baseline() -> None:
    rows = model_rows()
    pred = pd.read_csv(BASE / "per_cycle_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.75), constrained_layout=True)
    order = ["CNN", "LSTM", "Transformer", "G-MTC"]
    for ax, battery, label in zip(axes[0], ["B0007", "B0018"], ["a", "b"]):
        panel(ax, label)
        sub = rows[rows["BatteryID"].eq(battery)]
        med = sub.groupby("Model")["RMSE"].median().reindex(order)
        sd = sub.groupby("Model")["RMSE"].std().reindex(order)
        xs = np.arange(len(order))
        ax.bar(xs, med, yerr=sd, color=[COLORS["light"], COLORS["cyan"], COLORS["purple"], COLORS["orange"]], width=0.65, capsize=2, linewidth=0)
        for j, model in enumerate(order):
            values = sub[sub["Model"].eq(model)]["RMSE"].to_numpy()
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(np.full(len(values), j) + jitter, values, s=9, facecolor="white", edgecolor=COLORS["black"], linewidth=0.45, zorder=3)
        if battery == "B0018":
            ax.axhline(0.9123, color=COLORS["red"], linestyle="--", linewidth=0.9, label="Paper G-MTC reference")
            ax.legend(loc="upper left")
        ax.set_xticks(xs, order, rotation=18, ha="right", rotation_mode="anchor")
        ax.set_ylabel("RMSE (%)")
        ax.set_title(battery, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)

    for ax, battery, label in zip(axes[1], ["B0007", "B0018"], ["c", "d"]):
        panel(ax, label)
        sub = pred[pred["BatteryID"].eq(battery)]
        grouped = sub.groupby("Cycle")
        cycles = np.array(sorted(sub["Cycle"].unique()))
        truth = grouped["TrueSOH"].median().reindex(cycles).to_numpy()
        median = grouped["PredSOH"].median().reindex(cycles).to_numpy()
        lower = grouped["PredSOH"].quantile(0.25).reindex(cycles).to_numpy()
        upper = grouped["PredSOH"].quantile(0.75).reindex(cycles).to_numpy()
        ax.plot(cycles, truth, color=COLORS["black"], label="Measured SOH")
        ax.plot(cycles, median, color=COLORS["orange"], label="G-MTC median")
        ax.fill_between(cycles, lower, upper, color=COLORS["orange"], alpha=0.20, linewidth=0, label="Seed IQR")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.set_title(f"{battery}; five random seeds", loc="left")
        ax.legend(loc="best")
        ax.grid(color="#ECECEC", linewidth=0.45)
    save_bundle(fig, "fig2_baseline_energy_v2")


def figure_physics() -> None:
    metrics = pd.read_csv(PHYS / "ablation_metrics.csv")
    trend = pd.read_csv(PHYS / "trend_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.65), constrained_layout=True)
    for ax, battery, label in zip(axes[0], ["B0007", "B0018"], ["a", "b"]):
        panel(ax, label)
        pivot = metrics[metrics["BatteryID"].eq(battery)].pivot(index="Seed", columns="PhysicsMode", values="RMSE")
        for i, mode in enumerate(["monotonic", "smooth", "both"]):
            delta = pivot[mode] - pivot["none"]
            x = np.full(len(delta), i)
            ax.scatter(x, delta, color=MODE_COLORS[mode], s=18, zorder=3)
            ax.hlines(delta.median(), i - 0.25, i + 0.25, color=COLORS["black"], linewidth=1.5)
        ax.axhline(0, color=COLORS["black"], linewidth=0.7)
        ax.set_xticks(range(3), ["Monotonic", "Smooth", "Both"])
        ax.set_ylabel("Paired delta RMSE (%)")
        ax.set_title(f"{battery}; negative favors constraint", loc="left")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)

    ax = axes[1, 0]
    panel(ax, "c")
    order = ["none", "monotonic", "smooth", "both"]
    for bidx, battery in enumerate(["B0007", "B0018"]):
        sub = metrics[metrics["BatteryID"].eq(battery)]
        med = sub.groupby("PhysicsMode")["TrendViolationRate"].median().reindex(order)
        xs = np.arange(4) + (bidx - 0.5) * 0.18
        ax.plot(xs, med, marker="o", markersize=3.5, color=[COLORS["blue"], COLORS["orange"]][bidx], label=battery)
        for j, mode in enumerate(order):
            values = sub[sub["PhysicsMode"].eq(mode)]["TrendViolationRate"].to_numpy()
            ax.scatter(np.full(len(values), xs[j]), values, s=7, alpha=0.45, color=[COLORS["blue"], COLORS["orange"]][bidx])
    ax.set_xticks(range(4), ["Original", "Mono", "Smooth", "Both"])
    ax.set_ylabel("Trend violation rate")
    ax.set_title("Held-out recursive Trend trajectories", loc="left")
    ax.legend()
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)

    ax = axes[1, 1]
    panel(ax, "d")
    battery = "B0018"
    sub = trend[trend["BatteryID"].eq(battery)]
    cycles = np.array(sorted(sub["Cycle"].unique()))
    truth = sub.groupby("Cycle")["TrueTrend"].median().reindex(cycles).to_numpy()
    ax.plot(cycles, truth, color=COLORS["black"], label="STL Trend target")
    styles = {"none": "-", "monotonic": "--", "both": ":"}
    for mode in ["none", "monotonic", "both"]:
        s = sub[sub["PhysicsMode"].eq(mode)].groupby("Cycle")["PredTrend"].median().reindex(cycles)
        ax.plot(cycles, s, color=MODE_COLORS[mode], linestyle=styles[mode], label={"none": "Original", "monotonic": "Monotonic", "both": "Both"}[mode])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Trend component")
    ax.set_title(f"{battery}; five-seed medians", loc="left")
    ax.legend(ncol=2)
    ax.grid(color="#E8E8E8", linewidth=0.5)
    save_bundle(fig, "fig3_physics_energy_v2")


def figure_outer() -> None:
    paired = pd.read_csv(OUTER / "formal_outer_paired_deltas.csv")
    ci = pd.read_csv(OUTER / "formal_outer_bootstrap_ci.csv")
    pred = pd.read_csv(OUTER / "formal_outer_per_cycle_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.75), constrained_layout=True)

    ax = axes[0, 0]
    panel(ax, "a")
    for i, battery in enumerate(["B0043", "B0044"]):
        sub = paired[paired["BatteryID"].eq(battery)]
        x = np.full(len(sub), i) + np.linspace(-0.10, 0.10, len(sub))
        ax.scatter(x, sub["DeltaRMSE"], s=20, color=[COLORS["blue"], COLORS["orange"]][i], label=battery)
        ax.hlines(sub["DeltaRMSE"].median(), i - 0.25, i + 0.25, color=COLORS["black"], linewidth=1.5)
    ax.axhline(0, color=COLORS["black"], linewidth=0.7)
    ax.set_xticks([0, 1], ["B0043", "B0044"])
    ax.set_ylabel("CILA - Original RMSE (%)")
    ax.set_title("Five paired random seeds", loc="left")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)

    ax = axes[0, 1]
    panel(ax, "b")
    y = np.arange(len(ci))
    obs = ci["Observed"].to_numpy()
    lo = ci["CI95Lower"].to_numpy()
    hi = ci["CI95Upper"].to_numpy()
    ax.errorbar(obs, y, xerr=np.vstack([obs - lo, hi - obs]), fmt="o", color=COLORS["purple"], capsize=3)
    ax.axvline(0, color=COLORS["black"], linewidth=0.7)
    ax.set_yticks(y, ci["BatteryID"])
    ax.set_xlabel("Paired median delta RMSE (%)")
    ax.set_title("Circular time-block bootstrap, 95% CI", loc="left")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)

    for ax, battery, label in zip(axes[1], ["B0043", "B0044"], ["c", "d"]):
        panel(ax, label)
        sub = pred[pred["BatteryID"].eq(battery)]
        for mode, color in [("Original", COLORS["neutral"]), ("CILA", COLORS["orange"])]:
            s = sub[sub["Mode"].eq(mode)]
            s = s.assign(Error=s["Pred_SOH"] - s["True_SOH"])
            grouped = s.groupby("Cycle")["Error"]
            cycles = np.array(sorted(s["Cycle"].unique()))
            median = grouped.median().reindex(cycles).to_numpy()
            lower = grouped.quantile(0.25).reindex(cycles).to_numpy()
            upper = grouped.quantile(0.75).reindex(cycles).to_numpy()
            ax.plot(cycles, median, color=color, label=f"{mode} median")
            ax.fill_between(cycles, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.axhline(0, color=COLORS["black"], linewidth=0.7)
        ax.set_yscale("symlog", linthresh=1)
        # Symlog exponent glyphs render smaller than their parent tick label.
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Prediction error (%)")
        ax.set_title(f"{battery}; five-seed median and IQR", loc="left")
        ax.legend()
        ax.grid(color="#E8E8E8", linewidth=0.45)
    save_bundle(fig, "fig4_outer_validation_energy_v2")


def write_manifest() -> None:
    rows = [
        ("fig1_framework_energy_v2", "Schematic-led composite", "Frozen G-MTC backbone and two isolated intervention locations"),
        ("fig2_baseline_energy_v2", "Quantitative grid", "Auditable baseline does not reproduce the paper value and CNN is best in the current comparison"),
        ("fig3_physics_energy_v2", "Quantitative grid", "Trend constraints produce battery- and seed-dependent trade-offs"),
        ("fig4_outer_validation_energy_v2", "Quantitative grid", "CILA fails joint outer-battery confirmation and cannot repair recursive branch failure"),
    ]
    with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FigureStem", "Archetype", "CoreConclusion", "Formats"])
        for stem, archetype, conclusion in rows:
            writer.writerow([stem, archetype, conclusion, "png;pdf;svg;tiff"])


def main() -> None:
    figure_framework()
    figure_baseline()
    figure_physics()
    figure_outer()
    write_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
