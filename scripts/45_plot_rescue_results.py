"""Plot the auditable paper-protocol rescue results without overwriting old figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

cnn_path = ROOT / "output" / "paper_protocol_rescue" / "B0018" / "seed_42" / "augmented" / "compact"
gmtc_path = ROOT / "output" / "paper_direct_gmtc" / "B0018" / "seed_42" / "width_16"

cnn_metrics = pd.read_csv(cnn_path / "metrics.csv")
cnn_metrics = cnn_metrics[cnn_metrics["Model"] == "CNN"].iloc[0]
cnn_pred = pd.read_csv(cnn_path / "per_cycle_predictions.csv")
cnn_pred = cnn_pred[cnn_pred["Model"] == "CNN"]
gmtc_metrics = pd.read_csv(gmtc_path / "metrics.csv").iloc[0]
gmtc_pred = pd.read_csv(gmtc_path / "per_cycle_predictions.csv")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

names = ["Previous recursive\nG-MTC", "Direct CNN\n+ SCF-GAN", "Direct full\nG-MTC"]
values = [5.1367, float(cnn_metrics["RMSE"]), float(gmtc_metrics["RMSE"])]
colors = ["#8c8c8c", "#2878b5", "#c82423"]
bars = axes[0].bar(names, values, color=colors, width=0.62)
for bar, value in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.3f}", ha="center")
axes[0].axhline(0.9123, color="#f2a104", linestyle="--", linewidth=1.5, label="Paper G-MTC: 0.9123")
axes[0].set_ylabel("RMSE (percentage points)")
axes[0].set_title("B0018 development evaluation")
axes[0].legend(frameon=False)
axes[0].grid(axis="y", alpha=0.25)

axes[1].plot(cnn_pred["Cycle"], cnn_pred["TrueSOH"], color="black", linewidth=1.8, label="Measured SOH")
axes[1].plot(cnn_pred["Cycle"], cnn_pred["PredSOH"], color="#2878b5", linewidth=1.4,
             label=f"Direct CNN (RMSE {float(cnn_metrics['RMSE']):.3f})")
axes[1].plot(gmtc_pred["Cycle"], gmtc_pred["PredSOH"], color="#c82423", linewidth=1.1, alpha=0.85,
             label=f"Direct full G-MTC (RMSE {float(gmtc_metrics['RMSE']):.3f})")
axes[1].axvline(25, color="#666666", linestyle=":", label="First-20% boundary")
axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("SOH (%)")
axes[1].set_title("Saved per-cycle predictions")
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False, fontsize=8)

fig.suptitle("Paper-protocol rescue: successful inference correction, unresolved G-MTC extrapolation", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "paper_protocol_rescue_v1.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(FIGURES / "paper_protocol_rescue_v1.png")
