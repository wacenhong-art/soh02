"""Fixed five-seed ensemble for Original versus monotonic Trend constraint."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "candidate_ensemble"
SEEDS = [42, 52, 62, 72, 82]


def source(seed):
    return ROOT / "output" / ("candidate_stable_physics" if seed == 42 else f"candidate_ensemble/seed_{seed}")


def main():
    rows, predictions = [], []
    for battery in ["B0007", "B0018"]:
        for mode in ["none", "monotonic"]:
            frames, trends = [], []
            for seed in SEEDS:
                root = source(seed); frame = pd.read_csv(root / battery / mode / "gmtc_pred_test.csv")
                comp = pd.read_csv(root / battery / mode / "comp_pred_test.csv").set_index("Cycle")
                frames.append(frame); trends.append(comp.loc[frame.Cycle, "Pred_Trend"].to_numpy())
            cycles = frames[0].Cycle.to_numpy(); true = frames[0].True_SOH.to_numpy()
            if not all(np.array_equal(cycles, f.Cycle.to_numpy()) for f in frames): raise RuntimeError("Cycle mismatch")
            pred = np.mean([f.Pred_SOH.to_numpy() for f in frames], axis=0)
            trend = np.mean(trends, axis=0)
            rmse = float(np.sqrt(np.mean((true-pred)**2))); mae = float(np.mean(np.abs(true-pred)))
            rows.append({"BatteryID": battery, "PhysicsMode": mode, "Seeds": str(SEEDS),
                "RMSE": rmse, "MAE": mae, "MAPE": float(np.sum(np.abs(true-pred))/np.sum(np.abs(true))*100),
                "R2": float(1-np.sum((true-pred)**2)/np.sum((true-true.mean())**2)),
                "TrendViolationRate": float(np.mean(np.diff(trend)>0))})
            predictions.append(pd.DataFrame({"BatteryID":battery,"PhysicsMode":mode,"Cycle":cycles,
                                              "TrueSOH":true,"EnsemblePredSOH":pred,"EnsemblePredTrend":trend}))
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT/"ensemble_metrics.csv",index=False,encoding="utf-8-sig")
    pd.concat(predictions,ignore_index=True).to_csv(OUT/"ensemble_predictions.csv",index=False,encoding="utf-8-sig")
    print(metrics.to_string(index=False))


if __name__ == "__main__": main()
