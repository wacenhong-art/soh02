"""Causal monotonic projection on Trend only, followed by frozen-protocol fusion refit."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fusion", ROOT / "scripts" / "04_attention_fusion.py")
fusion = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(fusion)


def causal_nonincreasing(values):
    """Online projection using no future values."""
    return np.minimum.accumulate(np.asarray(values, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", default=str(ROOT / "output" / "candidate_lambda_0p001"))
    p.add_argument("--output-root", default=str(ROOT / "output" / "candidate_trend_projection"))
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    source, out = Path(a.source_root), Path(a.output_root); rows = []
    for battery in ["B0007", "B0018"]:
        src, dst = source / battery / "monotonic", out / battery / "projected"
        dst.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            frame = pd.read_csv(src / f"comp_pred_{split}.csv")
            frame["Pred_Trend"] = causal_nonincreasing(frame.Pred_Trend)
            frame.to_csv(dst / f"comp_pred_{split}.csv", index=False, encoding="utf-8-sig")
        metrics, pred = fusion.run_fusion(
            str(dst), str(dst), str(dst / "fusion_att_bilstm_projection.pth"),
            a.seed + int(battery[-2:]) + 1400, a.epochs, a.patience,
        )
        comp = pd.read_csv(dst / "comp_pred_test.csv").set_index("Cycle")
        trend = comp.loc[pred.Cycle, "Pred_Trend"].to_numpy()
        rows.append({"BatteryID": battery, "Module": "causal_trend_projection",
                     **metrics, "TrendViolationRate": float(np.mean(np.diff(trend) > 0))})
    result = pd.DataFrame(rows)
    result.to_csv(out / "projection_metrics.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
