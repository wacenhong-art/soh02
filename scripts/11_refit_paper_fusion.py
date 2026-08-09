"""Refit paper Attention Bi-LSTM on frozen component predictions."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fusion", ROOT / "scripts" / "04_attention_fusion.py")
fusion = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(fusion)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component-root", default=str(ROOT / "output" / "paper_aligned_physics"))
    p.add_argument("--output-root", default=str(ROOT / "output" / "paper_aligned_physics_regularized"))
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fusion-mode", choices=["paper", "iprf"], default="paper")
    a = p.parse_args(); source, out = Path(a.component_root), Path(a.output_root)
    rows = []
    for battery in ["B0007", "B0018"]:
        for mode in ["none", "monotonic", "smooth", "both"]:
            input_dir = source / battery / mode; output_dir = out / battery / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics, pred = fusion.run_fusion(
                str(input_dir), str(output_dir), str(output_dir / "fusion_att_bilstm_paper.pth"),
                a.seed + int(battery[-2:]) + 900, a.epochs, a.patience, a.fusion_mode,
            )
            comp = pd.read_csv(input_dir / "comp_pred_test.csv").set_index("Cycle")
            trend = comp.loc[pred.Cycle, "Pred_Trend"].to_numpy()
            violation = float(np.mean(np.diff(trend) > 0)) if len(trend) > 1 else 0.0
            rows.append({"BatteryID": battery, "PhysicsMode": mode,
                "LambdaPhy": 0.0 if mode == "none" else 0.01, **metrics,
                "TrendViolationRate": violation})
    result = pd.DataFrame(rows)
    result.to_csv(out / "paper_fusion_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
