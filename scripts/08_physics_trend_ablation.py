"""Physics-guided STL-Trend ablation, isolated from other project innovations.

Only the Trend TimesNet objective is changed:
L_total = L_prediction + lambda_phy * L_physics.
Adjacency is constructed inside a DatasetID/BatteryID trajectory sorted by Cycle;
it is never inferred from shuffled mini-batches.
"""
import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("time_series", ROOT / "scripts" / "03_time_series.py")
ts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ts)
FUSION_SPEC = importlib.util.spec_from_file_location("attention_fusion", ROOT / "scripts" / "04_attention_fusion.py")
fusion = importlib.util.module_from_spec(FUSION_SPEC)
FUSION_SPEC.loader.exec_module(fusion)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_COLS = ["HF1", "HF2", "HF3"]
BASE_SEED = ts.SEED


def reset_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def stable_seed(base, battery, component):
    """Independent of battery execution order and ablation list length."""
    battery_offset = {"B0007": 7000, "B0018": 18000}[battery]
    component_offset = {"Seasonal": 100, "Residual": 200, "Trend": 300}[component]
    return int(base + battery_offset + component_offset)


def physics_loss(pred, mode):
    """Loss on one ordered, continuous trajectory; pred shape is [time, 1]."""
    if mode == "none" or pred.shape[0] < 2:
        return pred.new_zeros(())
    terms = []
    if mode in {"monotonic", "both"}:
        terms.append(torch.relu(pred[1:] - pred[:-1]).mean())
    if mode in {"smooth", "both"} and pred.shape[0] >= 3:
        terms.append(torch.square(pred[2:] - 2 * pred[1:-1] + pred[:-2]).mean())
    return torch.stack(terms).sum() if terms else pred.new_zeros(())


def train_component(model, trajectories, val_xy, mode, lam, epochs, patience, log_path):
    opt = torch.optim.AdamW(model.parameters(), lr=ts.LR, weight_decay=ts.WEIGHT_DECAY)
    mse = nn.MSELoss()
    vx, vy = (torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for v in val_xy)
    best, wait, best_state, rows = float("inf"), 0, None, []
    for epoch in range(epochs):
        model.train(); pred_sum = phy_sum = 0.0
        order = list(range(len(trajectories))); random.shuffle(order)
        for idx in order:
            x, y = trajectories[idx]
            x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
            y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            opt.zero_grad(); pred = model(x)
            lp = mse(pred, y); lphy = physics_loss(pred, mode)
            (lp + lam * lphy).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pred_sum += lp.item(); phy_sum += lphy.item()
        model.eval()
        with torch.no_grad(): val = mse(model(vx), vy.unsqueeze(-1)).item()
        rows.append({"Epoch": epoch + 1, "PredictionLoss": pred_sum / len(order),
                     "PhysicsLoss": phy_sum / len(order), "ValidationPredictionLoss": val})
        if val < best - 1e-7:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    return model, best


def make_trajectories(aug, norm, component_state, values, mean, std):
    result = []
    required = {"DatasetID", "Cycle"}
    if not required.issubset(aug.columns):
        raise ValueError(f"增强数据缺少轨迹字段: {required - set(aug.columns)}")
    for dataset_id, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        cycles = group["Cycle"].to_numpy(int)
        if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
            raise ValueError(f"DatasetID={dataset_id} Cycle不连续，拒绝计算物理相邻损失")
        gx = np.concatenate([norm(group), component_state[cycles]], axis=1).astype(np.float32)
        gy = (values[cycles] - mean) / std
        result.append(ts.windows(gx, gy))
    return result


def metric_row(battery, mode, lam, true, pred, trend_pred):
    delta = np.diff(trend_pred)
    return {"BatteryID": battery, "PhysicsMode": mode, "LambdaPhy": lam,
            "RMSE": mean_squared_error(true, pred) ** 0.5,
            "MAE": mean_absolute_error(true, pred),
            # Paper Eq. (30): sum absolute error divided by sum reference SOH.
            "MAPE": np.sum(np.abs(true - pred)) / np.maximum(np.sum(np.abs(true)), 1e-8) * 100,
            "R2": r2_score(true, pred),
            "TrendViolationRate": float(np.mean(delta > 0)) if len(delta) else 0.0}


def run_battery(battery, modes, lam, epochs, patience, out_dir, seed):
    raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
    raw = raw[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    aug_path = ROOT / "output" / f"{battery}_aug_train_hf.csv"
    if not aug_path.exists():
        raise FileNotFoundError(f"缺少 {aug_path.name}；请先用SCF-GAN为该目标电池生成训练增强数据")
    aug = pd.read_csv(aug_path).dropna().reset_index(drop=True)
    n = len(raw); train_end, val_end = int(n * .6), int(n * .8)
    soh = raw.SOH.to_numpy(np.float32) * 100
    dec = STL(soh, period=ts.STL_PERIOD, robust=True).fit()
    comps = {"Trend": np.asarray(dec.trend, np.float32),
             "Seasonal": np.asarray(dec.seasonal, np.float32),
             "Residual": np.asarray(dec.resid, np.float32)}
    hf = raw[HF_COLS].to_numpy(np.float32); hm, hs = hf[:train_end].mean(0), hf[:train_end].std(0)
    hs[hs < 1e-8] = 1
    norm = lambda frame: (frame[HF_COLS].to_numpy(np.float32) - hm) / hs
    fixed_predictions = {}
    all_rows, pred_frames = [], []
    # Seasonal original TimesNet and Residual iTransformer are identical for all ablations.
    trained = {}
    component_modes = [("Seasonal", "none"), ("Residual", "none")] + [("Trend", m) for m in modes]
    for name, mode in component_modes:
        # Every component is independent of execution order; all Trend modes share initialization/order.
        reset_seed(stable_seed(seed, battery, name))
        values = comps[name]; mean, std = float(values[:train_end].mean()), max(float(values[:train_end].std()), 1e-8)
        state = ((values - mean) / std).reshape(-1, 1); lag = np.vstack([state[:1], state[:-1]])
        raw_x = np.concatenate([norm(raw), lag], axis=1).astype(np.float32)
        trajectories = make_trajectories(aug, norm, lag, values, mean, std)
        vx, vy = ts.windows(raw_x[train_end:val_end], (values[train_end:val_end] - mean) / std)
        if name == "Residual":
            model = ts.iTransformer().to(DEVICE)
        elif name == "Seasonal":
            model = ts.LightweightTimesNet().to(DEVICE)
        else:
            model = ts.TimesNet().to(DEVICE)
        model, best = train_component(model, trajectories, (vx, vy), mode if name == "Trend" else "none",
                                      lam if name == "Trend" else 0.0, epochs, patience,
                                      out_dir / f"{battery}_{name.lower()}_{mode}_train_log.csv")
        trained[(name, mode)] = {"model": model, "raw_x": raw_x, "values": values,
                                 "mean": mean, "std": std, "best": best}
    for mode in modes:
        mode_dir = out_dir / battery / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        selected = [("Trend", trained[("Trend", mode)]),
                    ("Seasonal", trained[("Seasonal", "none")]),
                    ("Residual", trained[("Residual", "none")])]
        split_ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
        split_frames = {}
        for split, (split_start, split_end) in split_ranges.items():
            # Two histories: one for component prediction and one for fusion windows.
            history_start = 0 if split == "train" else max(0, split_start - 2 * ts.SEQ_LEN)
            columns = {}
            for name, item in selected:
                xw, true_component = ts.windows(item["raw_x"][history_start:split_end],
                                                item["values"][history_start:split_end])
                prediction = ts.predict(item["model"], xw) * item["std"] + item["mean"]
                cycles = np.arange(history_start + ts.SEQ_LEN, split_end)
                if "Cycle" in columns and not np.array_equal(columns["Cycle"], cycles):
                    raise RuntimeError("分量Cycle未对齐")
                columns["Cycle"] = cycles
                columns[f"True_{name}"] = true_component
                columns[f"Pred_{name}"] = prediction
            for hf_col in HF_COLS:
                columns[hf_col] = raw.loc[columns["Cycle"], hf_col].to_numpy(np.float32)
            split_frames[split] = pd.DataFrame(columns)
            split_frames[split].to_csv(mode_dir / f"comp_pred_{split}.csv", index=False, encoding="utf-8-sig")
        fusion_seed = stable_seed(seed, battery, "Trend") + 900
        model_path = mode_dir / "fusion_att_bilstm_paper.pth"
        fused_metrics, fused_test = fusion.run_fusion(
            str(mode_dir), str(mode_dir), str(model_path), fusion_seed, epochs, patience
        )
        trend_lookup = split_frames["test"].set_index("Cycle")["Pred_Trend"]
        true_trend_lookup = split_frames["test"].set_index("Cycle")["True_Trend"]
        trend_pred = trend_lookup.loc[fused_test.Cycle].to_numpy()
        true_trend = true_trend_lookup.loc[fused_test.Cycle].to_numpy()
        true = fused_test.True_SOH.to_numpy(); pred = fused_test.Pred_SOH.to_numpy()
        row = metric_row(battery, mode, 0.0 if mode == "none" else lam, true, pred, trend_pred)
        row.update({f"Fusion{k}": value for k, value in fused_metrics.items() if k.startswith("MeanWeight")})
        all_rows.append(row)
        frame = fused_test.rename(columns={"True_SOH": "TrueSOH", "Pred_SOH": "PredSOH"}).copy()
        frame.insert(0, "PhysicsMode", mode); frame.insert(0, "BatteryID", battery)
        frame["TrueTrend"] = true_trend; frame["PredTrend"] = trend_pred
        pred_frames.append(frame)
    return all_rows, pred_frames, {"BatteryID": battery, "Rows": n, "TrainEnd": train_end,
        "ValidationEnd": val_end, "TestRows": n-val_end, "Protocol": "target-first-60%-labeled-adaptation; next-20%-validation; last-20%-strict-test"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batteries", default="B0007,B0018")
    p.add_argument("--modes", default="none,monotonic,smooth,both")
    p.add_argument("--lambda-phy", type=float, default=.01, choices=[0, .001, .01, .1])
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(ROOT / "output" / "physics_trend"))
    a = p.parse_args(); modes = a.modes.split(",")
    if not set(modes) <= {"none", "monotonic", "smooth", "both"}: raise ValueError("非法physics_mode")
    reset_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, frames, protocols = [], [], []
    for battery in a.batteries.split(","):
        r, f, protocol = run_battery(battery.strip(), modes, a.lambda_phy, a.epochs, a.patience, out, a.seed)
        rows += r; frames += f; protocols.append(protocol)
    metrics = pd.DataFrame(rows); predictions = pd.concat(frames, ignore_index=True)
    metrics.to_csv(out / "physics_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "physics_ablation_predictions.csv", index=False, encoding="utf-8-sig")
    config = {"physics_modes": modes, "lambda_phy": a.lambda_phy, "epochs": a.epochs, "seed": a.seed,
              "timesnet": "paper-deep-trend-light-seasonal", "fusion": "paper-attention-bilstm",
              "transfer_pretrain": False, "oselm": False,
              "causal_multirate": False, "scf_gan_algorithm_changed": False,
              "physics_applied_to": "STL Trend branch only", "protocols": protocols}
    (out / "physics_ablation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for battery, g in predictions.groupby("BatteryID"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for mode, q in g.groupby("PhysicsMode"):
            ax[0].plot(q.Cycle, q.PredSOH, label=mode); ax[1].plot(q.Cycle, q.PredTrend, label=mode)
        q = g[g.PhysicsMode == modes[0]]; ax[0].plot(q.Cycle, q.TrueSOH, "k--", label="true")
        ax[1].plot(q.Cycle, q.TrueTrend, "k--", label="true")
        ax[0].set_title(f"{battery} strict test SOH"); ax[1].set_title("STL Trend")
        for z in ax: z.set_xlabel("Cycle"); z.grid(alpha=.3); z.legend()
        fig.tight_layout(); fig.savefig(out / f"{battery}_physics_ablation.png", dpi=180); plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__": main()
