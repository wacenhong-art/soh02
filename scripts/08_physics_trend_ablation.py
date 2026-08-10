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
    battery_offset = {
        "B0007": 7000,
        "B0018": 18000,
        "B0043": 43000,
        "B0044": 44000,
    }[battery]
    component_offset = {"Seasonal": 100, "Residual": 200, "Trend": 300}[component]
    return int(base + battery_offset + component_offset)


def physics_loss(pred, mode):
    """Loss on one ordered trajectory in physical Trend percentage points."""
    if mode == "none" or pred.shape[0] < 2:
        return pred.new_zeros(())
    terms = []
    if mode in {"monotonic", "both"}:
        terms.append(torch.relu(pred[1:] - pred[:-1]).mean())
    if mode in {"smooth", "both"} and pred.shape[0] >= 3:
        terms.append(torch.square(pred[2:] - 2 * pred[1:-1] + pred[:-2]).mean())
    return torch.stack(terms).sum() if terms else pred.new_zeros(())


def causal_component_predict(model, norm_hf, values, mean, std, target_start, target_end,
                             known_end, use_component_lag=True, clip_recursive_state=False):
    """Recursive component inference without using labels at or after known_end."""
    if target_start < ts.SEQ_LEN:
        raise ValueError(f"target_start={target_start} 小于SEQ_LEN={ts.SEQ_LEN}")
    state = np.full(len(norm_hf), np.nan, dtype=np.float32)
    if use_component_lag:
        state[:known_end] = (np.asarray(values[:known_end], np.float32) - mean) / std
        observed = state[:known_end]
        lower, upper = float(np.min(observed)), float(np.max(observed))
        margin = 0.1 * max(upper - lower, 1e-6)
    predictions = []
    model.eval()
    with torch.no_grad():
        for cycle in range(target_start, target_end):
            indices = np.arange(cycle - ts.SEQ_LEN + 1, cycle + 1)
            if use_component_lag:
                lag_indices = np.maximum(indices - 1, 0)
                lag = state[lag_indices]
                if np.isnan(lag).any():
                    raise RuntimeError(f"Cycle={cycle} 的递归分量状态尚未生成")
                x = np.concatenate([norm_hf[indices], lag[:, None]], axis=1)[None, ...]
            else:
                x = norm_hf[indices][None, ...]
            pred_norm = float(ts.predict(model, x)[0])
            if use_component_lag and clip_recursive_state:
                pred_norm = float(np.clip(pred_norm, lower - margin, upper + margin))
            predictions.append(pred_norm * std + mean)
            if use_component_lag and cycle >= known_end:
                state[cycle] = pred_norm
    return np.asarray(predictions, dtype=np.float32)


def train_component(model, trajectories, validation_context, mode, lam, mean, std,
                    epochs, patience, log_path, use_component_lag=True,
                    clip_recursive_state=False):
    opt = torch.optim.AdamW(model.parameters(), lr=ts.LR, weight_decay=ts.WEIGHT_DECAY)
    mse = nn.MSELoss()
    norm_hf, values, val_start, val_end = validation_context
    val_true_norm = (np.asarray(values[val_start:val_end], np.float32) - mean) / std
    best, wait, best_state, rows = float("inf"), 0, None, []
    for epoch in range(epochs):
        model.train(); pred_sum = phy_sum = 0.0
        order = list(range(len(trajectories))); random.shuffle(order)
        for idx in order:
            x, y = trajectories[idx]
            x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
            y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            opt.zero_grad(); pred = model(x)
            lp = mse(pred, y)
            pred_physical = pred * std + mean
            lphy = physics_loss(pred_physical, mode)
            (lp + lam * lphy).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pred_sum += lp.item(); phy_sum += lphy.item()
        val_pred_physical = causal_component_predict(
            model, norm_hf, values, mean, std, val_start, val_end, val_start
            , use_component_lag, clip_recursive_state
        )
        val_pred_norm = (val_pred_physical - mean) / std
        val_prediction = float(np.mean(np.square(val_pred_norm - val_true_norm)))
        val_physics = float(physics_loss(
            torch.as_tensor(val_pred_physical[:, None], dtype=torch.float32, device=DEVICE), mode
        ).item())
        val_total = val_prediction + lam * val_physics
        rows.append({"Epoch": epoch + 1, "PredictionLoss": pred_sum / len(order),
                     "PhysicsLoss": phy_sum / len(order),
                     "ValidationPredictionLoss": val_prediction,
                     "ValidationPhysicsLoss": val_physics,
                     "ValidationTotalLoss": val_total})
        if val_total < best - 1e-7:
            best, wait = val_total, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), Path(log_path).with_suffix(".pth"))
    return model, best


def make_trajectories(aug, norm, component_state, values, mean, std,
                      use_component_lag=True):
    result = []
    required = {"DatasetID", "Cycle"}
    if not required.issubset(aug.columns):
        raise ValueError(f"增强数据缺少轨迹字段: {required - set(aug.columns)}")
    for dataset_id, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        cycles = group["Cycle"].to_numpy(int)
        if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
            raise ValueError(f"DatasetID={dataset_id} Cycle不连续，拒绝计算物理相邻损失")
        gx = norm(group).astype(np.float32)
        if use_component_lag:
            gx = np.concatenate([gx, component_state[cycles]], axis=1).astype(np.float32)
        gy = (values[cycles] - mean) / std
        result.append(ts.windows(gx, gy))
    return result


def metric_row(battery, mode, lam, evaluated_split, true, pred, trend_pred, true_trend, epsilon):
    delta = np.diff(trend_pred)
    positive = delta[delta > 0]
    second = np.diff(trend_pred, n=2)
    true_delta = np.diff(true_trend)
    return {"BatteryID": battery, "EvaluatedSplit": evaluated_split,
            "PhysicsMode": mode, "LambdaPhy": lam,
            "RMSE": mean_squared_error(true, pred) ** 0.5,
            "MAE": mean_absolute_error(true, pred),
            # Paper Eq. (30): sum absolute error divided by sum reference SOH.
            "MAPE": np.sum(np.abs(true - pred)) / np.maximum(np.sum(np.abs(true)), 1e-8) * 100,
            "R2": r2_score(true, pred),
            "TrendViolationRate": float(np.mean(delta > 0)) if len(delta) else 0.0,
            "TrendToleranceEpsilon": float(epsilon),
            "TrendToleranceViolationRate": float(np.mean(delta > epsilon)) if len(delta) else 0.0,
            "TrendMeanPositiveJump": float(positive.mean()) if len(positive) else 0.0,
            "TrendMaxPositiveJump": float(positive.max()) if len(positive) else 0.0,
            "TrendTotalPositiveVariation": float(positive.sum()) if len(positive) else 0.0,
            "TrendMeanAbsSecondDifference": float(np.mean(np.abs(second))) if len(second) else 0.0,
            "TrueTrendViolationRate": float(np.mean(true_delta > 0)) if len(true_delta) else 0.0}


def run_battery(battery, modes, lam, epochs, patience, out_dir, seed, evaluation_split,
                fusion_stable_init=False, use_component_lag=True,
                clip_recursive_state=False):
    raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
    raw = raw[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    aug_path = ROOT / "output" / f"{battery}_aug_train_hf.csv"
    if not aug_path.exists():
        raise FileNotFoundError(f"缺少 {aug_path.name}；请先用SCF-GAN为该目标电池生成训练增强数据")
    aug = pd.read_csv(aug_path).dropna().reset_index(drop=True)
    n = len(raw); train_end, val_end = int(n * .6), int(n * .8)
    soh_full = raw["SOH"].to_numpy(np.float32) * 100
    # Split first, then decompose: validation/test SOH must not influence any
    # component target, scaler, model parameter, or recursive state.
    dec_train = STL(soh_full[:train_end], period=ts.STL_PERIOD, robust=True).fit()
    comps = {"Trend": np.asarray(dec_train.trend, np.float32),
             "Seasonal": np.asarray(dec_train.seasonal, np.float32),
             "Residual": np.asarray(dec_train.resid, np.float32)}
    trend_delta_train = np.diff(comps["Trend"])
    trend_epsilon = max(1e-4, 0.05 * float(np.median(np.abs(trend_delta_train))))
    hf = raw[HF_COLS].to_numpy(np.float32)
    component_fit_end = max(ts.SEQ_LEN + 3, int(train_end * .8))
    hm, hs = hf[:component_fit_end].mean(0), hf[:component_fit_end].std(0)
    hs[hs < 1e-8] = 1
    norm = lambda frame: (frame[HF_COLS].to_numpy(np.float32) - hm) / hs
    norm_hf = norm(raw)
    all_rows, pred_frames = [], []
    # Seasonal original TimesNet and Residual iTransformer are identical for all ablations.
    trained = {}
    component_modes = [("Seasonal", "none"), ("Residual", "none")] + [("Trend", m) for m in modes]
    for name, mode in component_modes:
        # Every component is independent of execution order; all Trend modes share initialization/order.
        reset_seed(stable_seed(seed, battery, name))
        values = comps[name]
        mean = float(values[:component_fit_end].mean())
        std = max(float(values[:component_fit_end].std()), 1e-8)
        state = ((values - mean) / std).reshape(-1, 1); lag = np.vstack([state[:1], state[:-1]])
        fit_aug = aug.loc[aug["Cycle"].astype(int) < component_fit_end].copy()
        trajectories = make_trajectories(
            fit_aug, norm, lag, values, mean, std, use_component_lag
        )
        if name == "Residual":
            model = ts.iTransformer().to(DEVICE)
        elif name == "Seasonal":
            model = ts.TimesNet(input_dim=4 if use_component_lag else 3).to(DEVICE)
        else:
            model = ts.TimesNet(input_dim=4 if use_component_lag else 3).to(DEVICE)
        model, best = train_component(model, trajectories,
                                      (norm_hf, values, component_fit_end, train_end),
                                      mode if name == "Trend" else "none",
                                      lam if name == "Trend" else 0.0, mean, std, epochs, patience,
                                      out_dir / f"{battery}_{name.lower()}_{mode}_train_log.csv",
                                      use_component_lag, clip_recursive_state)
        trained[(name, mode)] = {"model": model, "values": values,
                                 "mean": mean, "std": std, "best": best}
    for mode in modes:
        mode_dir = out_dir / battery / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        selected = [("Trend", trained[("Trend", mode)]),
                    ("Seasonal", trained[("Seasonal", "none")]),
                    ("Residual", trained[("Residual", "none")])]
        split_ranges = {"train": (0, train_end), "val": (train_end, val_end)}
        if evaluation_split == "test":
            split_ranges["test"] = (val_end, n)
        split_frames = {}
        for split, (split_start, split_end) in split_ranges.items():
            target_start = ts.SEQ_LEN if split == "train" else split_start - ts.SEQ_LEN
            known_end = train_end
            columns = {}
            for name, item in selected:
                recursive_start = target_start if target_start < train_end else train_end - ts.SEQ_LEN
                recursive_prediction = causal_component_predict(
                    item["model"], norm_hf, item["values"], item["mean"], item["std"],
                    recursive_start, split_end, known_end, use_component_lag,
                    clip_recursive_state
                )
                offset = target_start - recursive_start
                prediction = recursive_prediction[offset:]
                cycles = np.arange(target_start, split_end)
                if "Cycle" in columns and not np.array_equal(columns["Cycle"], cycles):
                    raise RuntimeError("分量Cycle未对齐")
                columns["Cycle"] = cycles
                if split == "train":
                    columns[f"True_{name}"] = item["values"][cycles]
                columns[f"Pred_{name}"] = prediction
            columns["True_SOH"] = soh_full[columns["Cycle"]]
            for hf_col in HF_COLS:
                columns[hf_col] = raw.loc[columns["Cycle"], hf_col].to_numpy(np.float32)
            split_frames[split] = pd.DataFrame(columns)
            split_frames[split].to_csv(mode_dir / f"comp_pred_{split}.csv", index=False, encoding="utf-8-sig")
        fusion_seed = stable_seed(seed, battery, "Trend") + 900
        model_path = mode_dir / "fusion_att_bilstm_paper.pth"
        fused_metrics, fused_eval = fusion.run_fusion(
            str(mode_dir), str(mode_dir), str(model_path), fusion_seed, epochs, patience,
            evaluate_test=evaluation_split == "test", stable_init=fusion_stable_init,
        )
        eval_key = "test" if evaluation_split == "test" else "val"
        trend_lookup = split_frames[eval_key].set_index("Cycle")["Pred_Trend"]
        trend_pred = trend_lookup.loc[fused_eval.Cycle].to_numpy()
        # Full-series STL is computed only after inference, solely for an
        # explicitly labelled diagnostic curve. It never affects selection.
        dec_diagnostic = STL(soh_full, period=ts.STL_PERIOD, robust=True).fit()
        true_trend_all = np.asarray(dec_diagnostic.trend, np.float32)
        true_trend = true_trend_all[fused_eval.Cycle.to_numpy(int)]
        true = fused_eval.True_SOH.to_numpy(); pred = fused_eval.Pred_SOH.to_numpy()
        row = metric_row(battery, mode, 0.0 if mode == "none" else lam,
                         evaluation_split, true, pred, trend_pred, true_trend, trend_epsilon)
        row.update({f"Fusion{k}": value for k, value in fused_metrics.items() if k.startswith("MeanWeight")})
        all_rows.append(row)
        frame = fused_eval.rename(columns={"True_SOH": "TrueSOH", "Pred_SOH": "PredSOH"}).copy()
        frame.insert(0, "PhysicsMode", mode); frame.insert(0, "BatteryID", battery)
        frame["TrueTrend"] = true_trend; frame["PredTrend"] = trend_pred
        pred_frames.append(frame)
    return all_rows, pred_frames, {"BatteryID": battery, "Rows": n, "TrainEnd": train_end,
        "ValidationEnd": val_end, "TestRows": n-val_end, "EvaluationSplit": evaluation_split,
        "TestLabelsUsedAsFeatures": False,
        "STLFitEndExclusive": train_end,
        "ComponentFitEndExclusive": component_fit_end,
        "ValidationLabelsUsedForComponentTraining": False,
        "ComponentInput": "HF1/HF2/HF3+lagged-component" if use_component_lag else "HF1/HF2/HF3-only",
        "RecursiveStateClippedToTrainRange": bool(clip_recursive_state),
        "Protocol": "target-first-60%-labeled-adaptation; next-20%-validation; last-20%-strict-test"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batteries", default="B0007,B0018")
    p.add_argument("--modes", default="none,monotonic,smooth,both")
    p.add_argument("--lambda-phy", type=float, default=.01, choices=[0, .001, .01, .1])
    p.add_argument("--lambda-weight", type=float, default=.01)
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fusion-stable-init", action="store_true")
    p.add_argument("--hf-only-components", action="store_true")
    p.add_argument("--clip-recursive-state", action="store_true")
    p.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    p.add_argument("--output-dir", default=str(ROOT / "output" / "physics_trend"))
    a = p.parse_args(); modes = a.modes.split(",")
    if not set(modes) <= {"none", "monotonic", "smooth", "both"}: raise ValueError("非法physics_mode")
    reset_seed(a.seed)
    fusion.LAMBDA_WEIGHT = float(a.lambda_weight)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, frames, protocols = [], [], []
    for battery in a.batteries.split(","):
        r, f, protocol = run_battery(battery.strip(), modes, a.lambda_phy, a.epochs,
                                     a.patience, out, a.seed, a.evaluation_split,
                                     a.fusion_stable_init, not a.hf_only_components,
                                     a.clip_recursive_state)
        # Keep clipping as an explicit P1 stabilization, never an implicit default.
        rows += r; frames += f; protocols.append(protocol)
    metrics = pd.DataFrame(rows); predictions = pd.concat(frames, ignore_index=True)
    metrics_name = "physics_ablation_metrics.csv" if a.evaluation_split == "test" else "physics_validation_metrics.csv"
    predictions_name = "physics_ablation_predictions.csv" if a.evaluation_split == "test" else "physics_validation_predictions.csv"
    metrics.to_csv(out / metrics_name, index=False, encoding="utf-8-sig")
    predictions.to_csv(out / predictions_name, index=False, encoding="utf-8-sig")
    config = {"physics_modes": modes, "lambda_phy": a.lambda_phy, "epochs": a.epochs, "seed": a.seed,
              "lambda_weight": a.lambda_weight,
              "evaluation_split": a.evaluation_split, "test_accessed_for_selection": False,
              "timesnet": "task-preregistered-original-timesnet-both-branches", "fusion": "paper-attention-bilstm",
              "transfer_pretrain": False, "oselm": False,
              "causal_multirate": False, "scf_gan_algorithm_changed": False,
              "fusion_stable_init": a.fusion_stable_init,
              "component_input": "HF-only" if a.hf_only_components else "HF+lagged-component",
              "clip_recursive_state": a.clip_recursive_state,
              "physics_applied_to": "STL Trend branch only", "protocols": protocols}
    (out / "physics_ablation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for battery, g in predictions.groupby("BatteryID"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for mode, q in g.groupby("PhysicsMode"):
            ax[0].plot(q.Cycle, q.PredSOH, label=mode); ax[1].plot(q.Cycle, q.PredTrend, label=mode)
        q = g[g.PhysicsMode == modes[0]]; ax[0].plot(q.Cycle, q.TrueSOH, "k--", label="true")
        ax[1].plot(q.Cycle, q.TrueTrend, "k--", label="true")
        ax[0].set_title(f"{battery} {a.evaluation_split} SOH"); ax[1].set_title("STL Trend")
        for z in ax: z.set_xlabel("Cycle"); z.grid(alpha=.3); z.legend()
        fig.tight_layout(); fig.savefig(out / f"{battery}_physics_ablation.png", dpi=180); plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__": main()
