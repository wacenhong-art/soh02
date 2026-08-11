"""Isolated Physics-consistent STL Trend (PC-STL) experiment.

The P0 branch is the paper architecture without legacy additions:
STL -> original TimesNet (Trend/Seasonal) + iTransformer (Residual)
-> paper-mode Attention Bi-LSTM.  The proposed branch changes one training
target only.  A monotonically decreasing, smooth Trend target is fitted on the
labelled training prefix, and its correction is moved into Residual so that
STL additivity remains exact.  Validation/test labels are never used to fit
the decomposition, projection, normalisers, or model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "physics_consistent_trend_v1"
HF_COLS = ["HF1", "HF2", "HF3"]
SEEDS = [42, 52, 62, 72, 82]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ts = load_module("pcstl_ts", ROOT / "scripts" / "03_time_series.py")
fusion = load_module("pcstl_fusion", ROOT / "scripts" / "04_attention_fusion.py")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def windows(x: np.ndarray, y: np.ndarray):
    xs, ys, cycles = [], [], []
    for end in range(ts.SEQ_LEN - 1, len(y)):
        xs.append(x[end - ts.SEQ_LEN + 1:end + 1])
        ys.append(y[end]); cycles.append(end)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(cycles, int)


def smooth_monotone_trend(trend: np.ndarray, mu: float, iterations: int = 25) -> np.ndarray:
    """Alternating smooth projection and decreasing isotonic projection.

    This uses only a labelled training Trend.  The linear smoothing system is
    I + mu D2^T D2; isotonic regression then enforces T[t+1] <= T[t].
    """
    y = np.asarray(trend, dtype=float)
    if len(y) < 3 or mu <= 0:
        return IsotonicRegression(increasing=False).fit_transform(np.arange(len(y)), y).astype(np.float32)
    d2 = np.zeros((len(y) - 2, len(y)), dtype=float)
    for i in range(len(y) - 2): d2[i, i:i + 3] = [1.0, -2.0, 1.0]
    solver = np.linalg.inv(np.eye(len(y)) + float(mu) * (d2.T @ d2))
    x = y.copy(); index = np.arange(len(y))
    iso = IsotonicRegression(increasing=False)
    for _ in range(iterations):
        x = solver @ (y + float(mu) * (d2.T @ d2 @ x))
        x = iso.fit_transform(index, x)
    return x.astype(np.float32)


def component_targets(soh_train: np.ndarray, mu: float, branch: str):
    dec = STL(soh_train, period=ts.STL_PERIOD, robust=True).fit()
    trend = np.asarray(dec.trend, np.float32)
    seasonal = np.asarray(dec.seasonal, np.float32)
    if branch == "pcstl": trend = smooth_monotone_trend(trend, mu)
    residual = np.asarray(soh_train - trend - seasonal, np.float32)
    reconstruction = float(np.max(np.abs(soh_train - (trend + seasonal + residual))))
    return {"Trend": trend, "Seasonal": seasonal, "Residual": residual}, reconstruction


def model_for(name: str, width: int):
    if name in {"Trend", "Seasonal"}:
        return ts.TimesNet(input_dim=3, d_model=width, top_k=3).to(DEVICE)
    return ts.iTransformer(seq_len=ts.SEQ_LEN, d_model=width, nhead=4, layers=1).to(DEVICE)


def train_component(model, train_arrays, val_arrays, checkpoint: Path, epochs: int, patience: int):
    train_loader = DataLoader(ts.TSDataset(*train_arrays), batch_size=32, shuffle=True)
    val_loader = DataLoader(ts.TSDataset(*val_arrays), batch_size=32, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=ts.LR, weight_decay=ts.WEIGHT_DECAY)
    loss_fn = nn.MSELoss(); best = float("inf"); wait = 0; rows = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for x, y in train_loader:
            opt.zero_grad(); loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
        model.eval(); val = []
        with torch.no_grad():
            for x, y in val_loader: val.append(float(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item()))
        train_loss, val_loss = float(np.mean(losses)), float(np.mean(val))
        rows.append({"Epoch": epoch, "TrainLoss": train_loss, "ValidationLoss": val_loss})
        if val_loss < best - 1e-7:
            best, wait = val_loss, 0; torch.save(model.state_dict(), checkpoint)
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(rows), best


def infer_component(model, x: np.ndarray, mean: float, std: float) -> np.ndarray:
    xw, _, _ = windows(x, np.zeros(len(x), np.float32))
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(xw, dtype=torch.float32, device=DEVICE)).cpu().numpy().reshape(-1)
    return pred * std + mean


def make_frame(dataset_id, raw_or_aug: pd.DataFrame, predictions, start_cycle: int, end_cycle: int):
    cycles = np.arange(max(ts.SEQ_LEN - 1, start_cycle), end_cycle)
    offset = ts.SEQ_LEN - 1
    return pd.DataFrame({
        "DatasetID": dataset_id, "Cycle": cycles,
        "HF1": raw_or_aug.loc[cycles, "HF1"].to_numpy(float),
        "HF2": raw_or_aug.loc[cycles, "HF2"].to_numpy(float),
        "HF3": raw_or_aug.loc[cycles, "HF3"].to_numpy(float),
        "True_SOH": raw_or_aug.loc[cycles, "SOH"].to_numpy(float) * 100,
        "Pred_Trend": predictions["Trend"][cycles - offset],
        "Pred_Seasonal": predictions["Seasonal"][cycles - offset],
        "Pred_Residual": predictions["Residual"][cycles - offset],
    })


def fusion_arrays(frames, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std):
    pieces = [fusion.prepare_split(f, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)[0] for f in frames]
    return tuple(np.concatenate([piece[index] for piece in pieces]) for index in range(3))


def metric_dict(true, pred, trend):
    delta = np.diff(trend); positive = delta[delta > 0]; second = np.diff(trend, n=2)
    return {
        "RMSE": float(mean_squared_error(true, pred) ** .5), "MAE": float(mean_absolute_error(true, pred)),
        "MAPE": float(np.abs(true - pred).sum() / max(np.abs(true).sum(), 1e-8) * 100), "R2": float(r2_score(true, pred)),
        "TrendViolationRate": float(np.mean(delta > 0)) if len(delta) else 0.0,
        "TrendMeanPositiveJump": float(positive.mean()) if len(positive) else 0.0,
        "TrendMaxPositiveJump": float(positive.max()) if len(positive) else 0.0,
        "TrendMeanAbsSecondDifference": float(np.mean(np.abs(second))) if len(second) else 0.0,
    }


def run_one(battery: str, seed: int, branch: str, mu: float, epochs: int, patience: int,
            split: str, output_dir: Path, stable_init: bool = False):
    raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
    aug = pd.read_csv(ROOT / "output" / f"{battery}_aug_train_hf.csv")
    raw = raw[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    aug = aug[["DatasetID", "Cycle"] + HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="raise")
    n = len(raw); train_end, val_end = int(.6 * n), int(.8 * n)
    if train_end <= 2 * ts.SEQ_LEN: raise ValueError(f"{battery}: 60%训练段不足以支持两阶段10步窗口")
    if split not in {"validation", "test"}: raise ValueError("split must be validation/test")
    evaluation_end = val_end if split == "validation" else n
    soh_train = raw.SOH.iloc[:train_end].to_numpy(np.float32) * 100
    targets, reconstruction_error = component_targets(soh_train, mu, branch)
    hf_mean = raw.loc[:train_end - 1, HF_COLS].to_numpy(np.float32).mean(0)
    hf_std = raw.loc[:train_end - 1, HF_COLS].to_numpy(np.float32).std(0); hf_std[hf_std < 1e-8] = 1
    norm = lambda frame: (frame[HF_COLS].to_numpy(np.float32) - hf_mean) / hf_std
    raw_x = norm(raw); aug = aug.loc[aug.Cycle.astype(int) < train_end].copy()
    ids = sorted(aug.DatasetID.astype(int).unique())
    if ids != list(range(6)): raise ValueError(f"{battery}: expected DatasetID 0..5, got {ids}")
    groups = {i: aug.loc[aug.DatasetID.eq(i)].sort_values("Cycle").reset_index(drop=True) for i in ids}
    if any(not np.array_equal(g.Cycle.to_numpy(int), np.arange(train_end)) for g in groups.values()):
        raise ValueError(f"{battery}: augmented trajectories crossed/violated training boundary")
    component_predictions, logs = {}, []
    for index, name in enumerate(["Trend", "Seasonal", "Residual"]):
        set_seed(seed + 100 * (index + 1)); target = targets[name]
        mean, std = float(target.mean()), max(float(target.std()), 1e-8); target_z = (target - mean) / std
        train_parts = [windows(norm(groups[i]), target_z)[:2] for i in range(5)]
        val_x, val_y, _ = windows(norm(groups[5]), target_z)
        train_arrays = (np.concatenate([x for x, _ in train_parts]), np.concatenate([y for _, y in train_parts]))
        model, log, best = train_component(model_for(name, 64), train_arrays, (val_x, val_y), output_dir / f"{name.lower()}.pth", epochs, patience)
        log.insert(0, "Component", name); logs.append(log)
        component_predictions[name] = {}
        for dataset_id, group in groups.items(): component_predictions[name][dataset_id] = infer_component(model, norm(group), mean, std)
        component_predictions[name]["raw"] = infer_component(model, raw_x, mean, std)
        logs.append(pd.DataFrame([{"Component": name, "BestSyntheticValidationLoss": best, "TargetMean": mean, "TargetStd": std}]))
    train_frames = [make_frame(i, groups[i], {name: component_predictions[name][i] for name in component_predictions}, 0, train_end) for i in range(5)]
    raw_pred = {name: component_predictions[name]["raw"] for name in component_predictions}
    raw_frame = make_frame("raw", raw, raw_pred, 0, n)
    train_comp = np.concatenate([f[["Pred_Trend", "Pred_Seasonal", "Pred_Residual"]].to_numpy(np.float32) for f in train_frames])
    comp_mean, comp_std = fusion.zscore_fit(train_comp, axis=0)
    feat = np.concatenate([fusion.build_causal_features(f, comp_mean, comp_std) for f in train_frames])
    feat_mean, feat_std = fusion.zscore_fit(feat, axis=0)
    y_train = np.concatenate([fusion.true_soh_target(f) for f in train_frames]); y_mean, y_std = float(y_train.mean()), max(float(y_train.std()), 1e-8)
    train_arrays = fusion_arrays(train_frames, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)
    # The current split begins ten cycles earlier only to form causal fusion windows.
    eval_start = train_end if split == "validation" else val_end
    context_start = max(ts.SEQ_LEN - 1, eval_start - fusion.SEQ_LEN)
    eval_frame = raw_frame.loc[(raw_frame.Cycle >= context_start) & (raw_frame.Cycle < evaluation_end)].reset_index(drop=True)
    eval_arrays = fusion.prepare_split(eval_frame, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)[0]
    set_seed(seed + 999); fusion.LAMBDA_WEIGHT = 0.01
    model = fusion.AttentionBiLSTMFusion(comp_mean, comp_std, y_mean, y_std, fusion_mode="paper", stable_init=stable_init).to(DEVICE)
    loader = DataLoader(fusion.FusionDataset(*train_arrays), batch_size=min(32, len(train_arrays[0])), shuffle=True)
    # A synthetic sixth trajectory is used only for early stopping; real val/test labels stay outside training.
    val_frame = make_frame(5, groups[5], {name: component_predictions[name][5] for name in component_predictions}, 0, train_end)
    val_arrays = fusion.prepare_split(val_frame, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)[0]
    val_loader = DataLoader(fusion.FusionDataset(*val_arrays), batch_size=min(32, len(val_arrays[0])), shuffle=False)
    model = fusion.train_model(model, loader, val_loader, output_dir / "attention_bilstm.pth", epochs, patience)
    model.eval()
    with torch.no_grad():
        pred_z, weights = model(torch.as_tensor(eval_arrays[0], dtype=torch.float32, device=DEVICE), torch.as_tensor(eval_arrays[1], dtype=torch.float32, device=DEVICE))
    pred = pred_z.cpu().numpy().reshape(-1) * y_std + y_mean; weights = weights.cpu().numpy()
    scored = eval_frame.iloc[fusion.SEQ_LEN:].reset_index(drop=True)
    if not np.all(scored.Cycle.to_numpy(int) >= eval_start): raise RuntimeError("fusion evaluation crossed split boundary")
    truth = scored.True_SOH.to_numpy(float); trend = scored.Pred_Trend.to_numpy(float)
    result = pd.DataFrame({"BatteryID": battery, "Seed": seed, "Branch": branch, "Mu": mu, "Cycle": scored.Cycle.to_numpy(int), "TrueSOH": truth, "PredSOH": pred, "PredTrend": trend, "PredSeasonal": scored.Pred_Seasonal, "PredResidual": scored.Pred_Residual, "w_Trend": weights[:, 0], "w_Seasonal": weights[:, 1], "w_Residual": weights[:, 2]})
    row = {"BatteryID": battery, "Seed": seed, "Branch": branch, "Mu": mu, "EvaluatedSplit": split, **metric_dict(truth, pred, trend), "STLReconstructionMaxAbsError": reconstruction_error, "StableAttentionInit": stable_init, "TestLabelsUsedAsFeatures": False}
    pd.concat(logs, ignore_index=True).to_csv(output_dir / "training_logs.csv", index=False, encoding="utf-8-sig")
    result.to_csv(output_dir / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    return row, result


def audit(batteries: list[str], out: Path):
    rows = []
    for battery in batteries:
        raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
        numeric = raw[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce")
        n = len(numeric); train_end, val_end = int(n * .6), int(n * .8)
        dec = STL(numeric.SOH.iloc[:train_end].to_numpy(float) * 100, period=ts.STL_PERIOD, robust=True).fit()
        rec = float(np.max(np.abs(numeric.SOH.iloc[:train_end].to_numpy(float) * 100 - (dec.trend + dec.seasonal + dec.resid))))
        rows.append({"BatteryID": battery, "Rows": n, "TrainEndExclusive": train_end, "ValidationEndExclusive": val_end, "TestRows": n - val_end, "STLTrainOnly": True, "STLTrainReconstructionError": rec, "EnoughForTwo10StepStages": train_end > 2 * ts.SEQ_LEN})
    frame = pd.DataFrame(rows); frame.to_csv(out / "data_audit.csv", index=False, encoding="utf-8-sig")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "run"], required=True)
    parser.add_argument("--batteries", default="B0007,B0018")
    parser.add_argument("--branches", default="p0,pcstl")
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--stable-init", action="store_true")
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    batteries = [x.strip() for x in args.batteries.split(",") if x.strip()]
    if args.phase == "audit":
        report = audit(batteries, out); print(report.to_string(index=False)); return
    all_rows, all_predictions = [], []
    for battery in batteries:
        for branch in [x.strip() for x in args.branches.split(",")]:
            if branch not in {"p0", "pcstl"}: raise ValueError("branches must be p0,pcstl")
            run_dir = out / "training_logs" / f"{battery}_{branch}_seed{args.seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            row, pred = run_one(battery, args.seed, branch, args.mu, args.epochs, args.patience, args.split, run_dir, args.stable_init)
            all_rows.append(row); all_predictions.append(pred)
    pd.DataFrame(all_rows).to_csv(out / f"{args.split}_metrics_seed{args.seed}.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(out / f"{args.split}_predictions_seed{args.seed}.csv", index=False, encoding="utf-8-sig")
    config = {"schema": "pcstl/1", "timestamp": datetime.now().isoformat(), "batteries": batteries, "branches": args.branches, "mu": args.mu, "seed": args.seed, "epochs": args.epochs, "patience": args.patience, "split": args.split, "stable_init": args.stable_init, "architecture": "STL -> original TimesNet Trend + original TimesNet Seasonal + iTransformer Residual -> paper Attention Bi-LSTM", "innovation": "training-only monotonic smooth Trend target; correction reallocated to Residual", "final_soh_constrained": False, "scf_gan_changed": False, "test_used_for_selection": False, "artifacts": {str((ROOT / p).relative_to(ROOT)): sha256(ROOT / p) for p in ["scripts/03_time_series.py", "scripts/04_attention_fusion.py", "scripts/02_scf_gan.py"]}}
    canonical = json.dumps(config, sort_keys=True).encode(); config["config_hash"] = hashlib.sha256(canonical).hexdigest()
    (out / f"config_{args.split}_seed{args.seed}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__": main()
