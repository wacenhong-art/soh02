"""Direct-HF paper-protocol G-MTC diagnostic.

Architecture is unchanged: STL -> Trend/original TimesNet,
Seasonal/lightweight TimesNet, Residual/iTransformer -> Attention Bi-LSTM.
The correction is the inference protocol: every component is estimated from
the observed HF window ending at the current cycle, never by recursively
feeding a predicted component through the held-out trajectory.

DatasetID 0--4 are training trajectories and DatasetID 5 is an internal
synthetic validation trajectory.  The raw target cycles after the first 20%
are an already-contacted development evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_direct_gmtc"
HF_COLS = ["HF1", "HF2", "HF3"]
SEQ_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ts = load_module("paper_ts", ROOT / "scripts" / "03_time_series.py")
fusion = load_module("paper_fusion", ROOT / "scripts" / "04_attention_fusion.py")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def windows(x: np.ndarray, y: np.ndarray):
    xs, ys, cycles = [], [], []
    for end in range(SEQ_LEN - 1, len(y)):
        xs.append(x[end - SEQ_LEN + 1 : end + 1])
        ys.append(y[end])
        cycles.append(end)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(cycles, int)


class AnchoredTrendTimesNet(nn.Module):
    """Original TimesNet learns a bounded residual around a linear Trend anchor."""

    def __init__(self, coefficient, intercept, width: int, residual_bound: float = 0.5):
        super().__init__()
        self.backbone = ts.TimesNet(input_dim=3, d_model=width, top_k=3)
        self.register_buffer("anchor_coefficient", torch.as_tensor(coefficient, dtype=torch.float32))
        self.register_buffer("anchor_intercept", torch.tensor(float(intercept), dtype=torch.float32))
        self.residual_bound = float(residual_bound)

    def forward(self, x):
        anchor = x[:, -1, :] @ self.anchor_coefficient + self.anchor_intercept
        residual = self.residual_bound * torch.tanh(self.backbone(x).reshape(-1))
        return (anchor + residual).unsqueeze(-1)


def component_model(name: str, width: int, trend_anchor=None):
    if name == "Trend":
        if trend_anchor is not None:
            return AnchoredTrendTimesNet(
                trend_anchor.coef_, trend_anchor.intercept_, width=width
            ).to(DEVICE)
        return ts.TimesNet(input_dim=3, d_model=width, top_k=3).to(DEVICE)
    if name == "Seasonal":
        return ts.LightweightTimesNet(input_dim=3, d_model=width, top_k=3).to(DEVICE)
    return ts.iTransformer(seq_len=SEQ_LEN, d_model=width, nhead=4, layers=1).to(DEVICE)


def train_component(model, train_arrays, val_arrays, checkpoint: Path, epochs: int, patience: int):
    train_loader = DataLoader(ts.TSDataset(*train_arrays), 32, shuffle=True)
    val_loader = DataLoader(ts.TSDataset(*val_arrays), 32, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best, wait, best_epoch, rows = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()))
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        rows.append({"Epoch": epoch, "TrainLoss": train_loss, "ValidationLoss": val_loss})
        if val_loss < best - 1e-7:
            best, best_epoch, wait = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model, pd.DataFrame(rows), best, best_epoch


def infer_component(model, x: np.ndarray, mean: float, std: float) -> np.ndarray:
    dummy = np.zeros(len(x), np.float32)
    xw, _, _ = windows(x, dummy)
    model.eval()
    with torch.no_grad():
        pred_z = model(torch.as_tensor(xw, dtype=torch.float32, device=DEVICE)).cpu().numpy().reshape(-1)
    return pred_z * std + mean


def fusion_arrays(frames, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std):
    xs, comps, ys = [], [], []
    for frame in frames:
        (x, comp, y), _ = fusion.prepare_split(
            frame, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std
        )
        xs.append(x); comps.append(comp); ys.append(y)
    return np.concatenate(xs), np.concatenate(comps), np.concatenate(ys)


def metrics(truth, prediction):
    return {
        "RMSE": float(mean_squared_error(truth, prediction) ** 0.5),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "MAPE": float(np.abs(truth - prediction).sum() / np.abs(truth).sum() * 100),
        "R2": float(r2_score(truth, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", default="B0018")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--width", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument(
        "--component-guard", choices=["none", "seasonal_residual"], default="none"
    )
    parser.add_argument("--fusion-mode", choices=["paper", "iprf"], default="paper")
    parser.add_argument(
        "--trend-anchor",
        choices=["none", "ridge_residual", "soh_ridge_residual"],
        default="none",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    if args.battery in {"B0005", "B0006"}:
        raise ValueError("source batteries B0005/B0006 cannot be target batteries")
    run_dir = (
        OUT / args.battery / f"seed_{args.seed}" / f"width_{args.width}"
        / f"guard_{args.component_guard}" / f"anchor_{args.trend_anchor}"
        / f"fusion_{args.fusion_mode}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{args.battery}_hf.csv").dropna().reset_index(drop=True)
    aug = pd.read_csv(ROOT / "output" / f"{args.battery}_aug_train_hf.csv").dropna().reset_index(drop=True)
    for column in HF_COLS + ["SOH"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        aug[column] = pd.to_numeric(aug[column], errors="raise")
    train_end = int(len(raw) * 0.20)
    if train_end <= 2 * SEQ_LEN:
        raise ValueError("first-20% prefix is too short for component and fusion windows")
    aug = aug[aug["Cycle"].astype(int) < train_end].copy()
    ids = sorted(aug["DatasetID"].astype(int).unique())
    if ids != [0, 1, 2, 3, 4, 5]:
        raise ValueError(f"expected DatasetID 0..5, got {ids}")

    hf_mean = raw.loc[: train_end - 1, HF_COLS].to_numpy(np.float32).mean(0)
    hf_std = raw.loc[: train_end - 1, HF_COLS].to_numpy(np.float32).std(0)
    hf_std[hf_std < 1e-8] = 1
    norm = lambda frame: (frame[HF_COLS].to_numpy(np.float32) - hf_mean) / hf_std
    raw_x = norm(raw)

    soh_prefix = raw.loc[: train_end - 1, "SOH"].to_numpy(np.float32) * 100
    decomposition = STL(soh_prefix, period=ts.STL_PERIOD, robust=True).fit()
    component_targets = {
        "Trend": np.asarray(decomposition.trend, np.float32),
        "Seasonal": np.asarray(decomposition.seasonal, np.float32),
        "Residual": np.asarray(decomposition.resid, np.float32),
    }

    group_inputs = {}
    for dataset_id, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle").reset_index(drop=True)
        cycles = group["Cycle"].to_numpy(int)
        if not np.array_equal(cycles, np.arange(train_end)):
            raise ValueError(f"DatasetID={dataset_id} is not a continuous first-20% trajectory")
        group_inputs[int(dataset_id)] = (group, norm(group))

    component_predictions = {dataset_id: {} for dataset_id in ids}
    raw_predictions = {}
    component_diagnostics = []
    for index, (name, values) in enumerate(component_targets.items()):
        set_seed(args.seed + 101 * (index + 1))
        target_mean = float(values.mean()); target_std = max(float(values.std()), 1e-8)
        target_z = (values - target_mean) / target_std
        train_x, train_y = [], []
        for dataset_id in [0, 1, 2, 3, 4]:
            xw, yw, _ = windows(group_inputs[dataset_id][1], target_z)
            train_x.append(xw); train_y.append(yw)
        val_x, val_y, _ = windows(group_inputs[5][1], target_z)
        trend_anchor = None
        if name == "Trend" and args.trend_anchor in {"ridge_residual", "soh_ridge_residual"}:
            # Fit only on the untouched real target prefix (DatasetID=0).
            # Synthetic trajectories remain training augmentation for the
            # TimesNet residual but cannot alter the long-term linear anchor.
            anchor_target = target_z
            if args.trend_anchor == "soh_ridge_residual":
                anchor_target = (soh_prefix - target_mean) / target_std
            anchor_x, anchor_y, _ = windows(group_inputs[0][1], anchor_target)
            trend_anchor = Ridge(alpha=0.1).fit(anchor_x[:, -1, :], anchor_y)
        model = component_model(name, args.width, trend_anchor=trend_anchor)
        model, log, best, best_epoch = train_component(
            model, (np.concatenate(train_x), np.concatenate(train_y)),
            (val_x, val_y), run_dir / f"{name.lower()}.pth", args.epochs, args.patience,
        )
        log.to_csv(run_dir / f"{name.lower()}_training_log.csv", index=False, encoding="utf-8-sig")
        component_diagnostics.append({"Component": name, "BestValidationLoss": best, "BestEpoch": best_epoch})
        lower = float(values.min())
        upper = float(values.max())
        margin = 0.05 * max(upper - lower, 1e-6)

        def guarded(prediction: np.ndarray) -> np.ndarray:
            if args.component_guard == "seasonal_residual" and name in {"Seasonal", "Residual"}:
                return np.clip(prediction, lower - margin, upper + margin)
            return prediction

        for dataset_id in ids:
            component_predictions[dataset_id][name] = guarded(infer_component(
                model, group_inputs[dataset_id][1], target_mean, target_std
            ))
        raw_predictions[name] = guarded(infer_component(model, raw_x, target_mean, target_std))

    def trajectory_frame(dataset_id: int) -> pd.DataFrame:
        group = group_inputs[dataset_id][0]
        cycles = np.arange(SEQ_LEN - 1, train_end)
        return pd.DataFrame({
            "DatasetID": dataset_id,
            "Cycle": cycles,
            "HF1": group.loc[cycles, "HF1"].to_numpy(float),
            "HF2": group.loc[cycles, "HF2"].to_numpy(float),
            "HF3": group.loc[cycles, "HF3"].to_numpy(float),
            "True_SOH": group.loc[cycles, "SOH"].to_numpy(float) * 100,
            "Pred_Trend": component_predictions[dataset_id]["Trend"],
            "Pred_Seasonal": component_predictions[dataset_id]["Seasonal"],
            "Pred_Residual": component_predictions[dataset_id]["Residual"],
        })

    train_frames = [trajectory_frame(dataset_id) for dataset_id in [0, 1, 2, 3, 4]]
    val_frames = [trajectory_frame(5)]
    train_comp = np.concatenate([
        frame[["Pred_Trend", "Pred_Seasonal", "Pred_Residual"]].to_numpy(np.float32)
        for frame in train_frames
    ])
    comp_mean, comp_std = fusion.zscore_fit(train_comp, axis=0)
    train_feat = np.concatenate([fusion.build_causal_features(frame, comp_mean, comp_std) for frame in train_frames])
    feat_mean, feat_std = fusion.zscore_fit(train_feat, axis=0)
    train_soh = np.concatenate([fusion.true_soh_target(frame) for frame in train_frames])
    y_mean, y_std = float(train_soh.mean()), max(float(train_soh.std()), 1e-8)
    train_arrays = fusion_arrays(train_frames, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)
    val_arrays = fusion_arrays(val_frames, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)

    train_loader = DataLoader(fusion.FusionDataset(*train_arrays), min(32, len(train_arrays[0])), shuffle=True)
    val_loader = DataLoader(fusion.FusionDataset(*val_arrays), min(32, len(val_arrays[0])), shuffle=False)
    fusion.LAMBDA_WEIGHT = 0.01
    fusion_model = fusion.AttentionBiLSTMFusion(
        comp_mean, comp_std, y_mean, y_std, fusion_mode=args.fusion_mode, stable_init=True
    ).to(DEVICE)
    fusion_model = fusion.train_model(
        fusion_model, train_loader, val_loader, run_dir / "attention_bilstm.pth",
        epochs=args.epochs, patience=args.patience,
    )

    component_cycles = np.arange(SEQ_LEN - 1, len(raw))
    raw_component_frame = pd.DataFrame({
        "Cycle": component_cycles,
        "HF1": raw.loc[component_cycles, "HF1"].to_numpy(float),
        "HF2": raw.loc[component_cycles, "HF2"].to_numpy(float),
        "HF3": raw.loc[component_cycles, "HF3"].to_numpy(float),
        "True_SOH": raw.loc[component_cycles, "SOH"].to_numpy(float) * 100,
        "Pred_Trend": raw_predictions["Trend"],
        "Pred_Seasonal": raw_predictions["Seasonal"],
        "Pred_Residual": raw_predictions["Residual"],
    })
    test_frame = raw_component_frame[raw_component_frame["Cycle"] >= train_end - SEQ_LEN].reset_index(drop=True)
    test_arrays = fusion.prepare_split(
        test_frame, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std
    )[0]
    fusion_model.eval()
    with torch.no_grad():
        pred_z, weights = fusion_model(
            torch.as_tensor(test_arrays[0], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(test_arrays[1], dtype=torch.float32, device=DEVICE),
        )
    prediction = pred_z.cpu().numpy().reshape(-1) * y_std + y_mean
    weights = weights.cpu().numpy()
    scored = test_frame.iloc[SEQ_LEN:].reset_index(drop=True)
    truth = scored["True_SOH"].to_numpy(float)
    result = pd.DataFrame({
        "BatteryID": args.battery, "DatasetID": "raw", "Cycle": scored["Cycle"].to_numpy(int),
        "TrueSOH": truth, "PredSOH": prediction,
        "PredTrend": scored["Pred_Trend"].to_numpy(float),
        "PredSeasonal": scored["Pred_Seasonal"].to_numpy(float),
        "PredResidual": scored["Pred_Residual"].to_numpy(float),
        "w_Trend": weights[:, 0], "w_Seasonal": weights[:, 1], "w_Residual": weights[:, 2],
    })
    result.to_csv(run_dir / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    row = {"BatteryID": args.battery, "Seed": args.seed, "Width": args.width, **metrics(truth, prediction)}
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(component_diagnostics).to_csv(run_dir / "component_diagnostics.csv", index=False, encoding="utf-8-sig")
    config = {
        "architecture": "STL -> original TimesNet + lightweight TimesNet + iTransformer -> Attention Bi-LSTM",
        "protocol": "target first 20%; DatasetID 0-4 train; DatasetID 5 synthetic validation; remaining raw cycles development evaluation",
        "inference": "direct observed-HF windows; no recursive component or SOH feedback",
        "test_status": "already-contacted development evaluation, not confirmatory",
        "battery": args.battery, "seed": args.seed, "width": args.width,
        "epochs": args.epochs, "patience": args.patience, "train_end_exclusive": train_end,
        "physics_mode": "none", "lambda_phy": 0, "transfer": False, "oselm": False,
        "component_guard": args.component_guard,
        "component_guard_definition": "Seasonal/Residual clipped to first-20% STL range plus 5% margin",
        "fusion_mode": args.fusion_mode,
        "trend_anchor": args.trend_anchor,
        "trend_anchor_definition": "Ridge(alpha=0.1) on DatasetID=0 first-20% Trend or SOH scale; original TimesNet learns bounded Trend residual",
    }
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
