"""Paper-protocol direct-SOH checkpoint for the G-MTC reproduction.

This runner isolates a central protocol question before the expensive STL
fusion experiment: given the first 20% of a target battery, can a temporal
model estimate each later cycle from the health features observed at that
cycle?  It deliberately avoids recursive SOH/component feedback.

The script does not replace the G-MTC architecture.  It reproduces the
CNN/LSTM/Transformer checkpoint in Table 5 and writes separate, auditable
development results under ``output/paper_protocol_rescue``.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_protocol_rescue"
HF_COLS = ["HF1", "HF2", "HF3"]
SEQ_LEN = 10
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEC = importlib.util.spec_from_file_location(
    "paper_comparison_models", ROOT / "baseline_compare.py"
)
legacy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(legacy)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {*HF_COLS, "SOH"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    out = frame.copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(required)).reset_index(drop=True)
    if not np.isfinite(out[list(required)].to_numpy(float)).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def make_windows(
    features: np.ndarray,
    targets: np.ndarray,
    cycles: np.ndarray,
    start_cycle: int,
    end_cycle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows without crossing a trajectory or temporal boundary.

    A sample ending at cycle ``t`` uses observed HFs through cycle ``t`` and
    predicts SOH at ``t``.  It never uses an SOH label as an input.
    """
    xs, ys, cs = [], [], []
    for end in range(SEQ_LEN - 1, len(targets)):
        cycle = int(cycles[end])
        if cycle < start_cycle or cycle >= end_cycle:
            continue
        window_cycles = cycles[end - SEQ_LEN + 1 : end + 1]
        if not np.all(np.diff(window_cycles) == 1):
            continue
        xs.append(features[end - SEQ_LEN + 1 : end + 1])
        ys.append(targets[end])
        cs.append(cycle)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(cs, dtype=int),
    )


def prepare_data(battery: str, use_augmented: bool):
    raw_path = ROOT / "data" / "hf_csv" / f"{battery}_hf.csv"
    aug_path = ROOT / "output" / f"{battery}_aug_train_hf.csv"
    raw = clean_frame(pd.read_csv(raw_path), str(raw_path))
    raw["Cycle"] = np.arange(len(raw), dtype=int)

    n_cycles = len(raw)
    paper_train_end = int(n_cycles * 0.20)
    fit_end = int(paper_train_end * 0.80)
    if fit_end <= SEQ_LEN or paper_train_end - fit_end < 2:
        raise ValueError(
            f"{battery}: first-20% prefix is too short for seq_len={SEQ_LEN}"
        )

    # All scaling statistics are fitted inside the fit part of the first-20%
    # prefix.  Validation and development-test labels never affect scaling.
    fit_raw = raw.iloc[:fit_end]
    hf_mean = fit_raw[HF_COLS].to_numpy(float).mean(axis=0)
    hf_std = fit_raw[HF_COLS].to_numpy(float).std(axis=0)
    hf_std[hf_std < 1e-8] = 1.0
    y_physical = fit_raw["SOH"].to_numpy(float) * 100.0
    y_mean = float(y_physical.mean())
    y_std = max(float(y_physical.std()), 1e-8)

    def normalized_hf(frame: pd.DataFrame) -> np.ndarray:
        return (frame[HF_COLS].to_numpy(float) - hf_mean) / hf_std

    train_frames = []
    if use_augmented:
        aug = clean_frame(pd.read_csv(aug_path), str(aug_path))
        if not {"DatasetID", "Cycle"}.issubset(aug.columns):
            raise ValueError(f"{aug_path} lacks DatasetID/Cycle")
        for dataset_id, group in aug.groupby("DatasetID", sort=True):
            group = group.sort_values("Cycle").reset_index(drop=True)
            group["DatasetID"] = int(dataset_id)
            train_frames.append(group[group["Cycle"] < fit_end].copy())
    else:
        real = raw.iloc[:fit_end].copy()
        real["DatasetID"] = 0
        train_frames.append(real)

    train_xs, train_ys = [], []
    for group in train_frames:
        x, y, _ = make_windows(
            normalized_hf(group),
            (group["SOH"].to_numpy(float) * 100.0 - y_mean) / y_std,
            group["Cycle"].to_numpy(int),
            SEQ_LEN - 1,
            fit_end,
        )
        if len(x):
            train_xs.append(x)
            train_ys.append(y)
    if not train_xs:
        raise ValueError(f"{battery}: no training windows")

    raw_features = normalized_hf(raw)
    raw_targets_z = (raw["SOH"].to_numpy(float) * 100.0 - y_mean) / y_std
    cycles = raw["Cycle"].to_numpy(int)
    val = make_windows(raw_features, raw_targets_z, cycles, fit_end, paper_train_end)
    test = make_windows(raw_features, raw_targets_z, cycles, paper_train_end, n_cycles)
    if not len(val[0]) or not len(test[0]):
        raise ValueError(f"{battery}: empty validation or development-test split")

    summary = {
        "BatteryID": battery,
        "TotalCycles": n_cycles,
        "FitCycles": fit_end,
        "ValidationCycles": paper_train_end - fit_end,
        "DevelopmentTestCycles": n_cycles - paper_train_end,
        "PaperPrefixEndExclusive": paper_train_end,
        "SequenceLength": SEQ_LEN,
        "TrainingTrajectories": len(train_frames),
        "TrainingWindows": int(sum(len(item) for item in train_xs)),
        "ValidationWindows": int(len(val[0])),
        "DevelopmentTestWindows": int(len(test[0])),
    }
    return (
        (np.concatenate(train_xs), np.concatenate(train_ys)),
        val,
        test,
        y_mean,
        y_std,
        summary,
    )


def train_model(model, train, validation, checkpoint: Path, epochs: int, patience: int):
    train_loader = DataLoader(
        legacy.SOHTSDataset(*train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        legacy.SOHTSDataset(validation[0], validation[1]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best, wait, best_epoch = float("inf"), 0, 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            prediction = model(xb.to(DEVICE))
            loss = criterion(prediction, yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(
                    float(criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item())
                )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
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


def calculate_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(mean_squared_error(truth, prediction) ** 0.5),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "MAPE": float(np.abs(truth - prediction).sum() / np.abs(truth).sum() * 100.0),
        "R2": float(r2_score(truth, prediction)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", choices=["B0007", "B0018"], default="B0018")
    parser.add_argument(
        "--models", nargs="+", choices=["CNN", "LSTM", "Transformer"],
        default=["CNN", "LSTM", "Transformer"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--features", choices=["real", "augmented"], default="augmented")
    parser.add_argument(
        "--capacity",
        choices=["standard", "compact"],
        default="standard",
        help="Compact changes only hidden width/layer count, not model family.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = OUT / args.battery / f"seed_{args.seed}" / args.features / args.capacity
    run_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, y_mean, y_std, data_summary = prepare_data(
        args.battery, args.features == "augmented"
    )
    if args.capacity == "compact":
        # The PDF does not disclose the comparison-model hidden widths.  With
        # only 60 trajectory-safe windows, the standard 64-wide networks are
        # strongly over-parameterized; this is an explicit engineering
        # approximation selected on the first-20% internal validation only.
        factories = {
            "CNN": lambda: legacy.BaselineCNN(hidden=16),
            "LSTM": lambda: legacy.BaselineLSTM(hidden_size=16, num_layers=1),
            "Transformer": lambda: legacy.BaselineTransformer(
                d_model=16, nhead=4, num_layers=1
            ),
        }
    else:
        factories = {
            "CNN": legacy.BaselineCNN,
            "LSTM": legacy.BaselineLSTM,
            "Transformer": legacy.BaselineTransformer,
        }
    metric_rows, prediction_rows = [], []
    for model_name in args.models:
        set_seed(args.seed)
        model = factories[model_name]().to(DEVICE)
        model, log, best_val, best_epoch = train_model(
            model,
            train,
            validation,
            run_dir / f"{model_name}.pth",
            args.epochs,
            args.patience,
        )
        log.to_csv(run_dir / f"{model_name}_training_log.csv", index=False, encoding="utf-8-sig")
        model.eval()
        with torch.no_grad():
            pred_z = (
                model(torch.as_tensor(test[0], dtype=torch.float32, device=DEVICE))
                .cpu().numpy().reshape(-1)
            )
        prediction = pred_z * y_std + y_mean
        truth = test[1] * y_std + y_mean
        row = {
            "BatteryID": args.battery,
            "Seed": args.seed,
            "FeatureSet": args.features,
            "Capacity": args.capacity,
            "Model": model_name,
            "BestEpoch": best_epoch,
            "BestValidationLoss": best_val,
            **calculate_metrics(truth, prediction),
        }
        metric_rows.append(row)
        for cycle, actual, estimate in zip(test[2], truth, prediction):
            prediction_rows.append(
                {
                    "BatteryID": args.battery,
                    "DatasetID": "raw",
                    "Cycle": int(cycle),
                    "Seed": args.seed,
                    "FeatureSet": args.features,
                    "Capacity": args.capacity,
                    "Model": model_name,
                    "TrueSOH": float(actual),
                    "PredSOH": float(estimate),
                }
            )
        print(
            f"{args.battery} {args.features} {model_name}: "
            f"RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, "
            f"MAPE={row['MAPE']:.4f}%, R2={row['R2']:.4f}, "
            f"best_epoch={best_epoch}"
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(run_dir / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    config = {
        "purpose": "Table-5 direct-SOH checkpoint before G-MTC fusion",
        "protocol": "first 20% target prefix; first 80% of prefix fit; last 20% of prefix validation; remaining 80% development evaluation",
        "inference": "direct estimation from observed HF window ending at current cycle; no recursive SOH/component feedback",
        "development_set_warning": "B0007/B0018 have been inspected repeatedly and are not untouched confirmatory test sets",
        "battery": args.battery,
        "models": args.models,
        "feature_set": args.features,
        "capacity": args.capacity,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "device": str(DEVICE),
        "data_summary": data_summary,
    }
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
