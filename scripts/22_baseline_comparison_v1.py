"""Fixed-protocol CNN/LSTM/Transformer comparison for paper_baseline_v1."""

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
OUT = ROOT / "output" / "paper_baseline_v1"
LOGS = OUT / "training_logs" / "comparison_models"
SEEDS = [42, 52, 62, 72, 82]
BATTERIES = ["B0007", "B0018"]
HF_COLS = ["HF1", "HF2", "HF3"]
SEQ_LEN = 10
EPOCHS = 150
PATIENCE = 30
LR = .001
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEC = importlib.util.spec_from_file_location("legacy_baselines", ROOT / "baseline_compare.py")
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def windows(features, targets):
    xs, ys, cycles = [], [], []
    for index in range(len(targets) - SEQ_LEN):
        xs.append(features[index:index + SEQ_LEN])
        ys.append(targets[index + SEQ_LEN])
        cycles.append(index + SEQ_LEN)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(cycles, int)


def prepare(battery):
    raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv").dropna().reset_index(drop=True)
    aug = pd.read_csv(ROOT / "output" / f"{battery}_aug_train_hf.csv").dropna().reset_index(drop=True)
    n = len(raw); train_end, val_end = int(n * .6), int(n * .8)
    hf_mean = raw.loc[:train_end - 1, HF_COLS].to_numpy(float).mean(0)
    hf_std = raw.loc[:train_end - 1, HF_COLS].to_numpy(float).std(0)
    hf_std[hf_std < 1e-8] = 1
    y_train_physical = raw.loc[:train_end - 1, "SOH"].to_numpy(float) * 100
    y_mean, y_std = y_train_physical.mean(), max(y_train_physical.std(), 1e-8)

    def norm(frame):
        return (frame[HF_COLS].to_numpy(float) - hf_mean) / hf_std

    train_x, train_y = [], []
    for _, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        x, y, _ = windows(norm(group), (group["SOH"].to_numpy(float) * 100 - y_mean) / y_std)
        train_x.append(x); train_y.append(y)
    train = (np.concatenate(train_x), np.concatenate(train_y))

    val_context = raw.iloc[train_end - SEQ_LEN:val_end].reset_index(drop=True)
    test_context = raw.iloc[val_end - SEQ_LEN:].reset_index(drop=True)
    val_x, _, _ = windows(norm(val_context), np.zeros(len(val_context)))
    test_x, _, _ = windows(norm(test_context), np.zeros(len(test_context)))
    val_y = (raw.loc[train_end:val_end - 1, "SOH"].to_numpy(float) * 100 - y_mean) / y_std
    test_true = raw.loc[val_end:, "SOH"].to_numpy(float) * 100
    return train, (val_x, val_y), (test_x, test_true), y_mean, y_std, val_end


def train_model(model, train, val, save_path):
    x_train, y_train = train; x_val, y_val = val
    train_loader = DataLoader(legacy.SOHTSDataset(x_train, y_train), 32, shuffle=True)
    val_loader = DataLoader(legacy.SOHTSDataset(x_val, y_val), 32, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(); best = float("inf"); wait = 0; rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for x, y in train_loader:
            optimizer.zero_grad(); pred = model(x.to(DEVICE)); loss = criterion(pred, y.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item())
        model.eval(); validation = []
        with torch.no_grad():
            for x, y in val_loader:
                validation.append(criterion(model(x.to(DEVICE)), y.to(DEVICE)).item())
        val_loss = float(np.mean(validation)); rows.append({"Epoch": epoch, "TrainLoss": np.mean(losses), "ValidationLoss": val_loss})
        if val_loss < best - 1e-7:
            best, wait = val_loss, 0; torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, pd.DataFrame(rows), best


def metrics(true, pred):
    return {
        "RMSE": mean_squared_error(true, pred) ** .5,
        "MAE": mean_absolute_error(true, pred),
        "MAPE": np.abs(true - pred).sum() / np.abs(true).sum() * 100,
        "R2": r2_score(true, pred),
    }


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    model_factories = {
        "CNN": legacy.BaselineCNN,
        "LSTM": legacy.BaselineLSTM,
        "Transformer": legacy.BaselineTransformer,
    }
    metric_rows, prediction_rows = [], []
    for battery in BATTERIES:
        train, val, test, y_mean, y_std, test_start = prepare(battery)
        for seed in SEEDS:
            for name, factory in model_factories.items():
                set_seed(seed); run_dir = LOGS / battery / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
                model = factory().to(DEVICE)
                model, log, best = train_model(model, train, val, run_dir / f"{name}.pth")
                log.to_csv(run_dir / f"{name}_train_log.csv", index=False, encoding="utf-8-sig")
                model.eval()
                with torch.no_grad():
                    pred_z = model(torch.as_tensor(test[0], dtype=torch.float32, device=DEVICE)).cpu().numpy().reshape(-1)
                pred = pred_z * y_std + y_mean; row = {"BatteryID": battery, "Seed": seed, "Model": name, "BestValidationLoss": best, **metrics(test[1], pred)}
                metric_rows.append(row)
                for offset, (truth, estimate) in enumerate(zip(test[1], pred)):
                    prediction_rows.append({"BatteryID": battery, "DatasetID": "raw", "Seed": seed, "Model": name,
                                            "Cycle": test_start + offset, "TrueSOH": truth, "PredSOH": estimate})
    detailed = pd.DataFrame(metric_rows); predictions = pd.DataFrame(prediction_rows)
    detailed.to_csv(OUT / "comparison_multiseed_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "comparison_per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    summary = detailed.groupby(["BatteryID", "Model"])[["RMSE", "MAE", "MAPE", "R2"]].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(OUT / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    config = {"models": list(model_factories), "seeds": SEEDS, "epochs": EPOCHS, "patience": PATIENCE,
              "lr": LR, "weight_decay": WEIGHT_DECAY, "protocol": "target 60/20/20; SCF-GAN train-only; held-out labels real",
              "test_used_for_tuning": False, "paper_layer_details": "engineering approximation where PDF is silent"}
    (OUT / "configs" / "comparison_models_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
