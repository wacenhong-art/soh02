import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_PATH = "./"
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
SEED = 42


def metrics(y_true, y_pred):
    return {
        "MAE%": mean_absolute_error(y_true, y_pred),
        "MAPE%": np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100.0,
        "RMSE%": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def raw_features(df):
    hf = df[["HF1", "HF2", "HF3"]].to_numpy(dtype=np.float64)
    base = df["Pred_SOH"].to_numpy(dtype=np.float64)
    dhf = np.vstack([np.zeros((1, 3)), np.diff(hf, axis=0)])
    dbase = np.r_[0.0, np.diff(base)].reshape(-1, 1)
    return np.column_stack([hf, dhf, base, dbase])


class OSELM:
    """Single-hidden-layer ELM with recursive least-squares online updates."""
    def __init__(self, input_dim, hidden_dim, ridge, forgetting, seed=SEED):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, hidden_dim))
        self.b = rng.uniform(-0.5, 0.5, hidden_dim)
        self.ridge = ridge
        self.forgetting = forgetting
        self.beta = None
        self.p = None

    def hidden(self, x):
        return np.tanh(np.asarray(x) @ self.w + self.b)

    def fit(self, x, y):
        h = self.hidden(x)
        eye = np.eye(h.shape[1])
        self.p = np.linalg.inv(h.T @ h + self.ridge * eye)
        self.beta = self.p @ h.T @ np.asarray(y)
        return self

    def predict(self, x):
        return self.hidden(x) @ self.beta

    def update(self, x, y, error_cap=None):
        h = self.hidden(np.asarray(x).reshape(1, -1)).reshape(-1, 1)
        gain = self.p @ h / (self.forgetting + (h.T @ self.p @ h).item())
        error = float(y) - (h.T @ self.beta.reshape(-1, 1)).item()
        if error_cap is not None:
            error = float(np.clip(error, -error_cap, error_cap))
        self.beta = self.beta + gain.reshape(-1) * error
        self.p = (self.p - gain @ h.T @ self.p) / self.forgetting


def standardize(train_x, *others):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return [(x - mean) / std for x in (train_x,) + others]


def prequential(model, x, correction_target, base_pred, clip_value, update, update_cap=None):
    predictions = []
    corrections = []
    for row, target_correction, base in zip(x, correction_target, base_pred):
        correction = float(np.clip(model.predict(row.reshape(1, -1))[0], -clip_value, clip_value))
        predictions.append(base + correction)
        corrections.append(correction)
        if update:
            model.update(row, target_correction, error_cap=update_cap)
    return np.asarray(predictions), np.asarray(corrections)


def make_model(input_dim, hidden, ridge, forgetting):
    return OSELM(input_dim, hidden, ridge, forgetting, seed=SEED)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    splits = {
        name: pd.read_csv(os.path.join(OUTPUT_PATH, f"gmtc_pred_{name}.csv"))
        for name in ("train", "val", "test")
    }
    x_train_raw, x_val_raw, x_test_raw = [raw_features(splits[k]) for k in ("train", "val", "test")]
    x_train, x_val, x_test = standardize(x_train_raw, x_val_raw, x_test_raw)
    correction = {
        key: frame["True_SOH"].to_numpy() - frame["Pred_SOH"].to_numpy()
        for key, frame in splits.items()
    }

    # Hyperparameters are selected exclusively on the validation split.
    candidates = []
    clip_value = max(1.0, float(np.quantile(np.abs(correction["train"]), 0.995)))
    for hidden in (8, 16, 32, 64):
        for ridge in (0.01, 0.1, 1.0, 10.0):
            for forgetting in (0.98, 0.995, 1.0):
                probe = make_model(x_train.shape[1], hidden, ridge, forgetting).fit(
                    x_train, correction["train"]
                )
                train_error_scale = max(
                    0.05,
                    float(np.sqrt(np.mean((correction["train"] - probe.predict(x_train)) ** 2))),
                )
                for cap_multiplier in (0.25, 0.5, 1.0, 2.0):
                    model = make_model(x_train.shape[1], hidden, ridge, forgetting).fit(
                        x_train, correction["train"]
                    )
                    update_cap = cap_multiplier * train_error_scale
                    pred, _ = prequential(
                        model, x_val, correction["val"],
                        splits["val"]["Pred_SOH"].to_numpy(), clip_value,
                        update=True, update_cap=update_cap,
                    )
                    score = metrics(splits["val"]["True_SOH"].to_numpy(), pred)["RMSE%"]
                    candidates.append((score, hidden, ridge, forgetting, cap_multiplier))
    val_rmse, hidden, ridge, forgetting, cap_multiplier = min(candidates)

    # Validation is historical by test time, so the final model may be refit on train+validation.
    x_history_raw = np.vstack([x_train_raw, x_val_raw])
    x_history, x_test_final = standardize(x_history_raw, x_test_raw)
    correction_history = np.r_[correction["train"], correction["val"]]
    final_model = make_model(x_history.shape[1], hidden, ridge, forgetting).fit(
        x_history, correction_history
    )
    base_test = splits["test"]["Pred_SOH"].to_numpy()
    true_test = splits["test"]["True_SOH"].to_numpy()
    final_clip = max(1.0, float(np.quantile(np.abs(correction_history), 0.995)))
    history_error_scale = max(
        0.05,
        float(np.sqrt(np.mean((correction_history - final_model.predict(x_history)) ** 2))),
    )
    update_cap = cap_multiplier * history_error_scale

    frozen_pred, frozen_correction = prequential(
        final_model, x_test_final, correction["test"], base_test, final_clip, update=False,
    )
    online_model = make_model(x_history.shape[1], hidden, ridge, forgetting).fit(
        x_history, correction_history
    )
    online_pred, online_correction = prequential(
        online_model, x_test_final, correction["test"], base_test, final_clip,
        update=True, update_cap=update_cap,
    )

    rows = []
    for name, pred in (
        ("Light-GMTC", base_test),
        ("Light-GMTC+ELM-frozen", frozen_pred),
        ("Light-GMTC+OSELM-online", online_pred),
    ):
        rows.append({"Model": name, **metrics(true_test, pred)})
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_PATH, "innovation_ablation_result.csv"), index=False)

    detail = splits["test"][["Cycle", "True_SOH", "Pred_SOH"]].copy()
    detail = detail.rename(columns={"Pred_SOH": "Pred_Light_GMTC"})
    detail["Correction_ELM_Frozen"] = frozen_correction
    detail["Pred_ELM_Frozen"] = frozen_pred
    detail["Correction_OSELM_Online"] = online_correction
    detail["Pred_OSELM_Online"] = online_pred
    detail.to_csv(os.path.join(OUTPUT_PATH, "gmtc_oselm_pred_result.csv"), index=False)

    pd.DataFrame([{
        "validation_RMSE": val_rmse,
        "hidden_nodes": hidden,
        "ridge": ridge,
        "forgetting_factor": forgetting,
        "update_cap_multiplier": cap_multiplier,
        "update_error_cap": update_cap,
        "correction_clip": final_clip,
    }]).to_csv(os.path.join(OUTPUT_PATH, "oselm_selection.csv"), index=False)

    print("=" * 68)
    print("轻量G-MTC + OSELM 消融结果")
    print("=" * 68)
    print(
        f"验证集选择：hidden={hidden}, ridge={ridge}, forgetting={forgetting}, "
        f"update_cap={update_cap:.4f}, RMSE={val_rmse:.4f}"
    )
    print(result.round(4).to_string(index=False))
    print("说明：online 指标采用 predict-then-update，当前标签不会参与当前预测。")


if __name__ == "__main__":
    main()
