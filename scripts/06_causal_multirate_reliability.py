import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT_PATH = "./output"
SEED = 42
ENSEMBLE_SIZE = 5


def metrics(y, pred):
    return {
        "MAE%": mean_absolute_error(y, pred),
        "MAPE%": np.sum(np.abs(y - pred)) / np.sum(np.abs(y)) * 100.0,
        "RMSE%": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def lag_delta(values, lag):
    shifted = np.vstack([np.repeat(values[:1], lag, axis=0), values[:-lag]])
    return values - shifted


def causal_multirate_features(frame):
    """Current/past-only features at fast, medium and slow degradation rates."""
    hf = frame[["HF1", "HF2", "HF3"]].to_numpy(dtype=np.float64)
    base = frame[["Pred_SOH"]].to_numpy(dtype=np.float64)
    hf_df = pd.DataFrame(hf)
    mean3 = hf_df.rolling(3, min_periods=1).mean().to_numpy()
    mean10 = hf_df.rolling(10, min_periods=1).mean().to_numpy()
    std3 = hf_df.rolling(3, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    std10 = hf_df.rolling(10, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    return np.column_stack([
        hf,
        lag_delta(hf, 1),       # fast rate
        lag_delta(hf, 3),       # medium rate
        lag_delta(hf, 10),      # slow rate
        mean3, mean10,
        std3, std10,
        base,
        lag_delta(base, 1),
        lag_delta(base, 3),
    ])


class OSELM:
    def __init__(self, input_dim, hidden, ridge, forgetting, seed):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, hidden))
        self.b = rng.uniform(-0.5, 0.5, hidden)
        self.ridge = ridge
        self.forgetting = forgetting
        self.beta = None
        self.p = None

    def hidden(self, x):
        return np.tanh(np.asarray(x) @ self.w + self.b)

    def fit(self, x, y):
        h = self.hidden(x)
        self.p = np.linalg.inv(h.T @ h + self.ridge * np.eye(h.shape[1]))
        self.beta = self.p @ h.T @ y
        return self

    def predict(self, x):
        return self.hidden(x) @ self.beta

    def update(self, x, y, error_cap):
        h = self.hidden(np.asarray(x).reshape(1, -1)).reshape(-1, 1)
        gain = self.p @ h / (self.forgetting + (h.T @ self.p @ h).item())
        error = float(y) - (h.T @ self.beta.reshape(-1, 1)).item()
        error = float(np.clip(error, -error_cap, error_cap))
        self.beta = self.beta + gain.reshape(-1) * error
        self.p = (self.p - gain @ h.T @ self.p) / self.forgetting


def scale_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def run_member(model, x, correction, base, update_cap):
    corr_pred = []
    for row, target in zip(x, correction):
        corr_pred.append(float(model.predict(row.reshape(1, -1))[0]))
        model.update(row, target, update_cap)
    return np.asarray(corr_pred)


def run_ensemble(models, x, correction, base, update_cap, gamma):
    pred, uncertainty, reliability = [], [], []
    for row, target, base_value in zip(x, correction, base):
        member_corr = np.asarray([
            model.predict(row.reshape(1, -1))[0] for model in models
        ])
        unc = float(member_corr.std())
        gate = 1.0 / (1.0 + gamma * unc)
        pred.append(float(base_value + gate * member_corr.mean()))
        uncertainty.append(unc)
        reliability.append(gate)
        for model in models:
            model.update(row, target, update_cap)
    return np.asarray(pred), np.asarray(uncertainty), np.asarray(reliability)


def new_model(input_dim, config, seed):
    hidden, ridge, forgetting, _ = config
    return OSELM(input_dim, hidden, ridge, forgetting, seed)


def main():
    splits = {
        key: pd.read_csv(os.path.join(OUTPUT_PATH, f"gmtc_pred_{key}.csv"))
        for key in ("train", "val", "test")
    }
    sizes = [len(splits[key]) for key in ("train", "val", "test")]
    all_frame = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    all_x = causal_multirate_features(all_frame)
    n_train, n_val, _ = sizes
    x_train = all_x[:n_train]
    x_val = all_x[n_train:n_train + n_val]
    x_test = all_x[n_train + n_val:]
    mean, std = scale_fit(x_train)
    x_train, x_val, x_test = [(x - mean) / std for x in (x_train, x_val, x_test)]

    correction = {
        key: splits[key]["True_SOH"].to_numpy() - splits[key]["Pred_SOH"].to_numpy()
        for key in splits
    }
    tune_end = max(1, int(0.6 * n_val))
    x_tune, x_cal = x_val[:tune_end], x_val[tune_end:]
    corr_tune, corr_cal = correction["val"][:tune_end], correction["val"][tune_end:]
    base_tune = splits["val"]["Pred_SOH"].to_numpy()[:tune_end]
    true_tune = splits["val"]["True_SOH"].to_numpy()[:tune_end]

    candidates = []
    for hidden in (16, 32, 64):
        for ridge in (0.1, 1.0, 10.0):
            for forgetting in (0.995, 1.0):
                probe = OSELM(x_train.shape[1], hidden, ridge, forgetting, SEED).fit(
                    x_train, correction["train"]
                )
                scale = max(0.05, float(np.sqrt(np.mean(
                    (correction["train"] - probe.predict(x_train)) ** 2
                ))))
                for cap_multiplier in (0.25, 0.5, 1.0):
                    model = OSELM(x_train.shape[1], hidden, ridge, forgetting, SEED).fit(
                        x_train, correction["train"]
                    )
                    corr_pred = run_member(
                        model, x_tune, corr_tune, base_tune, cap_multiplier * scale
                    )
                    rmse = np.sqrt(mean_squared_error(true_tune, base_tune + corr_pred))
                    candidates.append((rmse, hidden, ridge, forgetting, cap_multiplier))
    _, hidden, ridge, forgetting, cap_multiplier = min(candidates)
    config = (hidden, ridge, forgetting, cap_multiplier)

    # Select an uncertainty shrinkage gate on tuning data only.
    probe = new_model(x_train.shape[1], config, SEED).fit(x_train, correction["train"])
    update_scale = max(0.05, float(np.sqrt(np.mean(
        (correction["train"] - probe.predict(x_train)) ** 2
    ))))
    update_cap = cap_multiplier * update_scale
    ensemble = [
        new_model(x_train.shape[1], config, SEED + i).fit(x_train, correction["train"])
        for i in range(ENSEMBLE_SIZE)
    ]
    # Member updates do not depend on gamma, so collect raw ensemble states once per gamma.
    gamma_scores = []
    for gamma in (0.0, 0.5, 1.0, 2.0, 4.0):
        models = [
            new_model(x_train.shape[1], config, SEED + i).fit(x_train, correction["train"])
            for i in range(ENSEMBLE_SIZE)
        ]
        tune_pred, _, _ = run_ensemble(
            models, x_tune, corr_tune, base_tune, update_cap, gamma
        )
        gamma_scores.append((np.sqrt(mean_squared_error(true_tune, tune_pred)), gamma))
    _, gamma = min(gamma_scores)

    # Calibration interval uses the held-out tail of validation, after tuning.
    models = [
        new_model(x_train.shape[1], config, SEED + i).fit(x_train, correction["train"])
        for i in range(ENSEMBLE_SIZE)
    ]
    run_ensemble(models, x_tune, corr_tune, base_tune, update_cap, gamma)
    base_cal = splits["val"]["Pred_SOH"].to_numpy()[tune_end:]
    true_cal = splits["val"]["True_SOH"].to_numpy()[tune_end:]
    cal_pred, _, _ = run_ensemble(models, x_cal, corr_cal, base_cal, update_cap, gamma)
    conformal_q90 = float(np.quantile(np.abs(true_cal - cal_pred), 0.9, method="higher"))

    # At test time the full validation split is historical and can be used for refitting.
    x_history_raw = all_x[:n_train + n_val]
    final_mean, final_std = scale_fit(x_history_raw)
    x_history = (x_history_raw - final_mean) / final_std
    x_test_final = (all_x[n_train + n_val:] - final_mean) / final_std
    corr_history = np.r_[correction["train"], correction["val"]]
    final_probe = new_model(x_history.shape[1], config, SEED).fit(x_history, corr_history)
    final_scale = max(0.05, float(np.sqrt(np.mean(
        (corr_history - final_probe.predict(x_history)) ** 2
    ))))
    final_cap = cap_multiplier * final_scale
    final_models = [
        new_model(x_history.shape[1], config, SEED + i).fit(x_history, corr_history)
        for i in range(ENSEMBLE_SIZE)
    ]
    base_test = splits["test"]["Pred_SOH"].to_numpy()
    true_test = splits["test"]["True_SOH"].to_numpy()
    test_pred, uncertainty, reliability = run_ensemble(
        final_models, x_test_final, correction["test"], base_test, final_cap, gamma
    )
    lower, upper = test_pred - conformal_q90, test_pred + conformal_q90
    coverage = float(np.mean((true_test >= lower) & (true_test <= upper)))

    result = pd.DataFrame([{
        "Model": "Light-GMTC+Causal-MultiRate-Reliable-OSELM",
        **metrics(true_test, test_pred),
        "PICP90": coverage,
        "MPIW90": 2.0 * conformal_q90,
        "MeanReliability": reliability.mean(),
    }])
    result.to_csv(os.path.join(OUTPUT_PATH, "causal_reliability_result.csv"), index=False)
    pd.DataFrame({
        "Cycle": splits["test"]["Cycle"],
        "True_SOH": true_test,
        "Pred_SOH": test_pred,
        "Lower90": lower,
        "Upper90": upper,
        "EnsembleStd": uncertainty,
        "ReliabilityGate": reliability,
    }).to_csv(os.path.join(OUTPUT_PATH, "causal_reliability_predictions.csv"), index=False)

    schema = []
    for rate, lag in (("fast", 1), ("medium", 3), ("slow", 10)):
        for hf in ("HF1", "HF2", "HF3"):
            schema.append({
                "Feature": f"delta_{lag}_{hf}", "Rate": rate,
                "LatestAllowedTime": "t", "UsesFuture": False,
                "Interpretation": f"{hf}(t)-{hf}(t-{lag})",
            })
    pd.DataFrame(schema).to_csv(os.path.join(OUTPUT_PATH, "causal_feature_schema.csv"), index=False)

    print("=" * 76)
    print("因果约束多速率 + 可靠性 OSELM")
    print("=" * 76)
    print(
        f"config: hidden={hidden}, ridge={ridge}, forgetting={forgetting}, "
        f"cap_multiplier={cap_multiplier}, gamma={gamma}"
    )
    print(result.round(4).to_string(index=False))
    print("因果边界：全部特征只使用 t 及其历史时刻；测试采用 predict-then-update。")


if __name__ == "__main__":
    main()
