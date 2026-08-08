import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===================== 全局配置（与G-MTC完全一致） =====================
ROOT_PATH = "./"
RAW_TARGET = os.path.join(ROOT_PATH, "data", "hf_csv", "B0018_hf.csv")
AUG_TRAIN = os.path.join(ROOT_PATH, "output", "B0018_aug_train_hf.csv")
HF_COLS = ["HF1", "HF2", "HF3"]
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
MODEL_DIR = os.path.join(ROOT_PATH, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEQ_LEN = 10
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 800
PATIENCE = 50
LAMBDA_DECAY = 1e-5

# ===================== 工具函数（与G-MTC完全对齐） =====================
def sliding_windows(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i+1:i+seq_len+1])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def temporal_split(data, train_ratio=0.6, val_ratio=0.2):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

def zscore_fit(data):
    return data.mean(), data.std() + 1e-8

def zscore_transform(data, mean, std):
    return (data - mean) / std

def calc_metrics(y_true, y_pred):
    """计算论文标准4项指标（百分比形式）"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE%": round(mae, 4),
        "MAPE%": round(mape, 4),
        "RMSE%": round(rmse, 4),
        "R2": round(r2, 4)
    }

# ===================== 数据集类 =====================
class SOHTSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===================== 深度学习基线模型 =====================
class BaselineCNN(nn.Module):
    """论文基线CNN：2层1D卷积 + 全局池化 + 全连接"""
    def __init__(self, seq_len=SEQ_LEN, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(3, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: [B, T, 1]
        x = x.transpose(1, 2)  # [B, 1, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.mean(x, dim=-1)  # [B, hidden]
        return self.fc(x)

class BaselineLSTM(nn.Module):
    """论文基线LSTM：2层LSTM + 全连接输出"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BaselineTransformer(nn.Module):
    """论文标准Transformer：2层编码器 + 全局池化 + 全连接"""
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model*2, dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.shape[1]]
        x = self.encoder(x)
        # Last-token readout preserves cycle order; global mean pooling erased it.
        return self.fc(self.norm(x[:, -1]))

# ===================== 通用深度学习训练函数 =====================
def train_dl_model(model, train_loader, val_loader, model_name, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=LAMBDA_DECAY)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    stop_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss_sum += criterion(pred, y).item()

        avg_train = train_loss_sum / len(train_loader)
        avg_val = val_loss_sum / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            stop_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            stop_cnt += 1
            if stop_cnt >= PATIENCE:
                break
    print(f"[{model_name}] 训练完成，最优验证损失：{best_val_loss:.6f}")
    return best_val_loss

def predict_dl_model(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy().flatten()
    return pred

# ===================== 主流程 =====================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # All methods use identical HF1/HF2/HF3 inputs. Augmentation is training-only.
    raw = pd.read_csv(RAW_TARGET).dropna().reset_index(drop=True)
    aug = pd.read_csv(AUG_TRAIN).dropna().reset_index(drop=True)
    n = len(raw)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    y_mean, y_std = zscore_fit(raw.loc[:train_end - 1, "SOH"].values * 100.0)
    hf_mean = raw.loc[:train_end - 1, HF_COLS].values.mean(axis=0)
    hf_std = raw.loc[:train_end - 1, HF_COLS].values.std(axis=0) + 1e-8

    def normalized_hf(frame):
        return (frame[HF_COLS].values - hf_mean) / hf_std

    train_xs, train_ys = [], []
    for _, group in aug.groupby("DatasetID", sort=True):
        group = group.sort_values("Cycle")
        xw, yw = sliding_windows(normalized_hf(group), SEQ_LEN)
        # sliding_windows's y is ignored because inputs are multivariate.
        target = (group["SOH"].values * 100.0 - y_mean) / y_std
        train_xs.append(xw)
        train_ys.append(target[SEQ_LEN:])
    X_train, y_train = np.concatenate(train_xs), np.concatenate(train_ys)

    # Use preceding observed history as context, but score every held-out point.
    val_context = raw.iloc[train_end - SEQ_LEN:val_end]
    test_context = raw.iloc[val_end - SEQ_LEN:]
    X_val, _ = sliding_windows(normalized_hf(val_context), SEQ_LEN)
    X_test, _ = sliding_windows(normalized_hf(test_context), SEQ_LEN)
    y_val = (raw["SOH"].values[train_end:val_end] * 100.0 - y_mean) / y_std
    y_test = (raw["SOH"].values[val_end:] * 100.0 - y_mean) / y_std
    y_test_true = raw["SOH"].values[val_end:] * 100.0

    # ===================== 经典机器学习基线 =====================
    print("\n" + "="*60)
    print("经典机器学习基线模型训练与评估")
    print("="*60)

    # 展平输入：[N, T] → [N, T]
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # SVR
    svr = SVR(kernel="rbf", C=10, gamma="scale")
    svr.fit(X_train_flat, y_train)
    svr_pred_norm = svr.predict(X_test_flat)
    svr_pred = svr_pred_norm * y_std + y_mean
    svr_metrics = calc_metrics(y_test_true, svr_pred)
    print("SVR:", svr_metrics)

    # GPR
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=False)
    gpr.fit(X_train_flat, y_train)
    gpr_pred_norm, _ = gpr.predict(X_test_flat, return_std=True)
    gpr_pred = gpr_pred_norm * y_std + y_mean
    gpr_metrics = calc_metrics(y_test_true, gpr_pred)
    print("GPR:", gpr_metrics)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_pred_norm = rf.predict(X_test_flat)
    rf_pred = rf_pred_norm * y_std + y_mean
    rf_metrics = calc_metrics(y_test_true, rf_pred)
    print("RF:", rf_metrics)

    # ===================== 深度学习基线 =====================
    print("\n" + "="*60)
    print("深度学习基线模型训练与评估")
    print("="*60)

    # 构建DataLoader
    def make_loader(X, y, shuffle):
        ds = SOHTSDataset(X, y)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # CNN
    cnn_model = BaselineCNN().to(device)
    train_dl_model(cnn_model, train_loader, val_loader, "CNN",
                   os.path.join(MODEL_DIR, "baseline_cnn.pth"))
    cnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "baseline_cnn.pth")))
    cnn_pred_norm = predict_dl_model(cnn_model, X_test)
    cnn_pred = cnn_pred_norm * y_std + y_mean
    cnn_metrics = calc_metrics(y_test_true, cnn_pred)
    print("CNN:", cnn_metrics)

    # LSTM
    lstm_model = BaselineLSTM().to(device)
    train_dl_model(lstm_model, train_loader, val_loader, "LSTM",
                   os.path.join(MODEL_DIR, "baseline_lstm.pth"))
    lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "baseline_lstm.pth")))
    lstm_pred_norm = predict_dl_model(lstm_model, X_test)
    lstm_pred = lstm_pred_norm * y_std + y_mean
    lstm_metrics = calc_metrics(y_test_true, lstm_pred)
    print("LSTM:", lstm_metrics)

    # Transformer
    trans_model = BaselineTransformer().to(device)
    train_dl_model(trans_model, train_loader, val_loader, "Transformer",
                   os.path.join(MODEL_DIR, "baseline_transformer.pth"))
    trans_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "baseline_transformer.pth")))
    trans_pred_norm = predict_dl_model(trans_model, X_test)
    trans_pred = trans_pred_norm * y_std + y_mean
    trans_metrics = calc_metrics(y_test_true, trans_pred)
    print("Transformer:", trans_metrics)

    # ===================== 汇总对比表格 =====================
    print("\n" + "="*60)
    print("所有模型测试集指标汇总（对齐论文Table 5~8格式）")
    print("="*60)

    result_dict = {
        "SVR": svr_metrics,
        "GPR": gpr_metrics,
        "RF": rf_metrics,
        "CNN": cnn_metrics,
        "LSTM": lstm_metrics,
        "Transformer": trans_metrics
    }

    result_df = pd.DataFrame(result_dict).T
    result_df.index.name = "Model"
    print(result_df)

    # 保存结果
    result_df.to_csv(os.path.join(OUTPUT_PATH, "baseline_compare_result.csv"))
    print(f"\n对比结果已保存：{os.path.join(OUTPUT_PATH, 'baseline_compare_result.csv')}")

    return result_df

if __name__ == "__main__":
    main()
