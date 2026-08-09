import os
import random
import re
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT_PATH = "./"
RAW_TARGET = os.path.join(ROOT_PATH, "data", "hf_csv", "B0018_hf.csv")
AUG_TRAIN = os.path.join(ROOT_PATH, "output", "B0018_aug_train_hf.csv")
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
MODEL_SAVE = os.path.join(ROOT_PATH, "model")
OUT_TS = os.path.join(OUTPUT_PATH, "stl_decomp.csv")
OUT_COMP_PRED = os.path.join(OUTPUT_PATH, "component_pred_result.csv")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE, exist_ok=True)

SEED = 42
SEQ_LEN = 10
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 800
PATIENCE = 50
WEIGHT_DECAY = 1e-5
STL_PERIOD = 12
HF_COLS = ["HF1", "HF2", "HF3"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def windows(x, y, seq_len=SEQ_LEN):
    xs, ys = [], []
    for i in range(len(y) - seq_len):
        # SOH estimation uses the health features measured in the current
        # charge cycle, while state channels remain one-cycle lagged.
        xs.append(x[i + 1:i + seq_len + 1])
        ys.append(y[i + seq_len])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


class TSDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class InceptionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        c1, c2 = channels // 3, channels // 3
        c3 = channels - c1 - c2
        self.conv1 = nn.Conv2d(channels, c1, 1)
        self.conv3 = nn.Conv2d(channels, c2, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, c3, 5, padding=2)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.cat([F.relu(self.conv1(x)), F.relu(self.conv3(x)), F.relu(self.conv5(x))], dim=1)
        return F.relu(self.bn(y))


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise spatial filtering followed by 1x1 channel mixing."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LightweightInceptionBlock(nn.Module):
    """Parameter-efficient TimesNet block with a residual connection."""
    def __init__(self, channels=64):
        super().__init__()
        c1, c2 = channels // 3, channels // 3
        c3 = channels - c1 - c2
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv2d(channels, c1, 1),
            DepthwiseSeparableConv2d(channels, c2, 3),
            DepthwiseSeparableConv2d(channels, c3, 5),
        ])
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.cat([branch(x) for branch in self.branches], dim=1)
        return F.relu(self.bn(y) + x)


class TimesNet(nn.Module):
    """Multi-period 2D model driven by the three health features."""
    def __init__(self, input_dim=4, d_model=64, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.input_proj = nn.Linear(input_dim, d_model)
        self.inception = InceptionBlock(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        amp = torch.abs(torch.fft.rfft(h, dim=1)).mean((0, 2))
        k = min(self.top_k, max(len(amp) - 1, 1))
        freq = torch.topk(amp[1:], k).indices[0] + 1
        period = max(int(x.shape[1] // int(freq)), 2)
        usable = (x.shape[1] // period) * period
        h2 = h[:, :usable].reshape(x.shape[0], usable // period, period, h.shape[-1]).permute(0, 3, 1, 2)
        h2 = self.inception(h2)
        return self.output(h2.mean(dim=(2, 3)))


class LightweightTimesNet(TimesNet):
    """TimesNet retaining FFT period discovery with separable 2D convolutions."""
    def __init__(self, input_dim=4, d_model=64, top_k=3):
        super().__init__(input_dim=input_dim, d_model=d_model, top_k=top_k)
        self.inception = LightweightInceptionBlock(d_model)


class iTransformer(nn.Module):
    """Inverted Transformer: three HFs plus component state are tokens."""
    def __init__(self, seq_len=SEQ_LEN, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.temporal_proj = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = self.temporal_proj(x.permute(0, 2, 1))
        return self.output(self.encoder(tokens).mean(dim=1))


def train_model(model, train_loader, val_loader, name, path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss = criterion(pred, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                losses.append(criterion(model(x.to(device)), y.to(device)).item())
        val = float(np.mean(losses))
        if val < best - 1e-7:
            best, wait = val, 0
            torch.save(model.state_dict(), path)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[{name}] 训练完成，最优验证损失：{best:.6f}")
    return model


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)


def pretrain_on_source_batteries():
    """Pretrain the lightweight temporal representation on B0005/06/07."""
    source_names = ("B0005", "B0006", "B0007")
    frames = []
    for battery in source_names:
        frame = pd.read_csv(os.path.join(ROOT_PATH, "data", "hf_csv", f"{battery}_hf.csv"))
        frame = frame[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        frames.append(frame)

    train_ends = [int(len(frame) * 0.8) for frame in frames]
    pooled_hf = np.concatenate([
        frame.loc[:end - 1, HF_COLS].to_numpy(np.float32)
        for frame, end in zip(frames, train_ends)
    ])
    pooled_y = np.concatenate([
        frame.loc[:end - 1, "SOH"].to_numpy(np.float32) * 100.0
        for frame, end in zip(frames, train_ends)
    ])
    hf_mean, hf_std = pooled_hf.mean(0), pooled_hf.std(0)
    hf_std[hf_std < 1e-8] = 1.0
    y_mean, y_std = float(pooled_y.mean()), max(float(pooled_y.std()), 1e-8)

    train_x, train_y, val_x, val_y = [], [], [], []
    for frame, end in zip(frames, train_ends):
        hf = (frame[HF_COLS].to_numpy(np.float32) - hf_mean) / hf_std
        target = (frame["SOH"].to_numpy(np.float32) * 100.0 - y_mean) / y_std
        state = np.r_[target[0], target[:-1]].reshape(-1, 1)
        features = np.concatenate([hf, state], axis=1).astype(np.float32)
        xw, yw = windows(features[:end], target[:end])
        train_x.append(xw); train_y.append(yw)
        xw, yw = windows(features[end - SEQ_LEN:], target[end - SEQ_LEN:])
        val_x.append(xw); val_y.append(yw)

    model = LightweightTimesNet().to(device)
    path = os.path.join(MODEL_SAVE, "light_timesnet_source_pretrain.pth")
    model = train_model(
        model,
        DataLoader(TSDataset(np.concatenate(train_x), np.concatenate(train_y)), BATCH_SIZE, shuffle=True),
        DataLoader(TSDataset(np.concatenate(val_x), np.concatenate(val_y)), BATCH_SIZE, shuffle=False),
        "Source-B0005/06/07-LightweightTimesNet", path,
    )
    transferable = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("input_proj.") or key.startswith("inception.")
    }
    print(f"迁移预训练完成：source={source_names}, transferable tensors={len(transferable)}")
    return transferable


def main():
    set_seed()
    print("=" * 60)
    variant = os.getenv("TIMESNET_VARIANT", "light").strip().lower()
    if variant not in {"original", "light"}:
        raise ValueError("TIMESNET_VARIANT 仅支持 original 或 light")
    print("步骤3：真实B0018 STL分解 + 增强HF驱动的三分量预测")
    print(f"TimesNet变体：{variant}")
    print("=" * 60)

    raw = pd.read_csv(RAW_TARGET)
    for col in HF_COLS + ["SOH"]:
        raw[col] = raw[col].map(lambda value: float(re.findall(r"[-+]?\d*\.?\d+", str(value))[0]))
    raw = raw.dropna().reset_index(drop=True)
    aug = pd.read_csv(AUG_TRAIN).dropna().reset_index(drop=True)
    n = len(raw)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    soh = raw["SOH"].to_numpy(np.float32) * 100.0
    decomp = STL(soh, period=STL_PERIOD, robust=True).fit()
    component_values = {
        "Trend": np.asarray(decomp.trend, np.float32),
        "Seasonal": np.asarray(decomp.seasonal, np.float32),
        "Residual": np.asarray(decomp.resid, np.float32),
    }
    pd.DataFrame({
        **{c: raw[c] for c in HF_COLS}, "SOH_raw": soh,
        **component_values,
    }).to_csv(OUT_TS, index=False, encoding="utf-8-sig")
    print(f"真实划分：train={train_end}, val={val_end-train_end}, test={n-val_end}")
    print(f"STL：Trend均值={decomp.trend.mean():.2f}, Seasonal标准差={decomp.seasonal.std():.2f}, Residual标准差={decomp.resid.std():.2f}")

    hf_mean = raw.loc[:train_end - 1, HF_COLS].to_numpy(np.float32).mean(0)
    hf_std = raw.loc[:train_end - 1, HF_COLS].to_numpy(np.float32).std(0)
    hf_std[hf_std < 1e-8] = 1.0

    def norm(frame):
        return (frame[HF_COLS].to_numpy(np.float32) - hf_mean) / hf_std

    split_ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
    saved = {key: {} for key in split_ranges}
    transfer_enabled = os.getenv("TRANSFER_PRETRAIN", "0").strip() == "1"
    transfer_components = {
        item.strip() for item in os.getenv("TRANSFER_COMPONENTS", "Trend").split(",")
        if item.strip()
    }
    transfer_state = pretrain_on_source_batteries() if transfer_enabled else None
    print(
        "迁移学习："
        f"{'B0005/06/07预训练 -> B0018选择性微调 ' + str(sorted(transfer_components)) if transfer_enabled else '关闭'}"
    )

    for name, values in component_values.items():
        target_mean, target_std = values[:train_end].mean(), values[:train_end].std()
        target_std = max(float(target_std), 1e-8)
        state = ((values - target_mean) / target_std).reshape(-1, 1)
        component_state = np.vstack([state[:1], state[:-1]])
        raw_x = np.concatenate([norm(raw), component_state], axis=1).astype(np.float32)

        # Each DatasetID is a separate trajectory; windows never cross dataset boundaries.
        train_xs, train_ys = [], []
        for _, group in aug.groupby("DatasetID", sort=True):
            group = group.sort_values("Cycle")
            cycles = group["Cycle"].to_numpy(int)
            # Preserve the paper's HF pathway while adding the observed history
            # of this component, which is essential for B0018 one-step tracking.
            gx = np.concatenate([norm(group), component_state[cycles]], axis=1).astype(np.float32)
            gy = (values[cycles] - target_mean) / target_std
            xw, yw = windows(gx, gy)
            train_xs.append(xw); train_ys.append(yw)
        x_train, y_train = np.concatenate(train_xs), np.concatenate(train_ys)

        x_val, y_val = windows(raw_x[train_end:val_end], (values[train_end:val_end] - target_mean) / target_std)
        if name == "Residual":
            model = iTransformer().to(device)
        else:
            model = (LightweightTimesNet() if variant == "light" else TimesNet()).to(device)
            if transfer_state is not None and variant == "light" and name in transfer_components:
                missing, unexpected = model.load_state_dict(transfer_state, strict=False)
                if unexpected:
                    raise RuntimeError(f"迁移权重包含未知参数：{unexpected}")
                print(f"[{name}] 已载入迁移表征，微调参数张量={len(model.state_dict()) - len(missing)}")
        path = os.path.join(MODEL_SAVE, f"{name.lower()}_{'timesnet' if name != 'Residual' else 'itransformer'}.pth")
        model = train_model(
            model,
            DataLoader(TSDataset(x_train, y_train), BATCH_SIZE, shuffle=True),
            DataLoader(TSDataset(x_val, y_val), BATCH_SIZE, shuffle=False),
            f"{name}-{type(model).__name__}", path,
        )

        for split, (start, end) in split_ranges.items():
            # Validation/test retain two windows of preceding real history:
            # one for the component forecaster and one for the fusion learner.
            # This evaluates every point in the held-out split without using
            # any future observation.
            history_start = 0 if split == "train" else max(0, start - 2 * SEQ_LEN)
            xw, yw = windows(raw_x[history_start:end], values[history_start:end])
            pred = predict(model, xw) * target_std + target_mean
            aligned = np.arange(history_start + SEQ_LEN, end)
            saved[split]["Cycle"] = aligned
            for hf_col in HF_COLS:
                saved[split][hf_col] = raw.loc[aligned, hf_col].to_numpy(np.float32)
            saved[split][f"True_{name}"] = yw
            saved[split][f"Pred_{name}"] = pred

    for split, columns in saved.items():
        frame = pd.DataFrame(columns)
        path = os.path.join(OUTPUT_PATH, f"comp_pred_{split}.csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{split}集真实分量预测保存：{path} ({len(frame)}行)")
    pd.DataFrame(saved["test"]).to_csv(OUT_COMP_PRED, index=False, encoding="utf-8-sig")

    original_params = sum(p.numel() for p in TimesNet().parameters())
    light_params = sum(p.numel() for p in LightweightTimesNet().parameters())
    pd.DataFrame([
        {"Model": "OriginalTimesNet", "Parameters": original_params},
        {"Model": "LightweightTimesNet", "Parameters": light_params},
    ]).to_csv(os.path.join(OUTPUT_PATH, "timesnet_efficiency.csv"), index=False)
    reduction = 100.0 * (1.0 - light_params / original_params)
    print(f"TimesNet参数量：original={original_params:,}, light={light_params:,}, 减少={reduction:.2f}%")


if __name__ == "__main__":
    main()
