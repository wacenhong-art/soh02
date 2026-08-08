import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re
import json

# 全局路径配置（匹配项目文件夹结构）
ROOT_PATH = "./"
CSV_PATH = os.path.join(ROOT_PATH, "data", "hf_csv")
MODEL_SAVE = os.path.join(ROOT_PATH, "model")
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 论文固定超参
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_DIM = 8
COND_DIM = 2
FEAT_DIM = 3
HID_DIM = 256
LAMBDA_FM = 0.1
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 800
SCALER_PATH = os.path.join(MODEL_SAVE, "scf_scaler.json")

# CSV脏数据清洗（处理[xx.xxx]字符串格式）
def clean_num(x):
    if isinstance(x, str):
        num = re.findall(r"\d+\.\d+|\d+", x)
        return float(num[0]) if num else np.nan
    return float(x)

# 数据集类：源电池B0005/B0006/B0007训练集
class BatteryHFDataset(Dataset):
    def __init__(self):
        # B0018 is the target battery; B0007 is also treated as a target in the
        # paper's NASA small-sample experiments.  Keep both out of GAN training.
        source_bats = ["B0005", "B0006"]
        all_data = []
        for name in source_bats:
            csv_file = os.path.join(CSV_PATH, f"{name}_hf.csv")
            df = pd.read_csv(csv_file)
            # 清洗所有特征列
            for col in ["HF1","HF2","HF3","SOH"]:
                df[col] = df[col].apply(clean_num)
            df = df.dropna()
            feat = df[["HF1","HF2","HF3"]].values
            # 条件1：归一化循环序号；条件2：源电池域标识
            cycle_norm = np.linspace(0, 1, len(df)).reshape(-1,1)
            battery_domain = np.full((len(df), 1), len(all_data), dtype=float)
            cond_arr = np.hstack([cycle_norm, battery_domain])
            combine = np.hstack([feat, cond_arr])
            all_data.append(combine)
        data = np.vstack(all_data).astype(np.float32)
        self.feature_min = data[:, :FEAT_DIM].min(axis=0)
        self.feature_max = data[:, :FEAT_DIM].max(axis=0)
        scale = np.maximum(self.feature_max - self.feature_min, 1e-8)
        # Generator uses Tanh, therefore real features must use the same [-1, 1] domain.
        data[:, :FEAT_DIM] = 2.0 * (data[:, :FEAT_DIM] - self.feature_min) / scale - 1.0
        self.data = data

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        feat = row[:FEAT_DIM]
        cond = row[FEAT_DIM:]
        return torch.from_numpy(feat), torch.from_numpy(cond)

class ConditionalBatchNorm1d(nn.Module):
    """Batch normalization modulated by the two conditioning variables."""
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.affine = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:num_features].fill_(1.0)
            self.affine.bias[num_features:].zero_()

    def forward(self, x, c):
        gamma, beta = self.affine(c).chunk(2, dim=1)
        return gamma * self.bn(x) + beta


# 生成器：论文表2 条件BN 8噪声+2条件 →256→512→256→4
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = NOISE_DIM + COND_DIM
        self.fc1 = nn.utils.spectral_norm(nn.Linear(self.in_dim, HID_DIM))
        self.bn1 = ConditionalBatchNorm1d(HID_DIM, COND_DIM)
        self.fc2 = nn.utils.spectral_norm(nn.Linear(HID_DIM, HID_DIM*2))
        self.bn2 = ConditionalBatchNorm1d(HID_DIM*2, COND_DIM)
        self.fc3 = nn.utils.spectral_norm(nn.Linear(HID_DIM*2, HID_DIM))
        self.bn3 = ConditionalBatchNorm1d(HID_DIM, COND_DIM)
        self.out = nn.utils.spectral_norm(nn.Linear(HID_DIM, FEAT_DIM))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.relu(self.bn1(self.fc1(x), c))
        x = self.relu(self.bn2(self.fc2(x), c))
        x = self.relu(self.bn3(self.fc3(x), c))
        return self.tanh(self.out(x))

# 判别器【修正维度，严格匹配论文表3】
# 输入6维(4特征+2条件) →256 →512 →256 →1
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = FEAT_DIM + COND_DIM
        self.fc1 = nn.utils.spectral_norm(nn.Linear(self.in_dim, HID_DIM))       # 6 → 256
        self.fc2 = nn.utils.spectral_norm(nn.Linear(HID_DIM, HID_DIM*2))         # 256 → 512
        self.fc3 = nn.utils.spectral_norm(nn.Linear(HID_DIM*2, HID_DIM))         # 512 → 256
        self.out = nn.utils.spectral_norm(nn.Linear(HID_DIM, 1))                # 256 → 1
        self.leaky = nn.LeakyReLU(0.2)
    def forward(self, x, c):
        inp = torch.cat([x, c], dim=1)
        h1 = self.leaky(self.fc1(inp))    # [32,256]
        h2 = self.leaky(self.fc2(h1))    # [32,512]
        h3 = self.leaky(self.fc3(h2))    # [32,256]
        score = self.out(h3)
        return score, h3  # h3用于特征匹配损失

# SCF-GAN训练函数（论文损失：对抗损失+0.1特征匹配损失）
def train_scf_gan():
    dataset = BatteryHFDataset()
    with open(SCALER_PATH, "w", encoding="utf-8") as f:
        json.dump({"min": dataset.feature_min.tolist(), "max": dataset.feature_max.tolist()}, f)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    opt_g = optim.Adam(gen.parameters(), lr=LR)
    opt_d = optim.Adam(dis.parameters(), lr=LR)
    loss_adv = nn.BCEWithLogitsLoss()

    print("==== 开始训练SCF-GAN ====")
    for ep in range(EPOCHS):
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_fm_loss = 0.0
        for feat, cond in loader:
            feat, cond = feat.to(device), cond.to(device)
            bs = feat.shape[0]
            real_label = torch.ones((bs,1), device=device)
            fake_label = torch.zeros((bs,1), device=device)

            # 更新判别器
            opt_d.zero_grad()
            real_score, _ = dis(feat, cond)
            loss_d_real = loss_adv(real_score, real_label)
            z = torch.randn((bs, NOISE_DIM), device=device)
            fake_feat = gen(z, cond)
            fake_score, _ = dis(fake_feat.detach(), cond)
            loss_d_fake = loss_adv(fake_score, fake_label)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            total_d_loss += loss_d.item()

            # 更新生成器
            opt_g.zero_grad()
            fake_score, h_fake = dis(fake_feat, cond)
            loss_g_adv = loss_adv(fake_score, real_label)
            with torch.no_grad():
                _, h_real = dis(feat, cond)
            # Match batch feature statistics; pairing individual real/fake rows is not meaningful.
            loss_fm = torch.mean(torch.square(h_fake.mean(0) - h_real.mean(0)))
            loss_g = loss_g_adv + LAMBDA_FM * loss_fm
            loss_g.backward()
            opt_g.step()
            total_g_loss += loss_g.item()
            total_fm_loss += loss_fm.item()

        if ep % 50 == 0:
            print(f"Epoch {ep:04} | D Loss:{total_d_loss/len(loader):.4f} | G Loss:{total_g_loss/len(loader):.4f} | FM Loss:{total_fm_loss/len(loader):.4f}")
    # 保存模型权重
    torch.save(gen.state_dict(), os.path.join(MODEL_SAVE, "scf_generator.pth"))
    torch.save(dis.state_dict(), os.path.join(MODEL_SAVE, "scf_discriminator.pth"))
    print("==== SCF-GAN训练完成，权重已保存至model文件夹 ====")

# 为B0018真实训练段生成5套HF特征；SOH标签始终来自真实目标电池。
def generate_aug_data():
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(os.path.join(MODEL_SAVE, "scf_generator.pth")))
    gen.eval()
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    feature_min = np.asarray(scaler["min"], dtype=np.float32)
    feature_max = np.asarray(scaler["max"], dtype=np.float32)
    df_raw = pd.read_csv(os.path.join(CSV_PATH, "B0018_hf.csv"))
    # 清洗数据
    for col in ["HF1","HF2","HF3","SOH"]:
        df_raw[col] = df_raw[col].apply(clean_num)
    df_raw = df_raw.dropna()
    train_len = int(len(df_raw) * 0.6)
    train_df = df_raw.iloc[:train_len].reset_index(drop=True)
    cycle_cond = np.arange(train_len, dtype=np.float32) / max(len(df_raw) - 1, 1)
    rows = []
    # Dataset 0 is the untouched real training trajectory.
    for i, row in train_df.iterrows():
        rows.append([0, i, row.HF1, row.HF2, row.HF3, row.SOH])
    with torch.no_grad():
        for dataset_id in range(1, 6):
            # 0.5 interpolates between the two source-battery domains.
            cond_np = np.column_stack([cycle_cond, np.full(train_len, 0.5)]).astype(np.float32)
            cond = torch.from_numpy(cond_np).to(device)
            z = torch.randn((train_len, NOISE_DIM), device=device)
            fake_scaled = gen(z, cond).cpu().numpy()
            fake_hf = (fake_scaled + 1.0) * 0.5 * (feature_max - feature_min) + feature_min
            fake_hf = np.maximum(fake_hf, 0.0)
            for i in range(train_len):
                # The label is the real target SOH at this training cycle.
                rows.append([dataset_id, i, *fake_hf[i], train_df.loc[i, "SOH"]])
    aug_df = pd.DataFrame(rows, columns=["DatasetID", "Cycle", "HF1", "HF2", "HF3", "SOH"])
    save_path = os.path.join(OUTPUT_PATH, "B0018_aug_train_hf.csv")
    aug_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"训练集增强完成：真实{train_len}条 + 5套合成HF，共{len(aug_df)}条；验证/测试未增强")

if __name__ == "__main__":
    train_scf_gan()
    generate_aug_data()
