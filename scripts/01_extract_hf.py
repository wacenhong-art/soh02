import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

# 全局路径配置
ROOT_PATH = "./"
RAW_MAT_PATH = os.path.join(ROOT_PATH, "data", "raw_mat")
SAVE_CSV_PATH = os.path.join(ROOT_PATH, "data", "hf_csv")
# 论文标准参数
NOMINAL_CAP = 2.0
CUTOFF_CURR = 0.02

os.makedirs(SAVE_CSV_PATH, exist_ok=True)


def smooth_feature_outliers(df, columns=("HF1", "HF2", "HF3"), window=7, threshold=4.0, verbose=True):
    """Replace isolated extraction glitches using a causal-data-safe Hampel filter."""
    out = df.copy()
    for col in columns:
        series = out[col].astype(float)
        median = series.rolling(window, center=True, min_periods=3).median()
        deviation = (series - median).abs()
        mad = deviation.rolling(window, center=True, min_periods=3).median()
        scale = 1.4826 * mad
        bad = (series <= 0) | ((scale > 0) & (deviation > threshold * scale))
        out.loc[bad, col] = np.nan
        out[col] = out[col].interpolate(limit_direction="both")
        if verbose:
            print(f"{col} Hampel修复异常点：{int(bad.sum())}")
    return out

def calc_h_features(voltage, current, time_seq):
    # 论文HF1 HF2 HF3计算公式
    idx_36 = np.flatnonzero(voltage >= 3.6)[0]
    idx_42 = np.flatnonzero(voltage >= 4.2)[0]
    if idx_42 <= idx_36:
        raise ValueError("未检测到有效的3.6V-4.2V充电区间")
    hf1 = time_seq[idx_42] - time_seq[idx_36]

    # CV截止电流必须在到达4.2V之后查找；从全序列起点搜索会命中
    # 充电前的零/负电流点并产生非物理的负HF2。
    post_cv = np.flatnonzero(np.abs(current[idx_42:]) <= CUTOFF_CURR)
    if len(post_cv) == 0:
        raise ValueError("4.2V之后未检测到截止电流")
    idx_cut = idx_42 + post_cv[0]
    hf2 = time_seq[idx_cut] - time_seq[idx_42]

    idx_38 = np.flatnonzero(voltage >= 3.8)[0]
    idx_40 = np.flatnonzero(voltage >= 4.0)[0]
    hf3 = time_seq[idx_40] - time_seq[idx_38]
    return hf1, hf2, hf3

def extract_battery_frame(mat_data, battery_name, verbose=True):
    """Extract the paper health features from an already loaded MAT mapping."""
    raw_data = mat_data[battery_name][0, 0]
    cycle_arr = raw_data["cycle"][0]
    # 兼容一维cycle数组
    total_cycles = len(cycle_arr)

    discharge_cap_dict = {}
    # 第一步：缓存所有放电循环容量
    for cycle_idx in range(total_cycles):
        cycle_info = cycle_arr[cycle_idx]
        cycle_type = cycle_info["type"][0]
        if cycle_type == "discharge":
            try:
                cap_val = float(np.asarray(cycle_info["data"]["Capacity"][0, 0]).squeeze())
                discharge_cap_dict[cycle_idx] = cap_val
            except Exception:
                continue

    feature_record = []
    # 第二步：遍历充电循环提取特征
    for cycle_idx in range(total_cycles):
        cycle_info = cycle_arr[cycle_idx]
        cycle_type = cycle_info["type"][0]
        if cycle_type != "charge":
            continue
        try:
            data_block = cycle_info["data"][0, 0]
            t = data_block["Time"].flatten()
            v = data_block["Voltage_measured"].flatten()
            i = data_block["Current_measured"].flatten()
            hf1, hf2, hf3 = calc_h_features(v, i, t)

            # 匹配后续放电容量
            real_cap = None
            for j in range(cycle_idx, total_cycles):
                if j in discharge_cap_dict:
                    real_cap = discharge_cap_dict[j]
                    break
            if real_cap is None:
                continue
            soh = real_cap / NOMINAL_CAP
            feature_record.append([hf1, hf2, hf3, soh])
        except Exception as err:
            if verbose:
                print(f"跳过{battery_name}第{cycle_idx}圈，异常：{str(err)}")
            continue
    df_out = pd.DataFrame(feature_record, columns=["HF1", "HF2", "HF3", "SOH"])
    df_out = smooth_feature_outliers(df_out, verbose=verbose)
    return df_out


def single_battery_extract(battery_name):
    mat_file = os.path.join(RAW_MAT_PATH, f"{battery_name}.mat")
    mat_data = loadmat(mat_file)
    df_out = extract_battery_frame(mat_data, battery_name)
    # 修复变量名：使用battery_name
    save_path = os.path.join(SAVE_CSV_PATH, f"{battery_name}_hf.csv")
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"==== {battery_name} 提取完成，有效循环：{len(df_out)} ====")

if __name__ == "__main__":
    bat_list = ["B0005", "B0006", "B0007", "B0018"]
    for bat in bat_list:
        single_battery_extract(bat)
