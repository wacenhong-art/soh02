import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置 =====================
ROOT_PATH = "./"
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
FIG_PATH = os.path.join(OUTPUT_PATH, "figures")
os.makedirs(FIG_PATH, exist_ok=True)

# 绘图风格：匹配论文学术风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["lines.linewidth"] = 1.5

def plot_soh_scatter():
    """图1：真实SOH vs 预测SOH散点图（对应论文Fig.7/8）"""
    df = pd.read_csv(os.path.join(OUTPUT_PATH, "gmtc_pred_result.csv"))
    true = df["True_SOH"].values
    pred = df["Pred_SOH"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=15, alpha=0.7, color="#1f77b4", label="G-MTC Estimation")
    
    # 理想估计参考线 y=x
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Estimation")
    
    plt.xlabel("Reference SOH (%)")
    plt.ylabel("Estimated SOH (%)")
    plt.title("SOH Estimation Result (G-MTC)")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "fig7_soh_scatter.png"), dpi=300)
    plt.close()
    print("✅ 绘制完成：SOH散点图")

def plot_components():
    """图2：STL三分量真实值与预测值时序（对齐样本长度）"""
    df_comp = pd.read_csv(os.path.join(OUTPUT_PATH, "component_pred_result.csv"))
    df_soh = pd.read_csv(os.path.join(OUTPUT_PATH, "gmtc_pred_result.csv"))
    if "Cycle" in df_comp.columns and "Cycle" in df_soh.columns:
        merged = df_comp.merge(
            df_soh[["Cycle", "True_SOH", "Pred_SOH"]], on="Cycle", how="inner"
        )
        df_comp = merged
        df_soh = merged
        cycles = merged["Cycle"].to_numpy()
    else:
        min_len = min(len(df_comp), len(df_soh))
        df_comp = df_comp.iloc[-min_len:].reset_index(drop=True)
        df_soh = df_soh.iloc[-min_len:].reset_index(drop=True)
        cycles = np.arange(min_len)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 总SOH
    ax = axes[0, 0]
    ax.plot(cycles, df_soh["True_SOH"], "k-", label="Real SOH")
    ax.plot(cycles, df_soh["Pred_SOH"], "r--", label="Predicted SOH")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH (%)")
    ax.set_title("(a) SOH Estimation")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    # 趋势分量
    ax = axes[0, 1]
    ax.plot(cycles, df_comp["True_Trend"], "k-", label="Real trend component")
    ax.plot(cycles, df_comp["Pred_Trend"], "r--", label="Predicted trend component")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Trend")
    ax.set_title("(b) Trend Component")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    # 季节分量
    ax = axes[1, 0]
    ax.plot(cycles, df_comp["True_Seasonal"], "k-", label="Real seasonal component")
    ax.plot(cycles, df_comp["Pred_Seasonal"], "r--", label="Predicted seasonal component")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Seasonal")
    ax.set_title("(c) Seasonal Component")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    # 残差分量
    ax = axes[1, 1]
    ax.plot(cycles, df_comp["True_Residual"], "k-", label="Real residual component")
    ax.plot(cycles, df_comp["Pred_Residual"], "r--", label="Predicted residual component")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Residual")
    ax.set_title("(d) Residual Component")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "fig9_components.png"), dpi=300)
    plt.close()
    print("✅ 绘制完成：三分量时序图")

def plot_weights():
    """图3：三分量动态权重变化曲线（对应论文Fig.10）"""
    df = pd.read_csv(os.path.join(OUTPUT_PATH, "gmtc_pred_result.csv"))
    cycles = np.arange(len(df))

    plt.figure(figsize=(8, 5))
    # 增加圆点标记，避免线太短看不见
    plt.plot(cycles, df["w_Trend"], "o-", color="#1f77b4", label="Trend component weight", linewidth=2, markersize=4)
    plt.plot(cycles, df["w_Seasonal"], "s-", color="#ff7f0e", label="Seasonal component weight", linewidth=2, markersize=4)
    plt.plot(cycles, df["w_Residual"], "^-", color="#2ca02c", label="Residual component weight", linewidth=2, markersize=4)
    
    plt.xlabel("Cycle number")
    plt.ylabel("Weight")
    plt.title("Dynamic Weight Distribution of Three Components")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    # 放宽y轴范围，避免线条贴边看不见
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "fig10_weights.png"), dpi=300)
    plt.close()
    print("✅ 绘制完成：动态权重曲线")

def plot_rmse_bar():
    """图4：多模型RMSE柱状对比图（对应论文Fig.11）"""
    df = pd.read_csv(os.path.join(OUTPUT_PATH, "baseline_compare_result.csv"), index_col=0)
    
    # 加入G-MTC结果
    gmtc_df = pd.read_csv(os.path.join(OUTPUT_PATH, "gmtc_pred_result.csv"))
    from sklearn.metrics import mean_squared_error
    gmtc_rmse = np.sqrt(mean_squared_error(gmtc_df["True_SOH"], gmtc_df["Pred_SOH"]))
    
    models = list(df.index) + ["G-MTC"]
    rmse_values = list(df["RMSE%"].values) + [gmtc_rmse]
    
    colors = ["#aec7e8"] * len(df.index) + ["#1f77b4"]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, rmse_values, color=colors, edgecolor="k", linewidth=0.8)
    
    plt.xlabel("Method")
    plt.ylabel("RMSE (%)")
    plt.title("RMSE Comparison of Different Methods")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "fig11_rmse_bar.png"), dpi=300)
    plt.close()
    print("✅ 绘制完成：RMSE对比柱状图")

def main():
    print("\n" + "="*60)
    print("开始绘制论文图表")
    print("="*60)
    
    plot_soh_scatter()
    plot_components()
    plot_weights()
    plot_rmse_bar()
    
    print("\n" + "="*60)
    print(f"所有图表已保存至：{FIG_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()
