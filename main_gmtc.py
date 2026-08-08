import os
import sys
import subprocess
import pandas as pd

# ===================== 路径配置 =====================
ROOT_PATH = "./"
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def run_step(step_name, script_path):
    """执行单步脚本并打印状态"""
    print("\n" + "="*70)
    print(f"执行步骤：{step_name}")
    print("="*70)
    result = subprocess.run([sys.executable, "-X", "utf8", script_path], check=False)
    if result.returncode != 0:
        print(f"[FAIL] {step_name} 执行失败，错误码：{result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {step_name} 执行完成")

def main():
    # Innovation edition defaults: lightweight backbone and validation-selected
    # Trend-only transfer, which avoids the observed Seasonal negative transfer.
    os.environ.setdefault("TIMESNET_VARIANT", "light")
    os.environ.setdefault("TRANSFER_PRETRAIN", "1")
    os.environ.setdefault("TRANSFER_COMPONENTS", "Trend")
    print("=" * 70)
    print("轻量G-MTC + OSELM 锂电池SOH估计创新框架")
    print("Paper: Data-augmented SOH estimation for lithium-ion batteries")
    print("=" * 70)

    # 按论文的分阶段流程执行，避免误用上一次运行遗留的中间结果。
    run_step("健康特征提取", "scripts/01_extract_hf.py")
    run_step("SCF-GAN数据增强", "scripts/02_scf_gan.py")
    run_step("STL分解与分量预测", "scripts/03_time_series.py")
    run_step("注意力Bi-LSTM动态融合", "scripts/04_attention_fusion.py")
    run_step("OSELM在线残差校正与消融", "scripts/05_oselm_correction.py")
    run_step("因果多速率与可靠性评估", "scripts/06_causal_multirate_reliability.py")
    run_step("创新模块消融汇总", "scripts/07_innovation_summary.py")

    # 步骤2：基线模型对比
    run_step("基线模型训练与对比", "baseline_compare.py")

    # 步骤3：结果可视化绘图
    run_step("论文图表绘制", "plot_results.py")
    run_step("创新结果图表与人工核验输出", "plot_innovation_results.py")

    # 严格验收：未达到论文指标或仍有逻辑偏差时主流程必须失败。
    run_step("论文一致性验收", "scripts/05_paper_compliance.py")

    # 最终汇总
    print("\n" + "="*70)
    print("完整流程执行完毕！")
    print("="*70)
    print("输出文件清单：")
    print(f"  1. G-MTC预测结果：{os.path.join(OUTPUT_PATH, 'gmtc_pred_result.csv')}")
    print(f"  2. 分量预测结果：{os.path.join(OUTPUT_PATH, 'component_pred_result.csv')}")
    print(f"  3. 基线对比表格：{os.path.join(OUTPUT_PATH, 'baseline_compare_result.csv')}")
    print(f"  4. 论文图表文件夹：{os.path.join(OUTPUT_PATH, 'figures')}")
    print(f"  5. 创新消融结果：{os.path.join(OUTPUT_PATH, 'innovation_ablation_result.csv')}")
    print(f"  6. 可靠性结果：{os.path.join(OUTPUT_PATH, 'causal_reliability_result.csv')}")
    print(f"  7. 完整消融表：{os.path.join(OUTPUT_PATH, 'innovation_complete_ablation.csv')}")
    print("="*70)

if __name__ == "__main__":
    main()
