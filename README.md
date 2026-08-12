# soh02：严格 G-MTC SOH 复现实验

本仓库只保留一条论文核心链和一条隔离的 Trend 物理约束消融链。历史上的
OSELM、因果多速率、CILA、测试后救援和重复报告脚本已移除；需要追溯时使用
Git 历史，不再让历史试验参与当前结果。

## 精简后的结构

```text
main_gmtc.py                    统一入口
soh_pipeline/                   数据边界、指标和预注册协议
scripts/01_extract_hf.py        HF1/HF2/HF3
scripts/02_scf_gan.py           仅增强训练数据
scripts/03_time_series.py       STL + TimesNet + iTransformer
scripts/04_attention_fusion.py  Attention Bi-LSTM
scripts/05_paper_compliance.py  论文核心验收
scripts/08_physics_trend_ablation.py  仅约束 STL Trend
scripts/09_paper_grid.py        构建/运行验证网格
scripts/18_data_quality_audit.py      原始数据审计
tests/                          泄漏、指标、模型结构与物理损失检查
```

## 为什么参考项目看起来更容易得到高分

- MVIP-Trans 将每块电池前 70% 放入训练，再预测同一电池后 30%；其公开代码只算
  RMSE，不算 R²，并对预测固定减去 0.01。这不是 `soh02` 的跨电池严格留出协议。
- SP-LSTM 用包含 B0018 全生命周期的特征拟合归一化器，并用 B0018 前 5 个真实
  测试标签计算预测偏移；公开主流程同样没有计算 R²。因此其低 RMSE 不能作为
  严格盲测 `R² > 0.95` 的证据。
- `soh02` 的 R² 为负表示模型 SSE 大于“把测试集真实均值作为常数预测”的 SSE。
  这通常来自跨电池分布偏移、递归分量误差和融合权重失稳，而不是指标程序错误。

## 命令

```powershell
python main_gmtc.py paper
python main_gmtc.py physics --evaluation-split validation --seed 42
python main_gmtc.py grid build
python main_gmtc.py grid one --config <config.json> --seed 42
python main_gmtc.py grid stage --max-configs 10 --seed 42
```

正式测试只能在配置和五种子验证结果锁定后运行一次。默认物理消融只看验证段，
避免不小心反复读取测试结果。

## 预注册实验与可复现性检查

研究问题：在保持 SCF-GAN、STL、TimesNet/iTransformer 和 Attention Bi-LSTM
不变时，表示参数或 Trend-only 物理约束能否同时改善 B0007 与 B0018 的验证
RMSE，并在锁定后提高严格测试 R²？

1. 固定源电池 B0005/B0006、目标电池 B0007/B0018、目标前 20% 标注适配。
2. 阶段 A 仅在验证段选择序列长度、STL 周期和模型宽度；两块目标电池必须同时改善。
3. 晋级配置运行种子 42/52/62；最终候选运行 42/52/62/72/82。
4. 锁定配置哈希后仅运行一次测试；不得依据测试结果创建新配置。
5. 每次保存配置哈希、随机种子、训练/验证/测试边界、指标 CSV、预测 CSV、训练日志和模型权重。
6. 检查归一化器只拟合训练段、STL 只拟合训练前缀、GAN 不生成验证/测试标签、相邻物理损失只用于同一连续轨迹。
7. 同时报告 RMSE、MAE、MAPE、R²、均值基线 RMSE、Trend 违反率；不只报告最优种子。
8. 随机实验复跑相对差异目标小于 5%；超出时报告不稳定，不挑选有利运行。

目标 `R² > 0.95` 是待检验假设，不是验收时必须伪造出来的数字。
