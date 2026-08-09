# Physics-guided Trend constraint：严格正式测试报告

生成时间：2026-08-09T20:21:00

## 协议锁定

- 模型结构保持：Trend/Seasonal 使用 original TimesNet，Residual 使用 iTransformer，之后由 Attention Bi-LSTM 融合。
- 物理损失只作用于 STL Trend 预测，不约束最终 SOH；未修改 SCF-GAN，未加入完整 PINN，也未使用 Frozen ELM。
- B0005/B0006 仅作为 SCF-GAN 源训练；B0007/B0018 分别按前 60% 训练适配、20% 验证、后 20% 严格测试。
- 这不是零标签纯跨电池协议：目标电池前 60% 的标签用于小样本适配，20% 标签用于模型/λ选择，最后 20% 仅用于一次锁定测试。
- λ=0.1 仅由验证集选择；正式测试种子为 42, 52, 62, 72, 82，测试后未调参、未按测试结果重跑。
- 测试预测采用递归方式，未知测试标签不作为下一时刻输入；因此比 teacher-forcing 结果更严格。

## 五随机种子结果

| Battery | Mode | RMSE mean±std | RMSE median | MAE mean | MAPE mean | R² mean | Trend violation mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0007 | Original | 85.6388±186.5934 | 2.0036 | 50.2879 | 69.8656% | -30043.2121 | 0.3939 |
| B0007 | Monotonic | 8.8130±14.9692 | 2.1360 | 7.1714 | 9.9633% | -218.3743 | 0.3273 |
| B0007 | Smooth | 2.0858±0.6559 | 1.8923 | 1.6966 | 2.3571% | -3.0084 | 0.2667 |
| B0007 | Both | 3.2934±2.7667 | 2.3154 | 2.6194 | 3.6392% | -13.4898 | 0.4121 |
| B0018 | Original | 9.1383±11.4385 | 4.2686 | 6.1389 | 8.8526% | -74.2210 | 0.5360 |
| B0018 | Monotonic | 4.3869±1.5133 | 4.1562 | 3.4556 | 4.9831% | -7.4252 | 0.5600 |
| B0018 | Smooth | 3.7764±0.9939 | 3.7437 | 2.9171 | 4.2066% | -5.0165 | 0.6240 |
| B0018 | Both | 7.4393±7.9185 | 4.8355 | 5.9948 | 8.6449% | -41.1738 | 0.4880 |

## 融合权重诊断

所有正式测试组合中，Seasonal 与 Residual 平均权重之和的最大值仅为 0.198490；融合层几乎全部选择 Trend（权重约为 1）。这属于注意力塌缩，说明当前严格结果不能用来证明多分支 Transformer 融合优于其他模型，也提示复现框架本身仍需先解决训练稳定性。

## Monotonic 与 Original 配对检验

- B0007：RMSE 改善 2/5 个种子；平均配对差值 -76.8258，中位配对差值 0.1081。时间块 bootstrap 95% CI [-233.3941, 0.8997]，改善概率 0.862。
- B0018：RMSE 改善 3/5 个种子；平均配对差值 -4.7513，中位配对差值 -0.2452。时间块 bootstrap 95% CI [-14.7653, 0.1020]，改善概率 0.938。

## 预注册成功标准

### B0007

- [PASS] Monotonic mean RMSE lower than Original；值=-76.82577881834644
- [FAIL] Monotonic improves RMSE in majority of 5 seeds；值=2/5
- [PASS] Mean strict Trend violation rate lower；值=-0.06666666666666665
- [PASS] Mean positive Trend variation lower；值=-285.72184295654296
- [FAIL] Paired block-bootstrap 95% CI below zero；值=[-233.394105, 0.899699]

### B0018

- [PASS] Monotonic mean RMSE lower than Original；值=-4.751341207820528
- [PASS] Monotonic improves RMSE in majority of 5 seeds；值=3/5
- [FAIL] Mean strict Trend violation rate lower；值=0.02400000000000002
- [FAIL] Mean positive Trend variation lower；值=0.8735824584960952
- [FAIL] Paired block-bootstrap 95% CI below zero；值=[-14.765307, 0.101997]

## 可用于论文的结论

本轮严格正式测试未通过全部预注册标准，不能声称该约束已经稳定降低 RMSE 或稳定改善物理一致性。

可以客观报告：验证集选择的 λ 在部分种子/电池上有效，但对初始化敏感；严格递归预测暴露了发散和跨种子不稳定。异常大 RMSE 使均值具有误导性，因此必须同时报告中位数、逐种子配对结果和 bootstrap 区间。

不能使用本次测试结果继续选择 λ、种子或架构；若要继续改进，应建立新的训练稳定化方案，并在新的未触碰外部测试电池/数据集上预注册后验证。

## 文件核验

- `final_multiseed_metrics.csv`：全部五种子、四模式原始指标。
- `paired_monotonic_vs_original.csv`：逐种子配对差值。
- `paired_block_bootstrap.csv`：时间块 bootstrap 置信区间。
- `success_criteria.csv`：预注册标准逐项 PASS/FAIL。
- `formal_test_predictions.csv`：全部逐循环真实值、预测值、Trend 与融合权重。
