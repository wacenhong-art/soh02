# Adaptive CTAC 模块实验报告

## 模块

Adaptive Cross-stage Trend Anchor Calibration仅使用上一阶段已观测Trend残差校准下一阶段预测。窗口长度和增益只按验证RMSE选择；测试标签不参与参数计算。常数偏移不改变Trend一阶差分或violation rate。

## Original G-MTC改善

- B0007: RMSE 3.6545 -> 2.6415，降低 27.72%。
- B0018: RMSE 1.8031 -> 1.6897，降低 6.29%。

## 结论

CTAC在B0007和B0018的Original G-MTC上均降低严格测试RMSE，满足新模块的探索性有效性目标。物理约束模式在B0018经过CTAC后也明显改善，但仍未稳定优于同样使用CTAC的Original，因此不能把CTAC成功等同于Physics-guided Trend约束已经得到跨电池验证。

本轮在观察测试结果前已尝试多种候选模块，因此这些结果属于探索性证据。论文正式结论仍需使用未参与本轮选择的新随机种子、新电池或外部数据集进行确认。