# Physics-guided Trend constraint报告（physics_trend_v3）

- PhysicsInnovationSupported: **false**
- λ=0.1仅由验证集选择；正式测试后未调参。
- 物理损失仅作用于STL Trend；未约束最终SOH。

## 五随机种子正式结果

| Battery | Mode | RMSE mean±std | RMSE median | Trend violation mean | improved seeds |
|---|---|---:|---:|---:|---:|
| B0007 | none | 5.9110±1.9770 | 5.2050 | 0.0545 | - |
| B0007 | monotonic | 8.4873±7.1560 | 5.9450 | 0.1212 | 2/5 |
| B0007 | smooth | 4.9721±1.1966 | 5.2186 | 0.0788 | 3/5 |
| B0007 | both | 4.6723±0.9630 | 5.0157 | 0.0788 | 4/5 |
| B0018 | none | 5.1770±0.5758 | 5.1367 | 0.0880 | - |
| B0018 | monotonic | 5.1754±0.6830 | 4.9920 | 0.1040 | 4/5 |
| B0018 | smooth | 5.1767±0.9149 | 5.1393 | 0.1200 | 2/5 |
| B0018 | both | 5.0926±0.9422 | 4.8323 | 0.0720 | 2/5 |

## 结论

没有任何单一physics_mode同时满足两电池中位RMSE下降、各至少3/5种子改善、Trend违规率下降且bootstrap区间完全低于0。
因此不能声称物理约束稳定降低RMSE。部分模式在B0007或个别种子显著改善，但B0018改善次数不足，monotonic还出现B0007种子82严重失败。
该结果可作为探索性/负结果写入论文，但不足以支持强创新有效性结论；确认性结论还需要未接触外部电池。