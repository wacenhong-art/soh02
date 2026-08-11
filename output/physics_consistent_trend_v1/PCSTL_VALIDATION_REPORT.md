# PC-STL 三随机种子开发验证报告

## 结论

`three_seed_gate_passed=False`。因此禁止继续进入五随机种子、配置锁定和任何确认测试。

PC-STL只使用训练前缀的STL Trend建立单调平滑监督目标，差额回写Residual；P0与PC-STL均使用相同P1 Attention初始化。验证集用于选择mu=1.0；没有读取测试标签。

| Battery | P0 median RMSE | PC-STL median RMSE | Improved seeds | Median violation delta | Gate |
| B0007 | 1.0354 | 0.7390 | 2/3 | +0.0000 | PASS |
| B0018 | 0.8681 | 2.6582 | 1/3 | +0.0400 | FAIL |

B0018未满足至少2/3种子RMSE改善；继续在同一开发电池调参会产生选择偏差，故该创新路线以负结果停止。
