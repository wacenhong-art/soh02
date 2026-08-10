# Physics-guided Trend v3实验报告

## 结论

物理损失已经按同BatteryID、DatasetID和连续Cycle正确作用于STL Trend，四模式和λ敏感性均已实际运行。但当前证据不支持“物理约束稳定降低RMSE”的论文主张。

`PhysicsInnovationSupported=false`。

## 公式与边界

- `L_total=L_prediction+lambda_phy*L_physics`
- monotonic：`mean(ReLU(T[t+1]-T[t]))`
- smooth：`mean((T[t+1]-2T[t]+T[t-1])^2)`
- both：monotonic+smooth
- physics只作用Trend，不作用Seasonal、Residual或最终SOH。
- 每个DatasetID整条有序训练轨迹独立计算，不在shuffle样本间构造相邻关系。

## B0034四模式，λ=0.001

| 模式 | RMSE | MAE | MAPE | R² | Trend violation |
|---|---:|---:|---:|---:|---:|
| Original | 2.3620 | 1.7640 | 2.5884% | 0.5371 | 38.93% |
| monotonic | 2.8141 | 2.3489 | 3.4467% | 0.3429 | 38.26% |
| smooth | 3.1145 | 2.7432 | 4.0253% | 0.1951 | 40.27% |
| both | **2.3141** | **1.6847** | **2.4720%** | **0.5557** | **38.26%** |

both相对Original的RMSE下降2.03%，但时间块bootstrap差值95%CI为[-0.1499, 0.0416]，跨过0，只能称为探索性改善。

## λ敏感性

- λ=0.001：RMSE 2.3141，唯一同时改善RMSE与违规率的B0034配置。
- λ=0.01：RMSE 2.9562，失败。
- λ=0.1：RMSE 3.1084，失败。
- 物理损失过强会牺牲小样本预测拟合。

## 冻结配对复核

| 电池 | Original RMSE | both RMSE | 差值 | 违规率变化 | Bootstrap 95%CI | 结论 |
|---|---:|---:|---:|---:|---|---|
| B0034 | 2.3620 | 2.3141 | -0.0479 | 38.93%→38.26% | [-0.1499, 0.0416] | 探索性改善 |
| B0036 | 4.2444 | 4.2098 | -0.0346 | 38.71%→37.90% | [-0.0996, 0.0235] | 探索性改善 |
| B0018 | 6.3577 | 6.6915 | +0.3338 | 37.25%→37.25% | [0.3146, 0.3658] | 明确恶化 |

B0018是已接触开发目标，不能用于调回λ或重新选择模式。固定锚点工程版本的B0018 RMSE 2.7334仍优于可训练锚点物理版本，但它不是公平physics配对基线。

## 投影诊断

- Trend硬cummin把违规率降为0，但RMSE恶化到2.9396。
- 25%软投影降低正向跳变幅度，但RMSE恶化到3.0185，违规率升高。
- 因而没有把投影包装为成功创新。

## 论文可写性

目前可以写“方法实现与负结果/探索性结果”，不能写“统计稳定地提升SOH精度”。还缺少：

1. 五随机种子中每个电池至少3/5改善；
2. B0007和B0018中位RMSE均低于Original；
3. bootstrap置信区间排除0；
4. 未接触确认电池上的一致改善；
5. 物理一致性显著下降且MAE、MAPE、R²不恶化。

## 可审计文件

- `ablation_metrics.csv`
- `lambda_sensitivity.csv`
- `paired_battery_metrics.csv`
- `per_cycle_predictions.csv`
- `trend_predictions.csv`
- `bootstrap_confidence_intervals.csv`
- `projection_diagnostics.csv`
- `configs/physics_v3_lock.json`
- `training_logs/ARTIFACT_INDEX.csv`
