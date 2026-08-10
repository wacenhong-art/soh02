# CILA Experiment Preregistration

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan → run
- Verification Status: PREREGISTERED, NOT YET EXECUTED
- Version Label: cila_v1

## Research Question

在保持SCF-GAN、STL、Trend/Seasonal TimesNet、Residual iTransformer和Attention Bi-LSTM不变时，对融合网络已有时间上下文表示增加跨BatteryID的CORAL协方差对齐，能否在小样本适配协议下改善跨工况电池SOH估计？

## Single Change

`L_total = L_prediction + lambda_weight * L_weight + lambda_align * L_CORAL`

- `L_CORAL`只作用于`AttentionBiLSTMFusion.encode_context()`返回的既有潜表示。冒烟尺度诊断后固定为跨域共享的批次标准化再计算协方差；不使用域标签分别标准化。
- 不新增预测网络，不替换任何分支，不修改SCF-GAN，不加入PINN，不约束最终SOH。
- `none, lambda_align=0`与`coral, lambda_align=0`必须在相同初始化和批次下数值等价。

## Domain and Split Lock

| Role | Batteries | Use |
|---|---|---|
| SCF-GAN frozen source | B0005, B0006 | 保持原源训练边界。 |
| Contacted development domains | B0007, B0018 | 各自前60%生成分支训练预测；中间20%选择lambda；后20%仅作已接触开发评估。 |
| Locked outer low-temperature group | B0049, B0050, B0051, B0052 | 4°C、2A、不同截止电压；配置锁定后才计算正式指标。 |
| Reserve | all other untouched candidates | 不参与本轮选择。 |

外层组仍允许每块目标电池前60%有标签小样本适配，因此不是零标签纯跨电池协议。论文主张必须写成“低温跨工况小样本适配中的方法有效性”，不能写成温度因果效应或零标签泛化。

## Hyperparameters

- `cila_mode`: `none`, `coral`
- `lambda_align`: `0`, `0.001`, `0.01`, `0.1`; default `0.01`
- seeds: `42, 52, 62, 72, 82`
- architecture, epoch budget, patience, learning rate, weight decay and `lambda_weight`: frozen baseline values
- official runs must pass `--lambda-weight 0.1`; earlier 2-epoch implementation smokes used the module default 0.01 and are excluded from accuracy comparison
- selection: development validation RMSE only; no outer-test label or curve may influence selection

## Gates

### Unit Gate

1. CORAL identical-domain loss is zero within tolerance.
2. CORAL loss is nonnegative and differentiable.
3. Every balanced batch contains at least two BatteryID domains and at least two samples per domain.
4. Sliding windows are created inside each battery and never cross BatteryID boundaries.
5. `lambda_align=0` gives exactly the base objective and unchanged two-output fusion API.
6. CORAL must be invariant to a shared affine rescaling of the pooled context within tolerance.

### Development Gate

1. B0007 and B0018 validation median RMSE both improve over pooled Original.
2. Each battery improves in at least 3/5 seeds.
3. Worst-battery RMSE decreases; MAE/MAPE/R2 show no material deterioration.
4. Paired block-bootstrap RMSE difference is reported. CI crossing zero means exploratory only.
5. Domain separability may be diagnostic only; lower separability with worse RMSE is negative transfer.

### Formal Outer Gate

After configuration hash lock, B0049–B0052 formal metrics are produced once for the five frozen seeds. If median RMSE fails to improve on every predeclared outer battery, or any battery materially deteriorates, CILA is not supported. The test protocol ends; no test-driven retuning is allowed.

## Known Confounds

- BatteryID, temperature, cutoff voltage and test campaign are not fully orthogonal.
- CORAL may erase SOH-relevant information.
- The frozen G-MTC baseline is auditable but did not match the paper's numerical result and did not establish Transformer superiority.
