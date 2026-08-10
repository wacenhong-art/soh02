# CILA Unit and Data Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run preparation
- Verification Status: VERIFIED AT UNIT/DATA LEVEL
- Version Label: cila_v1_pretraining

## Result

**PASS**。这只表示实现可以进入冒烟训练，不表示CILA改善RMSE。

## Synthetic Unit Gates

| Gate | Result |
|---|---|
| 相同协方差域的CORAL为0 | PASS |
| 不同协方差的CORAL大于0 | PASS |
| CORAL梯度存在且有限 | PASS |
| 每批两个域且每域4个样本 | PASS |
| 原两输出API与新增上下文API在评估模式下预测一致 | PASS |
| `lambda_align=0`精确退化为原目标 | PASS |
| 分电池构窗不跨边界 | PASS |

## Real Development Data Audit

| Battery | Train windows | Validation windows | Cycle order | Cross-battery windows | Labels used as input |
|---|---:|---:|---|---|---|
| B0007 | 80 | 34 | strictly increasing | no | no |
| B0018 | 56 | 26 | strictly increasing | no | no |

输入来自现有`candidate_ensemble/seed_52/*/none`原始分支结果，仅用于边界和形状检查。没有运行训练、没有读取正式外层B0049–B0052结果、没有产生拟合指标。

## Preserved Failed Checks

1. 文档运行时缺少PyTorch，改用项目既有训练运行时。
2. Python 3.14动态导入未登记模块，修复测试加载器。
3. 在训练模式比较连续前向时受原模型Dropout影响，改为评估模式验证API等价；未关闭训练阶段Dropout。

## Next Command Gate

下一步为2-epoch、单种子、开发验证集冒烟训练。其输出只检查训练可执行性，不用于选择lambda或论文结论。
