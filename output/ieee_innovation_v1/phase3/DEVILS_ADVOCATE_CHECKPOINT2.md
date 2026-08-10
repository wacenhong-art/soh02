# Devil's Advocate Report — Checkpoint 2

## Verdict

**PASS WITH MANDATORY SAFEGUARDS**，仅允许进入“用户确认候选方向”门槛，不允许直接宣称创新有效。

## Critical Issues

无不可修复关键问题，但下列主要问题必须写入预注册协议。

## Major Issues

1. **工况—电池身份混杂**：NASA温度、电流、截止条件和BatteryID并非完全正交；只能称跨工况电池泛化，不能称温度因果鲁棒性。
2. **负迁移**：分布对齐可能抹去SOH有效特征。若worst-condition RMSE或两块目标组中任一显著恶化，应判失败。
3. **目标域接触污染**：B0007/B0018已接触。CILA权重不得由它们的测试结果选择，确认性结论必须使用预注册未接触外层电池。
4. **弱基线归因**：BaselineComplete为true，但PaperNumericMatched和TransformerBestObserved均为false；相对改善不能包装为论文数值复现或Transformer优越。
5. **证据粒度**：10篇深读均为摘要级Level VI，方法细节和数据划分仍可能改变适配判断。

## Strongest Counter-Argument

CILA可能只是对一个表现较弱、Attention近静态的工程基线加常规正则；任何改善也可能来自训练组构成，而不是解决真实工况漂移。只有同一冻结架构、单因素变化、未接触外层组、简单模型对照和配对统计同时成立，才能缓解该批评。

## Required Stress Tests

| Stress test | Pre-registered interpretation |
|---|---|
| 删除最强支持论文 | 推荐仍需由至少3个独立方法簇支持。 |
| 对齐权重为0 | 必须精确回到冻结Original路径。 |
| 条件标签置乱 | 若仍同等改善，说明机制可能只是普通正则。 |
| 留一训练域 | 检查对某个源域的依赖。 |
| 域可分性下降但RMSE上升 | 判定负迁移，不得称成功。 |
| CNN/LSTM简单模型 | 不预设Transformer最好。 |

## Evidence Not Found

- 未找到近期IEEE论文证明CORAL损失在当前SCF-GAN + STL + TimesNet/iTransformer + Attention Bi-LSTM结构上有效。
- 未找到能够解除NASA工况与BatteryID混杂的元数据证据。
- 未找到允许用已接触B0007/B0018作强确认性测试的合理依据。

## Gate

Checkpoint 2通过，但只到用户确认门槛。确认前不修改模型；确认后先冻结外层角色和成功/失败标准，再实现。
