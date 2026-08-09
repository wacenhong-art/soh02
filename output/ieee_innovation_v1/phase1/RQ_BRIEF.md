# Research Question Brief

## Topic Area

面向真实BMS应用约束的锂离子电池SOH估计，并在冻结的G-MTC架构上实施单一、可关闭、可证伪的创新。

## Primary Research Question

在不改变G-MTC核心架构、只增加一个可关闭机制的条件下，IEEE于2024-08-10至2026-08-10发表的SOH研究所揭示的哪些现实问题导向干预，能够在无数据泄漏的NASA及预注册外部数据评估中，为冻结G-MTC基线带来最可信、可重复的改善？

## FINER Assessment

| Criterion | Score | Justification |
|---|---:|---|
| Feasible | 4/5 | 当前代码、可信基线和NASA完整数据可用；外部数据兼容性尚待核验。 |
| Interesting | 5/5 | 直接针对实验室SOH模型的真实部署障碍。 |
| Novel | 4/5 | 新颖性由近两年IEEE空缺和单机制隔离共同决定，而非网络堆叠。 |
| Ethical | 5/5 | 使用公开电池数据，不涉及人类受试者。 |
| Relevant | 5/5 | 直接服务BMS泛化、可靠性与可复现实证。 |
| **Average** | **4.6/5** | 达到ARS阈值。 |

## Scope Boundaries

**In scope:** 2024-08-10至2026-08-10 IEEE论文；锂离子电池SOH；小样本、跨工况、噪声/缺失、部分充电、在线可靠性等现实痛点；一个新增机制；公开且可审计数据。

**Out of scope:** 完整PINN；同时替换TimesNet/iTransformer/SCF-GAN；多个创新模块堆叠；测试集调参；依赖不可获得的专用传感器；把B0007/B0018当作未接触确认集。

**Key assumptions:** 冻结基线用于相对比较而非宣称复现论文数值；最终方向由文献证据和数据可验证性共同决定；正式测试划分在标签结果评估前冻结。

## Sub-questions

1. 近两年IEEE SOH论文集中暴露了哪些尚未解决的真实应用痛点？
2. 哪些痛点能够由现有NASA数据或合法公开数据严格验证？
3. 哪一个单一机制在现实价值、G-MTC契合度、可实现性和可证伪性方面综合最优？

## Phase Status

等待用户确认后进入Sider Scholar文献检索阶段。
