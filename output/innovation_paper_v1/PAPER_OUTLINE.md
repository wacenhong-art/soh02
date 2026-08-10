# Paper Outline

## Structure pattern

IMRaD + Related Work + Reproducibility and Integrity + Limitations。

## Overview

论文从G-MTC在小样本SOH估计中的结构动机出发，先给出可信复现审计，再分别检验两个互不叠加、可关闭的最小干预：仅作用于STL Trend分支的物理损失，以及只作用于既有融合上下文的协方差对齐。结果部分不以“寻找最好模型”为叙事，而以预注册门槛检验可重复性、物理一致性和跨电池泛化；讨论部分解释开发集微弱改善为何在未接触外层电池上失效。

## Detailed outline

### 1. 引言（约900字）

**Purpose:** 定义小样本、跨电池域偏移与可信评估问题。

- 1.1 SOH估计的工程意义与数据稀缺。
- 1.2 G-MTC的可解释分解优势及论文披露不足。
- 1.3 近期IEEE研究所指向的域偏移、部分标签和现场可靠性痛点。
- 1.4 研究问题、贡献和明确的负结果定位。

**Evidence:** Yang等原论文；Miao、Song、Yao、Luo等近期IEEE论文；本项目符合性与泄漏审计。

**Transition:** 从现实问题进入方法谱系和最小创新选择。

### 2. 相关工作（约1100字）

**Purpose:** 说明原架构、物理先验和域对齐三条线索及其边界。

- 2.1 SCF-GAN + STL + TimesNet/iTransformer + Attention Bi-LSTM。
- 2.2 Trend物理一致性与不约束最终SOH的理由。
- 2.3 跨电池/工况域偏移与CORAL式二阶统计对齐。
- 2.4 研究空缺：缺少冻结架构、五种子、预注册外层确认的负迁移证据。

**Transition:** 将文献空缺转化为可证伪实验设计。

### 3. 方法（约1700字）

**Purpose:** 给出可复现的原架构、数据协议和两个隔离干预。

- 3.1 HF1/HF2/HF3与SOH定义。
- 3.2 SCF-GAN训练边界和STL三分量建模。
- 3.3 P0/P1基线及工程稳定化分类。
- 3.4 Trend-only physics loss：none/monotonic/smooth/both。
- 3.5 CILA：融合上下文均值/协方差对齐，不替换任何主干。
- 3.6 60/20/20有标签适配协议、外层电池选择和泄漏门禁。
- 3.7 指标、五种子、配对时间块bootstrap和成功门槛。

**Transition:** 从设计承诺进入按预注册顺序报告的证据。

### 4. 结果（约1500字）

**Purpose:** 分层报告基线、物理约束、开发验证和外层确认。

- 4.1 基线可信性与原论文数值差距。
- 4.2 CNN/LSTM/Transformer比较和Attention诊断。
- 4.3 Trend物理约束消融及未通过门槛的原因。
- 4.4 CILA开发集配对结果。
- 4.5 B0043/B0044正式外层负结果与极端递归外推。

**Transition:** 将数值失败转化为机制边界解释，而非事后调参。

### 5. 讨论（约1200字）

**Purpose:** 解释为什么局部约束不能修复错误的递归分支轨迹。

- 5.1 原论文“Transformer/融合更优”叙事在本复现中的不稳定性。
- 5.2 Trend物理一致性与最终SOH准确率并非同一目标。
- 5.3 CILA只能对齐融合上下文，无法校正分支外推爆炸。
- 5.4 对真实BMS研究的启示：先验证可辨识性和外推稳定性，再做域对齐。

### 6. 局限性（约500字）

NASA实验室数据、目标标签适配、外层幸存者筛选、论文未披露参数、外层仅两电池、开发集已接触、置信区间和极端值解释限制。

### 7. 结论（约400字）

回答三个问题：可信基线成立但数值未复现；两个创新均未获得确认性支持；最强贡献是给出可审计失败边界和后续单一改进方向。

## Evidence map

| Section | Evidence | Role |
|---|---|---|
| 1–2 | 原论文PDF、10篇已核验IEEE文献 | 背景、问题与空缺 |
| 3 | method_compliance_matrix、protocol、配置哈希 | 方法与可复现性 |
| 4.1–4.2 | baseline_comparison、multiseed_metrics、attention_diagnostics | 基线与模型比较 |
| 4.3 | physics_trend_v3消融与逐循环Trend预测 | 物理约束证据 |
| 4.4–4.5 | CILA五种子开发/外层预测与bootstrap | 域对齐与负迁移证据 |
| 5–7 | attempt ledger、外层失败模式、11项统计谬误扫描 | 解释、限制与结论 |

