# Methodology Blueprint

## Research Paradigm

**Selected:** Pragmatist-positivist applied research.

**Justification:** 研究先用结构化文献证据识别工程问题，再用可重复计算实验检验一个明确机制；结论由可观察指标和预注册门槛决定。

## Method

**Type:** 结构化证据扫描 + 多准则决策分析 + 配对计算实验。

1. 使用Sider Scholar检索近两年IEEE论文，保留完整检索式、时间和筛选理由。
2. 至少20篇候选、8篇深入核验，提取WHY/HOW/WHAT、数据和现实局限。
3. 最多形成3个候选创新，以预先固定的100分量表选择唯一方向。
4. 用户确认后实现；Original与Proposed保持相同数据、种子、预算和架构，仅一个机制不同。
5. 先验证后锁定配置，正式测试只运行一次；保存逐循环预测和配对bootstrap。

## Data Strategy

- 文献：IEEE Xplore收录且发布日期位于2024-08-10至2026-08-10。
- 主数据：NASA完整归档，原始ZIP只读。
- 已接触开发评估：B0007、B0018。
- 未接触候选池：B0025-B0056；本阶段不指定正式测试角色，不比较其模型误差。
- 外部数据：仅当NASA不能回答最终研究问题时，从官方或带许可证研究存储库下载。

## Analytical Framework

- 文献：去重、来源核验、证据等级、现实痛点主题归纳和反证搜索。
- 方向选择：现实价值25、架构契合25、数据可行20、研究空缺15、隔离消融10、成本5。
- 实验：RMSE、MAE、MAPE、R2、专用痛点指标、5种子分布、配对时间块bootstrap、复杂度和推理时间。

## Validity Criteria

| Criterion | Strategy |
|---|---|
| Internal validity | 单因素改动；训练/验证/测试边界单元测试；配置哈希锁定。 |
| External validity | 预注册未接触电池或外部数据；分开报告跨电池、跨温度和跨数据集结果。 |
| Reliability | 固定五种子；逐循环预测可重算；训练日志和失败尝试保留。 |
| Construct validity | 每个候选必须有与现实痛点对应的专用指标，不能只看RMSE。 |

## Limitations by Design

- 冻结G-MTC基线未匹配论文精度，因此只能用于公平相对比较。
- NASA是实验室数据；即便跨温度/负载测试通过，也不能自动外推到真实车队。
- B0007/B0018已被接触，只能作为开发评估集。

## Ethics and Registration

- IRB：不适用，使用公开非个人数据。
- Reporting：PRISMA-inspired检索日志；不在未达到完整系统综述标准时声称PRISMA系统综述。
- Preregistration：推荐；至少在仓库内冻结protocol.json、数据角色、成功门槛和配置哈希。

## Phase Status

等待用户确认研究问题和方法蓝图。
