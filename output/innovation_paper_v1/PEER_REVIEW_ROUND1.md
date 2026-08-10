# ARS five-perspective peer review — Round 1

## Overall decision

**Major revision, score 7.0/10.** 方法审计和负结果透明度较强，但以现有数据不足以构成“新算法性能提升”论文。最合适定位是可信复现与跨电池鲁棒性边界研究。若目标期刊只接受正向性能创新，命中率较低；若强调reproducibility、robustness或negative transfer，具有可写性。

## 1 Journal-fit reviewer

- **Strength:** 研究问题清楚，结果与预注册门槛一致，未用测试集继续迭代。
- **Major:** 标题中的“创新”应避免暗示正向效果；当前题目已经改为泛化边界，方向合适。
- **Major:** 稿件缺作者/单位/资助等投稿元数据，必须在提交前补齐。
- **Score:** 6.8/10。

## 2 Methodology reviewer

- **Strength:** 数据按BatteryID/DatasetID/Cycle控制；GAN仅增强训练；标准化和STL边界清楚；逐循环重算和配对bootstrap恰当。
- **Major:** 外层只有两个可行电池，不能估计广泛电池总体效应；应把bootstrap解释限制在轨迹与种子，而非电池总体。
- **Major:** B0043/B0044绝对预测已严重失效，CILA的微小配对差不应被解释为有实际意义。
- **Minor:** 给出工程稳定化与论文原方法的明确列表。
- **Score:** 7.4/10。

## 3 Domain reviewer

- **Strength:** 只约束Trend而不约束最终SOH符合容量再生语义；HF2截止点处理正确。
- **Major:** NASA实验室数据无法覆盖真实EV中的标签漂移、部分充电和多温度因果效应。
- **Minor:** 不应把B0043/B0044称为严格“外部数据集”，宜称同归档未接触外层电池。
- **Score:** 7.1/10。

## 4 Perspective reviewer

- **Strength:** 从“追求更低RMSE”转向“何时局部约束失效”，对工程部署更有价值。
- **Major:** 读者可能误解60/20/20为跨电池无标签迁移；摘要、方法和讨论已三次声明其为有标签适配，保留这些限定。
- **Minor:** 对幸存者筛选和同归档外层限制应置于表3附近，而不仅在局限性中。
- **Score:** 7.2/10。

## 5 Devil's advocate

最强反对意见是：由于基线绝对误差在外层电池上已经灾难性失效，对下游CILA进行细小差异比较几乎没有应用意义；论文可能只是记录一个实现不稳定的复现，而非揭示一般化的科学规律。此外，原论文未公开源码和数据细节，当前实现的失败也可能源自工程近似，而不是G-MTC、Trend约束或域对齐本身。因此，任何关于“主导瓶颈”的表述必须保持为定位性推断，并清楚说明结果只适用于本实现和协议。

该反对意见成立，但不要求删除研究。应通过以下方式处理：（1）拒绝算法优越性主张；（2）把分支外推称为“本实现中的首要故障位置”；（3）保留原论文数值未复现和Attention近静态诊断；（4）把稿件定位为可审计复现/负迁移边界，而非新SOTA。

## Required revisions

| Priority | Location | Revision | Status |
|---|---|---|---|
| Major | Abstract/Conclusion | 明确两个创新均未获确认支持 | 已满足 |
| Major | Methods/Limitations | 限定bootstrap与外层两电池的推断范围 | 已满足 |
| Major | Discussion | 将“主导瓶颈”降格为本实现中的机制性推断 | 已满足 |
| Major | Declarations | 保留作者元数据占位并阻止直接投稿 | 已满足 |
| Minor | Results | 解释非配对方法中位数与配对中位差不同 | 已满足 |
| Minor | Figures | 增加图形数据来源、坐标尺度和局限清单 | 已满足 |

## Round-1 disposition

稿件在内容层面可进入一次修订后复核；不要求追加测试集调参或新模块。确认性正向创新仍不成立。

