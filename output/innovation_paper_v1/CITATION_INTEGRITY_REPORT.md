# Citation and data integrity report

## Verdict

**PASS WITH MANDATORY AUTHOR METADATA COMPLETION.** 未发现虚构参考文献、悬空正文引文或无法从保存文件重算的核心数值。作者、单位、资助、利益冲突和最终代码公开地址必须由真实作者在投稿前补充，当前占位不得直接投稿。

## Reference verification

| Ref. | Verification route | Verdict | Notes |
|---:|---|---|---|
| 1 | 用户提供的原论文PDF首页、方法页和表6 | VERIFIED | 题名、作者、Energy 342、139463、DOI和B0018数值核对 |
| 2–7, 13–14, 16 | Sider Scholar发现；Crossref IEEE member 263；Semantic Scholar DOI/题名/摘要矩阵 | VERIFIED | `source_verification_matrix.csv`十条均通过；部分为摘要级证据 |
| 8–9 | 原论文PDF参考文献[29]–[30]和正文；DOI/卷页信息 | VERIFIED | 本稿只借鉴思想，不声称复现完整PINN |
| 10 | DOI/出版元数据核验 | VERIFIED | Deep CORAL, DOI `10.1007/978-3-319-49409-8_35` |
| 11 | ICLR/OpenReview元数据 | VERIFIED | TimesNet作者、题名和2023会议一致 |
| 12 | ICLR/OpenReview元数据 | VERIFIED | iTransformer作者、题名和2024会议一致 |
| 15 | IEEE元数据与DOI `10.1109/TIM.2008.2005965` | VERIFIED | 作者、卷58(2)、页291–296一致 |

正文使用16条编号文献，参考文献表也有16条；扫描未发现孤儿条目或超出[1]–[16]的引文。IEEE条目2、3、5、6的最终卷期页在本地检索证据中未完整提供，因此没有臆造页码。

## Claim-to-evidence audit

| Claim | Evidence artifact | Result |
|---|---|---|
| 原论文B0018 RMSE=0.9123%、R²=0.8765 | 原论文PDF表6 | PASS |
| 基线B0018中位RMSE=5.1367% | `baseline_comparison.csv`, `multiseed_metrics.csv` | PASS |
| CNN在当前比较中最好 | `baseline_comparison.csv` | PASS；限定为当前实现与协议 |
| PhysicsInnovationSupported=false | `PHYSICS_TREND_REPORT.md`, `ablation_metrics.csv` | PASS |
| CILA开发效应小于0.001个百分点 | `five_seed_validation_summary.csv` | PASS |
| CILA外层确认失败 | `formal_outer_summary.csv`, `formal_outer_decision.json` | PASS |
| B0044配对中位差+0.000689且2/5改善 | `formal_outer_paired_deltas.csv` | PASS |
| 80项指标重算通过 | `metric_recomputation_audit.csv` | PASS |

## Seven AI-research failure modes

| Failure mode | Status | Control |
|---|---|---|
| Fabricated citations | PASS | 每条文献有PDF、DOI或已保存三方核验路径 |
| Fabricated data/results | PASS | 所有核心数值来自CSV，逐循环重算 |
| Test-set tuning | PASS | 配置哈希锁定；外层失败后未调参 |
| Cherry-picking seeds | PASS | 五个固定种子全部保留 |
| Leakage | PASS WITH LIMIT | 训练边界门禁通过；60/20/20明确为有标签适配 |
| Overclaiming causality/generalization | PASS | 未把BatteryID混杂解释为温度因果；外层失败明确 |
| Hidden AI authorship | PASS | 稿件含AI使用声明 |

## Data and figure checks

五张外层图均能追溯到CSV和`38_plot_cila_outer.py`。F2使用对数轴，F4使用symlog，均在清单中披露。未将图形视觉差异替代配对统计量。正式投稿前建议把图中英文说明与目标期刊字体统一，但这不是数据完整性问题。

