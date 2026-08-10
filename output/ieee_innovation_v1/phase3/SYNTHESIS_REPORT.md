# Synthesis Report — ARS Phase 3

## Claim Intent Manifest

```json
{
  "manifest_version": "1.0",
  "manifest_id": "M-2026-08-10T08:45:00+08:00-cila",
  "emitted_by": "synthesis_agent",
  "emitted_at": "2026-08-10T08:45:00+08:00",
  "claims": [
    {"claim_id": "C-001", "claim_text": "Recent IEEE SOH evidence converges on cross-condition domain shift as a practical problem.", "intended_evidence_kind": "empirical", "planned_refs": ["miao2025", "song2025", "yao2025", "lin2025"]},
    {"claim_id": "C-002", "claim_text": "No retrieved source establishes that latent covariance alignment improves the frozen G-MTC baseline.", "intended_evidence_kind": "empirical", "planned_refs": ["miao2025", "song2025"]},
    {"claim_id": "C-003", "claim_text": "CILA is the most feasible isolated innovation for the available project data and architecture.", "intended_evidence_kind": "normative", "planned_refs": ["miao2025", "song2025", "yao2025", "luo2026"]}
  ],
  "manifest_negative_constraints": [
    {"constraint_id": "MNC-1", "rule": "Do not claim causal temperature robustness from battery-confounded NASA groups."},
    {"constraint_id": "MNC-2", "rule": "Do not claim RMSE improvement before a locked held-out experiment."}
  ]
}
```

## Literature Matrix

| Evidence cluster | Sources | Convergent finding | Boundary for this project |
|---|---|---|---|
| Domain shift / transfer | Miao et al. (2025) <!--ref:miao2025--><!--anchor:quote:The%20discrepancy%20in%20working%20conditions%20gives%20rise%20to%20the%20phenomenon%20of%20domain%20shift-->; Song et al. (2025) <!--ref:song2025--><!--anchor:quote:insufficient%20labeled%20training%20data%20and%20different%20usage%20scenarios-->; Yao et al. (2025) <!--ref:yao2025--><!--anchor:quote:time-consuming%20data%20collection%20and%20poor%20adaptability%20to%20different%20batteries--> | 工况/电池变化使训练分布失配。 | 元学习、动态卷积或新特征映射网络不能整套搬入G-MTC。 |
| Field reliability | Luo et al. (2026) <!--ref:luo2026--><!--anchor:quote:segmented%20charging%2C%20domain%20shift%2C%20and%20weak%20interpretability--> | 现场标签、片段和域偏移同时存在。 | 摘要级证据不足以支持完整现场方案，且当前没有车队数据。 |
| Specialized sensing | Lin et al. (2025) <!--ref:lin2025--><!--anchor:quote:generalize%20across%20temperature%20and%20cell%20type-->; Hui et al. (2025) <!--ref:hui2025--><!--anchor:quote:external%20magnetic-field%20sensing%20with%20PLSR%20and%20CNN%20models--> | 新传感能提高可辨识性。 | 当前NASA字段不含EIS/磁场，不能虚构。 |
| Complexity counterweight | Olivero-Ortiz et al. (2025) <!--ref:olivero2025--><!--anchor:quote:cycle%20time%20outperformed%20impedance-only%20features--> | 特征质量可能比网络复杂度更重要。 | 必须保留CNN/LSTM/Transformer等简单对照，不能宣称Transformer必然最好。 |

## Key Themes

### Theme 1：域偏移是最稳定的现实问题

**Evidence strength：Moderate（5项Level VI单项研究，方法异质但方向收敛）。** 多目标工况、无标签目标电池、跨数据集特征映射、跨温度EIS迁移与多源异构学习共同表明，实验室内拟合精度不能自动转化为新工况泛化。锚点理由：这些研究分别把工作条件、标签缺失、跨电池差异或温度/电芯类型列为核心障碍。

### Theme 2：可部署性不等于再堆一个网络

**Evidence strength：Moderate。** 现场研究把不可靠标签、分段充电和不确定性捆绑处理，而传感研究改变了观测条件；把完整方法搬入当前项目会同时改变数据、网络和损失，无法隔离贡献。锚点理由：检索到的方法往往依赖额外目标数据、额外传感器或多模块联合训练。

### Theme 3：最小可检验机制是训练域潜表示对齐

**Evidence strength：Emerging。** 文献支持“分布失配需要处理”，但没有一篇检索文献证明CORAL式协方差约束会改善当前G-MTC。推荐CILA是基于证据与工程约束的推断，不是文献已证实结论。锚点理由：它能保持SCF-GAN、STL、TimesNet、iTransformer和Attention Bi-LSTM不变，并允许单因素消融。

## Contradictions & Resolutions

| Claim A | Claim B | Resolution |
|---|---|---|
| 域对齐可缩小跨工况差异。 | 对齐过强可能移除与SOH有关、同时又带工况特征的信息。 | 条件差异而非已解决矛盾；以验证集选择权重并预注册负迁移失败门槛。 |
| 目标域无标签数据可帮助适配。 | 真正未见工况测试不应在训练时接触目标域。 | 分成两种协议：无监督域适配与源域域泛化，主结论不得混合。 |
| 复杂网络报告较低误差。 | 两电芯GPR研究提示简单运营特征也可主导效果。 | 保留简单模型比较；本项目不以Transformer最好作为前提。 |

### Cross-Paper Tension Inventory

```yaml
cross_paper_tensions:
  - pair_id: CP-001
    paper_a: miao2025
    paper_b: song2025
    candidate_basis: shared RQ subtopic
    overlap_topic: cross-condition SOH domain shift
    a_finding: multisource meta-learning targets multiple unseen working conditions
    a_evidence_pointer: verified abstract in Phase 2 matrix
    b_finding: feature alignment uses 30% early unlabeled target charging data
    b_evidence_pointer: verified abstract in Phase 2 matrix
    pair_assessment: conditional_difference
    resolution_status: resolved_in_synthesis
    resolution_pointer: Synthesis Report > Contradictions & Resolutions, row 2
    scholar_confirmation: pending
  - pair_id: CP-002
    paper_a: yao2025
    paper_b: das2025
    candidate_basis: shared construct/outcome/measure
    overlap_topic: GAN augmentation for scarce battery data
    a_finding: GAN plus feature mapping addresses cross-battery adaptation
    a_evidence_pointer: verified abstract in Phase 2 matrix
    b_finding: GAF-DCGAN augmentation is evaluated for a downstream SOC task
    b_evidence_pointer: verified abstract in Phase 2 matrix
    pair_assessment: no_material_conflict
    resolution_status: not_applicable
    scholar_confirmation: pending
  - pair_id: CP-003
    paper_a: lin2025
    paper_b: hui2025
    candidate_basis: agent-noted cross-cluster
    overlap_topic: specialized sensing for SOH observability
    a_finding: EIS features support transfer across temperature and cell type
    a_evidence_pointer: verified abstract in Phase 2 matrix
    b_finding: external magnetic sensing supports SOH regression
    b_evidence_pointer: verified abstract in Phase 2 matrix
    pair_assessment: no_material_conflict
    resolution_status: not_applicable
    scholar_confirmation: pending
```

**Coverage Note**：10篇语料中评估3组候选配对；这是有范围的咨询扫描，不是完整成对矛盾检测。跨主题矛盾可能遗漏，书目耦合仅作为纳入信号；每项resolution_pointer等待研究者确认。

## Knowledge Gaps

1. **方法空缺**：未找到把轻量潜协方差对齐作为冻结G-MTC单一附加损失的近期IEEE验证。
2. **实证空缺**：摘要通常未披露完整数据划分、代码和跨工况混杂控制。
3. **外部效度空缺**：NASA实验室工况不能代表真实车队标签噪声、路况和维护事件。
4. **统计空缺**：检索摘要普遍没有五随机种子配对分布或跨工况bootstrap证据。

## Evidence Convergence Map

- Strong：无（没有系统综述或多中心一致证据）。
- Moderate：跨工况/跨电池域偏移（5项Level VI）。
- Emerging：现场不确定性与解释性；轻量潜表示对齐对G-MTC的适配。
- Gap：当前冻结G-MTC上的CILA真实效果。

## Theoretical Integration

CILA把问题定义为“训练域潜表示的二阶统计不一致”：保持预测器不变，仅用源训练组之间的均值/协方差距离作为辅助正则。若该机制有效，应同时降低未见工况RMSE和最差工况误差，而不仅是平均RMSE；若域可分性下降但SOH误差上升，则说明发生了负迁移。

## Synthesis Limitations

- 语料限定IEEE与近两年，且深入证据为摘要级，不是完整系统综述。
- Sider Scholar反证检索噪声高，未验证结果不能进入结论。
- 推荐分数包含工程判断；92分不是效果概率。
- 尚未运行CILA实验，因此没有新的拟合结果或RMSE改善结论。
