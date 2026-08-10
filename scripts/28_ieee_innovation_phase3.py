"""Build the auditable ARS Phase 3 synthesis and innovation decision gate.

This script is literature/decision support only. It does not import or modify
the SOH model, read candidate-battery model errors, or select an outer test set.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "ieee_innovation_v1"
PHASE3 = OUT / "phase3"


CANDIDATES = [
    {
        "rank": 1,
        "candidate_id": "CILA",
        "candidate": "Condition-Invariant Latent Alignment（条件不变潜表示对齐）",
        "mechanism": "在现有三分支送入Attention Bi-LSTM前的潜表示上增加训练域间CORAL协方差对齐损失；不增加预测网络。",
        "real_world_value_25": 24,
        "gmtc_fit_25": 22,
        "data_feasibility_20": 19,
        "research_gap_15": 14,
        "isolated_ablation_10": 9,
        "cost_5": 4,
        "total_100": 92,
        "decision": "唯一推荐，等待用户确认",
        "main_risk": "工况与电池身份混杂；过强对齐可能抹去SOH相关退化特征并造成负迁移。",
    },
    {
        "rank": 2,
        "candidate_id": "UQ",
        "candidate": "Uncertainty-aware calibration（不确定性校准）",
        "mechanism": "为现有预测增加异方差或后验校准，输出误差区间。",
        "real_world_value_25": 22,
        "gmtc_fit_25": 17,
        "data_feasibility_20": 16,
        "research_gap_15": 13,
        "isolated_ablation_10": 8,
        "cost_5": 3,
        "total_100": 79,
        "decision": "不选",
        "main_risk": "改变输出与训练目标，且当前小样本不足以证明区间校准在外部工况可靠。",
    },
    {
        "rank": 3,
        "candidate_id": "PWR",
        "candidate": "Partial-window robustness（部分充电窗口鲁棒性）",
        "mechanism": "对现有健康特征窗口做掩码/截断一致性训练。",
        "real_world_value_25": 20,
        "gmtc_fit_25": 20,
        "data_feasibility_20": 18,
        "research_gap_15": 7,
        "isolated_ablation_10": 9,
        "cost_5": 4,
        "total_100": 78,
        "decision": "不选",
        "main_risk": "原论文已以部分充电健康特征为前提，新增性有限，容易变成数据增强细节。",
    },
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PHASE3.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "innovation_decision_matrix.csv", CANDIDATES)
    write_csv(
        PHASE3 / "counter_evidence_search_log.csv",
        [
            {
                "provider": "Sider Scholar/OpenAlex",
                "scholar_result_id": "03394f48-c6bf-4dd9-a345-fc708cb50f31",
                "query": "lithium ion battery state of health domain adaptation negative transfer unseen working conditions",
                "window": "2024-08-10..2026-08-10",
                "result_count": 18,
                "assessment": "high noise; no verified paper guarantees benefit on G-MTC; used to define negative-transfer risk",
            },
            {
                "provider": "Sider Scholar/OpenAlex",
                "scholar_result_id": "a955f43e-8554-4089-aad2-432942abe81f",
                "query": "battery state of health uncertainty field conditions segmented charging label reliability",
                "window": "2024-08-10..2026-08-10",
                "result_count": 19,
                "assessment": "high noise; reinforces field-label and fragmented-profile concerns but does not establish a compatible minimal module",
            },
        ],
    )

    pain_points = """# 现实应用痛点与项目适配性

## 证据收敛

1. **跨温度、负载、材料和充放电策略的域偏移**：多源元学习、无标签域对齐、特征映射、EIS迁移和联邦异构数据五类研究从不同角度指向同一问题。对本项目而言，这是唯一同时具备近期IEEE证据、NASA多工况数据和单模块实现可能性的主痛点。
2. **现场标签不可靠与充电片段不完整**：现场物理引导研究把标签质量、分段充电和不确定性视为共同障碍，但完整方案需要新的标签构造、概率建模和现场数据，不适合作为本轮单一创新。
3. **可信度与可解释性**：不确定性、SHAP和物理先验均受关注，但它们不能代替跨工况性能证据。

## 不应选择的方向

- 不再增加GAN：与SCF-GAN重叠且破坏隔离创新。
- 不加入完整PINN/PIDGRU/DeepHPM：改变主架构，变量过多。
- 不加入EIS或磁场传感：NASA现有输入不具备这些观测，合成输入会制造证据。
- 不上联邦学习：现实价值存在，但分布式系统超出本论文一个创新点的范围。
- 不用SHAP单独充当预测创新：它是解释工具，不直接解决泛化。

## 推荐痛点

**未见电池/工况上的潜表示分布偏移。**

建议把论文问题严格写成“跨工况电池泛化”，而不是“温度因果鲁棒性”，因为NASA中温度、放电电流、截止条件和电池身份并未完全正交。
"""
    (OUT / "REAL_WORLD_PAIN_POINTS.md").write_text(pain_points, encoding="utf-8")

    synthesis = """# Synthesis Report — ARS Phase 3

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
"""
    (PHASE3 / "SYNTHESIS_REPORT.md").write_text(synthesis, encoding="utf-8")

    da = """# Devil's Advocate Report — Checkpoint 2

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
"""
    (PHASE3 / "DEVILS_ADVOCATE_CHECKPOINT2.md").write_text(da, encoding="utf-8")

    gate = """# Innovation Confirmation Gate

## 唯一推荐

**CILA：Condition-Invariant Latent Alignment（条件不变潜表示对齐）**，决策分数92/100。

## 单一改动

在现有Trend/Seasonal/Residual三分支形成并送入Attention Bi-LSTM的潜表示上，仅增加训练源域之间的CORAL二阶统计对齐损失：

`L_total = L_prediction + lambda_align * L_CORAL`

不替换TimesNet，不替换iTransformer，不修改SCF-GAN，不加入完整PINN，不约束最终SOH，不加入第二个新预测网络。

## 确认后才执行

1. 不查看候选电池模型误差，先按元数据冻结训练域、验证域和未接触外层测试域。
2. 模式仅`none/coral`，一次只改`lambda_align`；由验证集选择后锁定哈希。
3. 五固定种子；保存RMSE/MAE/MAPE/R2、worst-condition RMSE、跨工况退化差、域可分性、推理成本和逐循环预测。
4. 配对时间块bootstrap；B0007/B0018仅作已接触开发评估，不承担确认性主结论。
5. 若两个预注册目标工况的中位RMSE未同时下降，或任一显著恶化，保留负结果并判定不支持，不调测试集重试。

## 需要用户决定

回复“确认CILA”后才进入代码实现。也可明确拒绝；拒绝时只重新讨论决策矩阵，不会悄悄尝试其他模块。
"""
    (PHASE3 / "INNOVATION_CONFIRMATION_GATE.md").write_text(gate, encoding="utf-8")

    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update(
        {
            "status": "waiting_innovation_confirmation",
            "phase3_synthesis_started": True,
            "phase3_completed_at": datetime.now().astimezone().isoformat(),
            "phase3_gate": "PASS_WITH_MANDATORY_SAFEGUARDS",
            "recommended_innovation_id": "CILA",
            "recommended_innovation_score": 92,
            "innovation_selected": False,
            "model_code_modified": False,
            "candidate_model_errors_inspected": False,
            "formal_outer_test_selected": False,
        }
    )
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert CANDIDATES[0]["total_100"] == max(x["total_100"] for x in CANDIDATES)
    assert protocol["innovation_selected"] is False
    assert protocol["model_code_modified"] is False
    print("Phase 3 artifacts written; gate=PASS_WITH_MANDATORY_SAFEGUARDS; recommendation=CILA; model_code_modified=false")


if __name__ == "__main__":
    main()
