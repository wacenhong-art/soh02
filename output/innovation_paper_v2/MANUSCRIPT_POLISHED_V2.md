# 小样本电池SOH估计中物理趋势约束与融合域对齐的泛化边界

**English title:** Generalization limits of physics-guided trend constraints and fusion-domain alignment for small-sample battery SOH estimation

## 摘要

小样本锂离子电池健康状态（state of health, SOH）估计不仅受标签稀缺影响，还受跨电池分布偏移和递归外推误差累积制约。本文以SCF-GAN、季节-趋势分解（STL）、TimesNet/iTransformer分支预测和Attention Bi-LSTM融合组成的G-MTC为冻结主干，建立无测试集调参、可逐循环重算的复现实验。随后分别检验两项可移除的最小干预：仅作用于STL Trend预测的单调/平滑物理约束，以及作用于既有融合上下文的协方差不变潜表示对齐（CILA）。B0007和B0018采用前60%有标签适配、20%验证和后20%测试；该协议不是零标签跨电池迁移。配置锁定后，B0043和B0044作为外层电池各执行一次五随机种子评估。基线通过数据泄漏和指标重算门禁，但B0018中位RMSE为5.1367%，未达到原论文0.9123%的报告值；在当前实现中，CNN也优于Transformer类比较模型。Trend约束在部分电池和种子中降低RMSE，却没有同时满足两电池准确率、物理一致性和时间块bootstrap门槛。CILA在开发电池上的中位改善小于0.001个百分点，并在B0044外层评估中仅有2/5种子改善，配对中位差为+0.000689个百分点，95%置信区间跨越0。结果不支持两项干预具有稳定的跨电池优势，而将上游分量递归外推识别为优先于局部正则和下游对齐的工程瓶颈。本文提供的是可反驳、可审计的适用边界证据，而非以选择性结果宣称新的最低误差。

**关键词：** 锂离子电池；健康状态；小样本学习；TimesNet；iTransformer；物理一致性；域偏移；负迁移

## Abstract

Small-sample lithium-ion battery state-of-health (SOH) estimation is limited not only by scarce labels but also by cross-cell distribution shift and error accumulation during recursive extrapolation. We froze a G-MTC forecasting backbone comprising SCF-GAN augmentation, seasonal-trend decomposition using Loess (STL), TimesNet/iTransformer component predictors, and Attention Bi-LSTM fusion, and established an auditable protocol without test-set tuning. Two removable interventions were then evaluated independently: monotonicity and smoothness penalties applied only to the predicted STL Trend component, and covariance-invariant latent alignment (CILA) applied to the existing fusion context. B0007 and B0018 followed a labeled 60/20/20 adaptation protocol and therefore did not constitute zero-label cross-cell transfer. After configuration locking, B0043 and B0044 were evaluated once as outer batteries using five random seeds. The baseline passed leakage and metric-recomputation checks, but the median B0018 RMSE was 5.1367%, compared with 0.9123% reported in the source paper; CNN also outperformed the Transformer-family comparators in the present implementation. Trend constraints reduced RMSE for selected batteries and seeds but did not jointly satisfy the preregistered accuracy, physical-consistency, and time-block bootstrap criteria. CILA produced median gains below 0.001 percentage points on the development batteries and failed outer confirmation on B0044, where only two of five seeds improved, the paired median difference was +0.000689 percentage points, and the 95% confidence interval crossed zero. These observations do not support a stable cross-cell advantage for either intervention. Instead, they identify recursive component extrapolation as a bottleneck that local regularization and downstream alignment could not reliably repair. The study contributes an auditable boundary analysis rather than a selectively optimized state-of-the-art claim.

**Keywords:** lithium-ion battery; state of health; small-sample learning; physical consistency; domain shift; negative transfer; reproducibility

## 1. 引言

锂离子电池SOH描述可用容量相对额定或初始容量的保持程度，是寿命预测、维护决策和安全控制的关键状态量。数据驱动模型避免了显式求解复杂电化学方程，但其可信度依赖于带标签退化轨迹的数量、训练与部署分布的一致性，以及模型在生命周期后段的外推稳定性。公开数据通常只包含少量实验室电芯，实际电池系统还会经历温度、负载和充电片段变化。因此，单一数据集或单个随机种子上的低误差不足以支持部署层面的泛化结论。

Yang等提出的G-MTC使用SCF-GAN扩充健康特征，以STL将SOH分解为Trend、Seasonal和Residual，并分别使用TimesNet、TimesNet和iTransformer建模，最后通过Attention Bi-LSTM融合分量预测[1]。该框架的重要性质不是所有分量都单调，而是不同分量承担不同时间尺度：Trend表征长期衰减，Seasonal表征周期波动，Residual容纳随机扰动和容量再生。由此，物理先验应作用于Trend，而不应把最终SOH强制为严格单调。

原论文在多个数据集上报告低于1%的RMSE，但没有公开源码，且若干网络层级、损失权重和数据切分细节未完全披露。因此，“实现了论文主方法路径”与“复现了论文数值”必须分别验证。近期SOH研究还显示，跨工况域偏移、无标签目标电池和现场部分充电是现实难点[2-7]。这些研究为物理先验和域对齐提供了动机，却不能预先保证它们在冻结G-MTC上有效。

本文围绕一个可证伪问题展开：在不替换TimesNet、iTransformer、Attention Bi-LSTM或SCF-GAN的条件下，Trend物理约束或融合域对齐能否在固定协议中稳定改善跨电池SOH估计？本文的贡献包括：（1）区分方法符合性、数值复现和比较模型排序，建立逐循环可重算的可信基线；（2）实现只作用于连续Trend轨迹的单调/平滑损失；（3）在不改变预测主干的条件下检验CILA；（4）以五随机种子、配对差和未接触外层电池保留负结果，明确两项干预的适用边界。

<!-- FIGURE:fig1_framework_energy_v2|图1 | 冻结G-MTC主干与两项隔离干预。a，SCF-GAN仅增强训练数据；STL Trend和Seasonal由TimesNet建模，Residual由iTransformer建模，三分量进入Attention Bi-LSTM融合。b，物理损失只约束Trend预测，CILA只约束融合上下文；两者均不约束最终SOH，也不使用测试集调参。 -->

## 2. 方法

### 2.1 数据、健康特征与划分

NASA Ames PCoE数据包含逐循环电压、电流、温度和放电容量记录[15]。SOH按当前放电容量相对额定容量计算。依据原论文提取三个充电时间特征：HF1为恒流充电开始至电压首次达到4.2 V的时长；HF2为达到4.2 V后至电流下降至20 mA截止点的时长；HF3为电压由3.8 V上升至4.0 V的时长。HF2的截止电流搜索严格从4.2 V到达点之后开始，避免把早期低电流噪声识别为恒压终点。

样本按BatteryID、DatasetID和Cycle排序。B0005和B0006仅用于SCF-GAN源训练；B0007不进入源预训练。B0007和B0018分别按生命周期前60%、随后20%和最后20%划分训练适配、验证和测试。标准化器、STL拟合、特征处理和GAN增强均限制在训练前缀，滑动窗口不得跨越边界。由于目标电池前缀包含SOH标签，该协议是小样本有标签适配，而非零标签跨电池迁移。

### 2.2 冻结G-MTC基线

STL将观测SOH写为y=T+S+R。Trend和Seasonal分别输入原始TimesNet路径，Residual输入iTransformer；分量预测与历史上下文进入Attention Bi-LSTM并生成最终SOH。P0为论文接近实现；P1仅加入固定随机种子、梯度裁剪、验证早停、训练前缀标准化和数值稳定初始化。P1不改变主架构，所有稳定化措施均与论文原始方法分开记录。

原论文未公开完整源码，因此SCF-GAN层宽、iTransformer细节、融合上下文和部分辅助损失属于工程近似。本文不把“主路径已实现”表述为源代码级完全复刻。

### 2.3 Trend-only物理约束

对同一BatteryID和DatasetID内Cycle连续的Trend预测T_hat计算单调项

L_mono = mean[ReLU(T_hat(t+1)-T_hat(t))]

和平滑项

L_smooth = mean[(T_hat(t+1)-2T_hat(t)+T_hat(t-1))^2]。

总损失为L_total=L_prediction+lambda_phy L_physics。physics_mode支持none、monotonic、smooth和both；lambda_phy支持0、0.001、0.01和0.1。物理损失从不在shuffle后的随机样本之间计算，也不施加于最终SOH。Trend violation rate定义为连续预测点中T_hat(t+1)>T_hat(t)+epsilon的比例，并与正向跳变幅度共同保存。

### 2.4 CILA融合上下文对齐

CILA只使用Attention Bi-LSTM已有融合上下文Z_b。对两个训练电池域，计算均值和协方差差异：

L_align=||mu_1-mu_2||_2^2+||C_1-C_2||_F^2/(4d^2)，

其中d为上下文维数。融合目标为L=L_prediction+lambda_weight L_weight+lambda_align L_align。Original令lambda_align=0；CILA的lambda_align=0.1和lambda_weight=0.1仅由验证集锁定。两种模式共享分量输入、初始化、种子、150轮上限和早停耐心30。

### 2.5 评估与成功门槛

固定随机种子为42、52、62、72和82。报告RMSE、MAE、MAPE和R²，并从保存的逐循环预测重新计算。Trend实验还报告Trend violation rate。配对时间序列采用5000次圆形时间块bootstrap估计RMSE差的95%置信区间。

Trend创新需同时满足：B0007和B0018中位RMSE均低于Original；每个目标电池至少3/5种子改善；Trend违规率或正向跳变下降；MAE、MAPE和R²无明显恶化；bootstrap方向支持改善。CILA外层确认同样要求逐电池配对方向和至少3/5种子改善。B0007和B0018是已接触开发电池；B0043和B0044仅在配置锁定后执行一次正式外层评估。

## 3. 结果

### 3.1 可信基线未实现数值复现

五个种子均完成训练，无NaN或数值发散。硬性泄漏检查通过，原始TimesNet、iTransformer和Attention Bi-LSTM路径均参与计算。逐循环重算与保存指标的最大差异为1.32e-05，因此BaselineComplete为true。

可信过程没有转化为论文数值匹配。B0018中位RMSE为5.1367%，中位R²为-9.5473，而原论文分别报告0.9123%和0.8765，故PaperNumericMatched为false（表1）。在相同工程协议下，CNN在B0007和B0018的中位RMSE分别为0.6755%和2.6151%，均低于LSTM、Transformer和当前G-MTC。因此TransformerBestObserved为false。Attention三分支均参与融合，但10个电池-种子组合中9个近似静态，动态权重行为仍属于工程近似。

**表1  冻结G-MTC五随机种子结果**

| 电池 | RMSE均值±标准差/% | RMSE中位数/% | MAE均值/% | MAPE均值/% | R²中位数 |
|---|---:|---:|---:|---:|---:|
| B0007 | 5.9110±1.9770 | 5.2050 | 5.1995 | 7.2238 | -22.1318 |
| B0018 | 5.1770±0.5758 | 5.1367 | 4.6910 | 6.7648 | -9.5473 |

<!-- FIGURE:fig2_baseline_energy_v2|图2 | 可信基线的数值差距。a-b，各模型五随机种子的RMSE；柱为中位数，误差线为一个种子标准差，空心点为单个种子。B0018红色虚线为原论文G-MTC参考RMSE 0.9123%。c-d，冻结G-MTC逐循环预测；橙线和阴影分别为五种子中位数和四分位距，黑线为实测SOH。 -->

### 3.2 Trend约束呈现电池与种子依赖性

验证集选择lambda_phy=0.1后，正式测试只改变physics_mode。both使B0007中位RMSE由5.2050%降至5.0157%，且4/5种子改善；但B0018仅2/5种子改善。monotonic在B0018有4/5种子改善，却使B0007中位RMSE升至5.9450%，并出现单种子严重失稳。没有一种模式同时通过两电池联合门槛（表2）。

**表2  Trend-only物理约束五随机种子消融**

| 电池 | 模式 | RMSE中位数/% | RMSE均值±标准差/% | Trend违规率均值 | 改善种子 |
|---|---|---:|---:|---:|---:|
| B0007 | Original | 5.2050 | 5.9110±1.9770 | 0.0545 | - |
| B0007 | Monotonic | 5.9450 | 8.4873±7.1560 | 0.1212 | 2/5 |
| B0007 | Smooth | 5.2186 | 4.9721±1.1966 | 0.0788 | 3/5 |
| B0007 | Both | 5.0157 | 4.6723±0.9630 | 0.0788 | 4/5 |
| B0018 | Original | 5.1367 | 5.1770±0.5758 | 0.0880 | - |
| B0018 | Monotonic | 4.9920 | 5.1754±0.6830 | 0.1040 | 4/5 |
| B0018 | Smooth | 5.1393 | 5.1767±0.9149 | 0.1200 | 2/5 |
| B0018 | Both | 4.8323 | 5.0926±0.9422 | 0.0720 | 2/5 |

<!-- FIGURE:fig3_physics_energy_v2|图3 | Trend-only物理约束的隔离消融。a-b，五个固定种子相对Original的配对RMSE差，负值表示改善，黑横线为中位数。c，各模式的测试期Trend violation rate；实心点为中位数，淡点为单个种子。d，B0018五种子Trend预测中位数；多条预测线在约72附近重合，显示递归分支的近常数输出而非物理约束优势。 -->

### 3.3 CILA未通过外层联合确认

在已接触的B0007和B0018上，CILA的配对中位RMSE差分别为-0.000100和-0.000786个百分点。方向虽为改善，但实际幅度均小于0.001个百分点，不能据此宣称具有工程意义。

外层评估的结论更严格。B0043有3/5种子改善，配对中位差为-0.001025个百分点，95%时间块bootstrap区间为[-0.283230,-0.000313]。B0044仅2/5种子改善，配对中位差为+0.000689个百分点，区间[-0.002632,+0.001013]跨越0（表3）。因此CILAInnovationSupported为false。

**表3  CILA正式外层评估**

| 电池 | Original中位RMSE/% | CILA中位RMSE/% | 配对中位差/% | 改善种子 | 95%时间块bootstrap区间 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| B0043 | 134.255705 | 134.254680 | -0.001025 | 3/5 | [-0.283230,-0.000313] | 方向通过，效应可忽略 |
| B0044 | 60.856165 | 51.303511 | +0.000689 | 2/5 | [-0.002632,+0.001013] | 失败 |

不同方法的非配对中位数可能来自不同随机种子，不能替代同一种子内的配对差。逐循环误差还显示，部分分量递归预测偏离真实尺度达数百至数千个百分点。CILA只调整融合上下文，无法从已经失真的分量输入恢复真实SOH。

<!-- FIGURE:fig4_outer_validation_energy_v2|图4 | CILA外层电池确认。a，五个固定种子的配对RMSE差，负值有利于CILA。b，5000次圆形时间块bootstrap得到的配对中位差及95%置信区间。c-d，五种子逐循环预测误差的中位数与四分位距，纵轴采用对称对数尺度以同时呈现误差方向和递归失稳幅度。 -->

## 4. 讨论

### 4.1 方法符合性、数值匹配与模型排序是三个问题

本研究的基线在数据边界、主路径执行、多种子训练和指标重算方面可审计，但没有复现原论文数值。两者并不矛盾：在作者源码、精确划分和完整超参数缺失时，主架构复现可以成立，数值复现仍可能失败。当前CNN优于Transformer类模型也不是对原论文结论的普遍否定，只说明“Transformer必然最好”没有在本实现和协议中被观察到。

### 4.2 更平滑不等于更准确

Trend单调和平滑先验具有合理物理动机，但测试期表现还受STL边界、窗口长度和递归状态传播影响。某些种子中违规率下降而RMSE未下降，表明预测可能更平滑却仍存在水平偏差；另一些种子中RMSE下降但违规率上升，表明准确率与物理指标存在有限样本权衡。最终SOH包含Seasonal和Residual，局部容量再生可能是真实信号，因此把单调性扩展到最终输出不是对负结果的合法补救。

### 4.3 下游对齐不能修复上游外推失稳

CILA假设训练电池间的二阶统计差异包含可去除的域偏移。当域差异与衰减阶段、尺度或分量误差耦合时，对齐可能同时削弱SOH信息。更直接的限制是，CILA位于融合端；若Trend、Seasonal或Residual已经发生数量级偏移，融合只能重加权错误输入，无法恢复缺失的真实轨迹。由此得到的“上游分量外推是瓶颈”是由错误传播位置支持的机制性推断，而不是已由新干预证明的因果结论。

### 4.4 对SOH论文设计的启示

第一，必须明确目标域标签使用方式，60/20/20有标签适配不能表述为零标签迁移。第二，比较模型应在同一数据边界、同一特征和多随机种子下评估，网络名称不构成性能先验。第三，跨电池结论应报告逐电池配对差和未接触外层结果，合并平均值可能掩盖负迁移。第四，外层失败后不应继续针对同一电池试到成功；新的分支稳定化假设需要新的预注册外层数据。

## 5. 局限性

本研究仅使用NASA实验室数据，外层电池仍来自同一归档，不能代表跨机构、跨化学体系或真实车队。B0007和B0018在项目开发中已被反复接触，只能支持开发性结论。B0043和B0044从通过健康特征和窗口可行性检查的电池中选取，存在可用性选择偏差，但筛选发生在模型推理前。外层只有两个电池，bootstrap区间描述的是已保存轨迹和种子配对的不确定性，不是现场电池总体置信区间。

原论文未公开源码，部分SCF-GAN、iTransformer和融合细节为工程近似。外层递归误差较大，使均值对单种子失稳敏感，因此本文同时报告中位数、单种子点和配对差。最后，正式外层失败后没有继续试验新的分支稳定化方案，避免把B0043和B0044转化为新的调参集。

## 6. 结论

本文在冻结G-MTC主架构、禁止测试集调参和保留失败记录的条件下，完成了可信基线、Trend-only物理约束和CILA融合域对齐的多种子评估。基线在审计层面完成，但没有匹配原论文RMSE和R²，Transformer也不是当前比较中的最佳模型。Trend约束和CILA均在局部条件下出现改善，却未通过预注册的跨电池联合门槛。

因此，本研究不支持“新增模块稳定降低SOH RMSE”的强结论。现有证据更一致地表明：当分量递归外推失稳时，局部Trend正则和下游融合对齐难以提供可重复收益。该负结果界定了方法适用范围，并为下一轮只检验一个上游外推稳定化因素、使用新的未接触外层数据提供可审计起点。

## 数据、代码与声明

实验使用NASA公开电池老化数据。代码、配置、训练日志、逐循环预测、失败尝试和结果汇总保存在项目输出目录；数据归档不随仓库重新分发。作者、单位、资助、利益冲突和CRediT贡献须由研究团队在投稿前据实补充。本稿在语言组织、代码审计和图表生成中使用OpenAI Codex及本地学术工作流；全部数值均来自保存的实验文件并经过程序重算，最终解释和投稿责任由人类作者承担。

## 参考文献

[1] H. Yang et al., “Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach,” Energy, vol. 342, 139463, 2026, doi: 10.1016/j.energy.2025.139463.

[2] M. Miao, C. Xiao, and J. Yu, “Multisource domain metalearning network for battery state-of-health estimation under multitarget working conditions,” IEEE Trans. Power Electron., 2025, doi: 10.1109/TPEL.2025.3538469.

[3] C. Song, H. Yuan, G. Wang, and N. Cui, “State of health estimation for lithium-ion battery with label-free data based on deep domain feature independent alignment network,” IEEE Trans. Power Electron., 2025, doi: 10.1109/TPEL.2025.3580291.

[4] W. Yao et al., “State of health estimation of lithium-ion batteries using data augmentation and feature mapping,” IEEE Trans. Transp. Electrific., vol. 11, pp. 4895-4905, 2025, doi: 10.1109/TTE.2024.3471867.

[5] L. Luo et al., “Physics-informed deep gated recurrent unit network for lithium-ion battery state of health estimation under field conditions,” IEEE Trans. Transp. Electrific., 2026, doi: 10.1109/TTE.2026.3686077.

[6] M. Lin et al., “Transfer learning-based lithium-ion battery state of health estimation with electrochemical impedance spectroscopy,” IEEE Trans. Transp. Electrific., 2025, doi: 10.1109/TTE.2025.3533540.

[7] Z. Hui et al., “An innovative magnetic field sensor based method predicting lithium-ion battery state-of-health,” Proc. IEEE Sensors Applications Symposium, 2025, doi: 10.1109/SAS65169.2025.11105130.

[8] A. Tian et al., “A generic physics-informed neural network framework for lithium-ion batteries state of health estimation,” Energy, vol. 332, 137215, 2025, doi: 10.1016/j.energy.2025.137215.

[9] S. Wang et al., “A generalizable physics-informed neural network for lithium-ion battery SOH estimation utilizing partial charging segments,” J. Energy Chem., vol. 112, pp. 977-986, 2025.

[10] B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” ECCV Workshops, pp. 443-450, 2016.

[11] H. Wu et al., “TimesNet: Temporal 2D-variation modeling for general time series analysis,” Proc. ICLR, 2023.

[12] Y. Liu et al., “iTransformer: Inverted Transformers are effective for time series forecasting,” Proc. ICLR, 2024.

[13] O. Das et al., “Battery time-series data augmentation using a GAF-based generative framework,” Proc. IEEE EUROCON, 2025, doi: 10.1109/EUROCON64445.2025.11073369.

[14] V. Dimitrievska et al., “Battery state of health estimation via federated learning on diverse data from multiple sources,” Proc. FLICS, 2026, doi: 10.1109/FLICS70075.2026.11621953.

[15] B. Saha et al., “Prognostics methods for battery health monitoring using a Bayesian framework,” IEEE Trans. Instrum. Meas., vol. 58, no. 2, pp. 291-296, 2009.

[16] V. Olivero-Ortiz et al., “Aging and performance analysis of 18650 lithium-ion cell capacity degradation: Data-driven insights and modeling,” IEEE Access, 2025, doi: 10.1109/ACCESS.2025.3636613.

