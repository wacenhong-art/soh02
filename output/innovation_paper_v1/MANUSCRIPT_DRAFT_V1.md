# 从物理Trend约束到融合域对齐：小样本锂离子电池SOH估计的可信复现、负迁移与泛化边界

> 稿件类型：原创研究论文初稿（期刊中性格式）  
> 版本：innovation_paper_v1  
> 作者、单位和通讯信息：待作者补充  
> 实验配置哈希：`797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`

## 摘要

小样本条件下的锂离子电池健康状态（state of health, SOH）估计容易同时受到训练样本不足、跨电池分布偏移和长时域外推不稳定的影响。本文以SCF-GAN数据增强、季节—趋势分解（STL）、TimesNet/iTransformer分支建模及Attention Bi-LSTM融合构成的G-MTC框架为基础，建立无测试集调参、可逐循环重算的可信复现协议，并检验两个互不叠加的最小干预。第一项干预仅在STL Trend分支加入单调和平滑物理损失，不约束包含容量再生现象的最终SOH；第二项干预在保持全部预测主干不变的前提下，对既有融合上下文加入跨电池潜表示协方差对齐（CILA）。B0007和B0018采用前60%有标签训练适配、20%验证、后20%测试，因而不属于零标签跨电池协议；B0043和B0044作为预注册外层电池，仅在配置锁定后执行一次五随机种子测试。可信基线完成了数据泄漏门禁和指标重算，但B0018中位RMSE为5.1367%，未匹配原论文0.9123%的结果，且CNN在当前比较中优于Transformer类模型。Trend约束没有任何一种模式同时通过两电池中位RMSE、种子稳定性、Trend违规率和bootstrap门槛。CILA在开发集获得不足0.001个百分点的中位RMSE改善，但外层测试中B0044仅2/5种子改善，配对中位差为+0.000689个百分点，95%时间块bootstrap区间跨越0；同时观察到分支递归外推导致数百至数千个百分点的误差。结果不支持CILA或Trend约束具有稳定泛化优势，而提示上游分支外推稳定性是优先于局部物理正则和下游域对齐的瓶颈。本文的贡献是提供一个可审计的反证框架和跨电池负迁移边界，为后续BMS研究避免以单次低误差替代真实泛化证据。

**关键词：** 锂离子电池；健康状态估计；小样本学习；季节—趋势分解；物理约束；域偏移；负迁移

## Abstract

Small-sample lithium-ion battery state-of-health (SOH) estimation is jointly challenged by scarce degradation trajectories, cross-cell distribution shift, and unstable long-horizon extrapolation. This study builds an auditable reproduction protocol around the G-MTC framework, which combines SCF-GAN augmentation, seasonal-trend decomposition using Loess, TimesNet/iTransformer component predictors, and Attention Bi-LSTM fusion. Two isolated and removable interventions are evaluated without altering the forecasting backbone. The first applies monotonicity and smoothness penalties exclusively to the STL Trend prediction, leaving the final SOH unconstrained because its residual component may contain capacity regeneration. The second, termed covariance-invariant latent alignment (CILA), adds a cross-battery covariance-alignment loss to the existing fusion context. B0007 and B0018 follow a labeled 60/20/20 adaptation protocol and are therefore not zero-label transfer tests. After configuration locking, B0043 and B0044 are evaluated once as preregistered outer batteries across five random seeds. The auditable baseline passes leakage and metric-recomputation checks, but its B0018 median RMSE is 5.1367%, far from the reported 0.9123%, while CNN outperforms the Transformer-family baselines in this implementation. No Trend-constraint mode jointly satisfies the preregistered accuracy, seed-stability, physical-consistency, and bootstrap criteria. CILA yields sub-0.001-percentage-point median improvements on the contacted development batteries, yet fails outer confirmation: B0044 improves in only two of five seeds, its paired median RMSE difference is +0.000689 percentage points, and the 95% time-block bootstrap interval crosses zero. Several outer runs also exhibit component-level recursive extrapolation errors of hundreds to thousands of percentage points. These results do not support a stable generalization advantage for either intervention. Instead, they identify upstream component extrapolation as a bottleneck that downstream alignment and local physical regularization cannot reliably repair. The study contributes a falsifiable evaluation workflow and negative-transfer boundary for trustworthy battery SOH research.

**Keywords:** battery health management; limited labeled data; temporal decomposition; physical consistency; domain alignment; external validation; reproducibility

## 1 引言

锂离子电池SOH通常以当前可用容量相对额定或初始容量的比例表示，是寿命管理、安全控制和维护决策的重要状态量。数据驱动方法不要求显式求解复杂电化学方程，但其性能依赖于带标签全寿命轨迹的数量、训练与部署工况的一致性以及时间序列模型的外推能力。公开实验室数据往往只包含少量电芯，真实车辆又常出现不完整充电片段、标签稀疏、温度与负载变化等问题，因此实验室内的低误差并不自动等价于真实部署可靠性。

Yang等提出G-MTC框架，以SCF-GAN扩充健康特征样本，采用STL将SOH分解为Trend、Seasonal和Residual，再分别使用TimesNet、TimesNet和iTransformer建模，最后由Attention Bi-LSTM进行动态融合[1]。原论文在NASA、CALCE和清华数据上报告RMSE均低于1%，其中B0018的RMSE为0.9123%、R²为0.8765。该结构具有清晰的分量语义：Trend描述长期衰减，Seasonal描述周期变化，Residual容纳随机波动和容量再生。然而，论文没有公开源码，若干层级配置、融合损失和精确划分细节未充分披露，使“方法路径复现”和“数值复现”必须分开判断。

近两年的IEEE SOH研究将关注点进一步转向跨工况域偏移、目标域标签缺乏、跨数据集特征差异以及现场片段数据。Miao等针对多目标工况提出多源域元学习[2]；Song等研究无标签目标电池的特征独立对齐[3]；Yao等将数据增强与特征映射结合以改善跨电池适应[4]；Luo等则把现场分段充电、标签不确定性和域偏移共同纳入物理信息模型[5]。这些工作共同支持“分布失配是现实痛点”，但并不证明某种域对齐损失能改善冻结的G-MTC框架。同时，采用EIS或磁场传感的研究改变了观测条件[6], [7]，完整移植其方法会破坏本研究的单机制可归因性。

本文最初的问题是：在不替换TimesNet、iTransformer、Attention Bi-LSTM或SCF-GAN的条件下，能否用一个小而清晰的机制改善物理一致性或跨电池泛化？我们依次考察两项互不叠加的干预。其一借鉴物理信息SOH研究[8], [9]，只约束STL Trend预测；其二依据二阶统计域对齐思想[10]，只对既有融合上下文加入跨电池协方差正则。与只报告最优随机种子不同，本文预先固定成功门槛、保留失败尝试，并将已接触开发集与未接触外层测试分离。

本文的贡献不是宣称得到新的最低RMSE，而是：（1）建立可追溯的G-MTC复现矩阵，区分论文明确方法、工程近似和稳定化措施；（2）实现严格作用于连续Trend轨迹的单调/平滑物理损失，并以五种子隔离消融检验其有效性；（3）实现不替换主架构的CILA融合域对齐，并在预注册外层电池上进行一次性确认；（4）报告未通过门槛的完整证据，揭示局部约束与下游对齐无法修复上游递归外推失真的适用边界。

## 2 相关工作

### 2.1 小样本增强与多分量时序建模

G-MTC的三个关键思想分别对应样本空间扩充、时间尺度分离和分量自适应融合。SCF-GAN通过谱归一化条件GAN、特征匹配与条件批归一化生成健康特征；STL将观测序列写为Trend、Seasonal与Residual之和；TimesNet利用二维周期变化描述时间模式[11]，iTransformer则将变量视为token进行时间序列建模[12]。原论文强调Residual可能包含随机变化和容量再生，因此分解后的分量应保留不同的物理含义，而不能把最终SOH强制处理为严格单调序列。

数据增强本身并不能保证跨电池泛化。Yao等的研究显示，GAN增强与特征映射需要共同处理小样本和数据集差异[4]；Das等把GAF生成框架用于电池时间序列增强，但其下游任务和输入表示与本研究不同[13]。因此本文保留SCF-GAN算法，不再增加第二种生成模型，并规定合成数据只可进入训练集。

### 2.2 Trend物理一致性

物理信息神经网络可将SEI、生热、扩散或等效电路残差写入训练目标。Tian等提出通用物理信息SOH框架[8]，Wang等利用部分充电片段构造可泛化的物理信息估计器[9]。完整PINN需要额外状态、参数或微分方程，超出当前数据字段，也会改变原论文网络。本文采取更窄的先验：长期Trend不应频繁上升，且相邻变化不应突然反转。该约束只作用于Trend分支，不把容量再生错误地当作物理违规。

### 2.3 域偏移与负迁移

跨温度、跨电芯和跨使用策略会使健康特征及隐空间统计发生变化。近期IEEE方法通过元学习、无标签特征对齐、迁移学习或多源联合学习处理该问题[2], [3], [6], [14]。CORAL类方法以均值和协方差距离缩小源域表示差异，计算开销较低，适合作为冻结主干上的单一附加损失[10]。但域不变性并非越强越好：若域差异与SOH信息耦合，对齐可能消除有用信号并导致负迁移。本文因此将“每个目标电池至少3/5种子改善”和“外层配对bootstrap方向一致”写入门槛，而不以合并平均值代替逐电池证据。

### 2.4 研究空缺

现有摘要级证据表明跨工况是重要问题，却少有工作同时报告冻结架构、逐循环预测、五随机种子、失败尝试和未接触外层电池。更少研究区分物理一致性指标与SOH准确率，或在域对齐失败后保留完整负结果。本文填补的是评估和适用边界空缺，而不是以更复杂网络替换现有结构。

## 3 方法

### 3.1 数据与健康特征

NASA Ames PCoE 18650数据包含充放电循环的电压、电流、温度和容量记录[15]。SOH定义为当前放电容量与额定容量之比。依据原论文，提取三个充电时间特征：HF1为从恒流充电开始到电压首次达到4.2 V的时间；HF2为达到4.2 V后至充电电流下降到20 mA截止点的时间；HF3为电压从3.8 V上升到4.0 V的时间。实现中特别保证HF2只在达到4.2 V之后搜索截止电流，避免使用更早的低电流噪声点。

数据按`BatteryID/DatasetID/Cycle`排序。B0005和B0006仅作为SCF-GAN源训练电池；B0007和B0018不进入源预训练。目标电池采用前60%训练适配、随后20%验证和末20%测试。该协议使用目标前缀SOH标签，不是零标签纯跨电池迁移。标准化器、特征处理、GAN增强和STL拟合均限制在相应训练前缀，滑动窗口不得跨越训练、验证和测试边界。

### 3.2 冻结G-MTC基线

设STL分解得到`y=T+S+R`。Trend和Seasonal分别输入原始TimesNet路径，Residual输入iTransformer；三个分量预测及其上下文进入Attention Bi-LSTM，以归一化权重生成最终SOH预测。P0为论文接近版本；P1只在保持主架构不变的前提下加入训练前缀标准化、固定随机种子、梯度裁剪、验证早停、数值稳定初始化和训练范围递归状态裁剪。上述措施属于工程稳定化，不称为原论文方法。

由于原论文没有公开源码，SCF-GAN层宽、iTransformer细节、融合上下文维度和权重辅助损失属于工程近似。任务预注册还要求Trend和Seasonal均使用原始TimesNet；这与PDF部分关于轻量Seasonal路径的描述存在差异，因此本基线是用户指定的隔离基线，而非逐字源码复刻。

### 3.3 Trend-only物理损失

预测损失记为`L_prediction`。对同一`BatteryID/DatasetID`且Cycle连续的Trend预测序列`T_hat`计算：

`L_mono = mean(ReLU(T_hat[t+1] - T_hat[t]))`，

`L_smooth = mean((T_hat[t+1] - 2*T_hat[t] + T_hat[t-1])^2)`。

总损失为：

`L_total = L_prediction + lambda_phy * L_physics`。

`physics_mode`支持`none`、`monotonic`、`smooth`和`both`，`lambda_phy`支持0、0.001、0.01和0.1，默认0.01。物理项在按轨迹组织的训练序列上计算，绝不使用shuffle后随机样本的批内位置作为“相邻循环”。最终SOH不受该损失约束。

Trend violation rate定义为连续预测点中`T_hat[t+1]>T_hat[t]+epsilon`的比例；同时保存平均正向跳变、最大正向跳变和二阶差分幅度，以避免单一比例掩盖跳变强度。

### 3.4 CILA融合协方差对齐

CILA只使用Attention Bi-LSTM融合模块中已经存在的跨分量上下文`Z_b`，其中`b`表示训练电池域。对于两个域，计算均值与协方差差异：

`L_align = ||mu_1-mu_2||_2^2 + ||C_1-C_2||_F^2 / (4d^2)`，

其中`d`为上下文维数。融合训练目标为：

`L = L_prediction + lambda_weight*L_weight + lambda_align*L_align`。

Original令`lambda_align=0`；CILA在验证集锁定`lambda_align=0.1`，`lambda_weight=0.1`。两者共享相同分量输入、初始化、随机种子、训练轮数150和早停耐心30。CILA不改变SCF-GAN、不约束最终SOH，也不替换TimesNet、iTransformer或Attention Bi-LSTM。

### 3.5 实验协议与外层锁定

基线与Trend实验使用种子42、52、62、72和82。B0007/B0018已在项目开发过程中被接触，因此只能支持开发性结论。CILA外层候选从完整NASA归档中按元数据和冻结HF/窗口可行性筛选，筛选发生在模型推理前。B0043和B0044各有88行，划分为52/18/18；只有训练循环0–51可被SCF-GAN增强。正式外层配置哈希锁定后测试一次，结果不得触发参数、架构、电池或种子更换。

外层协议仍是60/20/20有标签目标适配，而不是无标签域泛化。其目的在于检验同一有限适配预算下，融合协方差对齐是否能稳定改善后段生命周期预测。

### 3.6 指标与门槛

报告RMSE、MAE、MAPE和R²，并从保存的逐循环文件重新计算。Trend实验额外报告Trend violation rate。五种子同时报告均值、标准差、中位数和改善种子数。相关时间序列采用5000次圆形时间块bootstrap，计算同一种子、同一循环上的配对RMSE差置信区间。

Trend创新只有在B0007和B0018中位RMSE均低于Original、每个电池至少3/5种子改善、Trend违规或正向跳变下降、其他指标无明显恶化且置信区间支持改善时才成立。CILA外层也要求逐电池配对方向和至少3/5种子改善。测试集不参与模式或权重选择。

## 4 结果

### 4.1 基线可信性与数值复现

所有五个种子均完成训练，无NaN或进程发散。逐循环预测可以重算RMSE、MAE、MAPE和R²，最大记录差异为`1.32e-05`。硬性泄漏检查通过，原始TimesNet、iTransformer和Attention Bi-LSTM路径均参与计算。因此`BaselineComplete=true`。

表1显示，可信性没有转化为论文数值匹配。B0018的中位RMSE为5.1367%，中位R²为-9.5473，而原论文分别为0.9123%和0.8765；预注册的5%相对RMSE与0.03绝对R²差门槛均未通过，故`PaperNumericMatched=false`。

**表1  五种子G-MTC锁定测试结果**

| 电池 | RMSE均值±标准差/% | RMSE中位数/% | MAE均值/% | MAPE均值/% | R²均值 |
|---|---:|---:|---:|---:|---:|
| B0007 | 5.9110±1.9770 | 5.2050 | 5.1995 | 7.2238 | -31.5028 |
| B0018 | 5.1770±0.5758 | 5.1367 | 4.6910 | 6.7648 | -9.8193 |

### 4.2 比较模型与Attention诊断

在相同工程协议下，CNN在B0007和B0018的中位RMSE分别为0.6755%和2.6151%，均低于LSTM和标准Transformer。Transformer中位RMSE分别为4.3966%和2.9100%。因此`TransformerBestObserved=false`，不能从原论文图11推导出本复现中Transformer必然最好。

Attention三分支均参与计算，未出现Trend权重完全为零；但10个电池—种子组合中9个权重随周期变化很小，未充分复现原论文所描述的明显动态分配。该诊断支持将融合实现标记为工程近似，并可能解释部分性能差距。

### 4.3 Trend物理约束消融

验证集选择`lambda_phy=0.1`后执行一次正式测试。表2显示，`both`在B0007的中位RMSE由5.2050%降至5.0157%，且4/5种子改善；但B0018仅2/5种子改善。`monotonic`在B0018有4/5种子改善，却使B0007中位RMSE升至5.9450%，平均RMSE还受种子82严重失败影响。没有单一模式同时满足全部门槛。

**表2  Trend-only物理约束五种子消融**

| 电池 | 模式 | RMSE均值±标准差/% | RMSE中位数/% | Trend违规率均值 | 改善种子 |
|---|---|---:|---:|---:|---:|
| B0007 | none | 5.9110±1.9770 | 5.2050 | 0.0545 | — |
| B0007 | monotonic | 8.4873±7.1560 | 5.9450 | 0.1212 | 2/5 |
| B0007 | smooth | 4.9721±1.1966 | 5.2186 | 0.0788 | 3/5 |
| B0007 | both | 4.6723±0.9630 | 5.0157 | 0.0788 | 4/5 |
| B0018 | none | 5.1770±0.5758 | 5.1367 | 0.0880 | — |
| B0018 | monotonic | 5.1754±0.6830 | 4.9920 | 0.1040 | 4/5 |
| B0018 | smooth | 5.1767±0.9149 | 5.1393 | 0.1200 | 2/5 |
| B0018 | both | 5.0926±0.9422 | 4.8323 | 0.0720 | 2/5 |

值得注意的是，损失项惩罚的是训练窗口中的Trend预测，而表中违规率来自递归测试预测。训练期物理一致性不能保证测试期递归轨迹一致性。这不是对最终SOH施加单调约束的理由，因为Residual中的容量再生仍应保留。

### 4.4 CILA开发验证

在已接触的B0007/B0018开发协议上，CILA的配对中位RMSE差分别为-0.000100和-0.000786个百分点，负值表示改善；改善种子数为4/5和5/5，bootstrap区间均位于0以下。然而改善幅度均低于0.001个百分点，B0007种子72反而恶化0.164408个百分点，使B0007平均差为+0.032741。域分类平衡准确率也未一致下降，说明协方差对齐没有稳定产生线性域不变表示。

这些结果只允许“方向性但实际幅度很小的开发集改善”表述，不能提前声称外部泛化有效。

### 4.5 预注册外层确认

正式外层五种子全部完成，80项保存指标重算检查通过。B0043有3/5种子改善，配对中位RMSE差为-0.001025个百分点，95%时间块bootstrap区间为[-0.283230, -0.000313]，方向门槛通过但实际幅度很小。B0044仅2/5种子改善，配对中位差为+0.000689，区间[-0.002632, +0.001013]跨越0，门槛失败。

**表3  CILA正式外层确认结果**

| 电池 | Original方法中位RMSE/% | CILA方法中位RMSE/% | 配对中位差/% | 改善种子 | 95%时间块bootstrap区间 | 门槛 |
|---|---:|---:|---:|---:|---:|---|
| B0043 | 134.255705 | 134.254680 | -0.001025 | 3/5 | [-0.283230, -0.000313] | 通过但效应可忽略 |
| B0044 | 60.856165 | 51.303511 | +0.000689 | 2/5 | [-0.002632, +0.001013] | 失败 |

B0044的CILA“方法中位RMSE”虽然低于Original，但两个方法的中位数可能来自不同随机种子。预注册主统计量是每个种子内的配对差中位数，因此不能用非配对方法中位数替代。总体判定为`CILAInnovationSupported=false`。

绝对误差还揭示更严重的问题：部分种子的分量递归外推RMSE达到数百至数千个百分点。例如种子52在B0044上的分支误差超过8000%，种子82在B0043上超过2700%。CILA仅作用于融合上下文，无法修复已经偏离真实尺度的Trend、Seasonal或Residual轨迹。外层结果没有触发裁剪、权重或网络调整。

## 5 讨论

### 5.1 “可信复现”与“高精度复现”必须分离

本研究完成了方法路径核验、数据边界审计、多种子训练和逐循环指标重算，因而基线在过程层面可信；但其数值与原论文差距显著，Attention动态性不足，CNN还在当前协议中优于Transformer。这表明，论文未披露的训练细节、数据协议、增强边界或融合实现可能对结果非常敏感。继续针对B0007/B0018调参也许能降低误差，却会把已接触测试信息混入开发决策，因此不能作为确认性复现。

原论文的低RMSE是待复现参照，不是必须通过优化强行达到的目标。若没有作者源码和完整超参数，最诚实的结论是“主架构已实现，若干细节为工程近似，数值未复现”。

### 5.2 物理一致性不是准确率的同义词

Trend单调和平滑先验具有明确物理直觉，但其效应受STL分解、窗口长度和递归预测误差共同影响。某些种子中违规率下降而RMSE不降，说明模型可能变得更平滑但偏离真实Trend水平；另一些种子中RMSE下降但违规率上升，说明误差函数和物理指标在有限测试段上存在权衡。只有同时报告准确率、违规率和跳变幅度，才能避免把“更平滑”误称为“更准确”。

最终SOH包含Seasonal与Residual。容量再生现象意味着其局部上升可能是真实信号，故把单调性直接施加于最终输出会改变原问题，不是本文负结果的合法补救。

### 5.3 融合域对齐的适用边界

CILA开发集效应很小，且外层结果高度依赖随机种子。B0043和B0044方向不同，合并平均会掩盖这种异质性。域对齐假设训练域之间的二阶统计差异包含可去除的工况偏移；当表示差异与衰减阶段、尺度或分支预测误差耦合时，强行对齐可能丢失SOH信息。本文没有把电池ID与温度因果混同，因此不能声称“温度导致失败”，只能报告在所选外层电池协议下观察到泛化失效。

更关键的是，CILA位于融合端，而外层失败首先发生在分量递归预测端。下游融合只能重加权现有输入，无法从数千个百分点的错误分支轨迹恢复真实SOH。该结果提示后续研究应首先以单一机制评估分支外推稳定性，例如训练域尺度不变的递归状态约束或非递归多步预测；但任何新方案必须换用新的预注册外层数据，而不能继续在B0043/B0044上试到成功。

### 5.4 对真实BMS研究的启示

第一，必须明确目标域标签使用方式。本文60/20/20协议是小样本有标签适配，不代表零标签迁移。第二，域泛化结论需要多电池、多工况和未接触外层证据；实验室NASA数据不能直接代表车队。第三，单次随机种子或合并中位数可能掩盖负迁移，应保存配对逐循环预测并报告种子分布。第四，复杂架构没有先验优越性；简单CNN在本实验中更好，说明特征质量、数据边界和外推方式可能比网络名称更重要，这与简单运行特征模型的近期证据相呼应[16]。

## 6 局限性

本研究仅使用NASA实验室数据，外层电池也来自同一归档，不能替代跨机构、跨化学体系或真实车队验证。B0007/B0018在此前实验中已被接触，只能视为开发评估。B0043/B0044从通过HF与窗口可行性筛选的幸存电池中选择，存在可用性选择偏差；但选择发生在模型误差计算前。外层只有两个电池，时间块bootstrap描述的是保存轨迹和种子配对的不确定性，不是现场电池总体置信区间。

原论文没有公开源码，若干SCF-GAN、iTransformer和融合细节属于工程近似，且用户指定的两支原始TimesNet与PDF中部分Seasonal轻量化描述不完全一致。极端外推误差使方法均值不稳定，因此本文同时报告中位数、逐种子差和失败轨迹。文献检索限定近两年IEEE，深入证据多为摘要级核验，不构成完整系统综述。最后，本文没有在正式外层失败后试验新的分支稳定化方法，因而“分支外推是主导瓶颈”是由错误传播位置支持的机制性推断，而非已经由新干预证明的因果结论。

## 7 结论

本文在冻结G-MTC主架构、禁止测试集调参和保留失败记录的前提下，完成了可信基线、Trend-only物理约束和CILA融合域对齐的五种子评估。基线在审计和可重算层面完成，但没有复现原论文数值，Transformer也不是实际比较中的最佳模型。Trend物理约束在个别模式和种子中改善RMSE或物理指标，却没有通过两电池联合门槛；CILA在已接触开发集上的微弱方向性改善没有通过B0043/B0044外层确认。

因此，本文不支持“新增模块稳定降低SOH RMSE”的强结论。更可靠的结论是：当分量递归外推本身失稳时，局部Trend正则或下游融合域对齐难以提供可重复的跨电池收益。完整保存这些负结果能够界定方法适用范围，并为下一轮只研究一个上游外推稳定化因素、使用新的未接触外层数据提供可审计起点。

## 数据与代码可用性

实验使用NASA公开电池老化数据。用户提供的完整归档在本地只读使用，不随仓库重新分发。代码、配置、训练日志、逐循环预测、失败尝试和结果汇总保存在本项目仓库的`output/paper_baseline_v1`、`output/physics_trend_v3`和`output/ieee_innovation_v1`目录。最终公开地址与许可证由作者投稿前补充。

## 伦理声明

本研究只使用公开、非个人电池实验数据，不涉及人类受试者或动物实验，伦理审批不适用。

## CRediT作者贡献声明

作者姓名与具体贡献尚未由研究团队确认。投稿前必须由所有作者依据真实贡献填写Conceptualization、Methodology、Software、Validation、Formal analysis、Writing和Supervision等角色；不得由自动化工具代填。

## 资助声明

待作者确认。在未获得正式项目信息前，本稿不声称任何资助。

## 利益冲突声明

待全体作者确认。当前草稿未发现可由实验记录判断的财务或非财务冲突，但这不替代作者声明。

## 生成式人工智能使用声明

本稿在代码审计、结果汇总和语言组织过程中使用了OpenAI Codex及本地Academic Research Skills工作流。所有数值来自保存的实验输出并经过逐循环重算；文献元数据通过DOI/出版方或已保存的来源核验矩阵检查。人工作者仍须对研究设计、数据合法性、文献解读、作者资格和最终文本承担全部责任。

## 参考文献

[1] H. Yang, S. Peng, C. Chen, A. Tang, Q. Yu, J. Zhang, and R. Li, “Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach,” *Energy*, vol. 342, Art. no. 139463, 2026, doi: 10.1016/j.energy.2025.139463.

[2] M. Miao, C. Xiao, and J. Yu, “Multisource domain metalearning network for battery state-of-health estimation under multitarget working conditions,” *IEEE Trans. Power Electron.*, 2025, doi: 10.1109/TPEL.2025.3538469.

[3] C. Song, H. Yuan, G. Wang, and N. Cui, “State of health estimation for lithium-ion battery with label-free data based on deep domain feature independent alignment network,” *IEEE Trans. Power Electron.*, 2025, doi: 10.1109/TPEL.2025.3580291.

[4] W. Yao, R. Lai, Y. Tian, X. Li, and J. Tian, “State of health estimation of lithium-ion batteries using data augmentation and feature mapping,” *IEEE Trans. Transp. Electrific.*, vol. 11, pp. 4895–4905, 2025, doi: 10.1109/TTE.2024.3471867.

[5] L. Luo, Y. Wang, Y. Li, and Q. Cui, “Physics-informed deep gated recurrent unit network for lithium-ion battery state of health estimation under field conditions,” *IEEE Trans. Transp. Electrific.*, 2026, doi: 10.1109/TTE.2026.3686077.

[6] M. Lin, D. Hu, J. Meng, and J. Wu, “Transfer learning-based lithium-ion battery state of health estimation with electrochemical impedance spectroscopy,” *IEEE Trans. Transp. Electrific.*, 2025, doi: 10.1109/TTE.2025.3533540.

[7] Z. Hui, M. Mamlouk, J. Zhang, and D. Bhowmik, “An innovative magnetic field sensor based method predicting lithium-ion battery state-of-health,” in *Proc. IEEE Sensors Applications Symposium*, 2025, doi: 10.1109/SAS65169.2025.11105130.

[8] A. Tian, L. He, T. Ding, K. Dong, Y. Wang, and J. Jiang, “A generic physics-informed neural network framework for lithium-ion batteries state of health estimation,” *Energy*, vol. 332, Art. no. 137215, 2025, doi: 10.1016/j.energy.2025.137215.

[9] S. Wang, R. Zhou, Y. Ren, H. Liu, Y. Lin, and C. Lian, “A generalizable physics-informed neural network for lithium-ion battery SOH estimation utilizing partial charging segments,” *J. Energy Chem.*, vol. 112, pp. 977–986, 2025.

[10] B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” in *ECCV Workshops*, 2016, pp. 443–450.

[11] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-variation modeling for general time series analysis,” in *Proc. Int. Conf. Learn. Representations*, 2023.

[12] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “iTransformer: Inverted Transformers are effective for time series forecasting,” in *Proc. Int. Conf. Learn. Representations*, 2024.

[13] O. Das, X. Sui, F. Sanfilippo, and S. Rudra, “Battery time-series data augmentation using a GAF-based generative framework,” in *Proc. IEEE EUROCON*, 2025, doi: 10.1109/EUROCON64445.2025.11073369.

[14] V. Dimitrievska, A. Urgolo, M. Freiberger, and K. Unger, “Battery state of health estimation via federated learning on diverse data from multiple sources,” in *Proc. 2nd Int. Conf. Federated Learning and Intelligent Computing Systems*, 2026, doi: 10.1109/FLICS70075.2026.11621953.

[15] B. Saha, K. Goebel, S. Poll, and J. Christophersen, “Prognostics methods for battery health monitoring using a Bayesian framework,” *IEEE Trans. Instrum. Meas.*, vol. 58, no. 2, pp. 291–296, 2009.

[16] V. Olivero-Ortiz, I. Oliveros, and C. A. Robles-Algarín, “Aging and performance analysis of 18650 lithium-ion cell capacity degradation: Data-driven insights and modeling,” *IEEE Access*, 2025, doi: 10.1109/ACCESS.2025.3636613.

<!--protected-hedges: 在本协议下未获得支持; 提示而非证明; 不属于零标签跨电池协议; 摘要级证据; 工程近似 -->
