# Innovation Confirmation Gate

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
