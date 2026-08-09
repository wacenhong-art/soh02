# G-MTC可信基线报告（paper_baseline_v1）

## 三个独立结论

- BaselineComplete: **true**。数据门禁、五种子训练、逐循环指标重算和锁定测试证据完整。
- PaperNumericMatched: **false**。项目预注册门槛为RMSE相对论文值5%且R²绝对差0.03。
- TransformerBestObserved: **false**。实际比较中CNN在B0007/B0018均为最低RMSE。

## 锁定正式测试

| Battery | RMSE mean±std | RMSE median | MAE mean | MAPE mean | R² mean |
|---|---:|---:|---:|---:|---:|
| B0007 | 5.9110±1.9770 | 5.2050 | 5.1995 | 7.2238% | -31.5028 |
| B0018 | 5.1770±0.5758 | 5.1367 | 4.6910 | 6.7648% | -9.8193 |

论文B0018报告RMSE=0.9123、R²=0.8765；当前中位RMSE=5.1367、中位R²=-9.5473，数值未复现。不能在同一测试集继续调参。

## 方法与近似

已运行SCF-GAN训练增强、STL、两支original TimesNet、Residual iTransformer、Attention Bi-LSTM；physics关闭。PDF实际写Seasonal为lightweight TimesNet，而任务预注册要求两支original TimesNet，因此这里是用户指定隔离基线，不是逐字PDF Seasonal配置。SCF-GAN层宽、iTransformer细节、9维融合误差特征和L_weight公式存在论文未披露/工程近似，详见method_compliance_matrix.csv。

P1稳定化包括训练前缀标准化、梯度裁剪、早停、融合加性重构初始化、L_weight=0.1和训练范围递归状态裁剪。这些均不是论文原始方法。Attention无Trend塌缩，但9/10组合近乎静态，论文动态权重行为未充分复现。

逐循环指标重算最大差异=1.32e-05；锁定配置哈希=`c9b526cab0f080aaf480c10798ca8c234d022b34e566d9ba2b6ba3e0748b670a`。
