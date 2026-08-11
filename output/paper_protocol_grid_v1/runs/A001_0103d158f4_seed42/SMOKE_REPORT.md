# A001 论文协议冒烟训练

状态：`completed_not_promoted`

配置：序列长度8；STL周期8；原始TimesNet宽度32、top-k=2；iTransformer 2头、1层；随机种子42；每个目标电池前20%标注数据训练，5套SCF-GAN增强。

| 目标电池 | 验证RMSE | 验证MAE | 验证MAPE | 验证R2 |
|---|---:|---:|---:|---:|
| B0007 | 4.8003 | 4.3133 | 4.8578% | -4.0682 |
| B0018 | 5.1904 | 4.8855 | 5.8925% | -8.3410 |

门禁：数值有限，数据边界清单显示 `TestLabelsUsedAsFeatures=false` 与 `ValidationLabelsUsedForComponentTraining=false`；但该20轮冒烟不具备晋级比较资格，也不访问测试段。A001不进入三随机种子验证。
