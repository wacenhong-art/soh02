# Figure manifest

| Figure | File | Source data | Transformation | Supported claim | Limitation |
|---|---|---|---|---|---|
| F1 外层电池冻结划分 | `output/figures/cila_outer_split_lock_v1.png` | `outer_test_lock.json`, protocol | frozen split plotting script | B0043/B0044在模型推理前锁定为52/18/18 | 仍为有标签60/20/20适配 |
| F2 五种子外层RMSE | `output/figures/cila_outer_rmse_multiseed_log_v1.png` | `formal_outer_metrics.csv` | `scripts/38_plot_cila_outer.py` | 绝对外推误差严重且高度依赖种子 | 对数轴会压缩Original/CILA的小差异 |
| F3 配对RMSE差 | `output/figures/cila_outer_paired_delta_v1.png` | `formal_outer_paired_deltas.csv` | `scripts/38_plot_cila_outer.py` | B0043为3/5改善，B0044为2/5改善 | 绿色仅表示方向，不表示实用显著性 |
| F4 逐循环预测离散 | `output/figures/cila_outer_prediction_iqr_v1.png` | `formal_outer_per_cycle_predictions.csv` | 五种子中位数和IQR，symlog纵轴 | 分支/融合递归外推偏离真实SOH | symlog仅用于展示极端正负值 |
| F5 配对bootstrap区间 | `output/figures/cila_outer_bootstrap_ci_v1.png` | `formal_outer_bootstrap_ci.csv` | 5000次圆形时间块bootstrap | B0044区间跨0，外层确认失败 | 区间不是现场电池总体区间 |

所有图均使用版本化文件名，未覆盖来源不明的旧图。

