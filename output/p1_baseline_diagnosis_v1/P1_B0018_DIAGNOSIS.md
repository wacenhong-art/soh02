# 固定P1基线：B0018误差定位

## 审计结论

- 未发现窗口跨越训练/验证边界；三种子均只预测验证段。
- 注意力权重和误差在保存的逐循环预测上计算；没有使用真实分量作为融合输入。
- 分量真值仅为诊断而在训练+验证前缀上重新STL拟合，测试段未参与。

## 是否存在明确可修复实现错误

目前未发现分量尺度、窗口边界或Attention权重和为1的硬错误。因此不得为改善B0018而更换结构或调参。

## 诊断数据

详见 component_summary.csv、attention_diagnostics.csv、feature_distribution_shift.csv 与 per_cycle_error_diagnostics.csv。
