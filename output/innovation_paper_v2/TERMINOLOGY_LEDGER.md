# Terminology ledger

| Canonical term | First-use definition | Variants avoided | Decision |
|---|---|---|---|
| state of health (SOH) | lithium-ion battery state of health | State-of-Health, health state | Define once, then use SOH |
| G-MTC | SCF-GAN + STL + TimesNet/iTransformer + Attention Bi-LSTM | proposed model, hybrid model | Use G-MTC for the frozen backbone |
| STL Trend | long-term component produced by STL | trend curve, degradation line | Use Trend as the component name |
| Trend-only physical constraint | monotonicity/smoothness loss applied to predicted Trend | physics loss, PINN | Do not call it a full PINN |
| Trend violation rate | fraction of consecutive predicted Trend increases above epsilon | monotonicity error | Preserve the metric definition |
| CILA | covariance-invariant latent alignment | domain adaptation module | Define once and retain CILA |
| labeled 60/20/20 adaptation | target prefix training, validation, held-out suffix testing | zero-label transfer | Explicitly reject zero-label wording |
| outer-battery evaluation | one-time locked evaluation on B0043/B0044 | external validation, independent cohort | Do not imply cross-institution validation |
| percentage point | arithmetic difference between percentage-valued RMSEs | percent improvement | Use for paired RMSE differences |
| engineering approximation | implementation choice not fully specified in the paper | exact reproduction | Keep separate from paper-specified modules |

