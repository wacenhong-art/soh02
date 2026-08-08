# Innovation result verification report

- Test samples: 26 real B0018 cycles
- Best experiment: A4 Selective-Trend-transfer+OSELM
- Final MAE: 0.4434
- Final RMSE: 1.1548
- Final R2: 0.4669
- TimesNet parameter reduction: 86.08%
- Reliability interval empirical coverage: 84.62%
- Future features used: No
- Online protocol: predict first, then update with the observed label

The causal multi-rate branch improves the no-transfer configuration, but it is not
additive with selective transfer. The final accuracy model therefore uses selective
Trend transfer plus robust OSELM, while the reliability branch is reported separately.
