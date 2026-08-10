# CILA formal outer validation report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED FROM SAVED PER-CYCLE PREDICTIONS
- Version Label: cila_outer_validation_v1
- Configuration hash: `797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`

## Verdict

Overall confidence is **RED_FLAG for a positive CILA generalization claim**. The frozen outer confirmation criterion failed. `CILAInnovationSupported=false`.

| Battery | Original median RMSE | CILA median RMSE | Median delta | Improved seeds | 95% block-bootstrap CI | Gate |
|---|---:|---:|---:|---:|---:|---|
| B0043 | 134.255705 | 134.254680 | -0.001025 | 3/5 | [-0.283230, -0.000313] | PASS, practically negligible |
| B0044 | 60.856165 | 51.303511 | +0.000689 | 2/5 | [-0.002632, +0.001013] | FAIL |

Negative delta favors CILA. B0044's CILA median is numerically lower because the median of each method can come from different seeds; the preregistered primary statistic is the median of paired seed deltas, which is positive. This distinction prevents an unpaired-median interpretation error.

## Reproducibility

- Five frozen seeds completed without NaN or process crashes.
- Every RMSE, MAE, MAPE and R2 value was recomputed from saved test predictions.
- Original and CILA used identical true labels and identical branch inputs within each seed.
- The largest absolute recomputation difference occurs on a very large float32 MAPE value; all 80 metric checks pass `atol=1e-4, rtol=1e-6` and are listed in `metric_recomputation_audit.csv`.
- Test results were not used for lambda, architecture, battery or seed selection.

## Observed failure modes

- Absolute generalization is poor and highly seed-dependent. Some seeds produce recursive component extrapolation with RMSE in the hundreds or thousands of percentage points.
- CILA cannot repair erroneous branch trajectories because its alignment term acts only on the fusion context; it does not stabilize recursive TimesNet/iTransformer component forecasts.
- Seed42 worsens both batteries; seed52 is mixed; seed62 produces negligible improvement; seed72 worsens both; seed82 improves both.
- The evidence supports a robustness boundary, not CILA superiority.

## Statistical fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | Per-battery directions differ; pooled averaging is prohibited. Results remain stratified by BatteryID. |
| Ecological fallacy | PASS | No individual-cell inference is made from battery-level aggregates. |
| Berkson's paradox | CAUTION | Outer batteries were selected from HF-compatible survivors; selection limits generalizability even though no model metric was used. |
| Collider bias | PASS | No post-outcome covariate adjustment is used. |
| Base-rate neglect | N/A | The task reports regression error, not diagnostic sensitivity/specificity. |
| Regression to the mean | PASS | Batteries were not selected by extreme model error; selection preceded inference. |
| Survivorship bias | CAUTION | Only 5/30 untouched candidates passed frozen HF/window feasibility; excluded batteries remain documented. |
| Look-elsewhere effect | PASS WITH NOTE | The outer battery pair and five seeds were locked before test inference; no selective deletion occurred. |
| Garden of forking paths | PASS WITH NOTE | Validation selected lambda; the outer test triggered no retuning. Failed candidate groups and attempts remain in the ledger. |
| Correlation is not causation | CAUTION | CILA-vs-Original differences are algorithmic associations under this protocol, not causal evidence about temperature mechanisms. |
| Reverse causality | N/A | No directional causal relation between physical variables is estimated. |

## Claim boundary for the manuscript

Allowed: covariance alignment produced weak development-set improvements but failed preregistered outer confirmation under severe low-temperature/domain-shift conditions; the experiment exposes branch-level extrapolation as the dominant bottleneck.

Prohibited: CILA significantly improves cross-condition SOH estimation; Transformer is best; the protocol is zero-label cross-battery; the result reproduces the Energy paper's numerical accuracy.
