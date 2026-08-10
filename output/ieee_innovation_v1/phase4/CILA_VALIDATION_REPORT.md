# CILA Five-Seed Development Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED ON CONTACTED DEVELOPMENT VALIDATION SETS
- Experiment ID: cila_five_seed_validation
- Test results inspected: no

## Locked Comparison

- Architecture: unchanged original TimesNet + original TimesNet + iTransformer + Attention Bi-LSTM
- Domains: B0007 and B0018 training prefixes pooled with explicit BatteryID labels
- Comparison: pooled-Original (`lambda_align=0`) versus CILA (`lambda_align=0.1`)
- Shared: component inputs, seed, initialization, `lambda_weight=0.1`, epochs=150, patience=30
- Seeds: 42, 52, 62, 72, 82

## Primary Validation Summary

| Battery | Original median RMSE | CILA median RMSE | median delta | improved seeds | mean delta | worst seed delta | block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0007 | 2.034534 | 2.034202 | -0.000100 | 4/5 | +0.032741 | +0.164408 | [-0.000364, -0.000003] |
| B0018 | 3.672001 | 3.671216 | -0.000786 | 5/5 | -0.091807 | -0.000027 | [-0.000883, -0.000661] |

Negative delta favors CILA. The confidence interval is for the median across seed-specific paired, circular time-block bootstrap RMSE differences (5000 replicates). It is not a confidence interval for a population of field batteries.

## Secondary Metrics

- Median MAE and MAPE changes favor CILA on both batteries.
- Median R2 changes favor CILA on both batteries.
- B0007 seed72 is a material negative-transfer case: RMSE rises by 0.164408, MAE by 0.155068 and R2 falls by 0.192437.
- The worst-battery RMSE improves in all five seeds because B0018 is the larger-error battery in each paired run.

## Domain Diagnostic

Validation BatteryID balanced accuracy decreases in seeds42 and52, is unchanged at 1.0 in seed62, and increases in seeds72 and82. CORAL aligns covariance, but the observed representations do not become consistently linearly domain-invariant. This diagnostic cannot be interpreted as causal temperature invariance because BatteryID and operating condition are confounded.

## Gate Decision

- Direction gate: PASS (B0007 4/5; B0018 5/5).
- Median RMSE gate: PASS for both batteries.
- Median block-bootstrap gate: PASS for both batteries.
- Practical-effect gate: FAIL/WEAK. Median improvements are below 0.001 percentage points.
- Stability warning: B0007 has one large adverse seed and a worse mean RMSE.

The locked configuration may proceed to the predeclared B0043-B0044 outer protocol after its data preparation and leakage audit passed. Earlier candidate groups failed length/HF compatibility before inference and are documented in the ledger. The outer result is confirmatory for this protocol and must be run once. Failure ends the protocol; it cannot trigger test-driven retuning.
