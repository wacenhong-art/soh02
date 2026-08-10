# CILA One-Seed Validation Screen

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: EXECUTED ON DEVELOPMENT VALIDATION ONLY
- Experiment ID: cila_seed42_lambda_screen
- Test results inspected: no

## Frozen Setup

- Inputs: `paper_baseline_v1/training_logs/b04r1_validation_lw0.1_seed_42/*/none`
- epochs: 30; patience: 10; seed: 42
- `lambda_weight=0.1`; stable additive initialization
- pooled domains: B0007 and B0018 train prefixes
- evaluation: validation prefixes only

## Results

| lambda_align | B0007 RMSE | delta vs pooled Original | B0018 RMSE | delta vs pooled Original | mean delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.858942 | 0 | 4.587895 | 0 | 0 |
| 0.001 | 3.858923 | -0.000020 | 4.587886 | -0.000009 | -0.000014 |
| 0.01 | 3.858926 | -0.000016 | 4.587877 | -0.000017 | -0.000017 |
| 0.1 | 3.858924 | -0.000018 | 4.587868 | -0.000027 | -0.000022 |

All runs selected epoch 1 and stopped at epoch 11 because the validation objective worsened. All three CORAL weights nominally improve both batteries, but every change is below `0.00003` percentage points and is not practically meaningful.

## Baseline Context

The pooled-Original is a new multi-domain control required by CILA. Against the frozen per-battery validation models, it is about `0.0213` points better on B0007 and `0.1561` points worse on B0018. It must not replace or overwrite the frozen paper baseline.

## Selection

`lambda_align=0.1` has the lowest mean validation RMSE and the largest mean reduction in this pre-registered screen. It advances to a three-seed validation pre-experiment. This is a selection decision, not evidence that CILA is effective. If the effect remains numerical-scale or fails the 3/5 direction gate, the innovation is unsupported.
