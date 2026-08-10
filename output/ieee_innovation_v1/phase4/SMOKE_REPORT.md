# CILA Smoke Experiment Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: EXECUTED, SCALE GATE FAILED
- Experiment ID: cila_smoke_raw_seed52
- Type: training smoke test
- Test results inspected: no

## Execution Status

Original and raw-CORAL commands both produced complete two-epoch logs, model weights, validation predictions and metrics. The first wrapper returned timeout code 124 after stopping its wait, but filesystem evidence and process inspection showed that the Original child process completed normally; it was not rerun.

## Validation Metrics

| Mode | Battery | RMSE | MAE | MAPE | R2 |
|---|---|---:|---:|---:|---:|
| Original | B0007 | 0.430554 | 0.306230 | 0.402061 | 0.902515 |
| raw CORAL, 0.01 | B0007 | 0.430553 | 0.306229 | 0.402060 | 0.902516 |
| Original | B0018 | 1.368553 | 1.161153 | 1.635228 | 0.234077 |
| raw CORAL, 0.01 | B0018 | 1.368550 | 1.161149 | 1.635223 | 0.234081 |

These differences are numerical noise and are not reported as an accuracy improvement.

## Failure Diagnosis

- Raw CORAL: approximately `1.7e-9` to `1.9e-9`.
- Weighted contribution at `lambda_align=0.01`: approximately `1.8e-11`.
- Prediction loss: approximately `1.5e-2` to `2.1e-2`.

The confirmed mechanism was therefore numerically inactive. Before any formal validation or outer-test access, attempt C01 fixes only the loss scale by applying a shared pooled-batch standardization to the existing context. Domain-specific standardization is forbidden because it could erase the discrepancy being measured.

## C01 Pre-training Scale Diagnostic

After the shared pooled standardization, without any training:

- initial prediction loss: `0.027161`
- initial CORAL: `0.052542`
- weighted CORAL at 0.01: `0.000525`
- weighted CORAL / prediction loss: `0.019345`
- domains per balanced batch: `2`

The repaired mechanism is numerically active but has not yet passed a training smoke test.

## C01 Scaled-CORAL Training Smoke

The two-epoch repaired run completed:

| Battery | Original RMSE | Scaled CORAL RMSE | CORAL - Original |
|---|---:|---:|---:|
| B0007 | 0.430554 | 0.430597 | +0.000043 |
| B0018 | 1.368553 | 1.368634 | +0.000081 |

Raw CORAL stayed near `0.053-0.058`; its weighted contribution stayed near `0.00053-0.00058`. The mechanism is active, but the two-epoch accuracy change is slightly negative and too small to interpret.

Post-run compliance audit found that these implementation smokes inherited `lambda_weight=0.01` from the module default, while frozen P1 uses `0.1`. All smoke results remain preserved but are excluded from official accuracy comparison. Attempt C02 makes `lambda_weight` an explicit hashed argument fixed to `0.1`; the next pooled-Original and CILA runs will share it.
