# CILA paper-readiness assessment

## Decision

The current CILA result is suitable for a transparent development-validation subsection and method/negative-transfer discussion. It is **not yet sufficient for a confirmatory innovation claim** because the preregistered B0043/B0044 outer test has not executed.

## What is implemented

- The paper backbone remains original TimesNet for Trend and Seasonal, iTransformer for Residual, and Attention Bi-LSTM fusion.
- SCF-GAN still trains from B0005/B0006 and augments only the target training prefix. Its network, objective and five-set augmentation algorithm were not changed.
- CILA adds one covariance-alignment loss on the existing pooled fusion context. It does not replace the paper backbone and does not constrain final SOH.
- Battery-balanced batches use explicit BatteryID and never construct windows across batteries.

## Development validation evidence

| Battery | Original median RMSE | CILA median RMSE | Median delta | Improved seeds | 95% block-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| B0007 | 2.034534 | 2.034202 | -0.000100 | 4/5 | [-0.000364, -0.000003] |
| B0018 | 3.672001 | 3.671216 | -0.000786 | 5/5 | [-0.000883, -0.000661] |

Negative delta favors CILA. These intervals summarize the median seed-specific paired development curves; they are not field-battery population intervals.

## Mandatory caveats

- Practical improvement is weak: both median reductions are below 0.001 percentage points.
- B0007 seed72 is negative transfer: RMSE worsens by 0.164408 percentage points, making the B0007 mean delta +0.032741.
- Domain balanced accuracy decreases only in 2/5 seeds, is unchanged in 1/5 and increases in 2/5. Full domain invariance is not demonstrated.
- B0007/B0018 are previously contacted development evaluation batteries.
- The protocol uses labeled target adaptation (60/20/20), not zero-label cross-battery transfer.

## Outer-test readiness

- B0043/B0044 are locked using metadata and frozen-HF feasibility only.
- Both have 88 rows and a 52/18/18 split; only cycles 0--51 were SCF-GAN augmented.
- Configuration hash: `797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`.
- No outer model metric has been inspected.
- The first execution request failed before process creation because the approval stream disconnected. Therefore no outer numerical conclusion exists.

## Paper claim boundary

At present the defensible claim is: "CILA showed statistically directional but practically small development-set improvements, with a documented seed-dependent negative-transfer case." A stronger claim that CILA improves generalization must wait for the frozen five-seed B0043/B0044 outer test. If that test fails, the protocol must be reported as a negative result and must not be tuned on the same test set.
