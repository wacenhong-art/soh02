# Formal outer-test status

## Current state

- Frozen configuration hash: `797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`.
- Locked outer batteries: B0043 and B0044.
- Protocol: target first 60% labeled adaptation, next 20% validation, last 20% strict test. This is not a zero-label cross-battery protocol.
- Formal outer metrics inspected: **No**.
- Formal output directory created: **No**.

## Completed gates

- Frozen HF extraction produced 88 usable rows for each battery.
- Split feasibility is 52/18/18, with 32 fusion training windows per battery.
- SCF-GAN augmentation is restricted to cycles 0--51 and preserves real target-prefix SOH labels.
- Original TimesNet, iTransformer and Attention Bi-LSTM validation-only smoke path completed.
- Pooled CILA shape audit found no cross-battery windows or target-label inputs.

## Launch incident

The first formal seed-42 launch was rejected before process creation because the execution approval stream disconnected. A read-only check found no Python process and no formal output directory. This is an external orchestration failure, not a training result. The command was not bypassed or silently retried.

## Scientific boundary

No outer-test claim is permitted until the five preregistered seeds complete. Current evidence supports only a weak validation-stage effect, with a material B0007 seed-72 negative-transfer warning. It is not yet sufficient to claim externally confirmed improvement.
