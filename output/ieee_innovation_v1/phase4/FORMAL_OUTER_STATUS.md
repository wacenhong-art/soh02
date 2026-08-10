# Formal outer-test status

## Current state

- Frozen configuration hash: `797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`.
- Locked outer batteries: B0043 and B0044.
- Protocol: target first 60% labeled adaptation, next 20% validation, last 20% strict test. This is not a zero-label cross-battery protocol.
- Formal outer branch metrics inspected: **Yes, seed 42 only**.
- Formal paired Original/CILA result available: **No**.
- Formal output directory created: **Yes**.

## Completed gates

- Frozen HF extraction produced 88 usable rows for each battery.
- Split feasibility is 52/18/18, with 32 fusion training windows per battery.
- SCF-GAN augmentation is restricted to cycles 0--51 and preserves real target-prefix SOH labels.
- Original TimesNet, iTransformer and Attention Bi-LSTM validation-only smoke path completed.
- Pooled CILA shape audit found no cross-battery windows or target-label inputs.

## Execution progress

- An earlier seed-42 launch was rejected before process creation because the execution approval stream disconnected.
- After renewed user authorization, the frozen seed-42 branch command completed.
- B0043 branch test: RMSE 46.119506, MAE 39.497387, MAPE 542.3120%, R2 -6.599162.
- B0044 branch test: RMSE 29.194232, MAE 21.434669, MAPE 300.8016%, R2 -2.248743.
- These are poor absolute fits and are preserved as formal negative evidence.
- The subsequent pooled Original launch was rejected before process creation by another approval-stream disconnect. No pooled Original or CILA result exists yet.

## Scientific boundary

No innovation comparison is permitted until paired Original and CILA runs complete. The outer test has now been accessed, so its values are permanently excluded from tuning. Current evidence supports only a weak validation-stage effect, with a material B0007 seed-72 negative-transfer warning and a seed-42 outer branch fitting failure. It is not sufficient to claim externally confirmed improvement.
