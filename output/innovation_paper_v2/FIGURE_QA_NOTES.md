# Figure contracts and QA notes

## Common export contract

- Backend: Python/matplotlib only.
- Final width: 183 mm.
- Exports: editable PDF/SVG plus 600 dpi PNG/TIFF.
- Font: Arial/Helvetica fallback; rendered glyph floor at least 5 pt.
- Source observations: all saved rows and all five fixed seeds; no missing-value exclusion and no aesthetic downsampling.
- Visual inheritance: style-only adaptation of the source paper's white background, multi-panel layout, compact legends, panel lettering and continuous-cycle traces. No source-paper panel or artwork was copied.

## Figure 1

- Core conclusion: the two interventions are isolated from the frozen forecasting backbone and act at different locations.
- Archetype: schematic-led composite.
- Panel a: the implemented G-MTC path.
- Panel b: Trend-only physical loss versus fusion-context CILA.
- Reviewer risk addressed: neither intervention is shown as a replacement for TimesNet/iTransformer, and no final-SOH monotonicity is implied.

## Figure 2

- Core conclusion: the auditable baseline does not match the reported B0018 value, and Transformer superiority is not observed in this implementation.
- Archetype: quantitative grid.
- Panels a-b: five seed-level RMSE values; bars are medians and error bars are one seed standard deviation (n=5 seeds).
- Panels c-d: measured SOH and five-seed prediction median with interquartile range.
- Source data: baseline comparison metrics and per-cycle predictions.

## Figure 3

- Core conclusion: Trend constraints yield battery- and seed-dependent trade-offs rather than a joint advantage.
- Archetype: quantitative grid.
- Panels a-b: within-seed paired RMSE difference relative to Original (n=5 seeds).
- Panel c: median and raw Trend violation rates across the same five seeds.
- Panel d: five-seed median Trend trajectories; overlapping near-constant predictions are retained because they are a diagnostic result.
- Source data: physics ablation metrics and Trend predictions.

## Figure 4

- Core conclusion: CILA does not pass joint outer-battery confirmation and does not repair recursive extrapolation failure.
- Archetype: quantitative grid.
- Panel a: within-seed paired RMSE differences (n=5 seeds).
- Panel b: paired median difference with a 95% circular time-block bootstrap interval (5000 replicates).
- Panels c-d: five-seed median prediction error and interquartile range; symmetric-log axes preserve sign and extreme magnitude.
- Source data: formal outer metrics, paired differences, bootstrap intervals and per-cycle predictions.

## Automated and visual QA

- Static source preflight: 20 PASS, 0 WARN, 0 FAIL.
- PDF glyph audit: all four figures pass the 5 pt minimum; smallest observed glyph is 5.25 pt.
- Panel-by-panel review: labels, legends and uncertainty marks are visible at final width; no clipping, overlap or hidden negative results was observed.

