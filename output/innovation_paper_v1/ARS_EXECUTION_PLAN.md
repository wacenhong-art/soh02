# ARS end-to-end execution plan

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: academic-pipeline + experiment-agent
- Version Label: ars_pipeline_v1
- Verification Status: IN_PROGRESS
- Repository HEAD at entry: `9fdd8a0`
- Frozen experiment configuration: `797a026852b95eecdc1ce2a284d21396bd95df0df5ba66e533805076d8933a10`

## Objective

Complete the frozen CILA outer experiment without test-set tuning, then produce a fully auditable innovation manuscript. Negative or null findings remain part of the paper; completion means a defensible conclusion, not forced superiority.

## Paper configuration

- Article type: original research article.
- Main language: Simplified Chinese, with independently written Chinese and English abstracts.
- Structure: IMRaD plus Related Work, Limitations, Data Availability, Ethics, CRediT, Funding, Conflict of Interest and AI-use statements.
- Citation style: IEEE numeric style.
- Target framing: journal-neutral battery SOH manuscript derived from the Energy G-MTC reproduction and recent IEEE application-gap review.
- Primary innovation under evaluation: CILA covariance alignment on the unchanged Attention Bi-LSTM fusion context.
- Secondary evidence: Trend-only physics-guided constraint is retained as a completed exploratory/negative-result module, not merged into CILA.

## Execution stages and gates

1. **Frozen experiment completion**
   - Finish seed 42 pooled Original and CILA.
   - Finish seeds 52/62/72/82 original branches and paired pooled fusions.
   - Keep TimesNet, iTransformer, SCF-GAN and Attention Bi-LSTM architecture fixed.
   - Gate: every seed has identical branch inputs for Original/CILA and no NaN/crash.

2. **Statistical validation**
   - Recompute RMSE, MAE, MAPE and R2 from saved per-cycle predictions.
   - Report seed-wise, battery-wise mean, standard deviation and median.
   - Use paired circular time-block bootstrap confidence intervals.
   - Run the complete 11-item ARS statistical fallacy scan.
   - Gate: all reported values reproduce from saved prediction files.

3. **Evidence freeze**
   - Produce final configuration and artifact hashes, leakage audit and immutable claim boundary.
   - Classify CILA as supported, exploratory, unsupported or harmful according to preregistered criteria.
   - Gate: no claim depends on test-driven retuning.

4. **Literature and citation verification**
   - Reuse the existing Sider Scholar/IEEE evidence matrix and paper PDF extraction.
   - Verify every cited DOI or authoritative metadata record; mark unavailable full text explicitly.
   - Gate: zero fabricated references and zero orphan citations.

5. **Manuscript drafting**
   - Write complete title, bilingual abstracts, Introduction, Related Work, Method, Experimental Design, Results, Discussion, Limitations and declarations.
   - Separate paper-specified, paper-inferred, engineering stabilization and new CILA mechanisms.
   - Gate: all numerical claims map to an artifact path and every external claim maps to a verified source.

6. **ARS integrity and peer review**
   - Run pre-review integrity verification including the seven AI research failure modes.
   - Run methodology, domain, perspective, devil's-advocate and editorial review roles independently.
   - Revise once, preserving unresolved issues as limitations.
   - Gate: final integrity report has no fabricated citation/data issue.

7. **Delivery**
   - Deliver versioned Markdown manuscript, references, tables/figures manifest, reproducibility appendix, reviewer reports and process record.
   - DOCX/PDF formatting is a later mandatory finalization checkpoint after content integrity passes.

## Stop rules

- Formal outer results never trigger parameter or architecture changes.
- A failed outer test ends the confirmatory claim; it does not trigger a replacement test battery.
- Missing paper details are labeled Paper-inferred or Engineering approximation.
- Transformer or CILA superiority is stated only when observed in saved results.

