# Annotated Bibliography - ARS Phase 2

Search window: 2024-08-10 to 2026-08-10. Discovery provider: Sider Scholar. Existence/date/venue: Crossref IEEE member metadata. Abstract cross-check: Semantic Scholar.

This is an abstract-level evidence corpus, not a claim of full-text human reading. All ten retained sources passed normalized title, DOI, venue, date-window, and abstract checks.

## 1. Multisource Domain Metalearning Network for Battery State-of-Health Estimation Under Multitarget Working Conditions

Mengqi Miao; Chaoang Xiao; Jianbo Yu (2025). Multisource Domain Metalearning Network for Battery State-of-Health Estimation Under Multitarget Working Conditions. *IEEE transactions on power electronics*. https://doi.org/10.1109/tpel.2025.3538469

- Relevance: Multiple target operating conditions cause domain shift; single-target adaptation must be retrained for each new condition.
- Method: Multisource domain meta-learning, meta-fusion, and an EMD recurrent-convolution feature extractor.
- Data: Combined lithium-ion battery dataset; exact constituent datasets not disclosed in the abstract metadata.
- Reported result: MAE 0.0474 and normalized RMSE 0.2187 over three target operating conditions (units/normalization require full-text confirmation).
- G-MTC boundary: High conceptual fit for cross-temperature/load validation, but the full meta-learning network would be too large as a single add-on.
- Verification/limitation: Use as evidence for the pain point, not as permission to replace TimesNet/iTransformer.

## 2. State of Health Estimation for Lithium-Ion Battery With Label-Free Data Based on Deep Domain Feature Independent Alignment Network

Chenxi Song; Haitao Yuan; G. Wang; Naxin Cui (2025). State of Health Estimation for Lithium-Ion Battery With Label-Free Data Based on Deep Domain Feature Independent Alignment Network. *IEEE transactions on power electronics*. https://doi.org/10.1109/tpel.2025.3580291

- Relevance: Target batteries lack SOH labels and differ by temperature, chemistry, and charge/discharge strategy.
- Method: Lightweight deep domain feature independent alignment with dynamic convolution and feature-space independent alignment.
- Data: Three lithium-ion datasets across temperatures, materials, and operating conditions; names absent from abstract metadata.
- Reported result: 30% early target charging data used without labels; average MAE 0.0098; relative improvements reported against GRU-FIAN, DANN, and LSTM.
- G-MTC boundary: Moderate-to-high conceptual fit, but current 60/20/20 protocol uses target labels and is not label-free.
- Verification/limitation: Would require a new protocol and domain-alignment loss; cannot be claimed using B0007/B0018 adaptation labels.

## 3. State of Health Estimation of Lithium-Ion Batteries Using Data Augmentation and Feature Mapping

W. Yao; Rucong Lai; Yong Tian; Xiaoyu Li; Jindong Tian (2025). State of Health Estimation of Lithium-Ion Batteries Using Data Augmentation and Feature Mapping. *IEEE Transactions on Transportation Electrification*. https://doi.org/10.1109/tte.2024.3471867

- Relevance: Small battery datasets are slow to collect and models adapt poorly across batteries.
- Method: GAN augmentation with consistent Gaussian noise plus a feature-mapping module to reduce cross-dataset differences.
- Data: Multiple battery datasets/samples; exact names absent from abstract metadata.
- Reported result: About 87% of 60 samples improved after augmentation; average AE below 2%; feature mapping RMSE below 1% and half below 0.5%.
- G-MTC boundary: Already overlaps the reproduced paper's SCF-GAN/small-sample premise; poor novelty as another main innovation.
- Verification/limitation: Retain as baseline lineage evidence, not a second GAN innovation.

## 4. Physics-Informed Deep Gated Recurrent Unit Network for Lithium-Ion Battery State of Health Estimation Under Field Conditions

Laijin Luo; Yu Wang; Yang Li; Qiushi Cui (2026). Physics-Informed Deep Gated Recurrent Unit Network for Lithium-Ion Battery State of Health Estimation Under Field Conditions. *IEEE Transactions on Transportation Electrification*. https://doi.org/10.1109/tte.2026.3686077

- Relevance: Field EV data have unreliable SOH labels, segmented charging, domain shift, and weak interpretability.
- Method: Capacity/uncertainty estimation with modified inverse ampere-hour integration and BEAST, feature screening, PIDGRU/DeepHPM, and Bayesian uncertainty-constrained training.
- Data: In-vehicle and cross-vehicle datasets; identifiers absent from abstract metadata.
- Reported result: MAE below 1.10% and RMSE below 1.31% across reported evaluations.
- G-MTC boundary: Pain point is highly relevant, but copying the full PIDGRU/DeepHPM would violate the one-module and no-full-PINN constraints.
- Verification/limitation: Possible inspiration for uncertainty-aware partial-trajectory handling only after compatible public data are secured.

## 5. Transfer Learning-Based Lithium-Ion Battery State of Health Estimation With Electrochemical Impedance Spectroscopy

Mingqiang Lin; Dingcai Hu; Jinhao Meng; Ji Wu (2025). Transfer Learning-Based Lithium-Ion Battery State of Health Estimation With Electrochemical Impedance Spectroscopy. *IEEE Transactions on Transportation Electrification*. https://doi.org/10.1109/tte.2025.3533540

- Relevance: SOH models must generalize across temperature and cell type while retaining physical interpretability.
- Method: Equivalent-circuit features from EIS, spatiotemporal LSTM, and transfer learning.
- Data: Two public EIS datasets; names absent from abstract metadata.
- Reported result: Overall RMSE below 1.9%; more than 80% accuracy improvement over ST-LSTM reported by authors.
- G-MTC boundary: Low direct feasibility because the current HF/STL pipeline does not use dense EIS spectra.
- Verification/limitation: Do not add EIS merely to chase accuracy; it changes sensing assumptions and available inputs.

## 6. Aging and Performance Analysis of 18650 Lithium-Ion Cell Capacity Degradation: Data-Driven Insights and Modeling

Víctor Olivero-Ortiz; Ingrid Oliveros; Carlos A. Robles-Algarín (2025). Aging and Performance Analysis of 18650 Lithium-Ion Cell Capacity Degradation: Data-Driven Insights and Modeling. *IEEE Access*. https://doi.org/10.1109/access.2025.3636613

- Relevance: Second-life screening needs interpretable, low-cost operational indicators and predictive uncertainty.
- Method: Cycle segmentation, cycle-time/internal-impedance indicators, and Gaussian-process regression with multiple kernels.
- Data: Two 18650 cells, 99 cycles, under 3C/1C and 2C/2C profiles from an open degradation dataset.
- Reported result: Best charge RMSE 2.217-4.606 mAh; discharge RMSE 3.166-3.518 mAh; cycle time outperformed impedance-only features.
- G-MTC boundary: Moderate fit for uncertainty or operational-feature diagnostics; small two-cell evidence limits generalization.
- Verification/limitation: Useful counter-evidence that feature relevance can matter more than a complex neural architecture.

## 7. Battery State of Health Estimation via Federated Learning on Diverse Data from Multiple Sources

V. Dimitrievska; Andrea Urgolo; Manuel Freiberger; K. Unger (2026). Battery State of Health Estimation via Federated Learning on Diverse Data from Multiple Sources. *2026 2nd International Conference on Federated Learning and Intelligent Computing Systems (FLICS)*. https://doi.org/10.1109/flics70075.2026.11621953

- Relevance: Models are confined to individual chemistries and laboratory datasets; distributed data cannot always be shared.
- Method: Comparison of federated-learning strategies over heterogeneous multi-source SOH datasets.
- Data: Three public datasets; names and quantitative results absent from abstract metadata.
- Reported result: Abstract reports a detailed performance assessment but provides no numeric metric.
- G-MTC boundary: Low-to-moderate: realistic but introduces a distributed training system, which is too broad for one clean G-MTC innovation.
- Verification/limitation: Evidence for heterogeneity/privacy as a gap; not yet a suitable minimal module.

## 8. Battery Time-Series Data Augmentation Using a GAF-Based Generative Framework

Opy Das; Xin Sui; Filippo Sanfilippo; S. Rudra (2025). Battery Time-Series Data Augmentation Using a GAF-Based Generative Framework. *EUROCON Conference*. https://doi.org/10.1109/eurocon64445.2025.11073369

- Relevance: Battery time-series data are scarce for downstream state estimation.
- Method: Gramian angular field encoding, DCGAN image synthesis, and inverse GAF reconstruction.
- Data: MIT battery dataset; voltage, current, and temperature sequences.
- Reported result: Augmented data improved downstream SOC estimation; abstract metadata gives no exact SOH/RMSE result.
- G-MTC boundary: Low novelty because G-MTC already contains SCF-GAN; replacing it would violate the frozen architecture.
- Verification/limitation: Exclude as the main innovation despite thematic relevance.

## 9. Explainable MFDWA-GRU Model for Battery State of Health Estimation

Anantha Padmanabhan; S. Mate; Rajeev Kumar Singh; S. Singh (2026). Explainable MFDWA-GRU Model for Battery State of Health Estimation. *IEEE Journal of Emerging and Selected Topics in Industrial Electronics*. https://doi.org/10.1109/jestie.2025.3618621

- Relevance: SOH models need robustness across initial conditions, battery combinations, horizons, and interpretable feature influence.
- Method: Multibranch dilated TCN with attention/dynamic weights, GRU, and SHAP explanations.
- Data: Multiple datasets; names and numeric errors absent from abstract metadata.
- Reported result: Authors report lower MAE/RMSE across multiple datasets, without numeric values in the retrieved abstract.
- G-MTC boundary: Moderate for diagnostics, low for replacing the existing TimesNet/iTransformer branches.
- Verification/limitation: SHAP could be a reporting tool, but is not by itself a predictive innovation.

## 10. An Innovative Magnetic Field Sensor based Method Predicting Lithium-ion Battery State-of-health

Ze Hui; Mohamed Mamlouk; Jie Zhang; Deepayan Bhowmik (2025). An Innovative Magnetic Field Sensor based Method Predicting Lithium-ion Battery State-of-health. *Sensors Applications Symposium*. https://doi.org/10.1109/sas65169.2025.11105130

- Relevance: Indirect BMS signals may miss structural aging and safety-relevant current inhomogeneity.
- Method: External magnetic-field sensing with PLSR and CNN models.
- Data: Dedicated magnetic sensor measurements during charge/discharge; sample details absent from abstract metadata.
- Reported result: Reported R2 = 0.92.
- G-MTC boundary: Very low implementation feasibility with current data despite strong application motivation.
- Verification/limitation: Exclude unless new hardware data are legally obtained; do not synthesize missing sensor values.
