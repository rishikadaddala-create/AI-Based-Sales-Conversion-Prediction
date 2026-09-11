# Cross-Validation Benchmark & Final Holdout Evaluation Report (Phase 5 & 6)

This report documents both the 5-fold Stratified Cross-Validation model selection results computed strictly on `X_train` (9,764 unique sessions) and the **first and only final evaluation** on the untouched holdout test set (`X_test`, 2,441 unique sessions).

---

## 1. Locked Model Architecture & Production Artifact

The single production model candidate was locked based **strictly** on training-set cross-validation performance. **No test set results were used to modify hyperparameters, select features, choose imbalance strategies, or calibrate probabilities.**

### Model Specification:
- **Algorithm**: **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters**:
  - `n_estimators = 200`
  - `max_depth = None`
  - `min_samples_leaf = 5`
  - `class_weight = None`
  - `random_state = 42`
- **Feature Engineering Configuration (Config A)**:
  - `include_pagevalues = True`
  - `page_count_config = "original"` (`Administrative`, `Informational`, `ProductRelated`)
  - `duration_config = "original"` (`Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`)
  - `include_product_page_ratio = False`
  - `include_avg_time_per_product_page = False`
- **Training Data**: Full deduplicated `X_train` / `y_train` (9,764 sessions, conversion rate: 15.63%).
- **Serialized Model Artifact**: `models/best_lead_scoring_model.joblib` (contains complete `ImbPipeline`).
- **Metadata Document**: `models/model_metadata.json`.

---

## 2. Phase 5: Cross-Validation Benchmark (Model Selection Phase)

The table below summarizes the 5-fold Stratified Cross-Validation metrics computed on `X_train` during Phase 5 model selection:

| Imbalance Strategy | Mean ROC-AUC ± SD | Mean PR-AUC ± SD | Mean Brier Score ± SD | Mean Precision | Mean Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **None (`class_weight=None`) [LOCKED]** | **0.9267 ± 0.0059** | **0.7474 ± 0.0199** | **0.0739 ± 0.0021** | **0.8077** | 0.4948 |
| **Balanced (`class_weight='balanced'`)** | **0.9279 ± 0.0040** | 0.7294 ± 0.0195 | 0.0874 ± 0.0023 | 0.5951 | **0.7844** |
| **SMOTE Resampling** | **0.9274 ± 0.0055** | 0.7213 ± 0.0266 | 0.0745 ± 0.0028 | 0.6882 | 0.6573 |

### Selection Hierarchy Rationale:
1. **Primary Metric (ROC-AUC)**: `balanced` (0.9279) and `none` (0.9267) are in an effective **near-tie** ($\Delta = +0.0012$, paired $t$-test $p = 0.4602 > 0.05$).
2. **Secondary Metrics**: `class_weight=None` achieves superior Precision-Recall AUC (**0.7474 vs 0.7294**), and `class_weight=None` achieves a lower Brier score in the evaluated CV experiments (**0.0739 vs 0.0874**).
3. **Selection Rule**: Following our predefined hierarchy ("prefer the simpler and/or better-calibrated model in near-ties"), `class_weight=None` was selected.

---

## 3. Phase 6: Final Holdout Test Set Performance (`X_test`)

For the **first and only time**, the locked Random Forest pipeline was evaluated on `data/processed/test.csv` (2,441 rows).

### A. Final Unbiased Test Set Metrics

| Metric Type | Metric Name | Training 5-Fold CV (Mean ± SD) | Final Holdout Test Set (`X_test`) |
| :--- | :--- | :--- | :--- |
| **PRIMARY** | **ROC-AUC** | **0.9267 ± 0.0059** | **0.9293** |
| **SECONDARY** | **PR-AUC (Average Precision)** | **0.7474 ± 0.0199** | **0.7458** |
| **SECONDARY** | **Brier Score** | **0.0739 ± 0.0021** | **0.0722** |
| Classification | Accuracy | 0.9026 ± 0.0031 | **0.9033** |
| Classification | Precision (0.50 Threshold) | 0.8077 ± 0.0210 | **0.7645** |
| Classification | Recall (0.50 Threshold) | 0.4948 ± 0.0250 | **0.5524** |
| Classification | F1-Score (0.50 Threshold) | 0.6133 ± 0.0230 | **0.6413** |

### B. Test Set Confusion Matrix (at Default 0.50 Threshold)

$$\begin{pmatrix} \text{TN} = 1994 & \text{FP} = 65 \\ \text{FN} = 171 & \text{TP} = 211 \end{pmatrix}$$

- **True Negatives (TN)**: `1,994` sessions correctly predicted as non-converting.
- **False Positives (FP)**: `65` non-converting sessions misclassified as converting.
- **False Negatives (FN)**: `171` converting sessions missed at 0.50 threshold.
- **True Positives (TP)**: `211` converting sessions correctly identified.

### C. Test Set Target Demographics
- **Positive Class Count (`Revenue=1`)**: `382`
- **Negative Class Count (`Revenue=0`)**: `2,059`
- **Total Holdout Sessions**: `2,441`
- **Actual Holdout Conversion Rate**: **`15.65%`** (consistent with training set `15.63%` and original dataset `15.47%`).

---

## 4. Generalization Gap & Performance Comparison

- **ROC-AUC Generalization**: The model exhibited exceptional generalization stability. Holdout test ROC-AUC (**0.9293**) closely matches training cross-validation (**0.9267 ± 0.0059**), sitting within 0.44 standard deviations of the CV mean ($\Delta = +0.0026$).
- **PR-AUC Generalization**: PR-AUC on holdout test data (**0.7458**) is virtually identical to cross-validation (**0.7474 ± 0.0199**).
- **Brier Calibration Loss**: Brier score on holdout test data (**0.0722**) is slightly lower (better) than training CV (**0.0739 ± 0.0021**), demonstrating that probability calibration generalizes cleanly without over-confidence.

---

## 5. Probability Calibration Assessment on Holdout Test Data

- **Evaluation Only**: Test set calibration was evaluated **strictly as a final descriptive diagnostic**. `X_test` was **NOT** used to calibrate or re-tune model weights.
- **Brier Score**: **`0.0722`**
- **Reliability Diagram**: Saved to `reports/figures/05_test_calibration_curve.png`. The reliability curve demonstrates that estimated conversion probabilities align closely with empirical conversion fractions across all probability deciles.

---

## 6. PageValues Sensitivity Reporting

As established in training cross-validation:
- **With `PageValues` (Model Set A)**: ROC-AUC = **0.9267**
- **Without `PageValues` (Model Set B)**: ROC-AUC = **0.7762**

**Methodological Statement**:  
*"The sensitivity experiment indicates that PageValues contributes substantial predictive signal, while the model without PageValues retains meaningful predictive performance."*

`PageValues` represents real-time session interaction data tracked prior to checkout. It remains documented as a core operational feature required during real-time inference.

---

## 7. Provenance & Reproducibility Audit

- **Raw Dataset Hash**: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` (100% verified immutable).
- **Train/Test Row Counts**: `9,764` train rows, `2,441` test rows (100% preserved).
- **Holdout Isolation**: `X_test` remained completely untouched until Phase 6.
- **Reload Test**: Reloaded `models/best_lead_scoring_model.joblib` and evaluated on `X_test`. `np.testing.assert_allclose` passed with **0 difference** (exact match).
