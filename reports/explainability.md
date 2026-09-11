# SHAP & LIME Model Explainability Report (Phase 7)

This report documents the global and local explainability analysis conducted on the locked production **Random Forest Classifier** (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `class_weight=None`) using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** on the holdout test dataset (`data/processed/test.csv`, 2,441 unique sessions).

**Model Locking Confirmation**:
- The model architecture, hyperparameters, preprocessing pipeline, and decision threshold (0.50) were **strictly locked** prior to Phase 7.
- No model parameters were modified, re-tuned, or re-trained during this explainability evaluation.
- The raw dataset `data/raw/online_shoppers_intention.csv` remains 100% immutable (SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`).

---

## 1. Feature-Name Mapping Validation

To ensure interpretable explanations without raw preprocessor artifacts, anonymous column identifiers were cleanly mapped back to meaningful domain labels:

- `num_raw__PageValues` $\rightarrow$ **`PageValues`**
- `num_raw__ExitRates` $\rightarrow$ **`ExitRates`**
- `num_raw__BounceRates` $\rightarrow$ **`BounceRates`**
- `num_raw__ProductRelated_Duration` $\rightarrow$ **`ProductRelated_Duration`**
- `cat__Month_Nov` $\rightarrow$ **`Month_Nov`**
- `cat__VisitorType_Returning_Visitor` $\rightarrow$ **`VisitorType_Returning_Visitor`**
- `cat__VisitorType_New_Visitor` $\rightarrow$ **`VisitorType_New_Visitor`**
- `bool__Weekend` $\rightarrow$ **`Weekend`**

All 73 one-hot encoded and numerical features are mapped cleanly in SHAP and LIME visual outputs.

---

## 2. SHAP Global Explainability Analysis

SHAP TreeExplainer was evaluated across all 2,441 holdout test sessions to measure the mean magnitude and directional influence of each feature on predicted conversion probabilities.

### Top 15 Most Influential Global SHAP Features

| Rank | Feature Name | Mean \|SHAP Value\| | Directional Impact / Model Prediction Association |
| :---: | :--- | :---: | :--- |
| **1** | **`PageValues`** | **`0.1255`** | **Strong Positive**: High PageValues strongly associated with higher model-predicted conversion probability. |
| **2** | **`Month_Nov`** | **`0.0219`** | **Strong Positive**: November sessions strongly associated with higher model-predicted conversion probability. |
| **3** | **`ExitRates`** | **`0.0199`** | **Strong Negative**: High exit rates strongly associated with lower model-predicted conversion probability. |
| **4** | **`BounceRates`** | **`0.0127`** | **Strong Negative**: High bounce rates associated with lower model-predicted conversion probability. |
| **5** | **`ProductRelated_Duration`** | **`0.0121`** | **Positive**: Longer time spent viewing products associated with higher model-predicted conversion probability. |
| **6** | **`Administrative`** | **`0.0098`** | **Positive**: Account/admin page interactions associated with positive model predictions. |
| **7** | **`ProductRelated`** | **`0.0081`** | **Positive**: Higher number of product page views associated with positive model predictions. |
| **8** | **`Month_May`** | **`0.0063`** | **Negative**: May sessions associated with lower model-predicted conversion probability. |
| **9** | **`VisitorType_Returning_Visitor`** | **`0.0059`** | **Positive**: Returning visitors associated with higher model-predicted conversion probability than new visitors. |
| **10** | **`Administrative_Duration`** | **`0.0058`** | **Positive**: Time spent on admin pages associated positively with model predictions. |
| **11** | **`VisitorType_New_Visitor`** | **`0.0052`** | **Negative**: New visitors associated with lower model-predicted conversion probability. |
| **12** | **`TrafficType_2`** | **`0.0041`** | **Positive**: Traffic source channel 2 associated with higher model-predicted conversion probability. |
| **13** | **`Informational`** | **`0.0031`** | **Mixed/Positive**: Informational page views associated positively with model predictions. |
| **14** | **`Informational_Duration`** | **`0.0029`** | **Mixed**: Time on info pages. |
| **15** | **`Month_Mar`** | **`0.0029`** | **Negative**: March sessions associated with lower model-predicted conversion probability. |

### Visual Artifacts:
- **Global SHAP Beeswarm Plot**: Saved to [`reports/figures/06_shap_summary_beeswarm.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/06_shap_summary_beeswarm.png)
- **Global SHAP Bar Importance Plot**: Saved to [`reports/figures/07_shap_bar_importance.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/07_shap_bar_importance.png)

---

## 3. Dedicated PageValues SHAP & Prediction-Time Analysis

- **Dominance**: `PageValues` is the single most influential feature in the model, with a mean $|SHAP|$ value of **0.1255** (nearly 6x higher than the second-ranking feature, `Month_Nov` at 0.0219).
- **Directionality**: SHAP values for `PageValues > 0` contribute positively to model predictions, raising predicted conversion probabilities upward by up to +0.45 in probability space. Conversely, sessions with `PageValues = 0` contribute negatively (-0.08 to -0.12).
- **Methodological Statement**:  
  *"PageValues is highly predictive in this dataset and is treated as an eligible prediction-time feature under the project's stated session-level prediction assumption. Because its strong predictive influence warrants additional scrutiny, a sensitivity experiment excluding PageValues was performed. The resulting model retained meaningful predictive performance but showed a substantial reduction in ROC-AUC."*

---

## 4. Explainability Consistency: SHAP vs. Random Forest MDI Importance

| Feature Name | SHAP Rank (Mean \|SHAP\|) | RF MDI Rank (Gini Impurity) | RF MDI Share |
| :--- | :---: | :---: | :---: |
| **`PageValues`** | **#1** (`0.1255`) | **#1** (`0.4965`) | **49.65%** |
| **`Month_Nov`** | **#2** (`0.0219`) | #8 (`0.0291`) | 2.91% |
| **`ExitRates`** | **#3** (`0.0199`) | #2 (`0.0730`) | 7.30% |
| **`BounceRates`** | #4 (`0.0127`) | #5 (`0.0443`) | 4.43% |
| **`ProductRelated_Duration`** | #5 (`0.0121`) | #3 (`0.0663`) | 6.63% |
| **`ProductRelated`** | #7 (`0.0081`) | #4 (`0.0572`) | 5.72% |

### Methodological Reconciliation:
Both SHAP and Random Forest MDI agree that **`PageValues` is the dominant feature**. However, `Month_Nov` ranks **#2 in SHAP** but **#8 in MDI**:
- **MDI Importance** measures total Gini impurity reduction across training splits. MDI systematically favors high-cardinality continuous features (such as `ExitRates` or `ProductRelated_Duration`) because they offer frequent split opportunities.
- **SHAP Importance** measures the average marginal contribution of a feature to out-of-fold/test predictions. Month_Nov makes a substantial contribution to the model's predicted conversion probability on test predictions.

---

## 5. Local SHAP Representative Lead Explanations

Representative test-set sessions were analyzed using local SHAP waterfall plots:

### Lead A: High-Probability Lead (Pos: `982`, Pred Prob: `0.9412`, Actual: `1`)
- **Top Positive SHAP Contributors**:
  1. `PageValues = 37.1` ($\Delta = +0.412$)
  2. `Month_Nov = 1` ($\Delta = +0.084$)
  3. `ProductRelated_Duration = 1,420s` ($\Delta = +0.038$)
  4. `ExitRates = 0.008` ($\Delta = +0.021$)
- **Business Interpretation**: High page-value accumulation combined with active product engagement during peak November shopping season contributes positively to a high model-predicted conversion score (94.1%).
- **Waterfall Visual**: Saved to [`reports/figures/08_shap_local_high_prob.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/08_shap_local_high_prob.png)

### Lead B: Borderline Medium-Probability Lead (Pos: `993`, Pred Prob: `0.4851`, Actual: `1`)
- **Top Positive Contributors**: `PageValues = 12.4` ($\Delta = +0.210$), `ProductRelated = 45` ($\Delta = +0.035$).
- **Top Negative Contributors**: `ExitRates = 0.035` ($\Delta = -0.062$), `VisitorType_New_Visitor = 1` ($\Delta = -0.028$).
- **Business Interpretation**: The lead visited valuable pages, but higher exit rates and new-visitor status contribute negatively, producing a borderline model prediction (48.5%). Sales intervention/discounts can target this lead.
- **Waterfall Visual**: Saved to [`reports/figures/09_shap_local_med_prob.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/09_shap_local_med_prob.png)

### Lead C: Low-Probability Non-Converting Lead (Pos: `0`, Pred Prob: `0.0041`, Actual: `0`)
- **Top Negative Contributors**: `PageValues = 0.0` ($\Delta = -0.112$), `ExitRates = 0.200` ($\Delta = -0.084$), `BounceRates = 0.200` ($\Delta = -0.052$).
- **Business Interpretation**: Zero page value combined with immediate bouncing (20% bounce rate) contributes negatively to the model prediction (0.4% conversion probability).
- **Waterfall Visual**: Saved to [`reports/figures/10_shap_local_low_prob.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/10_shap_local_low_prob.png)

---

## 6. Local LIME Surrogate Explanations

LIME Tabular Explainer was fitted on the transformed feature space (`random_state=42`).

- **Target Lead**: High-Probability Lead (Pos `982`, Pred Prob `0.9412`).
- **Top LIME Positive Rules**:
  1. `PageValues > 15.2` (Weight: `+0.384`)
  2. `Month_Nov = 1` (Weight: `+0.076`)
  3. `ExitRates <= 0.012` (Weight: `+0.032`)
  4. `ProductRelated_Duration > 850s` (Weight: `+0.028`)
- **LIME Visual Output**: Saved to [`reports/figures/11_lime_local_high_prob.png`](file:///c:/Users/rishi/OneDrive/Desktop/AI-Based%20Sales%20Conversion%20Probability%20Prediction/reports/figures/11_lime_local_high_prob.png)
- **Consistency Verification**: The major directional drivers identified by LIME are broadly consistent with the corresponding SHAP explanation for this example.

---

## 7. Methodological Limitations & Causal Constraints

1. **Non-Causal Interpretability**: SHAP and LIME explain model predictions, **not underlying causal consumer mechanisms**. A high SHAP value for `PageValues` or `Month_Nov` indicates strong statistical association with model predictions, not causality.
2. **LIME Local Sampling Variance**: LIME builds a linear surrogate around localized perturbations of a single sample. Weights can vary slightly depending on perturbation sample size.
3. **Multicollinearity**: Correlated session attributes (e.g., `ProductRelated` vs. `ProductRelated_Duration`) share attribution across collinear features.
4. **Categorical Feature Splitting**: One-hot encoding splits categorical attribution across individual level indicators (`Month_Nov`, `Month_May`).

---

## 8. Summary of Created Artifacts

1. **`src/explain.py`**: Clean, modular explainability generation script.
2. **`notebooks/05_explainability.ipynb`**: Interactive Jupyter Notebook.
3. **`reports/explainability.md`**: Complete SHAP & LIME documentation.
4. **`reports/figures/`**:
   - `06_shap_summary_beeswarm.png`
   - `07_shap_bar_importance.png`
   - `08_shap_local_high_prob.png`
   - `09_shap_local_med_prob.png`
   - `10_shap_local_low_prob.png`
   - `11_lime_local_high_prob.png`
