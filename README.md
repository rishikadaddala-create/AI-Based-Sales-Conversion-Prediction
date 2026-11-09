# AI-Based Sales Conversion Probability Prediction

A complete, academically reproducible machine learning project that predicts online shopping session **conversion probability** using the UCI Online Shoppers Purchasing Intention dataset. The project demonstrates end-to-end ML engineering: data preprocessing, feature engineering, stratified cross-validation model selection, SHAP/LIME explainability, and an interactive Streamlit web application.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Dataset](#dataset)
5. [Technologies Used](#technologies-used)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Running the Application](#running-the-application)
9. [Running Tests](#running-tests)
10. [Methodology](#methodology)
11. [Feature Engineering](#feature-engineering)
12. [Cross-Validation & Model Selection](#cross-validation--model-selection)
13. [Final Model Configuration](#final-model-configuration)
14. [Evaluation Results](#evaluation-results)
15. [Explainability (SHAP & LIME)](#explainability-shap--lime)
16. [Limitations](#limitations)
17. [Future Improvements](#future-improvements)
18. [Reproducibility](#reproducibility)
19. [License](#license)

---

## Project Overview

This project builds a probabilistic classifier that scores each e-commerce browsing session by its likelihood of ending in a purchase. The output is a **conversion probability score** (0–100 %) rather than a hard yes/no classification, enabling sales and marketing teams to prioritise high-value leads and trigger targeted interventions.

---

## Problem Statement

E-commerce platforms generate large volumes of browsing sessions, but only ~15 % convert to revenue. Identifying high-intent sessions in real time — using features available during the session — allows businesses to intervene with targeted promotions, retargeting ads, or personalised assistance before the user exits.

---

## Project Objectives

1. Build a production-ready binary classifier trained on real-world session data.
2. Conduct rigorous, leakage-free model selection using Stratified K-Fold Cross-Validation on training data only.
3. Evaluate the locked model once on a held-out test set.
4. Provide global and local explainability via SHAP and LIME.
5. Package the model in an interactive web application suitable for demonstration.
6. Document all methodology decisions transparently for academic review.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository — Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) |
| **Citation** | Sakar et al. (2019) |
| **License** | CC BY 4.0 |
| **Raw Sessions** | 12,330 |
| **Duplicates Removed** | 125 |
| **Unique Sessions** | 12,205 |
| **Training Set** | 9,764 sessions (80 %) |
| **Holdout Test Set** | 2,441 sessions (20 %) |
| **Target** | `Revenue` (bool) — 1 = session ended in purchase |
| **Class Ratio** | ~15 % positive (imbalanced) |
| **Raw Dataset SHA-256** | `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |

See [`data/README.md`](data/README.md) for full provenance details.

---

## Technologies Used

| Category | Library / Tool | Version Constraint |
|---|---|---|
| Data manipulation | pandas | `>=2.0.0,<3.0.0` |
| Numerical computing | numpy | `>=1.24.0,<2.0.0` |
| Machine learning | scikit-learn | `>=1.3.0,<1.7.0` |
| Imbalance handling | imbalanced-learn | `>=0.11.0,<0.14.0` |
| Statistical testing | scipy | `>=1.10.0,<2.0.0` |
| Global explainability | shap | `>=0.42.0,<0.47.0` |
| Local explainability | lime | `>=0.2.0` |
| Visualisation | matplotlib, seaborn | `>=3.7.0`, `>=0.12.0` |
| Web application | streamlit | `>=1.28.0,<1.45.0` |
| Model serialisation | joblib | `>=1.3.0` |
| Testing | pytest | `>=7.4.0,<10.0.0` |
| Language | Python | 3.12 |

---

## Project Structure

```
AI-Based Sales Conversion Probability Prediction/
├── data/
│   ├── raw/                          # Raw UCI dataset (gitignored)
│   ├── processed/                    # train.csv & test.csv (gitignored)
│   ├── synthetic_demo_sessions.csv   # 11-row demo file for Streamlit
│   └── README.md                     # Dataset provenance & splitting details
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Dataset schema, stats, quality checks
│   ├── 02_eda.ipynb                  # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb  # Feature design walkthrough
│   ├── 04_model_training.ipynb       # Model selection & CV results (read-only)
│   └── 05_explainability.ipynb       # SHAP & LIME analysis
├── src/
│   ├── data_preprocessing.py         # Loading, deduplication, train/test split
│   ├── feature_engineering.py        # Custom transformer & pipeline factory
│   ├── train.py                      # Full training entry-point (Phase 6)
│   ├── predict.py                    # Inference utilities (single + batch)
│   └── explain.py                    # SHAP TreeExplainer & LIME Tabular
├── models/
│   ├── best_lead_scoring_model.joblib  # Serialised ImbPipeline (tracked)
│   └── model_metadata.json           # Hyperparameters, CV & test metrics
├── reports/
│   ├── model_results.md              # Cross-validation & holdout evaluation report
│   ├── explainability.md             # SHAP & LIME findings
│   ├── cv_experiment_results.csv     # 168-row experiment matrix
│   └── figures/                      # All 11 generated plots (01–11)
├── app/
│   └── streamlit_app.py             # Streamlit web application (5 tabs)
├── tests/
│   └── test_app.py                  # 9 pytest tests (all passing)
├── scratch/                          # Experiment & analysis scripts (tracked)
│   ├── run_phase5.py                 # Phase 5 CV experiment runner
│   ├── execute_phase6.py             # Phase 6 final training & evaluation
│   ├── near_tie_analysis.py          # Focused balanced vs none comparison
│   ├── diagnose_discrepancy.py       # Audit script for report consistency
│   ├── parse_phase5_results.py       # CSV result parser
│   └── inspect_shap_rankings.py      # SHAP ranking verification
├── requirements.txt                  # Pinned dependency ranges
├── LICENSE                           # MIT License
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "AI-Based Sales Conversion Probability Prediction"

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Dataset note**: The raw dataset (`data/raw/online_shoppers_intention.csv`) is gitignored. Download it from [UCI](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) and place it in `data/raw/`. The processed train/test split files (`data/processed/`) are also gitignored for the same reason.

---

## Running the Application

The Streamlit app loads the pre-trained model from `models/best_lead_scoring_model.joblib` and requires no retraining.

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` and provides five tabs:

| Tab | Description |
|---|---|
| **Single Lead Scoring** | Enter session attributes manually and receive a conversion probability score |
| **Batch Scoring** | Upload a CSV of sessions and download a scored results file |
| **SHAP Explanation** | View global feature importance (beeswarm & bar charts) |
| **LIME Explanation** | View local surrogate explanation for a chosen test session |
| **Model Information** | Hyperparameters, CV results, and provenance details |

> **Note**: The sidebar displays an icon loaded from `https://img.icons8.com/color/96/combo-chart.png`. An internet connection is required to render this icon; the app remains fully functional without it.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **9 tests passed**.

| Test | Description |
|---|---|
| `test_model_load` | Verifies `best_lead_scoring_model.joblib` loads correctly |
| `test_single_lead_prediction` | Checks single-row inference returns valid probability |
| `test_schema_validation` | Rejects input missing required columns |
| `test_batch_scoring` | Verifies batch CSV scoring returns correct row count |
| `test_batch_missing_columns_rejection` | Confirms malformed batch input is rejected |
| `test_extra_columns_resilience` | Confirms extra columns are safely ignored |
| `test_shap_single_lead_explanation` | Validates SHAP output has correct feature count |
| `test_synthetic_demo_data_exists_and_scores` | Scores the 11-row demo file end-to-end |
| `test_holdout_test_not_referenced_in_app` | Confirms `test.csv` is never referenced in app code |

---

## Methodology

### Pipeline Architecture

```
Raw Session Data (17 columns)
        ↓
[ feature_engineer ]  — FeatureEngineeringTransformer (custom sklearn Transformer)
        ↓
[ preprocessor ]      — ColumnTransformer
                         ├── num_raw   → StandardScaler  (numerical features)
                         ├── bool      → FunctionTransformer (bool→int)
                         └── cat       → OneHotEncoder(handle_unknown='ignore')
        ↓
[ classifier ]        — RandomForestClassifier
        ↓
Conversion Probability Score (0.0 – 1.0)
```

All transformations are encapsulated inside an `ImbPipeline` (imbalanced-learn), preventing any leakage of test-set statistics into preprocessing.

### Train/Test Split

- `random_state=42`, `test_size=0.20`, stratified on `Revenue`
- Deduplication performed **before** splitting on the full dataset
- The holdout test set (`X_test`) remained untouched until final evaluation

---

## Feature Engineering

The `FeatureEngineeringTransformer` applies a configurable feature set. The production configuration is:

| Parameter | Value | Meaning |
|---|---|---|
| `include_pagevalues` | `True` | Include `PageValues` as a feature |
| `page_count_config` | `'original'` | Use `Administrative`, `Informational`, `ProductRelated` as-is |
| `duration_config` | `'original'` | Use raw duration columns without aggregation |
| `include_product_page_ratio` | `False` | Do not compute product-page ratio |
| `include_avg_time_per_product_page` | `False` | Do not compute average time per product page |

The pipeline passes **17 raw input columns** through feature engineering, resulting in **73 transformed features** at the classifier input (after one-hot encoding of categorical variables).

> **PageValues note**: `PageValues` is a Google Analytics metric computed from pages visited during a browsing session prior to a potential checkout. Its availability at real-time inference depends on the operational deployment context. Under this project's session-level prediction assumption, it is treated as an eligible input feature.

---

## Cross-Validation & Model Selection

### Experiment Matrix

A 168-configuration grid was evaluated using **5-Fold Stratified Cross-Validation** on `X_train` only (see `reports/cv_experiment_results.csv`). Axes varied:

- **Model type**: Random Forest, Logistic Regression
- **Feature config (`include_pagevalues`)**: True / False
- **Page count config**: `original` / `total`
- **Duration config**: `original` / `total`
- **Imbalance strategy**: `none` (class_weight=None), `balanced` (class_weight='balanced'), SMOTE
- **Hyperparameters**: `{n_estimators, max_depth, min_samples_leaf}`

### Imbalance Strategy Comparison (top-3 configurations, RF with PageValues)

| Strategy | Mean ROC-AUC ± SD | Mean PR-AUC ± SD | Mean Brier Score ± SD |
|---|---|---|---|
| **None** *(selected)* | **0.9267 ± 0.0059** | **0.7474 ± 0.0199** | **0.0739 ± 0.0021** |
| Balanced | 0.9279 ± 0.0040 | 0.7294 ± 0.0195 | 0.0874 ± 0.0023 |
| SMOTE | 0.9274 ± 0.0055 | 0.7213 ± 0.0266 | 0.0745 ± 0.0028 |

*All CV results are **VERIFIED** directly from `reports/cv_experiment_results.csv`.*

### Selection Rationale

1. **ROC-AUC near-tie**: `balanced` (0.9279) vs `none` (0.9267), Δ = +0.0012. A paired *t*-test returned *p* = 0.4602 > 0.05, confirming no statistically significant difference. *(p-value is **REPORTED** from `scratch/near_tie_analysis.py` output; fold-level values were not persisted.)*
2. **Secondary metrics decisive**: `class_weight=None` achieves superior PR-AUC (0.7474 vs 0.7294) and lower Brier score (0.0739 vs 0.0874), indicating better probability calibration.
3. **Selection rule**: Pre-defined hierarchy — prefer the simpler, better-calibrated model in near-ties.

---

## Final Model Configuration

```python
RandomForestClassifier(
    n_estimators    = 200,
    max_depth       = None,   # fully grown trees
    min_samples_leaf = 5,
    class_weight    = None,
    random_state    = 42
)
```

Serialised as an `ImbPipeline` in `models/best_lead_scoring_model.joblib`.  
Metadata: `models/model_metadata.json`.

---

## Evaluation Results

> **Important labelling**:
> - **VERIFIED** — directly confirmed from `reports/cv_experiment_results.csv` or the saved joblib.
> - **REPORTED** — stated in `models/model_metadata.json` and `scratch/execute_phase6.py` output; not independently re-verifiable without re-running Phase 6.

### Primary Metrics

| Metric | 5-Fold CV on `X_train` *(VERIFIED)* | Holdout `X_test` *(REPORTED)* |
|---|---|---|
| **ROC-AUC** | **0.9267 ± 0.0059** | **0.9293** |
| **PR-AUC** | **0.7474 ± 0.0199** | **0.7458** |
| **Brier Score** | **0.0739 ± 0.0021** | **0.0722** |
| Accuracy | 0.9026 ± 0.0031 | 0.9033 |
| Precision (@ 0.50) | 0.8077 ± 0.0210 | 0.7645 |
| Recall (@ 0.50) | 0.4948 ± 0.0250 | 0.5524 |
| F1-Score (@ 0.50) | 0.6133 ± 0.0230 | 0.6413 |

### Holdout Confusion Matrix *(REPORTED)*

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN = 1,994 | FP = 65 |
| **Actual Positive** | FN = 171 | TP = 211 |

Holdout test set: 2,441 sessions, 382 positives (15.65 % conversion rate).

### Figures

| Figure | Description |
|---|---|
| `reports/figures/01_target_distribution.png` | Class imbalance visualisation |
| `reports/figures/02_numerical_distributions.png` | Numerical feature distributions |
| `reports/figures/03_conversion_by_categorical.png` | Conversion rates by categorical feature |
| `reports/figures/04_cv_calibration_curve.png` | Calibration curve (CV fold) |
| `reports/figures/05_test_calibration_curve.png` | Calibration curve (holdout test) |
| `reports/figures/06_shap_summary_beeswarm.png` | SHAP beeswarm global importance |
| `reports/figures/07_shap_bar_importance.png` | SHAP bar global importance |
| `reports/figures/08_shap_local_high_prob.png` | SHAP waterfall — high-probability lead |
| `reports/figures/09_shap_local_med_prob.png` | SHAP waterfall — borderline lead |
| `reports/figures/10_shap_local_low_prob.png` | SHAP waterfall — low-probability lead |
| `reports/figures/11_lime_local_high_prob.png` | LIME explanation — high-probability lead |

---

## Explainability (SHAP & LIME)

### Top Global SHAP Features

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `PageValues` | 0.1255 |
| 2 | `Month_Nov` | 0.0219 |
| 3 | `ExitRates` | 0.0199 |
| 4 | `BounceRates` | 0.0127 |
| 5 | `ProductRelated_Duration` | 0.0121 |

`PageValues` dominates global importance (nearly 6× the second feature). See `reports/explainability.md` for the full analysis.

> **Caution**: SHAP and LIME explain model predictions, not causal consumer mechanisms. High SHAP values indicate strong *statistical association*, not causality.

---

## Limitations

1. **PageValues deployment**: The practical availability of `PageValues` at real-time inference depends on when the prediction is triggered relative to session events. Confirm with the operational system before deployment.
2. **Temporal generalisation**: The dataset covers a single year of sessions. Seasonal patterns may drift over time.
3. **Threshold sensitivity**: Default 0.50 threshold yields low recall (55 %). Business-driven threshold tuning on validation data is advisable before deployment.
4. **Dataset scope**: Trained on one e-commerce platform's sessions; generalisation to other domains is unverified.
5. **Calibration**: While Brier scores are low, formal calibration curves (Platt scaling / isotonic regression) were not applied. The model may benefit from post-hoc calibration in production.
6. **p-value evidence**: The near-tie *p* = 0.4602 is reported from experiment output. Fold-level values were not persisted; independent re-verification requires re-running `scratch/near_tie_analysis.py`.

---

## Future Improvements

- Threshold optimisation using precision-recall tradeoff on validation data
- Post-hoc probability calibration (Platt scaling / isotonic regression)
- Temporal cross-validation to account for session date ordering
- Additional feature engineering (session recency, path-level features)
- MLflow or DVC experiment tracking for persistent CV fold outputs
- CI/CD pipeline with automated regression tests on model performance
- Streamlit Cloud or Docker deployment with health-check endpoint

---

## Reproducibility

| Artefact | Detail |
|---|---|
| Raw dataset hash | `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |
| Train rows | 9,764 |
| Test rows | 2,441 |
| `random_state` | 42 (split, model, LIME) |
| Training script | `scratch/execute_phase6.py` |
| CV experiment matrix | `scratch/run_phase5.py` |
| Dependencies | `requirements.txt` (pinned ranges) |

To reproduce the final model evaluation (without retraining):
```bash
python scratch/execute_phase6.py   # requires raw data in data/raw/
```

> **Note**: Re-running `execute_phase6.py` will retrain and overwrite the saved model. The existing `models/best_lead_scoring_model.joblib` is the reference artefact and should be preserved unless intentional retraining is performed.

---

## License

**Code**: MIT License — see [`LICENSE`](LICENSE).  
**Dataset**: CC BY 4.0 — [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) by Sakar et al. (2019).
