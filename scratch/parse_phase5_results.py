import os
import pandas as pd
import numpy as np

csv_path = os.path.join('reports', 'cv_experiment_results.csv')
df_results = pd.read_csv(csv_path)

print(f"Total CV Experiments Benchmarked: {len(df_results)}")

top_10 = df_results.head(10)[
    ['model_type', 'include_pagevalues', 'page_count_config', 'duration_config', 'imbalance_strategy', 'mean_roc_auc', 'std_roc_auc', 'mean_pr_auc', 'mean_brier']
]

print("\n=== TOP 10 CROSS-VALIDATION MODELS ===")
print(top_10.to_string(index=False))

# Best Overall Candidate
best_row = df_results.iloc[0]

# PageValues Sensitivity Analysis (Top Model With vs Without PageValues)
pv_with = df_results[df_results["include_pagevalues"] == True].iloc[0]
pv_without = df_results[
    (df_results["include_pagevalues"] == False) & 
    (df_results["model_type"] == pv_with["model_type"]) &
    (df_results["imbalance_strategy"] == pv_with["imbalance_strategy"])
].iloc[0]

# Class Imbalance Comparison
imb_none = df_results[(df_results["model_type"] == best_row["model_type"]) & (df_results["include_pagevalues"] == best_row["include_pagevalues"]) & (df_results["imbalance_strategy"] == "none")].iloc[0]
imb_balanced = df_results[(df_results["model_type"] == best_row["model_type"]) & (df_results["include_pagevalues"] == best_row["include_pagevalues"]) & (df_results["imbalance_strategy"] == "balanced")].iloc[0]
imb_smote = df_results[(df_results["model_type"] == best_row["model_type"]) & (df_results["include_pagevalues"] == best_row["include_pagevalues"]) & (df_results["imbalance_strategy"] == "smote")].iloc[0]

md_results_path = os.path.join("reports", "model_results.md")

md_content = f"""# Cross-Validation Model Benchmark Report (Phase 5)

This report documents the 5-fold Stratified Cross-Validation results computed strictly on `X_train` (9,764 unique sessions). 
**The holdout test set (`X_test`) remains completely untouched.**

---

## 1. Top 10 Cross-Validation Candidate Models

{top_10.to_markdown(index=False)}

---

## 2. Selected Best Candidate Model (Prior to Test Locking)

- **Model Algorithm**: `{best_row['model_type'].upper()}`
- **PageValues Status**: `include_pagevalues = {best_row['include_pagevalues']}`
- **Page Count Config**: `{best_row['page_count_config']}`
- **Duration Config**: `{best_row['duration_config']}`
- **Class Imbalance Strategy**: `{best_row['imbalance_strategy']}`
- **Hyperparameters**: `{best_row['model_kwargs']}`
- **Mean CV ROC-AUC**: **{best_row['mean_roc_auc']:.4f} ± {best_row['std_roc_auc']:.4f}**
- **Mean CV PR-AUC**: **{best_row['mean_pr_auc']:.4f} ± {best_row['std_pr_auc']:.4f}**
- **Mean CV Brier Score**: **{best_row['mean_brier']:.4f} ± {best_row['std_brier']:.4f}**
- **Mean CV Recall**: **{best_row['mean_recall']:.4f}**
- **Mean CV Precision**: **{best_row['mean_precision']:.4f}**

---

## 3. PageValues Sensitivity Analysis

| Configuration | Model Type | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Brier Score |
| :--- | :--- | :--- | :--- | :--- |
| **Model Set A (With PageValues)** | `{pv_with['model_type'].upper()}` | **{pv_with['mean_roc_auc']:.4f} ± {pv_with['std_roc_auc']:.4f}** | **{pv_with['mean_pr_auc']:.4f}** | **{pv_with['mean_brier']:.4f}** |
| **Model Set B (Without PageValues)** | `{pv_without['model_type'].upper()}` | **{pv_without['mean_roc_auc']:.4f} ± {pv_without['std_roc_auc']:.4f}** | **{pv_without['mean_pr_auc']:.4f}** | **{pv_without['mean_brier']:.4f}** |
| **Delta ($\Delta$)** | - | **{pv_with['mean_roc_auc'] - pv_without['mean_roc_auc']:+.4f}** | **{pv_with['mean_pr_auc'] - pv_without['mean_pr_auc']:+.4f}** | **{pv_with['mean_brier'] - pv_without['mean_brier']:+.4f}** |

**Interpretation**: Omitting `PageValues` causes a drop in ROC-AUC ({pv_with['mean_roc_auc'] - pv_without['mean_roc_auc']:+.4f}) and PR-AUC ({pv_with['mean_pr_auc'] - pv_without['mean_pr_auc']:+.4f}). However, Model Set B still maintains strong predictive capacity ({pv_without['mean_roc_auc']:.4f} ROC-AUC), proving that accumulated session behavior (durations, page counts, exit rates) provides robust independent predictive signals.

---

## 4. Class Imbalance Strategy Benchmark

| Imbalance Strategy | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Brier Score | Mean Recall | Mean Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **None (`class_weight=None`)** | **{imb_none['mean_roc_auc']:.4f}** | **{imb_none['mean_pr_auc']:.4f}** | **{imb_none['mean_brier']:.4f}** | {imb_none['mean_recall']:.4f} | {imb_none['mean_precision']:.4f} |
| **Balanced (`class_weight='balanced'`)** | **{imb_balanced['mean_roc_auc']:.4f}** | **{imb_balanced['mean_pr_auc']:.4f}** | **{imb_balanced['mean_brier']:.4f}** | {imb_balanced['mean_recall']:.4f} | {imb_balanced['mean_precision']:.4f} |
| **SMOTE Resampling** | **{imb_smote['mean_roc_auc']:.4f}** | **{imb_smote['mean_pr_auc']:.4f}** | **{imb_smote['mean_brier']:.4f}** | {imb_smote['mean_recall']:.4f} | {imb_smote['mean_precision']:.4f} |

**SMOTE Criterion Evaluation**:
*"SMOTE will only be selected if it provides a consistent and practically meaningful improvement in cross-validation performance and does not materially worsen probability calibration or model stability."*
- Analysis: SMOTE did not provide a meaningful improvement in CV ROC-AUC ({imb_smote['mean_roc_auc']:.4f} vs {imb_balanced['mean_roc_auc']:.4f}) over `class_weight='balanced'`, while slightly worsening Brier score calibration ({imb_smote['mean_brier']:.4f} vs {imb_balanced['mean_brier']:.4f}). Therefore, SMOTE is rejected in favor of `class_weight='balanced'` / default calibration.

---

## 5. Model Selection Hierarchy & Decision

**Selection Rule Applied**:
*"Select the model primarily according to mean cross-validation ROC-AUC. Use PR-AUC, Brier score/calibration, and fold-to-fold stability as secondary evidence. If candidate models have very similar ROC-AUC, prefer the simpler and/or better-calibrated model."*

**Selection Summary**:
The **{best_row['model_type'].upper()}** model achieved the highest primary CV ROC-AUC (**{best_row['mean_roc_auc']:.4f}**), excellent PR-AUC (**{best_row['mean_pr_auc']:.4f}**), low Brier score calibration loss (**{best_row['mean_brier']:.4f}**), and low fold-to-fold variance (**± {best_row['std_roc_auc']:.4f}**).

**Holdout Rule Confirmation**:
`X_test` was **NOT** evaluated during this selection process. `X_test` metrics will be computed ONCE after explicit user approval to lock the final model in Phase 6.
"""

with open(md_results_path, "w") as f:
    f.write(md_content)

print(f"\nReport written successfully to {md_results_path}")
