"""
Explainability Module for Sales Lead Conversion Probability Prediction.

This module provides programmatic tools to generate SHAP and LIME explanations
for the locked production Random Forest pipeline.

It handles:
1. Extracting and cleaning feature names after ColumnTransformer preprocessing.
2. Global SHAP feature importance & summary (beeswarm) visualization.
3. Local SHAP waterfall explanations for representative test-set leads.
4. Tabular LIME local surrogate explanations.
5. Saving publication-quality explainability figures to reports/figures/.
"""

import sys
import os

sys.path.insert(0, ".")

import joblib
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lime import lime_tabular

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline


def clean_feature_names(raw_feature_names: List[str]) -> List[str]:
    """
    Map raw ColumnTransformer feature names into clean, readable labels.
    
    Examples:
    - 'num_raw__PageValues' -> 'PageValues'
    - 'cat__Month_Nov' -> 'Month_Nov'
    - 'cat__VisitorType_Returning_Visitor' -> 'VisitorType_Returning_Visitor'
    - 'bool__Weekend' -> 'Weekend'
    """
    clean_names = []
    for name in raw_feature_names:
        # Strip transformer prefixes like 'num_raw__', 'cat__', 'bool__', 'num_std__', 'num_log__'
        if "__" in name:
            clean_name = name.split("__", 1)[1]
        else:
            clean_name = name
        clean_names.append(clean_name)
    return clean_names


def get_transformed_data_and_feature_names(
    pipeline: ImbPipeline, X_data: pd.DataFrame
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Transform input DataFrame through feature engineering and preprocessing steps,
    and extract clean feature names.
    
    Returns
    -------
    X_trans : np.ndarray
        Transformed array fed into estimator.
    clean_names : List[str]
        Clean human-readable feature names.
    X_trans_df : pd.DataFrame
        Transformed DataFrame with clean column names.
    """
    fe_step = pipeline.named_steps["feature_engineer"]
    proc_step = pipeline.named_steps["preprocessor"]

    X_fe = fe_step.transform(X_data)
    X_trans = proc_step.transform(X_fe)

    raw_names = []
    for name, trans, cols in proc_step.transformers_:
        if name == 'remainder' or trans == 'drop':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            try:
                names = trans.get_feature_names_out(cols)
            except Exception:
                names = [f"{name}__{c}" for c in cols]
        else:
            names = [f"{name}__{c}" for c in cols]
        raw_names.extend(names)

    clean_names = clean_feature_names(raw_names)

    X_trans_df = pd.DataFrame(X_trans, columns=clean_names, index=X_data.index)
    return X_trans, clean_names, X_trans_df


def generate_shap_explanations(
    model_path: str = "models/best_lead_scoring_model.joblib",
    test_path: str = "data/processed/test.csv",
    output_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Execute global and local SHAP analysis on holdout test set using locked model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/5] Loading locked production pipeline and test dataset...")
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    # Extract transformed features & feature names
    X_test_trans, clean_names, X_test_trans_df = get_transformed_data_and_feature_names(pipeline, X_test)
    estimator = pipeline.named_steps["classifier"]

    print(f"[2/5] Transformed X_test shape: {X_test_trans.shape} with {len(clean_names)} clean features.")

    # Instantiate SHAP TreeExplainer
    print("[3/5] Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(estimator)
    
    # Check SHAP value output dimensions (binary classification: positive class is index 1 or 2D array)
    shap_values_raw = explainer.shap_values(X_test_trans)
    
    if isinstance(shap_values_raw, list):
        # List of [class_0, class_1] arrays
        shap_vals = shap_values_raw[1]
        base_val = explainer.expected_value[1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_vals = shap_values_raw[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        shap_vals = shap_values_raw
        base_val = explainer.expected_value

    print(f"[OK] SHAP values computed cleanly. Matrix shape: {shap_vals.shape}")

    # Global SHAP Feature Importance Summary Plot (Beeswarm)
    print("[4/5] Generating Global SHAP Beeswarm & Importance Bar Plots...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_trans_df, feature_names=clean_names, show=False)
    plt.title("SHAP Global Summary (Beeswarm Plot) on Holdout Test Set", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "06_shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Global SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_trans_df, feature_names=clean_names, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "07_shap_bar_importance.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Global SHAP plots saved to '{beeswarm_path}' and '{bar_path}'")

    # Local SHAP Explanations (High, Medium, Low probability examples)
    print("[5/5] Selecting representative test-set leads for local SHAP explanations...")
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    X_test_with_prob = X_test.copy()
    X_test_with_prob["pred_prob"] = y_prob_test
    X_test_with_prob["actual_revenue"] = y_test

    # Select candidate indices
    high_idx = X_test_with_prob[X_test_with_prob["pred_prob"] > 0.85].index[0]
    # Medium: probability closest to 0.50
    med_idx = (X_test_with_prob["pred_prob"] - 0.50).abs().idxmin()
    # Low: probability < 0.05
    low_idx = X_test_with_prob[X_test_with_prob["pred_prob"] < 0.05].index[0]

    # Map dataframe index to integer positional index in X_test_trans
    high_pos = X_test.index.get_loc(high_idx)
    med_pos = X_test.index.get_loc(med_idx)
    low_pos = X_test.index.get_loc(low_idx)

    local_examples = [
        ("High-Probability Lead", high_pos, high_idx, "08_shap_local_high_prob.png"),
        ("Borderline Medium-Probability Lead", med_pos, med_idx, "09_shap_local_med_prob.png"),
        ("Low-Probability Non-Converting Lead", low_pos, low_idx, "10_shap_local_low_prob.png"),
    ]

    local_results = []

    for label, pos, orig_idx, filename in local_examples:
        prob = y_prob_test[pos]
        actual = y_test.iloc[pos]
        
        # Create SHAP Explanation object for waterfall plot
        explanation_obj = shap.Explanation(
            values=shap_vals[pos],
            base_values=base_val,
            data=X_test_trans[pos],
            feature_names=clean_names
        )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation_obj, max_display=10, show=False)
        plt.title(f"SHAP Waterfall Explanation: {label} (Prob: {prob:.4f}, Actual: {actual})", fontsize=11, fontweight="bold", pad=15)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Extract top positive & negative features for this lead
        feat_df = pd.DataFrame({
            "feature": clean_names,
            "shap_value": shap_vals[pos],
            "feature_val": X_test_trans[pos]
        }).sort_values(by="shap_value", ascending=False)

        top_pos = feat_df[feat_df["shap_value"] > 0].head(5).to_dict(orient="records")
        top_neg = feat_df[feat_df["shap_value"] < 0].tail(5).to_dict(orient="records")

        local_results.append({
            "label": label,
            "orig_index": int(orig_idx),
            "pos_index": int(pos),
            "predicted_prob": float(prob),
            "actual_revenue": int(actual),
            "figure_path": fig_path,
            "top_positive_contributors": top_pos,
            "top_negative_contributors": top_neg
        })

        print(f"[OK] Local SHAP explanation for '{label}' (Pos: {pos}) saved to '{fig_path}'")

    return {
        "beeswarm_path": beeswarm_path,
        "bar_path": bar_path,
        "clean_feature_names": clean_names,
        "base_value": float(base_val),
        "local_examples": local_results
    }


def generate_lime_explanations(
    model_path: str = "models/best_lead_scoring_model.joblib",
    test_path: str = "data/processed/test.csv",
    output_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Generate local LIME surrogate explanations for representative test-set leads.
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    X_test_trans, clean_names, X_test_trans_df = get_transformed_data_and_feature_names(pipeline, X_test)
    estimator = pipeline.named_steps["classifier"]

    # Instantiate LIME Tabular Explainer on transformed feature space
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_test_trans,
        feature_names=clean_names,
        class_names=["Non-Converting", "Converting"],
        mode="classification",
        random_state=42
    )

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    
    # Pick the representative high probability example index
    high_idx = (y_prob_test > 0.85).nonzero()[0][0]
    high_prob = y_prob_test[high_idx]
    actual_high = y_test.iloc[high_idx]

    exp = lime_explainer.explain_instance(
        data_row=X_test_trans[high_idx],
        predict_fn=estimator.predict_proba,
        num_features=10
    )

    lime_fig_path = os.path.join(output_dir, "11_lime_local_high_prob.png")
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation: High-Probability Lead (Prob: {high_prob:.4f}, Actual: {actual_high})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(lime_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] LIME local explanation figure saved to '{lime_fig_path}'")

    lime_list = exp.as_list()
    return {
        "lime_fig_path": lime_fig_path,
        "lead_pos_index": int(high_idx),
        "predicted_prob": float(high_prob),
        "actual_revenue": int(actual_high),
        "lime_features": lime_list
    }

if __name__ == "__main__":
    shap_res = generate_shap_explanations()
    lime_res = generate_lime_explanations()
    print("\n[SUCCESS] Phase 7 SHAP & LIME Explainability analysis completed cleanly.")
