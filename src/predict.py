"""
Inference and Explainability Helper Module for Streamlit Production App.

This module encapsulates:
1. Safe loading of the locked production joblib model pipeline.
2. Schema validation for raw input DataFrames.
3. Single-session conversion probability estimation & non-guarantee interpretation.
4. Batch CSV scoring & lead prioritization ranking.
5. Single-lead SHAP attribution extraction with clean feature names.
"""

import os
import sys
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, ".")
from src.explain import get_transformed_data_and_feature_names

# The 17 original raw feature columns required by the pipeline
REQUIRED_INPUT_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def load_production_model(model_path: str = "models/best_lead_scoring_model.joblib"):
    """
    Safely load the locked production model pipeline.
    
    Parameters
    ----------
    model_path : str
        Path to joblib serialized model.
        
    Returns
    -------
    Pipeline
        Loaded Scikit-Learn ImbPipeline object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Production model artifact missing at '{model_path}'! "
            "Please ensure 'models/best_lead_scoring_model.joblib' is generated prior to running Streamlit."
        )
    pipeline = joblib.load(model_path)
    return pipeline


def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that input DataFrame contains all 17 required raw feature columns.
    
    Returns
    -------
    is_valid : bool
        True if all required columns exist, False otherwise.
    missing_cols : List[str]
        List of missing column names.
    """
    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def predict_single_lead(pipeline, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict conversion probability for a single browsing session.
    
    Parameters
    ----------
    pipeline : ImbPipeline
        Loaded production pipeline.
    input_dict : Dict[str, Any]
        Dictionary of raw feature values.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary containing probability, predicted class, and business narrative.
    """
    input_df = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    input_df = input_df[REQUIRED_INPUT_COLUMNS]
    
    y_prob = float(pipeline.predict_proba(input_df)[:, 1][0])
    
    # Default 0.50 threshold
    predicted_class = "Likely to Convert" if y_prob >= 0.50 else "Less Likely to Convert"
    
    if y_prob >= 0.70:
        interpretation = "Based on the model's estimated probability, this session has a high likelihood of conversion."
        priority_label = "High Priority Lead"
    elif y_prob >= 0.40:
        interpretation = "Based on the model's estimated probability, this session shows moderate conversion intent."
        priority_label = "Medium Priority Lead"
    else:
        interpretation = "Based on the model's estimated probability, this session has a low likelihood of conversion."
        priority_label = "Low Priority Lead"
        
    non_guarantee_note = "Note: Predictions are estimates from the trained machine-learning model and are not guarantees of customer behavior."

    return {
        "probability": y_prob,
        "probability_percent": f"{y_prob * 100:.1f}%",
        "predicted_class": predicted_class,
        "priority_label": priority_label,
        "interpretation": interpretation,
        "non_guarantee_note": non_guarantee_note,
        "input_df": input_df
    }


def predict_batch_leads(pipeline, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score a batch of browsing sessions and rank by conversion probability descending.
    """
    is_valid, missing_cols = validate_input_schema(df)
    if not is_valid:
        raise ValueError(
            f"Uploaded dataset is missing required schema columns: {missing_cols}. "
            f"Required columns: {REQUIRED_INPUT_COLUMNS}"
        )
    
    # Reorder columns cleanly to match expected input schema
    df_eval = df[REQUIRED_INPUT_COLUMNS].copy()
    
    y_probs = pipeline.predict_proba(df_eval)[:, 1]
    
    df_scored = df.copy()
    df_scored["conversion_probability"] = np.round(y_probs, 4)
    df_scored["conversion_probability_pct"] = (y_probs * 100).round(1).astype(str) + "%"
    df_scored["predicted_class"] = np.where(y_probs >= 0.50, "Likely to Convert", "Less Likely to Convert")
    
    # Prioritization tiering
    conditions = [
        (y_probs >= 0.70),
        (y_probs >= 0.40) & (y_probs < 0.70),
        (y_probs < 0.40)
    ]
    choices = ["High Priority", "Medium Priority", "Low Priority"]
    df_scored["lead_priority"] = np.select(conditions, choices, default="Low Priority")

    # Sort descending by conversion_probability
    df_scored = df_scored.sort_values(by="conversion_probability", ascending=False).reset_index(drop=True)
    
    summary = {
        "total_sessions": len(df_scored),
        "high_priority_count": int((df_scored["lead_priority"] == "High Priority").sum()),
        "medium_priority_count": int((df_scored["lead_priority"] == "Medium Priority").sum()),
        "low_priority_count": int((df_scored["lead_priority"] == "Low Priority").sum()),
        "mean_predicted_prob": float(y_probs.mean())
    }

    return df_scored, summary


def explain_single_lead_shap(pipeline, single_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Generate SHAP feature contributions for a single session using clean feature names.
    
    Returns
    -------
    shap_df : pd.DataFrame
        DataFrame of features, values, and SHAP attributions.
    base_val : float
        SHAP explainer base expected value.
    """
    X_trans, clean_names, _ = get_transformed_data_and_feature_names(pipeline, single_df)
    estimator = pipeline.named_steps["classifier"]
    
    explainer = shap.TreeExplainer(estimator)
    shap_vals_raw = explainer.shap_values(X_trans)
    
    if isinstance(shap_vals_raw, list):
        shap_vals = shap_vals_raw[1][0]
        base_val = float(explainer.expected_value[1])
    elif isinstance(shap_vals_raw, np.ndarray) and shap_vals_raw.ndim == 3:
        shap_vals = shap_vals_raw[0, :, 1]
        base_val = float(explainer.expected_value[1])
    else:
        shap_vals = shap_vals_raw[0]
        base_val = float(explainer.expected_value)
        
    shap_df = pd.DataFrame({
        "feature": clean_names,
        "transformed_val": X_trans[0],
        "shap_contribution": shap_vals
    }).sort_values(by="shap_contribution", key=abs, ascending=False).reset_index(drop=True)

    return shap_df, base_val
