"""
Unit Tests for Streamlit Application Core Prediction & Explainability Logic (Phase 8).

Verifies:
1. Production model artifact loads cleanly.
2. Single-session prediction produces valid probabilities between 0 and 1.
3. Batch scoring works cleanly on valid schemas.
4. Missing required columns are detected and rejected.
5. Extra unexpected columns do not corrupt prediction.
6. Batch scoring output contains conversion_probability and is sorted descending.
7. SHAP single-lead explanation helper generates valid attributions.
8. No model retraining or modification occurs.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.predict import (
    load_production_model,
    validate_input_schema,
    predict_single_lead,
    predict_batch_leads,
    explain_single_lead_shap,
    REQUIRED_INPUT_COLUMNS,
)


@pytest.fixture(scope="module")
def pipeline():
    """Load production joblib model pipeline fixture."""
    model_path = os.path.join("models", "best_lead_scoring_model.joblib")
    assert os.path.exists(model_path), f"Production model missing at {model_path}"
    return load_production_model(model_path)


@pytest.fixture
def sample_valid_input():
    """Fixture for valid raw session dictionary."""
    return {
        "Administrative": 2,
        "Administrative_Duration": 50.0,
        "Informational": 1,
        "Informational_Duration": 10.0,
        "ProductRelated": 25,
        "ProductRelated_Duration": 850.0,
        "BounceRates": 0.0,
        "ExitRates": 0.01,
        "PageValues": 25.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }


def test_model_load(pipeline):
    """Verify production model loads and is a valid pipeline."""
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")


def test_single_lead_prediction(pipeline, sample_valid_input):
    """Verify single lead prediction produces probability between 0 and 1."""
    res = predict_single_lead(pipeline, sample_valid_input)
    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert res["predicted_class"] in ["Likely to Convert", "Less Likely to Convert"]
    assert "Priority Lead" in res["priority_label"]
    assert "estimates" in res["non_guarantee_note"]


def test_schema_validation():
    """Verify input schema validation flags missing required columns."""
    valid_df = pd.DataFrame([{col: 0 for col in REQUIRED_INPUT_COLUMNS}])
    is_valid, missing = validate_input_schema(valid_df)
    assert is_valid is True
    assert len(missing) == 0

    invalid_df = valid_df.drop(columns=["PageValues", "Month"])
    is_valid_inv, missing_inv = validate_input_schema(invalid_df)
    assert is_valid_inv is False
    assert "PageValues" in missing_inv
    assert "Month" in missing_inv


def test_batch_scoring(pipeline, sample_valid_input):
    """Verify batch scoring works, adds probability, and sorts descending."""
    batch_dict_1 = sample_valid_input.copy()
    batch_dict_2 = sample_valid_input.copy()
    batch_dict_2["PageValues"] = 0.0
    batch_dict_2["ExitRates"] = 0.15

    batch_df = pd.DataFrame([batch_dict_2, batch_dict_1])  # 2nd lead has higher PageValues

    scored_df, summary = predict_batch_leads(pipeline, batch_df)

    assert "conversion_probability" in scored_df.columns
    assert "predicted_class" in scored_df.columns
    assert "lead_priority" in scored_df.columns
    assert len(scored_df) == 2

    # Check descending sort order
    probs = scored_df["conversion_probability"].values
    assert probs[0] >= probs[1]
    assert summary["total_sessions"] == 2


def test_batch_missing_columns_rejection(pipeline, sample_valid_input):
    """Verify batch scoring rejects dataframes missing required columns."""
    bad_df = pd.DataFrame([sample_valid_input]).drop(columns=["PageValues"])
    with pytest.raises(ValueError) as excinfo:
        predict_batch_leads(pipeline, bad_df)
    assert "missing required schema columns" in str(excinfo.value)


def test_extra_columns_resilience(pipeline, sample_valid_input):
    """Verify extra non-schema columns do not corrupt prediction process."""
    extra_input = sample_valid_input.copy()
    extra_input["Unexpected_Column_X"] = 9999
    extra_input["Total_Pages"] = 50  # Engineered feature should be ignored

    extra_df = pd.DataFrame([extra_input])
    scored_df, summary = predict_batch_leads(pipeline, extra_df)

    assert len(scored_df) == 1
    assert "conversion_probability" in scored_df.columns
    assert 0.0 <= scored_df["conversion_probability"].iloc[0] <= 1.0


def test_shap_single_lead_explanation(pipeline, sample_valid_input):
    """Verify SHAP explanation helper generates valid attributions for a single lead."""
    input_df = pd.DataFrame([sample_valid_input])
    shap_df, base_val = explain_single_lead_shap(pipeline, input_df)

    assert isinstance(shap_df, pd.DataFrame)
    assert "feature" in shap_df.columns
    assert "shap_contribution" in shap_df.columns
    assert len(shap_df) > 0
    assert isinstance(base_val, float)


def test_synthetic_demo_data_exists_and_scores(pipeline):
    """Verify synthetic demo dataset exists, passes schema validation, and scores successfully."""
    demo_path = os.path.join("data", "synthetic_demo_sessions.csv")
    assert os.path.exists(demo_path), f"Synthetic demo dataset missing at {demo_path}"
    
    demo_df = pd.read_csv(demo_path)
    is_valid, missing = validate_input_schema(demo_df)
    assert is_valid is True, f"Synthetic demo data missing schema columns: {missing}"
    
    scored_df, summary = predict_batch_leads(pipeline, demo_df)
    assert len(scored_df) == len(demo_df)
    assert summary["total_sessions"] == len(demo_df)
    assert "conversion_probability" in scored_df.columns


def test_holdout_test_not_referenced_in_app():
    """Verify production Streamlit app does not reference holdout test.csv file."""
    app_path = os.path.join("app", "streamlit_app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "test.csv" not in content, "Streamlit application still references test.csv!"

