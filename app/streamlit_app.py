"""
Production Streamlit Application for AI-Based Sales Conversion Probability Prediction.

This interactive web dashboard presents:
- Tab 1: Single Lead Prediction (Form inputs -> Locked ML Pipeline -> Probability & Priority)
- Tab 2: Batch Lead Scoring (CSV upload -> Validation -> Prioritization Ranking -> Download)
- Tab 3: Model Performance (Phase 5 CV & Phase 6 Holdout Test Benchmark Metrics)
- Tab 4: Explainability (SHAP Contribution Analysis & Local Attribution)
- Tab 5: Methodology & Data Provenance (UCI Dataset Provenance, Preprocessing & Governance)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend required on Streamlit Cloud (Linux/headless)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, ".")

from src.predict import (
    load_production_model,
    predict_single_lead,
    predict_batch_leads,
    explain_single_lead_shap,
    validate_input_schema,
    REQUIRED_INPUT_COLUMNS
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Sales Conversion Probability & Lead Scoring",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .priority-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .priority-medium {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .priority-low {
        background-color: #F3F4F6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load Model Safely with st.cache_resource
@st.cache_resource
def get_model():
    model_path = os.path.join("models", "best_lead_scoring_model.joblib")
    try:
        pipeline = load_production_model(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading production model: {str(e)}")
        st.stop()


pipeline = get_model()


# Sidebar Navigation & Model Info
# NOTE: The image below is loaded from an external URL and requires an internet connection to display.
# The app remains fully functional if the image fails to load.
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
st.sidebar.title("Lead Scoring System")
st.sidebar.markdown("**Locked Production Model**: Random Forest (Config A)")
st.sidebar.markdown("**Model Version**: `v1.0.0` (Phase 6 Locked)")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Holdout Policy**: Evaluation data (`X_test`) remains strictly isolated. "
    "Predictions are generated directly by the production pipeline fitted on `X_train`."
)

st.markdown('<div class="main-header">AI-Based Sales Conversion Probability Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Enterprise Lead Scoring & Real-Time Conversion Analytics System</div>', unsafe_allow_html=True)


# Application Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Single Lead Prediction",
    "📊 Batch Lead Scoring",
    "📈 Model Performance",
    "🔍 Explainability (SHAP)",
    "📚 Methodology & Data Provenance"
])


# ==============================================================================
# TAB 1 — SINGLE LEAD PREDICTION
# ==============================================================================
with tab1:
    st.markdown("### Enter Session Features for Real-Time Conversion Probability Estimation")
    st.markdown("Fill out the browsing session attributes below to estimate conversion probability.")

    with st.form("single_lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📄 Page View Activity")
            admin_pages = st.number_input("Administrative Page Views", min_value=0, max_value=50, value=2, step=1)
            admin_duration = st.number_input("Administrative Duration (sec)", min_value=0.0, max_value=5000.0, value=50.0, step=5.0)
            info_pages = st.number_input("Informational Page Views", min_value=0, max_value=30, value=1, step=1)
            info_duration = st.number_input("Informational Duration (sec)", min_value=0.0, max_value=3000.0, value=10.0, step=5.0)
            product_pages = st.number_input("Product-Related Page Views", min_value=0, max_value=700, value=25, step=1)
            product_duration = st.number_input("Product-Related Duration (sec)", min_value=0.0, max_value=70000.0, value=850.0, step=10.0)

        with col2:
            st.markdown("#### ⚡ Engagement & Intent Metrics")
            bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=0.20, value=0.0, step=0.005, help="Percentage of visitors who enter and leave on the same page")
            exit_rate = st.slider("Exit Rate", min_value=0.0, max_value=0.20, value=0.01, step=0.005, help="Percentage of total page views that ended in an exit")
            page_values = st.number_input("PageValues", min_value=0.0, max_value=400.0, value=25.0, step=1.0, help="Google Analytics score measuring value of pages visited before purchase")
            special_day = st.select_slider("Special Day Proximity", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0, help="Closeness of visit to a major commercial date (e.g. Mother's Day)")
            # Month options reflect months present in the UCI training dataset.
            # 'Jan' and 'Apr' are absent from the training data; 'Jan' is included here
            # for user convenience — the OHE pipeline handles unknown values gracefully
            # (handle_unknown='ignore'), treating unknown months as all-zero indicator columns.
            month = st.selectbox("Month of Visit", options=['Jan', 'Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=9)

        with col3:
            st.markdown("#### 🌐 Technical & Demographics")
            visitor_type = st.selectbox("Visitor Type", options=['Returning_Visitor', 'New_Visitor', 'Other'], index=0)
            operating_system = st.selectbox("Operating System ID", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1)
            browser = st.selectbox("Browser ID", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=1)
            region = st.selectbox("Geographic Region ID", options=[1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
            traffic_type = st.selectbox("Traffic Type ID", options=list(range(1, 21)), index=1)
            weekend = st.checkbox("Weekend Visit", value=False)

        submit_btn = st.form_submit_button("Predict Conversion Probability", use_container_width=True)

    if submit_btn:
        single_input = {
            "Administrative": admin_pages,
            "Administrative_Duration": admin_duration,
            "Informational": info_pages,
            "Informational_Duration": info_duration,
            "ProductRelated": product_pages,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rate,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        res = predict_single_lead(pipeline, single_input)
        st.session_state["last_single_res"] = res

    if "last_single_res" in st.session_state:
        res = st.session_state["last_single_res"]
        st.markdown("---")
        st.markdown("### 📊 Prediction Output & Score Breakdown")

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{res["probability_percent"]}</div><div class="metric-label">Estimated Conversion Probability</div></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{res["predicted_class"]}</div><div class="metric-label">Classification Output (Threshold 0.50)</div></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{res["priority_label"]}</div><div class="metric-label">Lead Prioritization Tier</div></div>', unsafe_allow_html=True)

        st.markdown(f"**Business Narrative**: {res['interpretation']}")
        st.markdown(f'<div class="disclaimer-box">{res["non_guarantee_note"]}</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2 — BATCH LEAD SCORING
# ==============================================================================
with tab2:
    st.markdown("### Upload Batch Session CSV File for Lead Prioritization")
    st.markdown("Upload a CSV file containing session attributes to score and rank multiple sales leads.")

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Choose a CSV File", type=["csv"])
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        demo_btn = st.button("Load Synthetic Demo Sessions", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV successfully loaded ({len(batch_df)} rows).")
        except Exception as e:
            st.error(f"Failed to parse CSV: {str(e)}")
    elif demo_btn:
        demo_csv_path = os.path.join("data", "synthetic_demo_sessions.csv")
        if os.path.exists(demo_csv_path):
            batch_df = pd.read_csv(demo_csv_path)
            st.info(f"Loaded synthetic demo sessions ({len(batch_df)} rows). Note: This dataset contains synthetic values created solely for demonstration purposes and was NOT used for model training or evaluation.")
        else:
            st.error(f"Synthetic demo CSV not found at '{demo_csv_path}'.")

    if batch_df is not None:
        is_valid, missing_cols = validate_input_schema(batch_df)
        if not is_valid:
            st.error(f"❌ **Schema Validation Failed**: Uploaded file is missing required columns: `{missing_cols}`.")
            st.warning(f"Required schema columns: `{REQUIRED_INPUT_COLUMNS}`")
        else:
            st.success("✅ **Schema Validation Passed**: All 17 required feature columns are present.")
            with st.spinner("Scoring batch leads using production pipeline..."):
                scored_df, summary = predict_batch_leads(pipeline, batch_df)

            st.markdown("### 📈 Batch Scoring Summary")
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            with bcol1:
                st.metric("Total Scored Sessions", summary["total_sessions"])
            with bcol2:
                st.metric("High Priority Leads (≥70%)", summary["high_priority_count"])
            with bcol3:
                st.metric("Medium Priority Leads (40-70%)", summary["medium_priority_count"])
            with bcol4:
                st.metric("Low Priority Leads (<40%)", summary["low_priority_count"])

            st.markdown("### 🏆 Scored Leads (Ranked by Conversion Probability Descending)")
            
            # Reorder columns for optimal viewing
            display_cols = ["conversion_probability_pct", "predicted_class", "lead_priority", "PageValues", "ProductRelated", "ProductRelated_Duration", "Month", "VisitorType"]
            remaining_cols = [c for c in scored_df.columns if c not in display_cols and c != "conversion_probability"]
            final_view_df = scored_df[display_cols + remaining_cols]

            st.dataframe(final_view_df, use_container_width=True)

            csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Leads CSV",
                data=csv_bytes,
                file_name="scored_sales_leads.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==============================================================================
# TAB 3 — MODEL PERFORMANCE
# ==============================================================================
with tab3:
    st.markdown("### Documented Phase 5 Cross-Validation & Phase 6 Holdout Test Metrics")
    st.markdown("Metrics reflect documented benchmark evaluations. **No metrics are recalculated inside Streamlit.**")

    col_cv, col_test = st.columns(2)

    with col_cv:
        st.markdown("#### 🔄 5-Fold Cross-Validation (Training Data)")
        st.info("Computed strictly on `X_train` (9,764 unique sessions) during Phase 5 model selection.")
        st.markdown("""
        | Metric Name | Mean 5-Fold CV Score ± SD |
        | :--- | :--- |
        | **Primary: ROC-AUC** | **`0.9267 ± 0.0059`** |
        | **Secondary: PR-AUC** | **`0.7474 ± 0.0199`** |
        | **Secondary: Brier Score** | **`0.0739 ± 0.0021`** |
        | Precision (0.50 Threshold) | `0.8077 ± 0.0210` |
        | Recall (0.50 Threshold) | `0.4948 ± 0.0250` |
        """)

    with col_test:
        st.markdown("#### 🎯 Final Holdout Test Performance (Untouched `X_test`)")
        st.success("Evaluated ONCE during Phase 6 on `X_test` (2,441 unique sessions).")
        st.markdown("""
        | Metric Name | Holdout Test Score |
        | :--- | :--- |
        | **Primary: ROC-AUC** | **`0.9293`** |
        | **Secondary: PR-AUC** | **`0.7458`** |
        | **Secondary: Brier Score** | **`0.0722`** |
        | Accuracy | `0.9033` |
        | Precision (0.50 Threshold) | `0.7645` |
        | Recall (0.50 Threshold) | `0.5524` |
        | F1-Score (0.50 Threshold) | `0.6413` |
        """)

    st.markdown("---")
    st.markdown("### 🔲 Final Holdout Confusion Matrix (Default 0.50 Threshold)")
    cm_col1, cm_col2 = st.columns([1, 2])
    with cm_col1:
        st.markdown("""
        ```
        [[1994   65]
         [ 171  211]]
        ```
        - **True Negatives (TN)**: `1,994`
        - **False Positives (FP)**: `65`
        - **False Negatives (FN)**: `171`
        - **True Positives (TP)**: `211`
        """)
    with cm_col2:
        st.markdown("""
        **Threshold Note**: Default classification threshold = **0.50**.  
        The confusion matrix reflects model discrimination at the standard default threshold. 0.50 is an illustrative baseline, not an optimized business cutoff.
        """)

    st.markdown("---")
    st.markdown("### 🔬 PageValues Sensitivity Analysis")
    st.markdown("""
    | Configuration | Model Strategy | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Brier Score |
    | :--- | :--- | :--- | :--- | :--- |
    | **Model Set A (With PageValues)** | `class_weight=None` | **`0.9267 ± 0.0059`** | **`0.7474`** | **`0.0739`** |
    | **Model Set B (Without PageValues)** | `class_weight=None` | **`0.7762 ± 0.0164`** | **`0.3852`** | **`0.1142`** |
    
    **Sensitivity Finding**: PageValues contributes substantial predictive signal in this dataset. The sensitivity experiment shows that predictive performance remains meaningful without PageValues, although performance decreases substantially.
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Locked Model Specification")
    st.markdown("""
    - **Algorithm**: **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)
    - **Hyperparameters**: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `class_weight=None`, `random_state=42`
    - **Feature Configuration**: Config A (`include_pagevalues=True`, `page_count_config="original"`, `duration_config="original"`, no ratio enhancements)
    """)


# ==============================================================================
# TAB 4 — EXPLAINABILITY (SHAP & LIME)
# ==============================================================================
with tab4:
    st.markdown("### Feature Attribution & Model Explainability (SHAP)")
    st.markdown("Explanations illustrate feature associations with model predictions. **SHAP values explain model predictions, not causal relationships.**")

    if "last_single_res" in st.session_state:
        res = st.session_state["last_single_res"]
        single_df = res["input_df"]
        st.markdown("#### 🎯 SHAP Feature Attribution for Current Single Lead Prediction")

        with st.spinner("Calculating SHAP feature attributions..."):
            shap_df, base_val = explain_single_lead_shap(pipeline, single_df)

        st.markdown(f"**Base Expected Value**: `{base_val:.4f}` | **Final Model-Predicted Probability**: `{res['probability_percent']}`")

        pos_df = shap_df[shap_df["shap_contribution"] > 0].head(5)
        neg_df = shap_df[shap_df["shap_contribution"] < 0].tail(5)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("##### 🟢 Top Positive Contributors (Increase Predicted Conversion)")
            for _, r in pos_df.iterrows():
                st.markdown(f"- **{r['feature']}**: `+{r['shap_contribution']:.4f}` (Value: `{r['transformed_val']:.2f}`) — *contributes positively to the model prediction*")
        with col_neg:
            st.markdown("##### 🔴 Top Negative Contributors (Decrease Predicted Conversion)")
            for _, r in neg_df.iterrows():
                st.markdown(f"- **{r['feature']}**: `{r['shap_contribution']:.4f}` (Value: `{r['transformed_val']:.2f}`) — *contributes negatively to the model prediction*")

    st.markdown("---")
    st.markdown("#### 🌐 Global Model SHAP Beeswarm & LIME Explanations")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        beeswarm_img = os.path.join("reports", "figures", "06_shap_summary_beeswarm.png")
        if os.path.exists(beeswarm_img):
            st.image(beeswarm_img, caption="Global SHAP Summary Beeswarm Plot (X_test)", use_container_width=True)
        else:
            st.info("Global SHAP plot available at `reports/figures/06_shap_summary_beeswarm.png`.")

    with img_col2:
        lime_img = os.path.join("reports", "figures", "11_lime_local_high_prob.png")
        if os.path.exists(lime_img):
            st.image(lime_img, caption="Representative LIME Local Explanation", use_container_width=True)
        else:
            st.info("LIME plot available at `reports/figures/11_lime_local_high_prob.png`.")


# ==============================================================================
# TAB 5 — METHODOLOGY & DATA PROVENANCE
# ==============================================================================
with tab5:
    st.markdown("### Project Methodology, Dataset Provenance & Data Governance")

    st.markdown("#### 1. Business Problem & Project Objective")
    st.markdown("""
    E-commerce sales teams require accurate, data-driven lead prioritization to identify high-intent browsing sessions 
    and optimize real-time marketing interventions. This project implements an academic, end-to-end Machine Learning 
    lead-scoring pipeline that predicts session conversion probability using behavioral analytics.
    """)

    st.markdown("#### 2. Dataset Provenance & Specifications")
    st.markdown("""
    | Attribute | Specification |
    | :--- | :--- |
    | **Source Repository** | UCI Machine Learning Repository |
    | **Dataset Name** | Online Shoppers Purchasing Intention Dataset |
    | **Dataset ID** | 468 |
    | **Target Variable** | `Revenue` (Boolean: True=Converted, False=Non-Converted) |
    | **Raw Observations** | 12,330 |
    | **Raw Features** | 18 (10 Numerical, 8 Categorical) |
    | **Raw Duplicate Records** | 125 exact duplicate rows removed prior to split |
    | **Total Unique Sessions** | 12,205 |
    | **Training Split (`X_train`)** | 9,764 sessions (80%) |
    | **Holdout Test Split (`X_test`)** | 2,441 sessions (20%) |
    | **Dataset License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
    """)

    st.markdown("#### 3. Preprocessing & Leakage Prevention")
    st.markdown("""
    - **Raw Data Immutability**: `data/raw/online_shoppers_intention.csv` remains 100% untouched (SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`).
    - **Strict Fold Isolation**: All feature scaling, median imputation, and categorical one-hot encoding are fitted strictly within cross-validation training folds.
    - **Holdout Test Set Isolation**: `X_test` remained completely untouched until Phase 6 final evaluation.
    """)

    st.markdown("#### 4. PageValues Prediction-Time Framing")
    st.markdown("""
    PageValues is highly predictive in this dataset and is treated as an eligible prediction-time feature under the project's stated session-level prediction assumption. Because its strong predictive influence warrants additional scrutiny, a sensitivity experiment excluding PageValues was performed. The resulting model retained meaningful predictive performance but showed a substantial reduction in ROC-AUC.
    """)

    st.markdown("#### 5. Methodological Limitations")
    st.markdown("""
    1. **Non-Causal Interpretability**: SHAP and LIME values explain model predictions, not underlying causal consumer mechanisms.
    2. **Sampling Variance**: LIME local linear surrogates vary with perturbation sample sizes.
    3. **Operational Thresholds**: The 0.50 classification boundary is an illustrative default. Commercial deployment requires cost-sensitive threshold optimization.
    """)
