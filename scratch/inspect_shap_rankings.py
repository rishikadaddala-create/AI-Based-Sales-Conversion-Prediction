import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap

sys.path.insert(0, ".")
from src.explain import get_transformed_data_and_feature_names

pipeline = joblib.load('models/best_lead_scoring_model.joblib')
test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop(columns=['Revenue'])
y_test = test_df['Revenue']

X_trans, clean_names, X_df = get_transformed_data_and_feature_names(pipeline, X_test)
estimator = pipeline.named_steps['classifier']

explainer = shap.TreeExplainer(estimator)
shap_vals_raw = explainer.shap_values(X_trans)

if isinstance(shap_vals_raw, list):
    shap_vals = shap_vals_raw[1]
elif isinstance(shap_vals_raw, np.ndarray) and shap_vals_raw.ndim == 3:
    shap_vals = shap_vals_raw[:, :, 1]
else:
    shap_vals = shap_vals_raw

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
importance_df = pd.DataFrame({'feature': clean_names, 'mean_abs_shap': mean_abs_shap}).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

print('=== TOP 15 GLOBAL SHAP FEATURES ===')
print(importance_df.head(15).to_string())

rf_importances = estimator.feature_importances_
rf_df = pd.DataFrame({'feature': clean_names, 'rf_mdi_importance': rf_importances}).sort_values(by='rf_mdi_importance', ascending=False).reset_index(drop=True)
print('\n=== TOP 10 RF MDI IMPORTANCE FEATURES ===')
print(rf_df.head(10).to_string())
