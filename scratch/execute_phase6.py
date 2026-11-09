import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '.')

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_preprocessing import build_preprocessing_pipeline

def run_phase6():
    print("==========================================================")
    print("=== PHASE 6: FINAL MODEL LOCKING & HOLDOUT EVALUATION ===")
    print("==========================================================")

    # 1. Provenance Verification
    raw_path = os.path.join("data", "raw", "online_shoppers_intention.csv")
    with open(raw_path, "rb") as f:
        raw_hash = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
    assert raw_hash == expected_hash, f"Raw dataset hash mismatch! {raw_hash} != {expected_hash}"
    print(f"[OK] Raw dataset provenance verified. SHA-256: {raw_hash}")

    # 2. Load Processed Datasets
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) == 9764, f"Unexpected train rows: {len(train_df)}"
    assert len(test_df) == 2441, f"Unexpected test rows: {len(test_df)}"

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    print(f"[OK] Training set loaded: {X_train.shape[0]} rows (Conversion rate: {y_train.mean()*100:.2f}%)")
    print(f"[OK] Holdout test set loaded: {X_test.shape[0]} rows (Conversion rate: {y_test.mean()*100:.2f}%)")

    # 3. Build & Fit Locked Model Pipeline on Full X_train
    print("\n--- Fitting Locked Pipeline on Full X_train ---")
    prep_pipe = build_preprocessing_pipeline(
        model_type="random_forest",
        include_pagevalues=True,
        page_count_config="original",
        duration_config="original"
    )

    fe_step = prep_pipe.named_steps["feature_engineer"]
    proc_step = prep_pipe.named_steps["preprocessor"]

    rf_kwargs = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": None,
        "random_state": 42
    }

    estimator = RandomForestClassifier(**rf_kwargs)

    pipeline = ImbPipeline([
        ("feature_engineer", fe_step),
        ("preprocessor", proc_step),
        ("classifier", estimator)
    ])

    pipeline.fit(X_train, y_train)
    print("[OK] Locked Random Forest Pipeline fitted on full X_train successfully.")

    # 4. Holdout Test Set Evaluation (FIRST & ONLY EVALUATION)
    print("\n--- Evaluating Locked Pipeline on Untouched Holdout Test Set ---")
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.50).astype(int)

    test_roc_auc = roc_auc_score(y_test, y_prob_test)
    test_pr_auc = average_precision_score(y_test, y_prob_test)
    test_brier = brier_score_loss(y_test, y_prob_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    pos_count = int(y_test.sum())
    neg_count = int((y_test == 0).sum())
    test_conv_rate = float(y_test.mean())

    print("\n=== FINAL HOLDOUT TEST SET PERFORMANCE METRICS ===")
    print(f"Primary Metric - ROC-AUC:            {test_roc_auc:.4f}")
    print(f"Secondary Metric - PR-AUC (Avg Prec): {test_pr_auc:.4f}")
    print(f"Secondary Metric - Brier Score:       {test_brier:.4f}")
    print(f"Accuracy:                            {test_accuracy:.4f}")
    print(f"Precision (at 0.50 threshold):       {test_precision:.4f}")
    print(f"Recall (at 0.50 threshold):          {test_recall:.4f}")
    print(f"F1-Score (at 0.50 threshold):        {test_f1:.4f}")
    print(f"\nConfusion Matrix (Default Threshold 0.50):\n{cm}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    print(f"\nTest Set Target Demographics:")
    print(f"  Positive Class Count (Revenue=1): {pos_count}")
    print(f"  Negative Class Count (Revenue=0): {neg_count}")
    print(f"  Actual Conversion Rate:          {test_conv_rate*100:.2f}%")

    # 5. Reliability / Calibration Assessment on Test Set
    fig_dir = os.path.join("reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    calib_plot_path = os.path.join(fig_dir, "05_test_calibration_curve.png")

    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (Ideal)")
    plt.plot(prob_pred, prob_true, "s-", color="#1f77b4", label=f"Locked RF Model (Brier: {test_brier:.4f})")
    plt.xlabel("Mean Predicted Probability (Holdout Test Set)")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Reliability Diagram (Probability Calibration) on Holdout Test Set", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(calib_plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Test calibration curve plot saved to '{calib_plot_path}'")

    # 6. Save Model Artifact & Metadata
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_lead_scoring_model.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(pipeline, model_path)
    print(f"[OK] Production model pipeline saved to '{model_path}'")

    metadata = {
        "model_name": "RandomForestClassifier",
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "hyperparameters": rf_kwargs,
        "feature_config": {
            "include_pagevalues": True,
            "page_count_config": "original",
            "duration_config": "original"
        },
        "training_data_ provenance": {
            "raw_dataset_sha256": raw_hash,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "random_state": 42
        },
        "cv_performance": {
            "mean_cv_roc_auc": 0.926655,
            "std_cv_roc_auc": 0.005868,
            "mean_cv_pr_auc": 0.747420,
            "mean_cv_brier": 0.073888,
            "mean_cv_precision": 0.807675,
            "mean_cv_recall": 0.494754
        },
        "holdout_test_performance": {
            "test_roc_auc": float(test_roc_auc),
            "test_pr_auc": float(test_pr_auc),
            "test_brier": float(test_brier),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "positive_count": pos_count,
            "negative_count": neg_count,
            "conversion_rate": test_conv_rate
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Model metadata saved to '{metadata_path}'")

    # 7. Reproducibility & Reload Check
    print("\n--- Executing Reload & Prediction Verification Check ---")
    reloaded_pipeline = joblib.load(model_path)
    reloaded_prob = reloaded_pipeline.predict_proba(X_test)[:, 1]
    reloaded_pred = reloaded_pipeline.predict(X_test)

    np.testing.assert_allclose(y_prob_test, reloaded_prob, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal((y_prob_test >= 0.50).astype(int), reloaded_pred)
    print("[OK] RELOAD VERIFICATION SUCCESSFUL: Reloaded model probabilities & predictions match in-memory model identically (0 difference).")

    return metadata

if __name__ == "__main__":
    run_phase6()
