"""
Model Training and Cross-Validation Module for Sales Lead Conversion Probability Prediction.

This module executes leak-free 5-fold Stratified Cross-Validation on the training set (X_train),
benchmarking Logistic Regression and Random Forest models across feature configurations,
PageValues sensitivity toggles, and class-imbalance strategies.

All preprocessing and resampling (SMOTE) are fitted strictly within each training fold.
Holdout test data (X_test) is NEVER accessed during cross-validation or model selection.
"""

import os
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.feature_engineering import FeatureEngineeringTransformer
from src.data_preprocessing import build_preprocessing_pipeline


def evaluate_model_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic_regression",
    model_kwargs: Optional[Dict[str, Any]] = None,
    include_pagevalues: bool = True,
    page_count_config: str = "original",
    duration_config: str = "original",
    include_product_page_ratio: bool = False,
    include_avg_time_per_product_page: bool = False,
    imbalance_strategy: str = "none",  # 'none', 'balanced', 'smote'
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute 5-fold Stratified Cross-Validation strictly on X_train.
    
    Returns a dictionary of mean metrics, standard deviations, and fold details.
    """
    if model_kwargs is None:
        model_kwargs = {}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
        X_val, y_val = X_train.iloc[val_idx].copy(), y_train.iloc[val_idx].copy()

        # Build feature preprocessing pipeline
        prep_pipe = build_preprocessing_pipeline(
            model_type=model_type,
            include_pagevalues=include_pagevalues,
            page_count_config=page_count_config,
            duration_config=duration_config,
            include_product_page_ratio=include_product_page_ratio,
            include_avg_time_per_product_page=include_avg_time_per_product_page,
        )

        # Extract steps from prep_pipe
        fe_step = prep_pipe.named_steps["feature_engineer"]
        proc_step = prep_pipe.named_steps["preprocessor"]

        # Configure estimator
        curr_model_kwargs = model_kwargs.copy()
        if imbalance_strategy == "balanced":
            curr_model_kwargs["class_weight"] = "balanced"
        elif "class_weight" in curr_model_kwargs and imbalance_strategy != "balanced":
            curr_model_kwargs.pop("class_weight", None)

        if model_type == "logistic_regression":
            if "random_state" not in curr_model_kwargs:
                curr_model_kwargs["random_state"] = random_state
            if "max_iter" not in curr_model_kwargs:
                curr_model_kwargs["max_iter"] = 1000
            estimator = LogisticRegression(**curr_model_kwargs)
        else:
            if "random_state" not in curr_model_kwargs:
                curr_model_kwargs["random_state"] = random_state
            estimator = RandomForestClassifier(**curr_model_kwargs)

        # Build full fold pipeline cleanly without nested Pipelines
        steps = [
            ("feature_engineer", fe_step),
            ("preprocessor", proc_step),
        ]
        if imbalance_strategy == "smote":
            steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("classifier", estimator))

        full_pipe = ImbPipeline(steps)

        # Fit strictly on fold training data
        full_pipe.fit(X_tr, y_tr)

        # Predict probabilities on validation fold
        y_prob = full_pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Record metrics
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        fold_metrics["pr_auc"].append(average_precision_score(y_val, y_prob))
        fold_metrics["brier_score"].append(brier_score_loss(y_val, y_prob))
        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))

    elapsed_time = time.time() - start_time

    results = {
        "model_type": model_type,
        "include_pagevalues": include_pagevalues,
        "page_count_config": page_count_config,
        "duration_config": duration_config,
        "imbalance_strategy": imbalance_strategy,
        "model_kwargs": str(model_kwargs),
        "mean_roc_auc": float(np.mean(fold_metrics["roc_auc"])),
        "std_roc_auc": float(np.std(fold_metrics["roc_auc"])),
        "mean_pr_auc": float(np.mean(fold_metrics["pr_auc"])),
        "std_pr_auc": float(np.std(fold_metrics["pr_auc"])),
        "mean_brier": float(np.mean(fold_metrics["brier_score"])),
        "std_brier": float(np.std(fold_metrics["brier_score"])),
        "mean_accuracy": float(np.mean(fold_metrics["accuracy"])),
        "mean_precision": float(np.mean(fold_metrics["precision"])),
        "mean_recall": float(np.mean(fold_metrics["recall"])),
        "mean_f1": float(np.mean(fold_metrics["f1"])),
        "elapsed_time_sec": float(elapsed_time),
    }

    return results


def run_experiment_matrix(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Execute complete structured experiment matrix over X_train folds:
    - Logistic Regression & Random Forest
    - PageValues True vs False
    - Feature Configs A, B, C, D
    - Class Imbalance (None, Balanced, SMOTE)
    - Hyperparameter Tuning (C for LR, n_estimators/max_depth for RF)
    """
    experiments = []

    # 1. LOGISTIC REGRESSION EXPERIMENTS
    lr_c_values = [0.01, 0.1, 1.0, 10.0]
    configs = [
        ("original", "original"),
        ("total", "original"),
        ("original", "total"),
        ("total", "total"),
    ]
    imbalance_strategies = ["none", "balanced", "smote"]

    print("Running Logistic Regression CV Experiments...")
    for pv in [True, False]:
        for p_cfg, d_cfg in configs:
            for imb in imbalance_strategies:
                for c_val in lr_c_values:
                    res = evaluate_model_cv(
                        X_train=X_train,
                        y_train=y_train,
                        model_type="logistic_regression",
                        model_kwargs={"C": c_val, "solver": "lbfgs"},
                        include_pagevalues=pv,
                        page_count_config=p_cfg,
                        duration_config=d_cfg,
                        imbalance_strategy=imb,
                        random_state=random_state,
                    )
                    experiments.append(res)

    # 2. RANDOM FOREST EXPERIMENTS
    rf_params_list = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    ]

    print("Running Random Forest CV Experiments...")
    for pv in [True, False]:
        for p_cfg, d_cfg in configs:
            for imb in imbalance_strategies:
                for rf_kw in rf_params_list:
                    res = evaluate_model_cv(
                        X_train=X_train,
                        y_train=y_train,
                        model_type="random_forest",
                        model_kwargs=rf_kw,
                        include_pagevalues=pv,
                        page_count_config=p_cfg,
                        duration_config=d_cfg,
                        imbalance_strategy=imb,
                        random_state=random_state,
                    )
                    experiments.append(res)

    df_results = pd.DataFrame(experiments)
    df_results = df_results.sort_values(by="mean_roc_auc", ascending=False).reset_index(drop=True)
    return df_results
