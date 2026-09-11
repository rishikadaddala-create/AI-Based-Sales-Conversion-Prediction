"""
Data Preprocessing Module for Sales Lead Conversion Probability Prediction.

This module manages:
1. Loading raw immutable data from data/raw/.
2. Pre-split deduplication (dropping exact duplicate records prior to train/test split).
3. Stratified train/test splitting (saving processed files to data/processed/).
4. Building algorithm-specific, configuration-driven Scikit-learn Pipelines:
   - Logistic Regression Pipeline: Numerical scaling + log1p for zero-heavy durations + OneHotEncoder for categoricals.
   - Random Forest Pipeline: Raw unscaled continuous distributions + OneHotEncoder for categoricals.
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from src.feature_engineering import FeatureEngineeringTransformer


RAW_DATA_PATH = os.path.join("data", "raw", "online_shoppers_intention.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def _convert_bool_to_int(x):
    """Picklable top-level function for converting boolean columns to integer 0/1."""
    return x.astype(int)


def _bool_feature_names_out(transformer, input_features):
    """Feature names out generator for boolean FunctionTransformer."""
    return input_features


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load immutable raw dataset from specified filepath.
    
    Parameters
    ----------
    filepath : str
        Path to raw CSV dataset.
        
    Returns
    -------
    pd.DataFrame
        Raw dataset DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at '{filepath}'.")
    df = pd.read_csv(filepath)
    return df


def handle_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Perform pre-split deduplication.
    
    "The dataset contains 125 exact duplicate observations. Because their origin
    cannot be established from the available data, they will be treated as duplicate
    records rather than assuming they represent repeated telemetry events."
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
        
    Returns
    -------
    Tuple[pd.DataFrame, int]
        (Deduplicated DataFrame, Number of duplicate rows removed)
    """
    num_duplicates = int(df.duplicated().sum())
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    return df_dedup, num_duplicates


def split_data(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    test_size: float = 0.2,
    random_state: int = 42,
    save_processed: bool = True,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split on deduplicated data and optionally save to disk.
    
    Parameters
    ----------
    df : pd.DataFrame
        Deduplicated dataset.
    target_col : str, default='Revenue'
        Name of target column.
    test_size : float, default=0.2
        Fraction of data reserved for holdout test set.
    random_state : int, default=42
        Random seed for reproducibility.
    save_processed : bool, default=True
        Whether to write train.csv and test.csv into data/processed/.
    processed_dir : str
        Target directory for processed CSV files.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if save_processed:
        os.makedirs(processed_dir, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(
    model_type: str = "logistic_regression",
    include_pagevalues: bool = True,
    page_count_config: str = "original",
    duration_config: str = "original",
    include_product_page_ratio: bool = False,
    include_avg_time_per_product_page: bool = False,
) -> Pipeline:
    """
    Build complete Scikit-learn Pipeline incorporating configuration-driven feature engineering
    and algorithm-specific column transformations.
    
    Parameters
    ----------
    model_type : {'logistic_regression', 'random_forest'}
        Algorithm choice determining numerical scaling and transformation rules.
    include_pagevalues : bool, default=True
        Include PageValues in feature set.
    page_count_config : {'original', 'total'}
        Use component page counts or aggregated Total_Pages.
    duration_config : {'original', 'total'}
        Use component durations or aggregated Total_Duration.
    include_product_page_ratio : bool
        Include Product_Page_Ratio feature.
    include_avg_time_per_product_page : bool
        Include Avg_Time_Per_Product_Page feature.
        
    Returns
    -------
    Pipeline
        Complete Scikit-learn Pipeline (FeatureEngineeringTransformer + ColumnTransformer).
    """
    valid_models = {"logistic_regression", "random_forest"}
    if model_type not in valid_models:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {valid_models}.")

    # Instantiate feature engineer
    fe_transformer = FeatureEngineeringTransformer(
        include_pagevalues=include_pagevalues,
        page_count_config=page_count_config,
        duration_config=duration_config,
        include_product_page_ratio=include_product_page_ratio,
        include_avg_time_per_product_page=include_avg_time_per_product_page,
    )

    # Determine column sets dynamically based on feature engineering parameters
    cat_nominal_features = [
        "Month",
        "VisitorType",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
    ]
    bool_features = ["Weekend"]

    # Base numerical features
    num_std_features: List[str] = ["SpecialDay", "BounceRates", "ExitRates"]
    num_log_features: List[str] = []
    num_raw_features: List[str] = ["SpecialDay", "BounceRates", "ExitRates"]

    # Page count features
    if page_count_config == "original":
        num_std_features.extend(["Administrative", "Informational", "ProductRelated"])
        num_raw_features.extend(["Administrative", "Informational", "ProductRelated"])
    elif page_count_config == "total":
        num_std_features.append("Total_Pages")
        num_raw_features.append("Total_Pages")

    # Duration features
    if duration_config == "original":
        if model_type == "logistic_regression":
            num_log_features.extend(
                ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
            )
        else:
            num_raw_features.extend(
                ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
            )
    elif duration_config == "total":
        if model_type == "logistic_regression":
            num_log_features.append("Total_Duration")
        else:
            num_raw_features.append("Total_Duration")

    # PageValues handling
    if include_pagevalues:
        num_std_features.append("PageValues")
        num_raw_features.append("PageValues")

    # Ratio enhancements
    if include_product_page_ratio:
        num_std_features.append("Product_Page_Ratio")
        num_raw_features.append("Product_Page_Ratio")
    if include_avg_time_per_product_page:
        num_std_features.append("Avg_Time_Per_Product_Page")
        num_raw_features.append("Avg_Time_Per_Product_Page")

    # Construct ColumnTransformer based on model algorithm choice
    if model_type == "logistic_regression":
        transformers = [
            (
                "num_std",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_std_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_nominal_features,
            ),
            ("bool", FunctionTransformer(_convert_bool_to_int, feature_names_out=_bool_feature_names_out), bool_features),
        ]
        if num_log_features:
            transformers.append(
                (
                    "num_log",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("log1p", FunctionTransformer(np.log1p)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_log_features,
                )
            )
    else:  # random_forest
        transformers = [
            (
                "num_raw",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_raw_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_nominal_features,
            ),
            ("bool", FunctionTransformer(_convert_bool_to_int), bool_features),
        ]

    column_transformer = ColumnTransformer(transformers=transformers)

    full_pipeline = Pipeline(
        [
            ("feature_engineer", fe_transformer),
            ("preprocessor", column_transformer),
        ]
    )

    return full_pipeline
