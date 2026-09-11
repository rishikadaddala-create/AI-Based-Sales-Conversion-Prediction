"""
Feature Engineering Module for Sales Lead Conversion Probability Prediction.

This module implements programmatic, configuration-driven feature engineering
transformations for the UCI Online Shoppers Purchasing Intention Dataset.

It ensures strict prevention of data leakage and eliminates redundancy by
evaluating aggregate features as explicit configuration alternatives rather than
combining aggregate sums with component variables.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for configuration-driven feature engineering.
    
    Parameters
    ----------
    include_pagevalues : bool, default=True
        Whether to include the PageValues feature. If False, PageValues and any
        features derived directly from it are strictly excluded.
    page_count_config : {'original', 'total'}, default='original'
        - 'original': Retains Administrative, Informational, and ProductRelated page counts.
        - 'total': Replaces the three component page counts with an aggregated Total_Pages feature.
    duration_config : {'original', 'total'}, default='original'
        - 'original': Retains Administrative_Duration, Informational_Duration, and ProductRelated_Duration.
        - 'total': Replaces the three component duration features with an aggregated Total_Duration feature.
    include_product_page_ratio : bool, default=False
        Whether to create Product_Page_Ratio = ProductRelated / (Total_Pages + 1e-5).
    include_avg_time_per_product_page : bool, default=False
        Whether to create Avg_Time_Per_Product_Page = ProductRelated_Duration / (ProductRelated + 1e-5).
    """

    def __init__(
        self,
        include_pagevalues: bool = True,
        page_count_config: str = "original",
        duration_config: str = "original",
        include_product_page_ratio: bool = False,
        include_avg_time_per_product_page: bool = False,
    ) -> None:
        self.include_pagevalues = include_pagevalues
        self.page_count_config = page_count_config
        self.duration_config = duration_config
        self.include_product_page_ratio = include_product_page_ratio
        self.include_avg_time_per_product_page = include_avg_time_per_product_page

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        valid_configs = {"original", "total"}
        if self.page_count_config not in valid_configs:
            raise ValueError(
                f"Invalid page_count_config '{self.page_count_config}'. Must be one of {valid_configs}."
            )
        if self.duration_config not in valid_configs:
            raise ValueError(
                f"Invalid duration_config '{self.duration_config}'. Must be one of {valid_configs}."
            )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringTransformer":
        """
        Fit method (stateless transformer).
        
        Returns self.
        """
        self._validate_config()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply configuration-driven feature engineering to DataFrame X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing session features.
            
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with selected feature configuration.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Create explicit shallow copy to prevent mutating caller DataFrame
        X_out = X.copy()

        # 1. Strict PageValues Handling
        if not self.include_pagevalues:
            if "PageValues" in X_out.columns:
                X_out = X_out.drop(columns=["PageValues"])

        # Identify component columns if present
        comp_pages = ["Administrative", "Informational", "ProductRelated"]
        comp_durations = [
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration",
        ]

        # Calculate Total_Pages for denominator if needed by ratio or if page_count_config=='total'
        has_comp_pages = all(col in X_out.columns for col in comp_pages)
        
        if has_comp_pages:
            total_pages_series = X_out["Administrative"] + X_out["Informational"] + X_out["ProductRelated"]
        elif "Total_Pages" in X_out.columns:
            total_pages_series = X_out["Total_Pages"]
        else:
            total_pages_series = None

        # 2. Ratio Enhancements (Calculated before component drop if needed)
        if self.include_product_page_ratio:
            if "ProductRelated" in X_out.columns and total_pages_series is not None:
                X_out["Product_Page_Ratio"] = X_out["ProductRelated"] / (total_pages_series + 1e-5)
            elif "ProductRelated" not in X_out.columns:
                raise KeyError("Cannot compute Product_Page_Ratio: 'ProductRelated' column missing.")

        if self.include_avg_time_per_product_page:
            if "ProductRelated_Duration" in X_out.columns and "ProductRelated" in X_out.columns:
                X_out["Avg_Time_Per_Product_Page"] = X_out["ProductRelated_Duration"] / (
                    X_out["ProductRelated"] + 1e-5
                )

        # 3. Page Count Configuration (Alternatives, not simultaneous co-existence)
        if self.page_count_config == "total":
            if has_comp_pages:
                X_out["Total_Pages"] = total_pages_series
                X_out = X_out.drop(columns=comp_pages)
        elif self.page_count_config == "original":
            # If original, keep comp_pages and do NOT keep Total_Pages
            if "Total_Pages" in X_out.columns:
                X_out = X_out.drop(columns=["Total_Pages"])

        # 4. Duration Configuration (Alternatives, not simultaneous co-existence)
        has_comp_durations = all(col in X_out.columns for col in comp_durations)
        if self.duration_config == "total":
            if has_comp_durations:
                X_out["Total_Duration"] = (
                    X_out["Administrative_Duration"]
                    + X_out["Informational_Duration"]
                    + X_out["ProductRelated_Duration"]
                )
                X_out = X_out.drop(columns=comp_durations)
        elif self.duration_config == "original":
            if "Total_Duration" in X_out.columns:
                X_out = X_out.drop(columns=["Total_Duration"])

        return X_out


def get_feature_config_summary(
    include_pagevalues: bool = True,
    page_count_config: str = "original",
    duration_config: str = "original",
    include_product_page_ratio: bool = False,
    include_avg_time_per_product_page: bool = False,
) -> Dict[str, Any]:
    """
    Generate metadata dictionary summarizing active feature configuration.
    """
    return {
        "include_pagevalues": include_pagevalues,
        "page_count_config": page_count_config,
        "duration_config": duration_config,
        "include_product_page_ratio": include_product_page_ratio,
        "include_avg_time_per_product_page": include_avg_time_per_product_page,
    }
