"""
Preprocessing module for EV Charging Intrusion Detection System.
Handles feature alignment, median imputation, label encoding, and hybrid feature synthesis.
"""

import json
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config import (
    IMPUTER_FILE,
    LABEL_ENCODER_FILE,
    FEATURE_COLUMNS_FILE,
    FEATURE_COLUMNS
)


class DataPreprocessor:
    """
    Manages data preprocessing lifecycle:
    - Feature column enforcement (61 standard NFStream flow features)
    - Median imputation of missing/infinite values
    - Label encoding for multiclass classification
    - Hybrid feature expansion (stacking IF anomaly scores)
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """
        Fits the imputer on X and label encoder on y.
        """
        # Ensure correct column ordering
        X_aligned = self._align_columns(X)
        X_numeric = X_aligned.apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        self.imputer.fit(X_numeric)

        if y is not None:
            self.label_encoder.fit(y)

        self.is_fitted = True
        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Aligns columns and imputes missing values using the fitted imputer.
        Returns a clean NumPy array of shape (N, 61).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming features.")

        X_aligned = self._align_columns(X)
        X_numeric = X_aligned.apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        return self.imputer.transform(X_numeric)

    def fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Fits and transforms input features."""
        self.fit(X)
        return self.transform_features(X)

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """Transforms string labels to integer classes."""
        return self.label_encoder.transform(y)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Transforms integer classes back to human-readable strings."""
        return self.label_encoder.inverse_transform(y_encoded.astype(int))

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures X has exactly the expected columns in the exact order,
        filling missing columns with NaN.
        """
        aligned = X.reindex(columns=self.feature_columns)
        return aligned

    @staticmethod
    def create_hybrid_features(X_clean: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Appends the Isolation Forest anomaly score as the 62nd feature column.
        Preserves exact feature engineering from research Colab notebook.
        """
        scores_col = anomaly_scores.reshape(-1, 1)
        return np.column_stack([X_clean, scores_col])

    def save(
        self,
        imputer_path: Path = IMPUTER_FILE,
        encoder_path: Path = LABEL_ENCODER_FILE,
        columns_path: Path = FEATURE_COLUMNS_FILE
    ):
        """Saves fitted preprocessor artifacts to disk."""
        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.label_encoder, encoder_path)
        with open(columns_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)

    @classmethod
    def load(
        cls,
        imputer_path: Path = IMPUTER_FILE,
        encoder_path: Path = LABEL_ENCODER_FILE,
        columns_path: Path = FEATURE_COLUMNS_FILE
    ) -> "DataPreprocessor":
        """Loads fitted preprocessor artifacts from disk."""
        if not imputer_path.exists() or not encoder_path.exists() or not columns_path.exists():
            raise FileNotFoundError("Preprocessor artifacts not found on disk.")

        with open(columns_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        preprocessor = cls(feature_columns=feature_cols)
        preprocessor.imputer = joblib.load(imputer_path)
        preprocessor.label_encoder = joblib.load(encoder_path)
        preprocessor.is_fitted = True
        return preprocessor

