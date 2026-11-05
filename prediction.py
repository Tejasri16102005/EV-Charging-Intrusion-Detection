"""
Prediction Module for Live Intrusion Detection.
Executes inference on uploaded network traffic CSV files using saved model artifacts.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from config import (
    ISOLATION_FOREST_FILE,
    XGBOOST_MODEL_FILE,
    MODEL_METADATA_FILE,
    PRIMARY_UI_CLASSES
)
from modules.preprocessing import DataPreprocessor


class IntrusionPredictor:
    """
    Manages end-to-end inference for newly uploaded EV network traffic CSVs.
    Strictly follows the research pipeline:
    Input CSV -> Reindex 61 Features -> Median Imputer -> Isolation Forest Anomaly Score
    -> 62nd Feature Stack -> XGBoost Classification -> Confidence & Severity Analysis.
    """

    def __init__(self):
        self.preprocessor: Optional[DataPreprocessor] = None
        self.isolation_model = None
        self.xgb_model: Optional[XGBClassifier] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.is_ready = False

    def check_artifacts_available(self) -> Tuple[bool, list]:
        """Checks if all required model artifacts exist on disk."""
        missing = []
        if not ISOLATION_FOREST_FILE.exists():
            missing.append("isolation_forest.pkl")
        if not XGBOOST_MODEL_FILE.exists():
            missing.append("xgboost_model.json")
        try:
            DataPreprocessor.load()
        except Exception:
            missing.extend(["imputer.pkl", "label_encoder.pkl", "feature_columns.json"])

        return len(missing) == 0, missing

    def load_artifacts(self) -> "IntrusionPredictor":
        """Loads preprocessor, Isolation Forest, and XGBoost models."""
        ready, missing = self.check_artifacts_available()
        if not ready:
            raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

        self.preprocessor = DataPreprocessor.load()
        self.isolation_model = joblib.load(ISOLATION_FOREST_FILE)

        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(str(XGBOOST_MODEL_FILE))

        if MODEL_METADATA_FILE.exists():
            import json
            with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.is_ready = True
        return self

    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates the structure and compatibility of an uploaded CSV file.
        """
        if self.preprocessor is None:
            self.load_artifacts()

        expected_features = self.preprocessor.feature_columns
        present_features = [col for col in expected_features if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_features]

        coverage_ratio = len(present_features) / len(expected_features) if expected_features else 0

        # Check missing values
        null_count = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1] if df.shape[0] > 0 else 1
        null_percentage = round((null_count / total_cells) * 100, 2)

        is_compatible = coverage_ratio >= 0.50  # At least 50% features present to be reliably imputable

        return {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "expected_count": len(expected_features),
            "present_count": len(present_features),
            "missing_count": len(missing_features),
            "missing_columns": missing_features,
            "extra_columns": extra_columns,
            "coverage_percentage": round(coverage_ratio * 100, 1),
            "null_count": null_count,
            "null_percentage": null_percentage,
            "is_compatible": is_compatible
        }

    def predict(
        self,
        df: pd.DataFrame,
        filename: str = "uploaded_traffic.csv"
    ) -> Dict[str, Any]:
        """
        Executes live inference on the uploaded DataFrame.
        Returns detailed per-record predictions and high-level summary metrics.
        """
        if not self.is_ready:
            self.load_artifacts()

        start_time = time.time()
        original_df = df.copy()

        # Step 1: Align columns to 61 standard features and impute
        X_clean = self.preprocessor.transform_features(df)

        # Step 2: Isolation Forest Anomaly Scoring
        # Positive score is more anomalous with -decision_function
        if_scores = -self.isolation_model.decision_function(X_clean)
        if_preds = self.isolation_model.predict(X_clean)
        if_status = np.where(if_preds == -1, "Anomaly", "Normal")

        # Step 3: Hybrid Feature Stacking
        X_hybrid = self.preprocessor.create_hybrid_features(X_clean, if_scores)

        # Step 4: XGBoost Multiclass Classification
        y_prob = self.xgb_model.predict_proba(X_hybrid)
        y_pred_codes = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1) * 100.0

        # Decode attack names
        predicted_classes = self.preprocessor.decode_labels(y_pred_codes)

        # Step 5: Final Normal / Malicious Classification
        final_classifications = np.where(predicted_classes == "Normal", "Normal", "Malicious")

        inference_time = time.time() - start_time

        # Build detailed record results DataFrame
        records_df = pd.DataFrame({
            "Record Number": np.arange(1, len(df) + 1),
            "Original Filename": filename,
            "IF Anomaly Score": np.round(if_scores, 4),
            "IF Status": if_status,
            "XGBoost Predicted Class": predicted_classes,
            "Final Classification": final_classifications,
            "Confidence (%)": np.round(confidences, 2)
        })

        # Summary Metrics
        total_records = len(df)
        normal_mask = final_classifications == "Normal"
        normal_count = int(np.sum(normal_mask))
        malicious_count = int(total_records - normal_count)

        normal_pct = round((normal_count / total_records) * 100, 2) if total_records > 0 else 0
        malicious_pct = round((malicious_count / total_records) * 100, 2) if total_records > 0 else 0

        # Attack-wise breakdown
        class_series = pd.Series(predicted_classes)
        attack_counts_all = class_series.value_counts().to_dict()

        # Primary 6 categories table
        primary_table = []
        for cat in PRIMARY_UI_CLASSES:
            cnt = int(attack_counts_all.get(cat, 0))
            pct = round((cnt / total_records) * 100, 2) if total_records > 0 else 0
            cls_type = "Benign" if cat == "Normal" else "Malicious"
            primary_table.append({
                "Attack Name": cat,
                "Detected Records": cnt,
                "Percentage (%)": pct,
                "Classification": cls_type
            })

        primary_df = pd.DataFrame(primary_table)

        # Other attack types outside the 6 categories
        other_attacks = {k: v for k, v in attack_counts_all.items() if k not in PRIMARY_UI_CLASSES}

        # Most frequent attack category (excluding Normal)
        malicious_series = class_series[class_series != "Normal"]
        most_frequent_attack = malicious_series.mode()[0] if not malicious_series.empty else "None"

        # Prepared Export DataFrame: keeps all original columns and prepends/appends prediction columns
        export_df = original_df.copy()
        export_df["IF_Anomaly_Score"] = np.round(if_scores, 4)
        export_df["IF_Status"] = if_status
        export_df["Predicted_Class"] = predicted_classes
        export_df["Final_Classification"] = final_classifications
        export_df["Prediction_Confidence_Pct"] = np.round(confidences, 2)
        export_df["Inference_Timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "summary": {
                "filename": filename,
                "total_records": total_records,
                "normal_count": normal_count,
                "normal_pct": normal_pct,
                "malicious_count": malicious_count,
                "malicious_pct": malicious_pct,
                "most_frequent_attack": most_frequent_attack,
                "inference_time_sec": round(inference_time, 3),
                "throughput_records_per_sec": round(total_records / max(inference_time, 0.001), 1)
            },
            "records_df": records_df,
            "primary_table_df": primary_df,
            "all_attack_counts": attack_counts_all,
            "other_attacks": other_attacks,
            "export_df": export_df
        }

