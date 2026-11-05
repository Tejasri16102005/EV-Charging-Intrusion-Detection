"""
Model Training Module for Hybrid Intrusion Detection System.
Trains Isolation Forest, Individual XGBoost, and Hybrid IF+XGBoost models
strictly reproducing the experimental methodology from the Colab research notebook.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from config import (
    ISOLATION_FOREST_FILE,
    XGBOOST_MODEL_FILE,
    XGBOOST_INDIVIDUAL_FILE,
    MODEL_METADATA_FILE,
    EVALUATION_RESULTS_FILE,
    RANDOM_SEED,
    IF_PARAMS,
    XGB_PARAMS
)
from modules.preprocessing import DataPreprocessor
from modules.data_loader import load_dataset_for_training


class ModelTrainer:
    """
    Executes the training and scientific evaluation pipeline for:
    1. Isolation Forest (Unsupervised Anomaly Detection)
    2. Individual XGBoost (Supervised 61-feature Baseline)
    3. Hybrid IF + XGBoost (62-feature Anomaly-Augmented Classifier)
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.preprocessor = DataPreprocessor()
        self.isolation_model: Optional[IsolationForest] = None
        self.xgb_individual: Optional[XGBClassifier] = None
        self.xgb_hybrid: Optional[XGBClassifier] = None
        self.evaluation_results: Dict[str, Any] = {}
        self.training_metadata: Dict[str, Any] = {}

    def train_and_evaluate(
        self,
        dataset_dir: Optional[Path] = None,
        test_size: float = 0.20,
        sample_fraction: Optional[float] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Runs the full research training pipeline:
        1. Loads 28 CSV files
        2. Fits preprocessor (median imputer, label encoder)
        3. Stratified split (80% train, 20% test)
        4. Trains Isolation Forest and extracts anomaly scores
        5. Trains Baseline XGBoost (61 features)
        6. Constructs 62nd hybrid feature (stacked anomaly score)
        7. Trains Hybrid XGBoost (62 features)
        8. Evaluates all 3 models on the identical test split
        9. Serializes artifacts to disk
        """
        start_total = time.time()

        if progress_callback:
            progress_callback("Loading dataset from 28 CSV files...", 0.10)

        X, y, metadata_df = load_dataset_for_training(
            dataset_dir=dataset_dir,
            sample_fraction=sample_fraction
        )

        total_records = len(X)

        if progress_callback:
            progress_callback(f"Preprocessing {total_records:,} records and splitting train/test...", 0.25)

        # Stratified train/test split matching research notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y
        )

        # Fit preprocessor on training data only (strictly prevent data leakage)
        self.preprocessor.fit(X_train, y_train)
        X_train_clean = self.preprocessor.transform_features(X_train)
        X_test_clean = self.preprocessor.transform_features(X_test)

        y_train_encoded = self.preprocessor.encode_labels(y_train)
        y_test_encoded = self.preprocessor.encode_labels(y_test)

        classes = list(self.preprocessor.label_encoder.classes_)
        num_classes = len(classes)

        # ---------------------------------------------------------
        # 1. Isolation Forest Training
        # ---------------------------------------------------------
        if progress_callback:
            progress_callback("Training Isolation Forest Anomaly Detector...", 0.40)

        start_if = time.time()
        self.isolation_model = IsolationForest(**IF_PARAMS)
        self.isolation_model.fit(X_train_clean)
        time_if_train = time.time() - start_if

        # Decision function: higher negative value = more anomalous
        # Research Colab uses: -isolation_model.decision_function(X)
        IF_train_scores = -self.isolation_model.decision_function(X_train_clean)
        IF_test_scores = -self.isolation_model.decision_function(X_test_clean)

        # Binary anomaly evaluation for Isolation Forest:
        # Ground truth: Normal = 0, Attack = 1
        y_test_binary = (y_test != "Normal").astype(int)
        # IF prediction: 1 = Inlier (0 = Normal), -1 = Outlier (1 = Attack)
        if_preds = self.isolation_model.predict(X_test_clean)
        y_if_binary = (if_preds == -1).astype(int)

        if_acc = float(accuracy_score(y_test_binary, y_if_binary) * 100)
        if_prec = float(precision_score(y_test_binary, y_if_binary, zero_division=0) * 100)
        if_rec = float(recall_score(y_test_binary, y_if_binary, zero_division=0) * 100)
        if_f1 = float(f1_score(y_test_binary, y_if_binary, zero_division=0) * 100)

        # ---------------------------------------------------------
        # 2. Individual Baseline XGBoost Training (61 Features)
        # ---------------------------------------------------------
        if progress_callback:
            progress_callback("Training Baseline XGBoost Classifier (61 features)...", 0.55)

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_encoded)

        start_xgb_ind = time.time()
        xgb_ind_params = dict(XGB_PARAMS)
        xgb_ind_params["num_class"] = num_classes

        self.xgb_individual = XGBClassifier(**xgb_ind_params)
        self.xgb_individual.fit(X_train_clean, y_train_encoded, sample_weight=sample_weights)
        time_xgb_ind_train = time.time() - start_xgb_ind

        y_pred_ind = self.xgb_individual.predict(X_test_clean)

        ind_acc = float(accuracy_score(y_test_encoded, y_pred_ind) * 100)
        ind_prec = float(precision_score(y_test_encoded, y_pred_ind, average="macro", zero_division=0) * 100)
        ind_rec = float(recall_score(y_test_encoded, y_pred_ind, average="macro", zero_division=0) * 100)
        ind_f1 = float(f1_score(y_test_encoded, y_pred_ind, average="macro", zero_division=0) * 100)

        # ---------------------------------------------------------
        # 3. Hybrid IF + XGBoost Training (62 Features)
        # ---------------------------------------------------------
        if progress_callback:
            progress_callback("Synthesizing hybrid feature space and training Hybrid XGBoost...", 0.70)

        X_train_hybrid = self.preprocessor.create_hybrid_features(X_train_clean, IF_train_scores)
        X_test_hybrid = self.preprocessor.create_hybrid_features(X_test_clean, IF_test_scores)

        start_xgb_hyb = time.time()
        xgb_hyb_params = dict(XGB_PARAMS)
        xgb_hyb_params["num_class"] = num_classes

        self.xgb_hybrid = XGBClassifier(**xgb_hyb_params)
        self.xgb_hybrid.fit(X_train_hybrid, y_train_encoded, sample_weight=sample_weights)
        time_xgb_hyb_train = time.time() - start_xgb_hyb

        # Inference timing
        start_test = time.time()
        y_pred_hyb = self.xgb_hybrid.predict(X_test_hybrid)
        test_duration = time.time() - start_test

        # Hybrid Multiclass Metrics
        hyb_acc = float(accuracy_score(y_test_encoded, y_pred_hyb) * 100)
        hyb_prec_macro = float(precision_score(y_test_encoded, y_pred_hyb, average="macro", zero_division=0) * 100)
        hyb_rec_macro = float(recall_score(y_test_encoded, y_pred_hyb, average="macro", zero_division=0) * 100)
        hyb_f1_macro = float(f1_score(y_test_encoded, y_pred_hyb, average="macro", zero_division=0) * 100)
        hyb_f1_weighted = float(f1_score(y_test_encoded, y_pred_hyb, average="weighted", zero_division=0) * 100)

        # Binary False Positive Rate calculation (Benign misclassified as Attack)
        benign_indices = np.where(y_test_encoded == self.preprocessor.label_encoder.transform(["Normal"])[0])[0]
        if len(benign_indices) > 0:
            benign_preds = y_pred_hyb[benign_indices]
            normal_code = self.preprocessor.label_encoder.transform(["Normal"])[0]
            false_positives = np.sum(benign_preds != normal_code)
            fpr = float((false_positives / len(benign_indices)) * 100)
        else:
            fpr = 0.0

        # Detailed per-class metrics
        report_dict = classification_report(
            y_test_encoded,
            y_pred_hyb,
            labels=range(num_classes),
            target_names=classes,
            output_dict=True,
            zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(
            y_test_encoded,
            y_pred_hyb,
            labels=range(num_classes)
        ).tolist()

        if progress_callback:
            progress_callback("Saving model artifacts and evaluation results...", 0.90)

        # Save artifacts
        self._save_artifacts(
            total_records=total_records,
            train_count=len(X_train),
            test_count=len(X_test),
            classes=classes
        )

        total_duration = time.time() - start_total

        results = {
            "dataset_info": {
                "total_records": total_records,
                "train_records": len(X_train),
                "test_records": len(X_test),
                "features_count": len(self.preprocessor.feature_columns),
                "hybrid_features_count": X_train_hybrid.shape[1],
                "classes": classes
            },
            "timings": {
                "if_training_sec": round(time_if_train, 2),
                "xgb_individual_training_sec": round(time_xgb_ind_train, 2),
                "xgb_hybrid_training_sec": round(time_xgb_hyb_train, 2),
                "test_inference_sec": round(test_duration, 2),
                "total_sec": round(total_duration, 2)
            },
            "three_model_comparison": {
                "metrics": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
                "isolation_forest": [round(if_acc, 2), round(if_prec, 2), round(if_rec, 2), round(if_f1, 2)],
                "xgb_individual": [round(ind_acc, 2), round(ind_prec, 2), round(ind_rec, 2), round(ind_f1, 2)],
                "hybrid_model": [round(hyb_acc, 2), round(hyb_prec_macro, 2), round(hyb_rec_macro, 2), round(hyb_f1_macro, 2)]
            },
            "hybrid_metrics": {
                "accuracy": round(hyb_acc, 4),
                "precision_macro": round(hyb_prec_macro, 4),
                "recall_macro": round(hyb_rec_macro, 4),
                "f1_macro": round(hyb_f1_macro, 4),
                "f1_weighted": round(hyb_f1_weighted, 4),
                "false_positive_rate": round(fpr, 4)
            },
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "classes": classes
        }

        self.evaluation_results = results

        with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if progress_callback:
            progress_callback("Training and evaluation successfully completed!", 1.0)

        return results

    def _save_artifacts(
        self,
        total_records: int,
        train_count: int,
        test_count: int,
        classes: list
    ):
        """Serializes models, preprocessor, and metadata to disk."""
        # 1. Preprocessor
        self.preprocessor.save()

        # 2. Isolation Forest
        joblib.dump(self.isolation_model, ISOLATION_FOREST_FILE)

        # 3. XGBoost Models
        self.xgb_hybrid.save_model(str(XGBOOST_MODEL_FILE))
        if self.xgb_individual is not None:
            self.xgb_individual.save_model(str(XGBOOST_INDIVIDUAL_FILE))

        # 4. Model Metadata
        metadata = {
            "model_version": "1.0.0",
            "model_architecture": "Hybrid Isolation Forest (100 estimators) + XGBoost Hist (300 estimators, max_depth=6)",
            "dataset_name": "CIC-EVSE-2024",
            "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": total_records,
            "train_records": train_count,
            "test_records": test_count,
            "feature_count_original": len(self.preprocessor.feature_columns),
            "feature_count_hybrid": len(self.preprocessor.feature_columns) + 1,
            "feature_order": self.preprocessor.feature_columns + ["if_anomaly_score"],
            "classes": classes,
            "class_mapping": {cls_name: int(idx) for idx, cls_name in enumerate(classes)},
            "hyperparameters": {
                "isolation_forest": IF_PARAMS,
                "xgboost": XGB_PARAMS
            }
        }

        with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

