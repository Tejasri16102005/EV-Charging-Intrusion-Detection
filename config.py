"""
Configuration module for EV Charging Station Hybrid Intrusion Detection System.
Defines paths, feature schemas, class labels, and model parameters.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Dataset directories
DATASET_DIR = BASE_DIR / "dataset"
FALLBACK_DATASET_DIR = BASE_DIR.parent  # In case the 28 CSVs are in parent folder
DEMO_DATA_DIR = BASE_DIR / "demo_data"

# Artifact directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
BLOCKCHAIN_DIR = BASE_DIR / "blockchain_data"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
for folder in [MODELS_DIR, RESULTS_DIR, BLOCKCHAIN_DIR, ASSETS_DIR, DEMO_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Model Artifact Filenames
ISOLATION_FOREST_FILE = MODELS_DIR / "isolation_forest.pkl"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.json"
XGBOOST_INDIVIDUAL_FILE = MODELS_DIR / "xgboost_individual.json"
IMPUTER_FILE = MODELS_DIR / "imputer.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Blockchain Ledger File
LEDGER_FILE = BLOCKCHAIN_DIR / "ledger.json"

# Evaluation Results File
EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

# Target Attack Categories
PRIMARY_UI_CLASSES = [
    "Normal",
    "SYN Flood",
    "ICMP Flood",
    "Port Scan",
    "Aggressive Scan",
    "OS Fingerprinting"
]

ALL_RESEARCH_CLASSES = [
    "Aggressive Scan",
    "ICMP Flood",
    "ICMP Fragmentation",
    "Normal",
    "OS Fingerprinting",
    "Port Scan",
    "Push ACK Flood",
    "SYN Flood",
    "SYN Stealth",
    "Service Detection",
    "Slowloris",
    "Synonymous IP",
    "TCP Flood",
    "UDP Flood",
    "Vulnerability Scan"
]

# 61 Flow Features extracted by NFStream and validated in Colab notebook
FEATURE_COLUMNS = [
    "src_port", "dst_port", "protocol", "ip_version",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps", "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms", "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms", "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms", "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "bidirectional_syn_packets", "bidirectional_cwr_packets", "bidirectional_ece_packets", "bidirectional_urg_packets",
    "bidirectional_ack_packets", "bidirectional_psh_packets", "bidirectional_rst_packets", "bidirectional_fin_packets",
    "src2dst_syn_packets", "src2dst_cwr_packets", "src2dst_ece_packets", "src2dst_urg_packets",
    "src2dst_ack_packets", "src2dst_psh_packets", "src2dst_rst_packets", "src2dst_fin_packets",
    "dst2src_syn_packets", "dst2src_ece_packets", "dst2src_ack_packets", "dst2src_psh_packets",
    "dst2src_rst_packets", "dst2src_fin_packets",
    "application_is_guessed", "application_confidence"
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# XGBoost Hyperparameters matching Colab implementation
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_SEED,
    "tree_method": "hist"
}

# Isolation Forest Hyperparameters matching Colab implementation
IF_PARAMS = {
    "n_estimators": 100,
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

