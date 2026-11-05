"""
Data Loader Module for CIC-EVSE-2024 Dataset.
Handles discovery, metadata inspection, and memory-efficient loading of the 28 CSV files.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

from config import DATASET_DIR, FALLBACK_DATASET_DIR, FEATURE_COLUMNS


def get_attack_label(filename: str) -> str:
    """
    Extracts the ground-truth research attack category from the filename,
    strictly adhering to the CICEVSE24 dataset naming convention.
    """
    name = filename.lower()
    if "benign" in name:
        return "Normal"
    elif "aggressive-scan" in name:
        return "Aggressive Scan"
    elif "icmp-flood" in name:
        return "ICMP Flood"
    elif "icmp-fragmentation" in name:
        return "ICMP Fragmentation"
    elif "os-fingerprinting" in name:
        return "OS Fingerprinting"
    elif "portscan" in name:
        return "Port Scan"
    elif "push-ack-flood" in name:
        return "Push ACK Flood"
    elif "service-detection" in name:
        return "Service Detection"
    elif "slowloris" in name:
        return "Slowloris"
    elif "syn-flood" in name:
        return "SYN Flood"
    elif "syn-stealth" in name:
        return "SYN Stealth"
    elif "synonymous-ip" in name:
        return "Synonymous IP"
    elif "tcp-flood" in name:
        return "TCP Flood"
    elif "udp-flood" in name:
        return "UDP Flood"
    elif "vulnerability-scan" in name:
        return "Vulnerability Scan"
    else:
        return "Unknown"


def resolve_dataset_directory(custom_path: Optional[str] = None) -> Path:
    """
    Resolves the active dataset directory containing the 28 CSV files.
    Checks custom path, then ./dataset, then parent directory.
    """
    if custom_path and os.path.isdir(custom_path):
        p = Path(custom_path)
        if len(list(p.glob("*.csv"))) > 0:
            return p

    # Check local dataset/ directory
    if DATASET_DIR.is_dir() and len(list(DATASET_DIR.glob("*.csv"))) >= 28:
        return DATASET_DIR

    # Check fallback (parent directory)
    if FALLBACK_DATASET_DIR.is_dir() and len(list(FALLBACK_DATASET_DIR.glob("*.csv"))) >= 28:
        return FALLBACK_DATASET_DIR

    # Check if any csv files are in local dataset
    if DATASET_DIR.is_dir() and len(list(DATASET_DIR.glob("*.csv"))) > 0:
        return DATASET_DIR

    return FALLBACK_DATASET_DIR


def list_dataset_files(dataset_dir: Optional[Path] = None) -> List[Path]:
    """
    Returns a sorted list of all CSV files in the resolved dataset directory.
    """
    d = dataset_dir or resolve_dataset_directory()
    csv_files = sorted(list(d.glob("*.csv")))
    # Exclude files inside ev_intrusion_detection if searching parent
    filtered = [f for f in csv_files if "ev_intrusion_detection" not in str(f)]
    return filtered


def get_dataset_summary(dataset_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Inspects each CSV file and returns a detailed summary DataFrame and stats dict.
    """
    files = list_dataset_files(dataset_dir)
    file_records = []
    total_rows = 0
    total_size_bytes = 0

    for file_path in files:
        fname = file_path.name
        size_bytes = file_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)
        total_size_bytes += size_bytes

        # Fast line count
        with open(file_path, "rb") as fp:
            num_rows = sum(1 for _ in fp) - 1
            if num_rows < 0:
                num_rows = 0

        total_rows += num_rows
        label = get_attack_label(fname)
        mode = "Charging" if "charging" in fname.lower() else "Idle"

        file_records.append({
            "Filename": fname,
            "Attack Class": label,
            "Operating Mode": mode,
            "Records": num_rows,
            "Size (MB)": size_mb,
            "Path": str(file_path)
        })

    summary_df = pd.DataFrame(file_records)

    stats = {
        "total_files": len(files),
        "total_records": total_rows,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "dataset_path": str(dataset_dir or resolve_dataset_directory()),
        "classes_count": summary_df["Attack Class"].nunique() if not summary_df.empty else 0
    }

    return summary_df, stats


def load_dataset_for_training(
    dataset_dir: Optional[Path] = None,
    sample_fraction: Optional[float] = None,
    progress_callback=None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads all 28 CSV files into a consolidated DataFrame using only the 61 relevant flow features.
    Prevents leakage by omitting IPs, MACs, IDs, and timestamps.
    Returns:
        X: DataFrame of 61 features
        y: Series of research labels
        metadata_df: DataFrame containing Operating_Mode and Source_File
    """
    files = list_dataset_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir or resolve_dataset_directory()}")

    all_dfs = []
    total_files = len(files)

    for i, file_path in enumerate(files):
        fname = file_path.name
        label = get_attack_label(fname)
        mode = "Charging" if "charging" in fname.lower() else "Idle"

        # Inspect columns present in this CSV
        head = pd.read_csv(file_path, nrows=1)
        use_cols = [c for c in FEATURE_COLUMNS if c in head.columns]

        df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

        # Ensure all 61 standard features are present (fill missing columns with NaN)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")

        # Reindex to enforce strict column order
        df = df[FEATURE_COLUMNS].copy()

        # Optional downsampling if memory or speed constrained
        if sample_fraction and 0 < sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)

        df["Research_Label"] = label
        df["Operating_Mode"] = mode
        df["Source_File"] = fname

        all_dfs.append(df)

        if progress_callback:
            progress_callback(i + 1, total_files, fname)

    combined = pd.concat(all_dfs, ignore_index=True)

    y = combined["Research_Label"].copy()
    metadata_df = combined[["Operating_Mode", "Source_File"]].copy()
    X = combined[FEATURE_COLUMNS].copy()

    # Convert to numeric, replace infs with nan
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    return X, y, metadata_df

