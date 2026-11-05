# AI-Enabled Distributed Intrusion Detection in EV Charging Stations

### Final-Year Major Project | Cybersecurity & Machine Learning Research Implementation

---

## 📌 1. Project Overview

Electric Vehicle Supply Equipment (**EVSE**) forms the backbone of modern electromobility infrastructure, connecting smart electric power grids, Central System Management Servers (**CSMS**), and intelligent Connected Electric Vehicles (**EVs**). Networked protocols such as **OCPP 1.6J/2.0.1** and **ISO 15118** expose charging controllers to critical network-based cyber threats including volumetric DDoS/floods, reconnaissance port sweeps, and unauthorized protocol exploitation.

This system implements a production-grade, locally deployable **Hybrid Intrusion Detection System (IDS)** that synergizes:
1. **Unsupervised Anomaly Detection:** An ensemble **Isolation Forest** that isolates anomalous network patterns without ground-truth labels and generates continuous anomaly scores.
2. **Supervised Multiclass Classification:** An optimized **XGBoost (Histogram Gradient Boosting)** model that consumes the original 61 NFStream flow features augmented with the Isolation Forest anomaly score as a 62nd meta-feature to classify fine-grained attack types.
3. **Cryptographic Blockchain Logging:** A local permissioned **SHA-256 tamper-evident distributed ledger** that commits an immutable audit block for every live detection session.

---

## 📊 2. Research Dataset: CIC-EVSE-2024

- **Dataset Source:** Canadian Institute for Cybersecurity (CIC-EVSE-2024 / CICEVSE24).
- **Physical Testbed:** Grizzl-E Smart Connect (EVSE-A) running OCPP client firmware over Wi-Fi and Ethernet.
- **Dataset Composition:** Exactly **28 empirical CSV files** totaling **547,854 network traffic flow records**.
- **Conditions Captured:** Active Charging Mode & Standby Idle Mode under both baseline benign and adversarial traffic.
- **Features Extracted:** **61 numerical NFStream flow features** (packet lengths, inter-arrival times, TCP flag counters, bidirectional byte rates).
- **Leakage Prevention:** Identifiers such as IP addresses (`src_ip`, `dst_ip`), MAC addresses, packet IDs, source filenames, and timestamps are strictly stripped prior to modeling.

### Target Threat Categories:
1. **Normal** (Benign Operational Traffic)
2. **SYN Flood** (TCP SYN Exhaustion Denial-of-Service)
3. **ICMP Flood** (Ping Flooding Attack)
4. **Port Scan** (Horizontal/Vertical Port Sweep Reconnaissance)
5. **Aggressive Scan** (High-Intensity Multi-Port Probe)
6. **OS Fingerprinting** (TCP/IP Stack Identification Probe)
*Plus fine-grained variants: SYN Stealth, Slowloris, UDP Flood, Push-ACK Flood, Synonymous IP Flood, Service Detection, Vulnerability Scanning.*

---

## 🏗️ 3. Machine Learning & Security Architecture

```
Input Network Traffic Flow Capture (.csv)
                  │
                  ▼
   [Data Preprocessing Pipeline]
   - Schema Alignment (61 NFStream Features)
   - Median Imputation (Fitted on Training Data)
   - Identifiers Stripped (Zero Data Leakage)
                  │
                  ▼
   [Isolation Forest Model (100 Trees)]
   - Unsupervised Structural Divergence
   - Continuous Anomaly Score Calculation (-decision_function)
                  │
                  ▼
   [Hybrid Feature Synthesis]
   - Stack Anomaly Score as Feature #62
                  │
                  ▼
   [XGBoost Multiclass Classifier (300 Estimators, Depth=6)]
   - Histogram-based Tree Splits
   - Cost-Sensitive Balanced Class Weighting
                  │
                  ▼
   [Hybrid Classification Result]
   - Normal vs. Malicious Determination
   - 6-Category Threat Classification
   - Prediction Confidence Assessment (%)
                  │
                  ▼
   [SHA-256 Distributed Audit Blockchain]
   - Canonical JSON Block Serialization
   - Cryptographic Hash Linking (previous_block_hash)
   - Real-Time Tamper-Evidence Verification
```

---

## 📂 4. Project Directory Structure

```
ev_intrusion_detection/
│
├── app.py                     # Master 9-Page Streamlit Web Application
├── config.py                  # Global configurations, paths, parameters & schemas
├── requirements.txt           # Verified Python dependency specifications
├── README.md                  # Comprehensive research & deployment manual
│
├── dataset/                   # Dedicated folder for 28 CIC-EVSE-2024 CSV files
│   └── (28 CSV files or auto-fallback to parent directory)
│
├── demo_data/                 # Ready-to-use benchmark CSV files for live testing
│   ├── demo_benign_traffic.csv
│   ├── demo_syn_flood_attack.csv
│   ├── demo_port_scan_attack.csv
│   ├── demo_mixed_traffic.csv
│   └── demo_corrupted_columns.csv
│
├── models/                    # Serialized machine learning models & preprocessors
│   ├── isolation_forest.pkl   # Fitted Isolation Forest model
│   ├── xgboost_model.json     # Trained Hybrid XGBoost model
│   ├── xgboost_individual.json# Trained Baseline XGBoost model (61 features)
│   ├── imputer.pkl            # Fitted median imputer
│   ├── label_encoder.pkl      # Fitted class label encoder
│   ├── feature_columns.json   # 61-feature strict ordered schema
│   └── model_metadata.json    # Complete research metadata & class mappings
│
├── modules/                   # Modular Python backend architecture
│   ├── __init__.py
│   ├── data_loader.py         # Memory-efficient loader & metadata analyzer
│   ├── preprocessing.py      # Imputation, alignment & hybrid stacking
│   ├── model_training.py      # Leakage-free training & evaluation engine
│   ├── prediction.py          # Real-time inference engine
│   ├── evaluation.py          # Scientific benchmark & report manager
│   ├── blockchain.py          # SHA-256 tamper-evident distributed ledger
│   └── visualization.py       # Publication-quality cybersecurity charts
│
├── blockchain_data/
│   └── ledger.json            # Persistent JSON audit ledger
│
├── results/
│   └── evaluation_results.json# Empirical metrics, reports & confusion matrix
│
└── assets/
    └── style.css              # Custom dark navy cybersecurity UI styling
```

---

## 💻 5. Local Windows Deployment Instructions

### Prerequisites:
- Windows 10 or 11 (64-bit)
- Python 3.10, 3.11, 3.12, or 3.13 installed
- Command Prompt (CMD) or Windows PowerShell

### Step-by-Step Launch:

#### Step 1: Open Terminal and Navigate to Project
```cmd
cd /d "d:\disease prediction\complete dataset\Network Traffic\EVSE-A\csv\ev_intrusion_detection"
```

#### Step 2: Install Dependencies
```cmd
pip install -r requirements.txt
```

#### Step 3: Run the Streamlit Application
```cmd
streamlit run app.py
```

#### Step 4: Access Application in Browser
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🧪 6. Testing & Validation Scenarios

The system includes pre-configured benchmark datasets in `demo_data/` for immediate validation on **Page 3 (Live CSV Detection)**:

1. **Compatible Benign Traffic (`demo_benign_traffic.csv`):** Validates normal vehicle charging flow recognition.
2. **SYN Flood Attack (`demo_syn_flood_attack.csv`):** Tests high-volume TCP connection starvation detection.
3. **Port Scan Attack (`demo_port_scan_attack.csv`):** Tests horizontal reconnaissance probe identification.
4. **Mixed Network Traffic (`demo_mixed_traffic.csv`):** Demonstrates concurrent multi-attack classification and Normal vs Malicious volume ratios.
5. **Corrupted Schema (`demo_corrupted_columns.csv`):** Verifies robust input validation and defensive error handling.
6. **Blockchain Tamper Demonstration (Page 8):** Click *"Simulate Unauthorized Tampering"* followed by *"Verify Blockchain Integrity"* to observe cryptographic link verification in action.

---

## 📈 7. Empirical Research Results Summary

Metrics calculated on the 20% held-out test partition (109,571 records):

| Model Architecture | Input Features | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** (Unsupervised) | 61 Flow Features | 24.27% | 99.94% | 24.28% | 39.06% |
| **Individual XGBoost** (Baseline) | 61 Flow Features | 99.95% | 94.09% | 95.09% | 94.54% |
| **Hybrid IF + XGBoost** (Ours) | **62 Stacked Features** | **99.96%** | **94.09%** | **95.09%** | **94.54%** |

*Note: Isolation Forest is evaluated as a binary anomaly detector (`Normal` vs `Attack`), whereas XGBoost and Hybrid are evaluated across all multi-class attack categories.*

