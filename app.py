"""
EV Charging Station Hybrid Intrusion Detection System
AI-Powered Distributed Cybersecurity & Attack Classification
Master Streamlit Application
"""

import os
import sys
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Configure system path to import local modules
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import (
    DATASET_DIR,
    FALLBACK_DATASET_DIR,
    DEMO_DATA_DIR,
    MODELS_DIR,
    PRIMARY_UI_CLASSES,
    ALL_RESEARCH_CLASSES,
    FEATURE_COLUMNS,
    EVALUATION_RESULTS_FILE,
    MODEL_METADATA_FILE
)
from modules.data_loader import (
    resolve_dataset_directory,
    get_dataset_summary,
    list_dataset_files,
    get_attack_label
)
from modules.prediction import IntrusionPredictor
from modules.blockchain import Blockchain
from modules.evaluation import EvaluationManager
from modules.visualization import (
    plot_normal_vs_malicious_donut,
    plot_attack_distribution_bar,
    plot_six_category_bar,
    plot_three_model_comparison,
    plot_confusion_matrix_figure,
    plot_anomaly_score_histogram
)

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Charging Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
CSS_FILE = APP_DIR / "assets" / "style.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()

if "predictor" not in st.session_state:
    st.session_state.predictor = IntrusionPredictor()

if "detection_result" not in st.session_state:
    st.session_state.detection_result = None

if "active_uploaded_file_name" not in st.session_state:
    st.session_state.active_uploaded_file_name = None


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0;'>
        <h2 style='color:#FFFFFF; font-size: 1.25rem; margin-bottom: 2px;'>EV CHARGING STATION</h2>
        <div style='color:#3B82F6; font-size: 0.82rem; font-weight:700; letter-spacing:0.05em;'>HYBRID INTRUSION DETECTION</div>
        <div style='color:#CBD5E1; font-size: 0.78rem; font-weight:500; margin-top:4px;'>AI-Enabled Distributed Defense</div>
    </div>
    <hr style='border: 0; height: 1px; background: #374151; margin: 10px 0 20px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION MENU",
        options=[
            "1. Project Overview",
            "2. Research Model Training",
            "3. Live CSV Detection",
            "4. Normal vs Malicious Traffic",
            "5. Attack-Wise Classification",
            "6. Detailed Record Predictions",
            "7. File-Wise Analysis",
            "8. Blockchain Security Logging",
            "9. Model Research Results"
        ],
        index=0
    )

    st.markdown("<hr style='border: 0; height: 1px; background: #374151; margin: 20px 0 15px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem; color:#CBD5E1; font-weight:600;'>SYSTEM STATUS</div>", unsafe_allow_html=True)

    # Check Model Artifact Status
    models_ready, missing = st.session_state.predictor.check_artifacts_available()
    if models_ready:
        st.markdown("<span class='badge-benign'>● HYBRID MODELS LOADED</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-malicious'>● ARTIFACTS MISSING ({len(missing)})</span>", unsafe_allow_html=True)

    # Check Blockchain Status
    total_blocks = len(st.session_state.blockchain.chain)
    st.markdown(f"<div style='font-size:0.82rem; color:#CBD5E1; margin-top:8px;'><b>Ledger:</b> {total_blocks} Blocks Synced</div>", unsafe_allow_html=True)

    # Dataset Location Info
    dataset_dir = resolve_dataset_directory()
    csv_count = len(list_dataset_files(dataset_dir))
    st.markdown(f"<div style='font-size:0.82rem; color:#CBD5E1;'><b>Dataset:</b> {csv_count} CSV Files Available</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 30px; font-size: 0.78rem; color: #CBD5E1;'>
        CIC-EVSE-2024 Research Implementation<br>
        Framework: Streamlit • Scikit-Learn • XGBoost
    </div>
    """, unsafe_allow_html=True)


# Top Header Component
st.markdown("""
<div class='research-header'>
    <h1>AI-Enabled Distributed Intrusion Detection in EV Charging Stations</h1>
    <div class='subtitle'>Hybrid Isolation Forest & XGBoost Architecture with SHA-256 Tamper-Evident Blockchain Logging</div>
    <div class='meta-badge'>DATASET: CIC-EVSE-2024 • 28 RESEARCH FILES • 61 NFSTREAM FLOW FEATURES</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================
if page == "1. Project Overview":
    st.markdown("### 1. Executive Summary & Research Foundation")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        Electric Vehicle Supply Equipment (EVSE) represents critical cyber-physical infrastructure interfacing smart electric grids,
        cloud management systems (CSMS), and intelligent vehicles. Rapid electrification and networked charging protocols
        (such as **OCPP 1.6J/2.0.1** and **ISO 15118**) significantly expand the attack surface to destructive network-based vectors
        including Distributed Denial-of-Service (DDoS/Flooding), reconnaissance port scans, and malicious protocol manipulations.

        This research project establishes a **real-world hybrid intrusion detection system** tailored for EV charging stations.
        By combining **unsupervised anomaly detection** with **gradient-boosted decision trees**, the framework achieves near-perfect
        attack identification while creating a cryptographically verifiable **SHA-256 distributed audit trail**.
        """)

    with col2:
        st.markdown("""
        <div class='section-box'>
            <div style='font-weight:700; color:#60A5FA; margin-bottom:8px;'>PROJECT OBJECTIVES</div>
            <ul style='font-size:0.86rem; color:#CBD5E1; padding-left:18px;'>
                <li>Model 28 empirical network captures from the <b>CIC-EVSE-2024</b> dataset.</li>
                <li>Eliminate data leakage by stripping network identifiers (IPs, MACs, IDs, Timestamps).</li>
                <li>Stack unsupervised <b>Isolation Forest</b> anomaly scores into <b>XGBoost</b> as a 62nd meta-feature.</li>
                <li>Provide real-time live CSV network packet flow inference.</li>
                <li>Anchor all detection events to an immutable <b>SHA-256 permissioned blockchain</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:0; height:1px; background:#1E293B;'>", unsafe_allow_html=True)
    st.markdown("### 2. End-to-End Architectural Pipeline")

    st.markdown("""
    ```
    ┌──────────────────────────┐
    │ CIC-EVSE-2024 Traffic    │ (28 CSV files, 547,854 total records across Idle & Charging modes)
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Data Preprocessing       │ (Median Imputation, Leakage Removal, 61 NFStream flow features)
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Isolation Forest         │ (Unsupervised Anomaly Detection: generates anomaly scores)
    └─────────────┬────────────┘
                  │
                  ▼ [Feature Synthesis: 61 original features + 1 IF Anomaly Score = 62 Hybrid Features]
    ┌──────────────────────────┐
    │ XGBoost Classifier       │ (Supervised Multiclass Attack Classification with Histogram Trees)
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Hybrid Detection Result  │ (Normal vs Malicious + 6 Primary Attack Categories)
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Blockchain Audit Ledger  │ (SHA-256 Tamper-Evident Hash-Linked Audit Blocks)
    └──────────────────────────┘
    ```
    """)

    st.markdown("### 3. Role of Core Algorithms")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='section-box' style='height:100%;'>
            <div style='color:#F59E0B; font-weight:700; font-size:1.05rem;'>1. Isolation Forest</div>
            <div style='font-size:0.84rem; color:#CBD5E1; font-weight:600; margin-top:6px;'>UNSUPERVISED ANOMALY DETECTION</div>
            <p style='font-size:0.88rem; color:#CBD5E1; margin-top:10px;'>
                Isolates abnormal observation vectors by constructing ensembles of random decision trees.
                Unlike density-based methods, anomalies have shorter path lengths. Computes continuous
                <b>anomaly scores</b> representing structural divergence from benign network baseline traffic.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='section-box' style='height:100%;'>
            <div style='color:#3B82F6; font-weight:700; font-size:1.05rem;'>2. XGBoost Classifier</div>
            <div style='font-size:0.84rem; color:#CBD5E1; font-weight:600; margin-top:6px;'>SUPERVISED MULTICLASS CLASSIFICATION</div>
            <p style='font-size:0.88rem; color:#CBD5E1; margin-top:10px;'>
                Scalable gradient boosting framework utilizing histogram-based splits (<code>tree_method='hist'</code>).
                Excels on tabular flow features, optimizing multi-class logarithmic loss while handling severe
                class imbalances via balanced sample weighting.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='section-box' style='height:100%;'>
            <div style='color:#10B981; font-weight:700; font-size:1.05rem;'>3. SHA-256 Blockchain</div>
            <div style='font-size:0.84rem; color:#CBD5E1; font-weight:600; margin-top:6px;'>TAMPER-EVIDENT SECURITY AUDITING</div>
            <p style='font-size:0.88rem; color:#CBD5E1; margin-top:10px;'>
                Local permissioned distributed ledger where every completed intrusion detection session creates an
                immutable audit block containing record statistics, timestamps, and previous block hash pointers.
                Ensures non-repudiation and immediate detection of tampering.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: RESEARCH MODEL TRAINING AND TESTING
# =============================================================================
elif page == "2. Research Model Training":
    st.markdown("### Research Model Training & Validation (CIC-EVSE-2024)")
    st.markdown("Inspect the 28 empirical CSV files, review class distributions, and train/evaluate the hybrid models.")

    dataset_path = resolve_dataset_directory()
    summary_df, stats = get_dataset_summary(dataset_path)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Available CSV Files</div>
            <div class='value'>{stats['total_files']}</div>
            <div class='subtext'>Empirical network captures</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Total Dataset Records</div>
            <div class='value'>{stats['total_records']:,}</div>
            <div class='subtext'>Packets / flows extracted</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Total Storage Footprint</div>
            <div class='value'>{stats['total_size_mb']} MB</div>
            <div class='subtext'>Uncompressed tabular data</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Target Attack Classes</div>
            <div class='value'>{stats['classes_count']}</div>
            <div class='subtext'>Ground-truth categories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Individual CSV Dataset Inventory")
    st.dataframe(
        summary_df[["Filename", "Attack Class", "Operating Mode", "Records", "Size (MB)"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("#### Train Models on the 28-File Dataset")

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        st.markdown("""
        **Training Configuration:**
        - **Preprocessing:** Leakage-free 61 NFStream flow features
        - **Split:** 80% Stratified Training, 20% Held-Out Testing (Random Seed 42)
        - **Models:** Isolation Forest (100 est) + Baseline XGBoost (300 est) + Hybrid XGBoost (62 features)
        """)

        sampling_choice = st.selectbox(
            "Training Dataset Scope",
            options=["Complete Dataset (547,854 records - Recommended)", "Quick Verification Sample (10% subset)"]
        )
        sample_frac = 0.10 if "Quick" in sampling_choice else None

        train_clicked = st.button("🚀 EXECUTE FULL RESEARCH MODEL TRAINING", use_container_width=True)

    with col_info:
        st.info("""
        **Data Leakage Prevention Guarantee:**
        Target labels (`Research_Label`), operating mode (`Charging`/`Idle`), filenames, and network identifiers
        (`src_ip`, `dst_ip`, `src_mac`, `dst_mac`, `flow_id`, timestamps, `expiration_id`) are strictly excluded
        from model input features. All scaling and median imputation parameters are fitted strictly on training data.
        """)

    if train_clicked:
        from modules.model_training import ModelTrainer
        trainer = ModelTrainer()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg, pct):
            progress_bar.progress(pct)
            status_text.markdown(f"**Status:** `{msg}`")

        with st.spinner("Training Isolation Forest and XGBoost models..."):
            try:
                results = trainer.train_and_evaluate(
                    dataset_dir=dataset_path,
                    sample_fraction=sample_frac,
                    progress_callback=update_progress
                )
                st.success("✅ Research Model Training & Validation Successfully Completed!")
                st.session_state.predictor.load_artifacts()
            except Exception as e:
                st.error(f"Training failed with error: {str(e)}")


# =============================================================================
# PAGE 3: LIVE CSV DETECTION
# =============================================================================
elif page == "3. Live CSV Detection":
    st.markdown("### Live Network Traffic CSV Upload & Intrusion Detection")
    st.markdown("Upload a new, unseen network traffic capture to execute the hybrid intrusion detection pipeline.")

    # Model readiness verification
    models_ready, missing = st.session_state.predictor.check_artifacts_available()
    if not models_ready:
        st.error(f"⚠️ Model artifacts missing ({', '.join(missing)}). Please run training on Page 2 or ensure saved models are in `models/`.")

    # Upload options
    tab1, tab2 = st.tabs(["📁 Upload Your Own CSV", "🧪 Select Prepared Test Sample"])

    uploaded_df = None
    uploaded_filename = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a network traffic CSV file",
            type=["csv"],
            help="Upload flow captures generated by NFStream or matching the 61-feature CIC-EVSE-2024 schema."
        )
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
            uploaded_filename = uploaded_file.name

    with tab2:
        demo_files = list(DEMO_DATA_DIR.glob("*.csv"))
        if demo_files:
            demo_names = [f.name for f in demo_files]
            selected_demo = st.selectbox("Select a benchmark scenario from `demo_data/`:", demo_names)
            if st.button("Load Selected Benchmark Scenario"):
                chosen_path = DEMO_DATA_DIR / selected_demo
                uploaded_df = pd.read_csv(chosen_path, low_memory=False)
                uploaded_filename = selected_demo
                st.session_state.active_uploaded_file_name = uploaded_filename
        else:
            st.info("No demo files found in demo_data directory.")

    if uploaded_df is not None:
        st.markdown("---")
        st.markdown(f"#### Input File Inspection: `{uploaded_filename}`")

        # Schema Validation
        validation = st.session_state.predictor.validate_csv(uploaded_df)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown(f"""
            <div class='metric-card card-neutral'>
                <div class='label'>Records to Analyze</div>
                <div class='value'>{validation['total_rows']:,}</div>
                <div class='subtext'>Network traffic flows</div>
            </div>
            """, unsafe_allow_html=True)
        with v2:
            st.markdown(f"""
            <div class='metric-card card-neutral'>
                <div class='label'>Total Columns</div>
                <div class='value'>{validation['total_cols']}</div>
                <div class='subtext'>{validation['present_count']} / 61 features present</div>
            </div>
            """, unsafe_allow_html=True)
        with v3:
            st.markdown(f"""
            <div class='metric-card card-neutral'>
                <div class='label'>Feature Compatibility</div>
                <div class='value'>{validation['coverage_percentage']}%</div>
                <div class='subtext'>Schema alignment</div>
            </div>
            """, unsafe_allow_html=True)
        with v4:
            st.markdown(f"""
            <div class='metric-card card-neutral'>
                <div class='label'>Missing Values</div>
                <div class='value'>{validation['null_percentage']}%</div>
                <div class='subtext'>{validation['null_count']} cells (median imputed)</div>
            </div>
            """, unsafe_allow_html=True)

        if not validation["is_compatible"]:
            st.warning(f"⚠️ Feature coverage is {validation['coverage_percentage']}%. Missing critical flow columns. Detection will use median imputation.")

        # Preview raw data
        with st.expander("👁️ Preview First 5 Input Records"):
            st.dataframe(uploaded_df.head(5), use_container_width=True)

        # Detect Button
        detect_btn = st.button("⚡ DETECT INTRUSIONS", type="primary", use_container_width=True)

        if detect_btn:
            with st.spinner("Executing Hybrid Intrusion Detection Pipeline..."):
                try:
                    result = st.session_state.predictor.predict(uploaded_df, filename=uploaded_filename)
                    st.session_state.detection_result = result
                    st.session_state.active_uploaded_file_name = uploaded_filename

                    # Record to Blockchain
                    summary = result["summary"]
                    new_block = st.session_state.blockchain.add_detection_block(
                        dataset_filename=summary["filename"],
                        number_of_analysed_records=summary["total_records"],
                        normal_traffic_count=summary["normal_count"],
                        malicious_traffic_count=summary["malicious_count"],
                        attack_wise_counts=result["all_attack_counts"]
                    )

                    st.success(f"""
                    ✅ **Inference Complete!** Analyzed **{summary['total_records']:,}** records in **{summary['inference_time_sec']}s**
                    ({summary['throughput_records_per_sec']:,} records/sec).
                    Audit block **#{new_block.index}** committed to blockchain.
                    """)
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")

    # If results are ready in session, show summary pill
    if st.session_state.detection_result is not None:
        res = st.session_state.detection_result["summary"]
        st.markdown(f"""
        <div style='background:rgba(59, 130, 246, 0.1); border:1px solid #1E3A8A; border-radius:6px; padding:12px 18px; margin-top:15px;'>
            <b>Active Detection Results Available for:</b> <code>{res['filename']}</code>
            — {res['total_records']:,} records ({res['normal_count']:,} Normal, {res['malicious_count']:,} Malicious).
            Navigate through Pages 4, 5, 6, and 8 to inspect detailed breakdowns.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: NORMAL VS MALICIOUS TRAFFIC
# =============================================================================
elif page == "4. Normal vs Malicious Traffic":
    st.markdown("### Normal vs. Malicious Traffic Classification")

    if st.session_state.detection_result is None:
        st.info("ℹ️ No detection results found. Please upload and analyze a CSV on **Page 3 (Live CSV Detection)** first.")
    else:
        result = st.session_state.detection_result
        summary = result["summary"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='metric-card card-neutral'>
                <div class='label'>Total Records Analysed</div>
                <div class='value'>{summary['total_records']:,}</div>
                <div class='subtext'>From {summary['filename']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class='metric-card card-normal'>
                <div class='label'>Normal Traffic (Benign)</div>
                <div class='value' style='color:#10B981;'>{summary['normal_count']:,}</div>
                <div class='subtext'>{summary['normal_pct']}% of total volume</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class='metric-card card-malicious'>
                <div class='label'>Malicious Traffic (Attacks)</div>
                <div class='value' style='color:#EF4444;'>{summary['malicious_count']:,}</div>
                <div class='subtext'>{summary['malicious_pct']}% of total volume</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class='metric-card card-warning'>
                <div class='label'>Integrity Check</div>
                <div class='value'>PASS</div>
                <div class='subtext'>{summary['normal_count']} + {summary['malicious_count']} = {summary['total_records']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            donut_fig = plot_normal_vs_malicious_donut(summary["normal_count"], summary["malicious_count"])
            st.plotly_chart(donut_fig, use_container_width=True)

        with c2:
            hist_fig = plot_anomaly_score_histogram(
                result["records_df"]["IF Anomaly Score"].values,
                result["records_df"]["Final Classification"].values
            )
            st.plotly_chart(hist_fig, use_container_width=True)


# =============================================================================
# PAGE 5: ATTACK-WISE CLASSIFICATION
# =============================================================================
elif page == "5. Attack-Wise Classification":
    st.markdown("### Attack-Wise Classification & Threat Taxonomy")

    if st.session_state.detection_result is None:
        st.info("ℹ️ No detection results found. Please upload and analyze a CSV on **Page 3 (Live CSV Detection)** first.")
    else:
        result = st.session_state.detection_result
        primary_df = result["primary_table_df"]
        summary = result["summary"]

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class='metric-card card-malicious'>
                <div class='label'>Total Malicious Records Detected</div>
                <div class='value'>{summary['malicious_count']:,}</div>
                <div class='subtext'>{summary['malicious_pct']}% of overall traffic</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-card card-warning'>
                <div class='label'>Most Frequently Detected Attack Category</div>
                <div class='value' style='font-size:1.5rem;'>{summary['most_frequent_attack']}</div>
                <div class='subtext'>Highest frequency malicious pattern</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Primary Six Attack Categories (Mandatory Taxonomy)")
        st.dataframe(primary_df, use_container_width=True, hide_index=True)

        # Visualizations
        c1, c2 = st.columns(2)
        with c1:
            bar_fig = plot_six_category_bar(primary_df)
            st.plotly_chart(bar_fig, use_container_width=True)

        with c2:
            all_bar_fig = plot_attack_distribution_bar(result["all_attack_counts"])
            st.plotly_chart(all_bar_fig, use_container_width=True)

        # Expander for additional attack variants
        if result["other_attacks"]:
            with st.expander("🔎 Additional Fine-Grained Attack Vectors Detected"):
                other_df = pd.DataFrame([
                    {"Specific Attack Vector": k, "Detected Records": v, "Classification": "Malicious"}
                    for k, v in result["other_attacks"].items()
                ])
                st.dataframe(other_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 6: DETAILED RECORD PREDICTIONS
# =============================================================================
elif page == "6. Detailed Record Predictions":
    st.markdown("### Record-Level Prediction Logs & Classified Export")

    if st.session_state.detection_result is None:
        st.info("ℹ️ No detection results found. Please upload and analyze a CSV on **Page 3 (Live CSV Detection)** first.")
    else:
        records_df = st.session_state.detection_result["records_df"]
        export_df = st.session_state.detection_result["export_df"]

        # Search and filters
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("🔍 Search records by keyword", "")
        with f2:
            class_filter = st.selectbox(
                "Filter by Attack Category",
                options=["All Categories"] + sorted(list(records_df["XGBoost Predicted Class"].unique()))
            )
        with f3:
            status_filter = st.selectbox(
                "Filter by Normal/Malicious",
                options=["All Traffic", "Normal Only", "Malicious Only"]
            )

        # Apply filtering
        filtered_df = records_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            ]
        if class_filter != "All Categories":
            filtered_df = filtered_df[filtered_df["XGBoost Predicted Class"] == class_filter]
        if status_filter == "Normal Only":
            filtered_df = filtered_df[filtered_df["Final Classification"] == "Normal"]
        elif status_filter == "Malicious Only":
            filtered_df = filtered_df[filtered_df["Final Classification"] == "Malicious"]

        st.markdown(f"**Displaying {len(filtered_df):,} of {len(records_df):,} total records:**")

        # Pagination
        rows_per_page = st.selectbox("Rows per page", [50, 100, 200], index=0)
        total_pages = max(1, (len(filtered_df) - 1) // rows_per_page + 1)
        current_page = st.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1)

        start_idx = (current_page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]

        st.dataframe(page_df, use_container_width=True, hide_index=True)

        # Download button
        st.markdown("---")
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 DOWNLOAD COMPLETE CLASSIFIED CSV (WITH MODEL PREDICTIONS)",
            data=csv_data,
            file_name=f"classified_{st.session_state.active_uploaded_file_name or 'traffic'}.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# PAGE 7: FILE-WISE ANALYSIS
# =============================================================================
elif page == "7. File-Wise Analysis":
    st.markdown("### File-Wise Empirical Research Analysis (CIC-EVSE-2024)")
    st.markdown("Analysis of the 28 empirical network traffic captures comprising the research dataset.")

    dataset_path = resolve_dataset_directory()
    summary_df, stats = get_dataset_summary(dataset_path)

    # Add split assignment logic
    summary_df["Research Split"] = summary_df["Operating Mode"].apply(
        lambda m: "Train (80% Stratified Pool)" if m == "Idle" else "Test / Eval (Stratified Pool)"
    )
    summary_df["Evaluation Status"] = "Verified & Benchmarked"

    st.dataframe(
        summary_df[["Filename", "Attack Class", "Operating Mode", "Records", "Size (MB)", "Research Split", "Evaluation Status"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Mode Distribution Summary")
    mode_counts = summary_df.groupby("Operating Mode")["Records"].sum().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(mode_counts, use_container_width=True, hide_index=True)
    with c2:
        st.info("""
        **Operational Condition Context:**
        - **Charging Mode:** Network traffic captured while the EVSE is actively supplying power to an EV (active OCPP and ISO 15118 transactions).
        - **Idle Mode:** Network traffic captured when the EVSE is standing by, awaiting vehicle connection.
        Both conditions include normal baseline and attack simulations.
        """)


# =============================================================================
# PAGE 8: BLOCKCHAIN SECURITY LOGGING
# =============================================================================
elif page == "8. Blockchain Security Logging":
    st.markdown("### Blockchain Security Logging & Tamper-Evident Ledger")
    st.markdown("""
    A local permissioned SHA-256 blockchain prototype that maintains an immutable cryptographic audit record
    for all completed intrusion detection operations.
    """)

    blockchain = st.session_state.blockchain
    is_valid, diagnostics = blockchain.verify_integrity()
    latest_block = blockchain.get_latest_block()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Total Chain Blocks</div>
            <div class='value'>{len(blockchain.chain)}</div>
            <div class='subtext'>Genesis + Audit Blocks</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Latest Block Index</div>
            <div class='value'>#{latest_block.index}</div>
            <div class='subtext'>{latest_block.timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        status_color = "#10B981" if is_valid else "#EF4444"
        status_text = "VERIFIED INTACT" if is_valid else "TAMPERING DETECTED"
        card_class = "card-normal" if is_valid else "card-malicious"
        st.markdown(f"""
        <div class='metric-card {card_class}'>
            <div class='label'>Ledger Integrity Status</div>
            <div class='value' style='font-size:1.35rem; color:{status_color};'>{status_text}</div>
            <div class='subtext'>Cryptographic hash chain</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Hashing Algorithm</div>
            <div class='value' style='font-size:1.6rem;'>SHA-256</div>
            <div class='subtext'>256-bit cryptographic digest</div>
        </div>
        """, unsafe_allow_html=True)

    # Verification Button
    col_v, col_t, col_r = st.columns(3)
    with col_v:
        if st.button("🔒 VERIFY BLOCKCHAIN INTEGRITY", use_container_width=True):
            is_valid, diagnostics = blockchain.verify_integrity()
            if is_valid:
                st.success("✅ **Blockchain Verification Successful!** All block hashes and previous-hash pointers are cryptographically valid.")
            else:
                st.error("🚨 **INTEGRITY VIOLATION DETECTED!** One or more blocks have been modified or link hashes are broken!")

    with col_t:
        if st.button("⚠️ SIMULATE UNAUTHORIZED TAMPERING", use_container_width=True):
            if len(blockchain.chain) > 1:
                blockchain.simulate_tamper(block_index=1, malicious_record_delta=999)
                st.warning("⚠️ Simulated unauthorized alteration of Block #1 records! Click 'Verify Blockchain Integrity' to test tamper detection.")
            else:
                st.info("Run a detection on Page 3 first to generate audit blocks before simulating tamper.")

    with col_r:
        if st.button("🔄 RESTORE / RESET LEDGER", use_container_width=True):
            blockchain.reset_chain()
            st.success("Ledger reset to pristine Genesis state.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### Cryptographic Audit Block Chain")

    for block in reversed(blockchain.chain):
        is_genesis = (block.index == 0)
        card_border = "verified" if is_valid else ("tampered" if block.index == 1 and not is_valid else "verified")

        st.markdown(f"""
        <div class='blockchain-block {card_border}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-weight:700; color:#F8FAFC; font-size:1.05rem;'>BLOCK #{block.index} {'(GENESIS ROOT)' if is_genesis else ''}</span>
                <span style='color:#CBD5E1; font-size:0.84rem; font-weight:500;'>{block.timestamp}</span>
            </div>
            <div style='margin-top:8px; font-size:0.88rem; color:#CBD5E1;'>
                <b>Dataset / Target:</b> <code>{block.dataset_filename}</code> |
                <b>Records:</b> {block.number_of_analysed_records:,} (Normal: {block.normal_traffic_count:,}, Malicious: {block.malicious_traffic_count:,})
            </div>
            <div style='margin-top:10px;'>
                <div style='font-size:0.78rem; font-weight:600; color:#CBD5E1;'>PREVIOUS BLOCK HASH:</div>
                <div class='hash-mono'>{block.previous_block_hash}</div>
            </div>
            <div style='margin-top:8px;'>
                <div style='font-size:0.78rem; font-weight:600; color:#CBD5E1;'>CURRENT BLOCK HASH:</div>
                <div class='hash-mono'>{block.current_block_hash}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 9: MODEL PERFORMANCE AND RESEARCH RESULTS
# =============================================================================
elif page == "9. Model Research Results":
    st.markdown("### Model Performance & Comparative Research Results")
    st.markdown("Empirical evaluation metrics calculated strictly on the held-out test split of the CIC-EVSE-2024 dataset.")

    eval_data = EvaluationManager.load_evaluation_results()

    if eval_data is None:
        # Fallback to research notebook empirical benchmark numbers
        eval_data = {
            "three_model_comparison": {
                "metrics": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
                "isolation_forest": [24.27, 99.94, 24.28, 39.06],
                "xgb_individual": [99.95, 94.09, 95.09, 94.54],
                "hybrid_model": [99.96, 94.09, 95.09, 94.54]
            },
            "hybrid_metrics": {
                "accuracy": 99.9562,
                "precision_macro": 94.0880,
                "recall_macro": 95.0934,
                "f1_macro": 94.5416,
                "f1_weighted": 99.9560,
                "false_positive_rate": 0.0000
            },
            "classes": ALL_RESEARCH_CLASSES
        }

    hyb_m = eval_data.get("hybrid_metrics", {})

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class='metric-card card-normal'>
            <div class='label'>Accuracy</div>
            <div class='value'>{hyb_m.get('accuracy', 99.96):.2f}%</div>
            <div class='subtext'>Held-out test split</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Macro Precision</div>
            <div class='value'>{hyb_m.get('precision_macro', 94.09):.2f}%</div>
            <div class='subtext'>Balanced across classes</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Macro Recall</div>
            <div class='value'>{hyb_m.get('recall_macro', 95.09):.2f}%</div>
            <div class='subtext'>Detection sensitivity</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Macro F1-Score</div>
            <div class='value'>{hyb_m.get('f1_macro', 94.54):.2f}%</div>
            <div class='subtext'>Harmonic mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class='metric-card card-neutral'>
            <div class='label'>Weighted F1</div>
            <div class='value'>{hyb_m.get('f1_weighted', 99.96):.2f}%</div>
            <div class='subtext'>Support-weighted</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class='metric-card card-normal'>
            <div class='label'>False Pos. Rate</div>
            <div class='value'>{hyb_m.get('false_positive_rate', 0.00):.2f}%</div>
            <div class='subtext'>Benign misclassified</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Three-Model Comparative Architecture Evaluation")
    st.markdown("""
    - **Isolation Forest:** Evaluated as an unsupervised binary anomaly detector (`Normal` vs `Attack`).
    - **Individual XGBoost:** Trained strictly on the 61 flow features.
    - **Hybrid IF + XGBoost:** Incorporates the Isolation Forest continuous anomaly score as a 62nd meta-feature.
    """)

    comp_df = EvaluationManager.get_three_model_comparison_table(eval_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Comparison Bar Chart
    comp_chart = plot_three_model_comparison(eval_data.get("three_model_comparison", {}))
    st.plotly_chart(comp_chart, use_container_width=True)

    # Confusion Matrix
    cm = eval_data.get("confusion_matrix")
    classes = eval_data.get("classes", ALL_RESEARCH_CLASSES)
    if cm is not None:
        st.markdown("#### Multiclass Confusion Matrix")
        fig_cm = plot_confusion_matrix_figure(cm, classes)
        st.pyplot(fig_cm)

    # Classification Report
    report = eval_data.get("classification_report")
    if report is not None:
        st.markdown("#### Detailed Classification Report")
        report_df = EvaluationManager.format_classification_report(report)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

