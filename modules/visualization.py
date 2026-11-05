"""
Visualization Module for EV Charging Intrusion Detection System.
Provides publication-quality cybersecurity charts using dark navy, slate, and cyan/blue palettes.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# Color Theme Constants
COLOR_NORMAL = "#10B981"      # Emerald Green for Benign / Normal
COLOR_MALICIOUS = "#EF4444"   # Crimson Red for Attacks
COLOR_NAVY_DARK = "#0F172A"   # Slate 900
COLOR_BLUE_ACCENT = "#3B82F6" # Electric Blue
COLOR_CYAN = "#06B6D4"        # Cyan Accent
COLOR_INDIVIDUAL_IF = "#F59E0B"
COLOR_INDIVIDUAL_XGB = "#3B82F6"
COLOR_HYBRID = "#10B981"


def plot_normal_vs_malicious_donut(normal_count: int, malicious_count: int) -> go.Figure:
    """
    Renders a donut chart comparing Normal vs Malicious traffic volumes.
    Green = Normal, Red = Malicious.
    """
    labels = ["Normal Traffic (Benign)", "Malicious Traffic (Attacks)"]
    values = [normal_count, malicious_count]
    colors = [COLOR_NORMAL, COLOR_MALICIOUS]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#1E293B", width=2)),
                textinfo="label+percent+value",
                textposition="outside",
                hoverinfo="label+value+percent",
                pull=[0, 0.05] if malicious_count > 0 else [0, 0]
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Traffic Composition: Normal vs. Malicious</b>",
            font=dict(size=16, color="#F8FAFC")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="#E2E8F0")
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        height=380
    )
    return fig


def plot_attack_distribution_bar(attack_counts: Dict[str, int]) -> go.Figure:
    """
    Renders a bar chart showing the frequency of each attack type.
    """
    df = pd.DataFrame(list(attack_counts.items()), columns=["Attack Category", "Count"])
    # Sort descending
    df = df.sort_values(by="Count", ascending=True)

    # Assign colors: Normal is Green, all attacks are shades of Red/Orange/Purple
    colors = [
        COLOR_NORMAL if cat == "Normal" else COLOR_BLUE_ACCENT
        for cat in df["Attack Category"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["Count"],
            y=df["Attack Category"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
            text=df["Count"],
            textposition="auto"
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Attack-Wise Distribution (Frequency of Detections)</b>",
            font=dict(size=16, color="#F8FAFC")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Number of Network Flow Records",
            gridcolor="#334155",
            color="#CBD5E1"
        ),
        yaxis=dict(
            title="",
            color="#E2E8F0"
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=400
    )
    return fig


def plot_six_category_bar(primary_df: pd.DataFrame) -> go.Figure:
    """
    Renders bar chart for the six primary target categories.
    """
    colors = [
        COLOR_NORMAL if row["Attack Name"] == "Normal" else COLOR_MALICIOUS
        for _, row in primary_df.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            x=primary_df["Attack Name"],
            y=primary_df["Detected Records"],
            marker=dict(color=colors, line=dict(color="#1E293B", width=1.5)),
            text=primary_df["Detected Records"],
            textposition="outside"
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Primary Six Categories Detection Count</b>",
            font=dict(size=16, color="#F8FAFC")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Category",
            color="#E2E8F0"
        ),
        yaxis=dict(
            title="Records Detected",
            gridcolor="#334155",
            color="#CBD5E1"
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=380
    )
    return fig


def plot_three_model_comparison(comparison_dict: Dict[str, Any]) -> go.Figure:
    """
    Visualizes the comparative performance of:
    1. Isolation Forest (Binary Anomaly Detection)
    2. Baseline XGBoost (61 Features)
    3. Hybrid IF + XGBoost (62 Features)
    """
    metrics = comparison_dict.get("metrics", ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"])
    if_scores = comparison_dict.get("isolation_forest", [24.27, 99.94, 24.28, 39.06])
    xgb_scores = comparison_dict.get("xgb_individual", [99.95, 94.09, 95.09, 94.54])
    hyb_scores = comparison_dict.get("hybrid_model", [99.96, 94.09, 95.09, 94.54])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=if_scores,
        name="Individual Isolation Forest",
        marker_color=COLOR_INDIVIDUAL_IF,
        text=[f"{v:.1f}%" for v in if_scores],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=xgb_scores,
        name="Individual XGBoost (61 Features)",
        marker_color=COLOR_INDIVIDUAL_XGB,
        text=[f"{v:.1f}%" for v in xgb_scores],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=hyb_scores,
        name="Hybrid IF + XGBoost (62 Features)",
        marker_color=COLOR_HYBRID,
        text=[f"{v:.1f}%" for v in hyb_scores],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Comparative Evaluation: Individual vs. Hybrid IDS Architecture</b>",
            font=dict(size=16, color="#F8FAFC")
        ),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#E2E8F0"),
        yaxis=dict(
            title="Score (%)",
            range=[0, 115],
            gridcolor="#334155",
            color="#CBD5E1"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="#E2E8F0")
        ),
        margin=dict(l=20, r=20, t=50, b=60),
        height=420
    )
    return fig


def plot_confusion_matrix_figure(cm: List[List[int]], classes: List[str]):
    """
    Generates a Matplotlib / Seaborn heatmap of the multiclass confusion matrix.
    """
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0B132B")
    ax.set_facecolor("#0B132B")

    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=True,
        cbar_kws={"label": "Traffic Record Count"}
    )

    ax.set_title("Confusion Matrix — Hybrid IF + XGBoost Evaluation", color="#F8FAFC", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Traffic Class", color="#E2E8F0", fontsize=11, labelpad=8)
    ax.set_ylabel("Ground Truth Class", color="#E2E8F0", fontsize=11, labelpad=8)

    plt.xticks(rotation=45, ha="right", color="#CBD5E1", fontsize=9)
    plt.yticks(rotation=0, color="#CBD5E1", fontsize=9)
    plt.tight_layout()
    return fig


def plot_anomaly_score_histogram(scores: np.ndarray, classifications: np.ndarray) -> go.Figure:
    """
    Visualizes distribution of Isolation Forest anomaly scores separated by Normal vs Malicious.
    """
    df = pd.DataFrame({
        "Score": scores,
        "Class": classifications
    })

    fig = px.histogram(
        df,
        x="Score",
        color="Class",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Normal": COLOR_NORMAL, "Malicious": COLOR_MALICIOUS},
        opacity=0.75
    )

    fig.update_layout(
        title=dict(
            text="<b>Isolation Forest Anomaly Score Distribution</b>",
            font=dict(size=16, color="#F8FAFC")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Anomaly Score (-decision_function)",
            gridcolor="#334155",
            color="#CBD5E1"
        ),
        yaxis=dict(
            title="Record Count",
            gridcolor="#334155",
            color="#CBD5E1"
        ),
        legend=dict(
            title="Classification",
            font=dict(color="#E2E8F0")
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=360
    )
    return fig

