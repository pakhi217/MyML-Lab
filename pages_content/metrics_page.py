import streamlit as st
import numpy as np

from components.cards import section_title, section_caption, metric_row, empty_state, info_card
from components.charts import (
    confusion_matrix_chart, roc_curve_chart, actual_vs_predicted_chart, loss_curve_chart,
    feature_importance_chart, pca_projection_chart, cluster_visualization_chart, elbow_chart,
)
from components.theme import get_theme
from utils.data_utils import format_seconds


def render():
    theme, _ = get_theme()
    results = st.session_state.get("results")

    section_title("Metrics & Visualizations", "📚")
    section_caption("A full breakdown of the most recently trained model.")

    if not results:
        empty_state("Train a model from any algorithm page to see metrics here.", "📚")
        return

    info_card(
        f"{results['algorithm']}",
        f"Task type: {results['type'].title()}",
        badge="Latest Run", badge_type="pink",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if results["type"] in ("regression", "classification"):
        section_title("Performance Metrics", "📊")
        items = []
        for label, value in results["metrics"].items():
            if isinstance(value, float) and abs(value) < 1.5 and label not in ("RMSE", "MAE", "MSE"):
                display = f"{value*100:.2f}%" if label == "Accuracy" else f"{value:.4f}"
            else:
                display = f"{value:.4f}"
            items.append({"label": label, "value": display})
        items.append({"label": "Train Time", "value": format_seconds(results["train_time"]), "icon": "⏱️"})
        items.append({"label": "Predict Time", "value": format_seconds(results["predict_time"]), "icon": "⚡"})
        metric_row(items, columns=min(4, len(items)))

    elif results["type"] == "clustering":
        section_title("Clustering Metrics", "📊")
        metric_row([
            {"label": "Clusters", "value": f"{len(np.unique(results['labels']))}"},
            {"label": "Inertia", "value": f"{results['inertia']:.2f}"},
            {"label": "Iterations", "value": f"{results['n_iter']}"},
            {"label": "Train Time", "value": format_seconds(results["train_time"])},
        ])

    elif results["type"] == "pca":
        section_title("PCA Metrics", "📊")
        evr = results["explained_variance_ratio"]
        metric_row([
            {"label": "PC1 Variance", "value": f"{evr[0]*100:.1f}%"},
            {"label": "PC2 Variance", "value": f"{evr[1]*100:.1f}%" if len(evr) > 1 else "-"},
            {"label": "Total Explained", "value": f"{sum(evr)*100:.1f}%"},
            {"label": "Compute Time", "value": format_seconds(results["train_time"])},
        ])

    section_title("Visualizations", "📈")
    charts = []

    if results["type"] == "regression":
        charts.append(actual_vs_predicted_chart(theme, results["y_test"], results["y_pred"]))
        if results.get("loss_history"):
            charts.append(loss_curve_chart(theme, results["loss_history"]))

    elif results["type"] == "classification":
        charts.append(confusion_matrix_chart(theme, results["confusion_matrix"], results["labels"]))
        if results.get("roc_data"):
            rd = results["roc_data"]
            charts.append(roc_curve_chart(theme, rd["fpr"], rd["tpr"], rd["auc"]))
        if results.get("loss_history"):
            charts.append(loss_curve_chart(theme, results["loss_history"]))
        if results.get("feature_importances") is not None:
            charts.append(feature_importance_chart(theme, results["feature_names"], results["feature_importances"]))

    elif results["type"] == "clustering":
        charts.append(cluster_visualization_chart(theme, results["X"], results["labels"], results["centroids"]))
        charts.append(elbow_chart(theme, results["k_range"], results["inertias"]))

    elif results["type"] == "pca":
        charts.append(pca_projection_chart(theme, results["components"], results.get("color")))

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, chart in zip(cols, charts[i:i + 2]):
            with col:
                st.plotly_chart(chart, use_container_width=True)
