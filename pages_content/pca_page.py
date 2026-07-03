import streamlit as st
import numpy as np
import pandas as pd

from components.cards import section_title, section_caption, formula_box, metric_row, empty_state
from components.charts import pca_projection_chart
from components.theme import get_theme
from core.pca import PCAScratch
from utils.data_utils import get_numeric_columns, StandardScaler, format_seconds, encode_labels


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    section_title("Principal Component Analysis (PCA)", "📉")
    section_caption("Projects high-dimensional data onto the directions of maximum variance.")

    with st.expander("📘 Simple Explanation", expanded=False):
        st.write(
            "PCA finds new axes (principal components) that are linear combinations of the original "
            "features, ordered by how much variance in the data they capture. Projecting onto the top "
            "2 components lets us visualize high-dimensional data in 2D."
        )
    formula_box("Cov(X) = (1/n)·XᵀX &nbsp;&nbsp;|&nbsp;&nbsp; components = eigenvectors of Cov(X), sorted by eigenvalue")

    if df is None:
        empty_state("Upload a dataset on the Dataset page first.", "📂")
        return

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for PCA.")
        return

    section_title("Feature Selection", "🎯")
    default_feats = [f for f in st.session_state.get("selected_features", []) if f in numeric_cols] or numeric_cols
    chosen = st.multiselect("Features to reduce", numeric_cols, default=default_feats)
    color_col = st.selectbox("Color points by (optional)", ["-- none --"] + list(df.columns))

    if len(chosen) < 2:
        st.warning("Select at least 2 numeric features.")
        return

    X = df[chosen].to_numpy(dtype=float)

    if st.button("🚀 Run PCA", use_container_width=False):
        progress = st.progress(0, text="Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        progress.progress(50, text="Computing principal components...")

        model = PCAScratch(n_components=2)
        components = model.fit_transform(X_scaled)
        progress.progress(100, text="Done")
        import time as _t
        _t.sleep(0.15)
        progress.empty()

        color = None
        if color_col != "-- none --":
            if pd.api.types.is_numeric_dtype(df[color_col]):
                color = df[color_col].to_numpy()
            else:
                encoded, _ = encode_labels(df[color_col])
                color = encoded

        st.session_state.results = {
            "type": "pca",
            "algorithm": "PCA",
            "components": components,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "train_time": model.train_time_,
            "color": color,
            "feature_names": chosen,
        }
        st.toast("PCA complete!", icon="🎉")

    results = st.session_state.get("results")
    if results and results.get("type") == "pca":
        section_title("Results", "📊")
        evr = results["explained_variance_ratio"]
        metric_row([
            {"label": "PC1 Variance", "value": f"{evr[0]*100:.1f}%", "icon": "🧭"},
            {"label": "PC2 Variance", "value": f"{evr[1]*100:.1f}%" if len(evr) > 1 else "-", "icon": "🧭"},
            {"label": "Total Explained", "value": f"{sum(evr)*100:.1f}%", "icon": "📦"},
            {"label": "Compute Time", "value": format_seconds(results["train_time"]), "icon": "⏱️"},
        ])

        section_title("PCA Projection", "📈")
        st.plotly_chart(pca_projection_chart(theme, results["components"], results.get("color")), use_container_width=True)
