import streamlit as st
import numpy as np

from components.cards import section_title, section_caption, formula_box, metric_row, empty_state
from components.charts import cluster_visualization_chart, elbow_chart
from components.theme import get_theme
from core.kmeans import KMeansScratch
from utils.data_utils import get_numeric_columns, StandardScaler, format_seconds


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    section_title("K-Means Clustering", "🔍")
    section_caption("Groups unlabeled data points into k clusters based on feature similarity.")

    with st.expander("📘 Simple Explanation", expanded=False):
        st.write(
            "K-Means repeatedly assigns each point to its nearest centroid, then recomputes centroids "
            "as the mean of assigned points, until assignments stop changing."
        )
    formula_box("assign: argmin_k ||x − μₖ||² &nbsp;&nbsp;|&nbsp;&nbsp; update: μₖ = mean(points in cluster k)")

    if df is None:
        empty_state("Upload a dataset on the Dataset page first.", "📂")
        return

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for clustering.")
        return

    section_title("Feature Selection", "🎯")
    default_feats = st.session_state.get("selected_features") or numeric_cols[:2]
    default_feats = [f for f in default_feats if f in numeric_cols][:2] or numeric_cols[:2]
    chosen = st.multiselect("Features used for clustering (first 2 are plotted)", numeric_cols, default=default_feats)
    if len(chosen) < 2:
        st.warning("Select at least 2 numeric features.")
        return

    X = df[chosen].to_numpy(dtype=float)

    section_title("Hyperparameters", "🎛️")
    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Number of Clusters (k)", 2, 10, 3)
    with c2:
        max_iter = st.slider("Max Iterations", 10, 300, 100, step=10)

    if st.button("🚀 Run Clustering", use_container_width=False):
        progress = st.progress(0, text="Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        progress.progress(35, text="Running K-Means...")

        model = KMeansScratch(n_clusters=k, max_iterations=max_iter)
        model.fit(X_scaled)
        progress.progress(70, text="Computing elbow curve...")

        inertias = []
        k_range = range(2, min(9, len(X_scaled)) + 1) if len(X_scaled) > 2 else range(2, 3)
        for kk in k_range:
            m = KMeansScratch(n_clusters=kk, max_iterations=max_iter)
            m.fit(X_scaled)
            inertias.append(m.inertia_history[-1])

        progress.progress(100, text="Done")
        import time as _t
        _t.sleep(0.15)
        progress.empty()

        st.session_state.results = {
            "type": "clustering",
            "algorithm": "K-Means",
            "X": X_scaled,
            "labels": model.labels_,
            "centroids": model.centroids,
            "inertia": model.inertia_history[-1],
            "n_iter": model.n_iter_,
            "train_time": model.train_time_,
            "k_range": list(k_range),
            "inertias": inertias,
            "feature_names": chosen,
        }
        st.toast("Clustering complete!", icon="🎉")

    results = st.session_state.get("results")
    if results and results.get("type") == "clustering":
        section_title("Results", "📊")
        metric_row([
            {"label": "Clusters", "value": f"{len(np.unique(results['labels']))}", "icon": "🔵"},
            {"label": "Inertia", "value": f"{results['inertia']:.2f}", "icon": "📉"},
            {"label": "Iterations", "value": f"{results['n_iter']}", "icon": "🔁"},
            {"label": "Train Time", "value": format_seconds(results["train_time"]), "icon": "⏱️"},
        ])

        section_title("Visualizations", "📈")
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(
                cluster_visualization_chart(theme, results["X"], results["labels"], results["centroids"]),
                use_container_width=True,
            )
        with v2:
            st.plotly_chart(elbow_chart(theme, results["k_range"], results["inertias"]), use_container_width=True)
