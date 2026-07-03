"""
MyML Lab — main application entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from components.theme import inject_global_css
from components.sidebar import render_sidebar
from components.navbar import render_navbar

from pages_content import (
    dashboard, dataset, regression, classification, clustering,
    pca_page, metrics_page, compare, reports, about,
)

st.set_page_config(
    page_title="MyML Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- session state defaults -------------------------------------------------
defaults = {
    "theme_mode": "dark",
    "active_page": "Dashboard",
    "df": None,
    "filename": None,
    "selected_features": [],
    "selected_target": None,
    "results": None,
    "compare_results": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- theme + chrome ----------------------------------------------------------
inject_global_css()
active_page = render_sidebar()

PAGE_META = {
    "Dashboard": ("Dashboard", "Welcome back — here's what's happening in your workspace."),
    "Dataset": ("Dataset", "Upload, preview, and prepare your data for training."),
    "Regression": ("Regression", "Predict continuous values with Linear Regression."),
    "Classification": ("Classification", "Predict categories with 4 different algorithms."),
    "Clustering": ("Clustering", "Discover structure in unlabeled data with K-Means."),
    "PCA": ("Dimensionality Reduction", "Visualize high-dimensional data in 2D with PCA."),
    "Metrics": ("Metrics & Visualizations", "Deep dive into your most recent model run."),
    "Compare": ("Compare Models", "MyML from-scratch implementation vs. scikit-learn."),
    "Reports": ("Reports", "Export predictions, metrics, models, and PDF summaries."),
    "About": ("About", "What MyML Lab is and how it was built."),
}

title, subtitle = PAGE_META.get(active_page, ("MyML Lab", ""))
render_navbar(title, subtitle)

PAGE_RENDERERS = {
    "Dashboard": dashboard.render,
    "Dataset": dataset.render,
    "Regression": regression.render,
    "Classification": classification.render,
    "Clustering": clustering.render,
    "PCA": pca_page.render,
    "Metrics": metrics_page.render,
    "Compare": compare.render,
    "Reports": reports.render,
    "About": about.render,
}

PAGE_RENDERERS.get(active_page, dashboard.render)()
