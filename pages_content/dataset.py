import streamlit as st
import pandas as pd
from components.cards import section_title, section_caption, empty_state, metric_row
from components.charts import correlation_heatmap
from components.theme import get_theme
from utils.data_utils import missing_values_summary, get_numeric_columns


def _generate_demo_dataset(n_per_class=50, seed=7):
    """Creates a small synthetic 3-class flower-measurement dataset for demo purposes only."""
    import numpy as np
    rng = np.random.RandomState(seed)
    centers = [(5.0, 3.4, 1.5, 0.25), (6.0, 2.8, 4.3, 1.3), (6.6, 3.0, 5.5, 2.0)]
    names = ["setosa_like", "versicolor_like", "virginica_like"]
    rows = []
    for (cx1, cx2, cx3, cx4), name in zip(centers, names):
        for _ in range(n_per_class):
            rows.append([
                round(cx1 + rng.normal(0, 0.35), 2),
                round(cx2 + rng.normal(0, 0.35), 2),
                round(cx3 + rng.normal(0, 0.4), 2),
                round(cx4 + rng.normal(0, 0.2), 2),
                name,
            ])
    return pd.DataFrame(rows, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


def render():
    theme, _ = get_theme()

    section_title("Upload Dataset", "📂")
    section_caption("CSV files only. Your data stays in-session and is never sent anywhere.")

    uploaded = st.file_uploader("Drop a CSV file here or click to browse", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.selected_features = []
            st.session_state.selected_target = None
            st.toast(f"Loaded {uploaded.name}", icon="✅")
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    df = st.session_state.get("df")
    if df is None:
        empty_state("Upload a CSV to preview it here.", "🗂️")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Use demo dataset instead"):
            st.session_state.df = _generate_demo_dataset()
            st.session_state.filename = "flower_demo.csv"
            st.session_state.selected_features = []
            st.session_state.selected_target = None
            st.rerun()
        return

    metric_row([
        {"label": "Rows", "value": f"{len(df):,}", "icon": "🧾"},
        {"label": "Columns", "value": f"{df.shape[1]}", "icon": "📐"},
        {"label": "Missing Cells", "value": f"{int(df.isnull().sum().sum())}", "icon": "⚠️"},
        {"label": "Numeric Columns", "value": f"{len(get_numeric_columns(df))}", "icon": "🔢"},
    ])

    section_title("Dataset Preview", "👁️")
    st.dataframe(df.head(50), use_container_width=True, height=280)

    tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "🧩 Missing Values", "🔤 Data Types"])

    with tab1:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] > 0:
            st.dataframe(numeric_df.describe().T, use_container_width=True)
        else:
            st.info("No numeric columns found for summary statistics.")

    with tab2:
        st.dataframe(missing_values_summary(df), use_container_width=True)

    with tab3:
        dtypes_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True)

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) >= 2:
        section_title("Correlation Heatmap", "🧠")
        st.plotly_chart(correlation_heatmap(theme, df[numeric_cols].corr()), use_container_width=True)

    section_title("Feature Selection", "🎯")
    section_caption("Choose the input features and target column used for training on the algorithm pages.")

    c1, c2 = st.columns([2, 1])
    with c1:
        features = st.multiselect(
            "Input Features (X)",
            options=list(df.columns),
            default=st.session_state.get("selected_features", []),
        )
    with c2:
        target = st.selectbox(
            "Target Column (y)",
            options=["-- none --"] + list(df.columns),
            index=(list(df.columns).index(st.session_state.selected_target) + 1)
            if st.session_state.get("selected_target") in df.columns else 0,
        )

    if st.button("💾 Save Feature Selection", use_container_width=False):
        st.session_state.selected_features = features
        st.session_state.selected_target = None if target == "-- none --" else target
        st.toast("Feature selection saved", icon="🎯")
