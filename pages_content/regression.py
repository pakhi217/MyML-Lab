import time
import streamlit as st
import numpy as np
import pandas as pd

from components.cards import section_title, section_caption, formula_box, metric_row, empty_state
from components.charts import loss_curve_chart, actual_vs_predicted_chart
from components.theme import get_theme
from core.linear_regression import LinearRegressionScratch
from utils.data_utils import prepare_xy, train_test_split, StandardScaler, format_seconds
from utils.metrics import mse, rmse, mae, r2_score


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    section_title("Linear Regression", "📈")
    section_caption("Predicts a continuous value from a linear combination of input features.")

    with st.expander("📘 Simple Explanation", expanded=False):
        st.write(
            "Linear Regression finds the straight line (or hyperplane, with multiple features) "
            "that best fits the relationship between inputs and a continuous target, by minimizing "
            "the squared distance between predicted and actual values."
        )
    formula_box("ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b &nbsp;&nbsp;|&nbsp;&nbsp; Loss = (1/n)·Σ(y − ŷ)²")

    if df is None:
        empty_state("Upload a dataset on the Dataset page first.", "📂")
        return

    features = st.session_state.get("selected_features", [])
    target = st.session_state.get("selected_target")
    if not features or not target:
        empty_state("Select input features and a target column on the Dataset page first.", "🎯")
        return

    X, y_raw, feature_names, warn = prepare_xy(df, features, target)
    if warn:
        st.warning(warn)
    if not pd.api.types.is_numeric_dtype(y_raw):
        st.error("Target column must be numeric for regression. Pick a numeric target on the Dataset page.")
        return
    y = y_raw.to_numpy(dtype=float)

    if X.shape[1] == 0:
        st.error("No numeric features selected.")
        return

    section_title("Hyperparameters", "🎛️")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lr = st.select_slider("Learning Rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.03)
    with c2:
        iterations = st.slider("Iterations", 50, 2000, 500, step=50)
    with c3:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05)
    with c4:
        l2 = st.select_slider("L2 Penalty", options=[0.0, 0.001, 0.01, 0.1, 1.0], value=0.0)

    train_clicked = st.button("🚀 Train Model", use_container_width=False)

    if train_clicked:
        progress = st.progress(0, text="Preparing data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        progress.progress(30, text="Training model...")

        model = LinearRegressionScratch(learning_rate=lr, n_iterations=iterations, l2_penalty=l2)
        model.fit(X_train_s, y_train)
        progress.progress(75, text="Evaluating...")

        start = time.time()
        y_pred = model.predict(X_test_s)
        predict_time = time.time() - start
        progress.progress(100, text="Done")
        time.sleep(0.15)
        progress.empty()

        metrics = {
            "RMSE": rmse(y_test, y_pred),
            "MAE": mae(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "MSE": mse(y_test, y_pred),
        }

        st.session_state.results = {
            "type": "regression",
            "algorithm": "Linear Regression",
            "metrics": metrics,
            "train_time": model.train_time_,
            "predict_time": predict_time,
            "y_test": y_test,
            "y_pred": y_pred,
            "loss_history": model.loss_history,
            "feature_names": feature_names,
        }
        st.toast("Model trained successfully!", icon="🎉")

    results = st.session_state.get("results")
    if results and results.get("type") == "regression":
        section_title("Results", "📊")
        metric_row([
            {"label": "RMSE", "value": f"{results['metrics']['RMSE']:.4f}", "icon": "📏"},
            {"label": "MAE", "value": f"{results['metrics']['MAE']:.4f}", "icon": "📐"},
            {"label": "R² Score", "value": f"{results['metrics']['R2']:.4f}", "icon": "🎯"},
            {"label": "Train Time", "value": format_seconds(results["train_time"]), "icon": "⏱️"},
        ])

        section_title("Visualizations", "📈")
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(actual_vs_predicted_chart(theme, results["y_test"], results["y_pred"]), use_container_width=True)
        with v2:
            st.plotly_chart(loss_curve_chart(theme, results["loss_history"]), use_container_width=True)
