"""
Compare Models page.

This is the ONLY place in the entire app that imports scikit-learn —
used purely as a benchmark reference to show how the from-scratch
NumPy implementations stack up against a production-grade library.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd

from components.cards import section_title, section_caption, metric_row, empty_state
from components.theme import get_theme
from utils.data_utils import prepare_xy, train_test_split, StandardScaler, encode_labels, format_seconds
from utils.metrics import accuracy_score, precision_score, r2_score, rmse

from core.linear_regression import LinearRegressionScratch
from core.logistic_regression import LogisticRegressionScratch
from core.knn import KNNScratch
from core.decision_tree import DecisionTreeScratch
from core.random_forest import RandomForestScratch


ALGOS = {
    "Linear Regression": "regression",
    "Logistic Regression": "classification",
    "KNN": "classification",
    "Decision Tree": "classification",
    "Random Forest": "classification",
}


def _run_scratch(algo, X_train, X_test, y_train, y_test):
    if algo == "Linear Regression":
        model = LinearRegressionScratch(learning_rate=0.05, n_iterations=800)
    elif algo == "Logistic Regression":
        model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=800)
    elif algo == "KNN":
        model = KNNScratch(k=5)
    elif algo == "Decision Tree":
        model = DecisionTreeScratch(max_depth=6)
    else:
        model = RandomForestScratch(n_estimators=25, max_depth=6)

    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    return y_pred, model.train_time_, predict_time


def _run_sklearn(algo, X_train, X_test, y_train, y_test):
    if algo == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif algo == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    elif algo == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
    elif algo == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=6)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=25, max_depth=6)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    return y_pred, train_time, predict_time


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    section_title("Compare Models", "⚖️")
    section_caption("MyML (from-scratch NumPy) vs. scikit-learn, trained on the same split of your data.")

    if df is None:
        empty_state("Upload a dataset on the Dataset page first.", "📂")
        return

    features = st.session_state.get("selected_features", [])
    target = st.session_state.get("selected_target")
    if not features or not target:
        empty_state("Select input features and a target column on the Dataset page first.", "🎯")
        return

    algo = st.selectbox("Algorithm", list(ALGOS.keys()))
    task = ALGOS[algo]

    X, y_raw, feature_names, warn = prepare_xy(df, features, target)
    if warn:
        st.warning(warn)
    if X.shape[1] == 0:
        st.error("No numeric features selected.")
        return

    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y_raw):
            st.error("Target column must be numeric for this algorithm.")
            return
        y = y_raw.to_numpy(dtype=float)
    else:
        y, _ = encode_labels(y_raw)
        y = y.astype(float)

    if st.button("🚀 Run Comparison", use_container_width=False):
        with st.spinner("Training both implementations..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            y_train_c = y_train.astype(int) if task == "classification" else y_train
            y_test_c = y_test.astype(int) if task == "classification" else y_test

            scratch_pred, scratch_train_t, scratch_predict_t = _run_scratch(
                algo, X_train_s, X_test_s, y_train_c, y_test_c
            )
            sklearn_pred, sklearn_train_t, sklearn_predict_t = _run_sklearn(
                algo, X_train_s, X_test_s, y_train_c, y_test_c
            )

            if task == "regression":
                scratch_score = r2_score(y_test_c, scratch_pred)
                sklearn_score = r2_score(y_test_c, sklearn_pred)
                score_label = "R² Score"
            else:
                scratch_score = accuracy_score(y_test_c, scratch_pred)
                sklearn_score = accuracy_score(y_test_c, sklearn_pred)
                score_label = "Accuracy"

        st.session_state.compare_results = {
            "algo": algo, "task": task, "score_label": score_label,
            "scratch_score": scratch_score, "sklearn_score": sklearn_score,
            "scratch_train_t": scratch_train_t, "sklearn_train_t": sklearn_train_t,
            "scratch_predict_t": scratch_predict_t, "sklearn_predict_t": sklearn_predict_t,
        }
        st.toast("Comparison complete!", icon="🎉")

    cr = st.session_state.get("compare_results")
    if cr and cr["algo"] == algo:
        section_title("Results", "📊")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="badge badge-pink">MyML (Scratch)</div>', unsafe_allow_html=True)
            metric_row([
                {"label": cr["score_label"], "value": f"{cr['scratch_score']:.4f}"},
                {"label": "Train Time", "value": format_seconds(cr["scratch_train_t"])},
            ], columns=2)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="badge badge-purple">scikit-learn</div>', unsafe_allow_html=True)
            metric_row([
                {"label": cr["score_label"], "value": f"{cr['sklearn_score']:.4f}"},
                {"label": "Train Time", "value": format_seconds(cr["sklearn_train_t"])},
            ], columns=2)
            st.markdown("</div>", unsafe_allow_html=True)

        diff = cr["scratch_score"] - cr["sklearn_score"]
        diff_pct = diff * 100
        section_title("Performance Difference", "📐")
        sign = "+" if diff >= 0 else ""
        st.markdown(
            f"""
            <div class="card">
                MyML scores <b>{sign}{diff_pct:.2f} points</b> {'higher' if diff >= 0 else 'lower'} than scikit-learn
                on {cr['score_label']}, and trains
                {'faster' if cr['scratch_train_t'] < cr['sklearn_train_t'] else 'slower'} by
                {format_seconds(abs(cr['scratch_train_t'] - cr['sklearn_train_t']))}.
            </div>
            """,
            unsafe_allow_html=True,
        )
