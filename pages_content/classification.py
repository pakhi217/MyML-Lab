import time
import streamlit as st
import numpy as np

from components.cards import section_title, section_caption, formula_box, metric_row, empty_state
from components.charts import (
    loss_curve_chart, confusion_matrix_chart, roc_curve_chart, feature_importance_chart,
)
from components.theme import get_theme
from core.logistic_regression import LogisticRegressionScratch
from core.knn import KNNScratch
from core.decision_tree import DecisionTreeScratch
from core.random_forest import RandomForestScratch
from utils.data_utils import prepare_xy, train_test_split, StandardScaler, encode_labels, format_seconds
from utils.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc_score

ALGO_INFO = {
    "Logistic Regression": {
        "explanation": "Estimates the probability of a class using a sigmoid function applied to a linear combination of features.",
        "formula": "p = 1 / (1 + e^-(w·x + b)) &nbsp;&nbsp;|&nbsp;&nbsp; Loss = Binary Cross-Entropy",
    },
    "KNN": {
        "explanation": "Classifies a point by majority vote among its k closest neighbors in feature space.",
        "formula": "distance(a, b) = √Σ(aᵢ − bᵢ)² &nbsp;&nbsp;|&nbsp;&nbsp; ŷ = mode(labels of k nearest)",
    },
    "Decision Tree": {
        "explanation": "Recursively splits the data on the feature/threshold that most reduces class impurity (Gini).",
        "formula": "Gini = 1 − Σ pᵢ² &nbsp;&nbsp;|&nbsp;&nbsp; Gain = Gini(parent) − weighted Gini(children)",
    },
    "Random Forest": {
        "explanation": "Trains many decision trees on bootstrapped samples with random feature subsets, then combines them by majority vote.",
        "formula": "ŷ = mode(tree₁(x), tree₂(x), ..., treeₙ(x))",
    },
}


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    section_title("Classification", "📊")
    section_caption("Choose an algorithm to predict a categorical target.")

    algo = st.selectbox("Algorithm", list(ALGO_INFO.keys()))
    info = ALGO_INFO[algo]

    with st.expander("📘 Simple Explanation", expanded=False):
        st.write(info["explanation"])
    formula_box(info["formula"])

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
    if X.shape[1] == 0:
        st.error("No numeric features selected.")
        return

    y, class_mapping = encode_labels(y_raw)
    class_names = list(class_mapping.keys())
    is_binary = len(class_names) == 2

    section_title("Hyperparameters", "🎛️")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05, key=f"ts_{algo}")

    params = {}
    if algo == "Logistic Regression":
        with c2:
            params["lr"] = st.select_slider("Learning Rate", [0.001, 0.01, 0.05, 0.1, 0.3], value=0.1)
        with c3:
            params["iterations"] = st.slider("Iterations", 50, 2000, 500, step=50)
    elif algo == "KNN":
        with c2:
            params["k"] = st.slider("k (neighbors)", 1, 25, 5)
    elif algo == "Decision Tree":
        with c2:
            params["max_depth"] = st.slider("Max Depth", 1, 15, 5)
        with c3:
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
    elif algo == "Random Forest":
        with c2:
            params["n_estimators"] = st.slider("Number of Trees", 5, 100, 20, step=5)
        with c3:
            params["max_depth"] = st.slider("Max Depth", 1, 15, 5)

    if not is_binary and algo != "KNN":
        st.info(f"Detected {len(class_names)} classes: {', '.join(map(str, class_names))} (multi-class).")

    if st.button("🚀 Train Model", use_container_width=False):
        progress = st.progress(0, text="Preparing data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y.astype(float), test_size=test_size)
        y_train, y_test = y_train.astype(int), y_test.astype(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        progress.progress(30, text=f"Training {algo}...")

        loss_history = None
        y_score = None

        if algo == "Logistic Regression":
            model = LogisticRegressionScratch(learning_rate=params["lr"], n_iterations=params["iterations"])
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            loss_history = model.loss_history
            if is_binary:
                y_score = model.predict_proba(X_test_s)
        elif algo == "KNN":
            model = KNNScratch(k=params["k"])
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            if is_binary:
                y_score = model.predict_proba(X_test_s)
        elif algo == "Decision Tree":
            model = DecisionTreeScratch(max_depth=params["max_depth"], min_samples_split=params["min_samples_split"])
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:  # Random Forest
            model = RandomForestScratch(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

        progress.progress(80, text="Evaluating...")
        start = time.time()
        _ = model.predict(X_test_s[:1]) if len(X_test_s) else None
        predict_time = time.time() - start
        progress.progress(100, text="Done")
        time.sleep(0.15)
        progress.empty()

        avg = "binary" if is_binary else "macro"
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average=avg),
            "Recall": recall_score(y_test, y_pred, average=avg),
            "F1 Score": f1_score(y_test, y_pred, average=avg),
        }

        cm, labels = confusion_matrix(y_test, y_pred)
        feature_importances = getattr(model, "feature_importances_", None)

        roc_data = None
        if is_binary and y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": auc_score(fpr, tpr)}

        st.session_state.results = {
            "type": "classification",
            "algorithm": algo,
            "metrics": metrics,
            "train_time": model.train_time_,
            "predict_time": predict_time,
            "confusion_matrix": cm,
            "labels": [class_names[i] if i < len(class_names) else i for i in labels],
            "loss_history": loss_history,
            "roc_data": roc_data,
            "feature_importances": feature_importances,
            "feature_names": feature_names,
        }
        st.toast("Model trained successfully!", icon="🎉")

    results = st.session_state.get("results")
    if results and results.get("type") == "classification":
        section_title("Results", "📊")
        metric_row([
            {"label": "Accuracy", "value": f"{results['metrics']['Accuracy']*100:.2f}%", "icon": "🎯"},
            {"label": "Precision", "value": f"{results['metrics']['Precision']:.3f}", "icon": "📐"},
            {"label": "Recall", "value": f"{results['metrics']['Recall']:.3f}", "icon": "🔁"},
            {"label": "F1 Score", "value": f"{results['metrics']['F1 Score']:.3f}", "icon": "⚖️"},
        ])

        section_title("Visualizations", "📈")
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(
                confusion_matrix_chart(theme, results["confusion_matrix"], results["labels"]),
                use_container_width=True,
            )
        with v2:
            if results.get("roc_data"):
                st.plotly_chart(
                    roc_curve_chart(theme, results["roc_data"]["fpr"], results["roc_data"]["tpr"], results["roc_data"]["auc"]),
                    use_container_width=True,
                )
            elif results.get("loss_history"):
                st.plotly_chart(loss_curve_chart(theme, results["loss_history"]), use_container_width=True)
            elif results.get("feature_importances") is not None:
                st.plotly_chart(
                    feature_importance_chart(theme, results["feature_names"], results["feature_importances"]),
                    use_container_width=True,
                )
