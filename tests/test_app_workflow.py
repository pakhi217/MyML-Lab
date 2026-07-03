"""
Full verification pass for MyML Lab.
Runs every page, every algorithm, theme toggle, and navigation click
through Streamlit's AppTest harness and reports PASS/FAIL per check.
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == 'FAIL' else ""))


def make_demo_df():
    rng = np.random.RandomState(7)
    centers = [(5.0, 3.4, 1.5, 0.25), (6.0, 2.8, 4.3, 1.3), (6.6, 3.0, 5.5, 2.0)]
    names = ["setosa_like", "versicolor_like", "virginica_like"]
    rows = []
    for (c1, c2, c3, c4), name in zip(centers, names):
        for _ in range(50):
            rows.append([c1 + rng.normal(0, 0.35), c2 + rng.normal(0, 0.35),
                         c3 + rng.normal(0, 0.4), c4 + rng.normal(0, 0.2), name])
    return pd.DataFrame(rows, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


df = make_demo_df()


def ss_get(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


def fresh(page=None, target=None, with_data=True):
    at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"))
    at.run(timeout=30)
    if with_data:
        at.session_state["df"] = df
        at.session_state["filename"] = "demo.csv"
        at.session_state["selected_features"] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        at.session_state["selected_target"] = target
    if page:
        at.session_state["active_page"] = page
    at.run(timeout=30)
    return at


# 1. Cold start, no data
at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"))
at.run(timeout=30)
check("Cold start (no dataset)", not at.exception)

# 2. Theme toggle
theme_btn = [b for b in at.button if "Dark" in b.label or "Light" in b.label]
check("Theme toggle button present", len(theme_btn) == 1)
if theme_btn:
    at.button(key=theme_btn[0].key).click().run(timeout=30)
    check("Theme toggle click", not at.exception, str(at.exception))
    check("Theme actually switched", at.session_state["theme_mode"] == "light")

# 3. Sidebar navigation via button click (not session_state injection)
nav_btn = [b for b in at.button if "Dataset" in b.label]
if nav_btn:
    at.button(key=nav_btn[0].key).click().run(timeout=30)
    check("Sidebar nav click -> Dataset", at.session_state["active_page"] == "Dataset" and not at.exception)

# 4. Every page with no dataset
for page in ["Dashboard", "Dataset", "Regression", "Classification", "Clustering", "PCA", "Metrics", "Compare", "Reports", "About"]:
    at = fresh(page=page, with_data=False)
    check(f"Page '{page}' renders with empty state (no data)", not at.exception, str(at.exception))

# 5. Dataset page with data + demo button path
at = fresh(page="Dataset")
check("Dataset page renders with data loaded", not at.exception, str(at.exception))

# 6. Regression train
at = fresh(page="Regression", target="sepal_length")
tb = [b for b in at.button if b.label == "🚀 Train Model"]
check("Regression Train button present", len(tb) == 1)
if tb:
    at.button(key=tb[0].key).click().run(timeout=60)
    check("Regression training executes", not at.exception, str(at.exception))
    res = ss_get(at, "results")
    check("Regression results stored", res is not None and res.get("type") == "regression")
    if res:
        check("Regression metrics sane (R2 is finite)", np.isfinite(res["metrics"]["R2"]))

# 7. Classification, all 4 algorithms
for algo in ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]:
    at = fresh(page="Classification", target="species")
    at.selectbox[0].set_value(algo).run(timeout=30)
    tb = [b for b in at.button if b.label == "🚀 Train Model"]
    check(f"Classification [{algo}] Train button present", len(tb) == 1)
    if tb:
        at.button(key=tb[0].key).click().run(timeout=60)
        check(f"Classification [{algo}] training executes", not at.exception, str(at.exception))
        res = ss_get(at, "results")
        check(f"Classification [{algo}] results stored", res is not None and res.get("type") == "classification")
        if res:
            acc = res["metrics"]["Accuracy"]
            check(f"Classification [{algo}] accuracy in [0,1]", 0.0 <= acc <= 1.0, str(acc))

# 8. Clustering
at = fresh(page="Clustering", target="species")
tb = [b for b in at.button if b.label == "🚀 Run Clustering"]
check("Clustering Run button present", len(tb) == 1)
if tb:
    at.button(key=tb[0].key).click().run(timeout=60)
    check("Clustering executes", not at.exception, str(at.exception))
    res = ss_get(at, "results")
    check("Clustering results stored", res is not None and res.get("type") == "clustering")

# 9. PCA
at = fresh(page="PCA", target="species")
tb = [b for b in at.button if b.label == "🚀 Run PCA"]
check("PCA Run button present", len(tb) == 1)
if tb:
    at.button(key=tb[0].key).click().run(timeout=60)
    check("PCA executes", not at.exception, str(at.exception))
    res = ss_get(at, "results")
    check("PCA results stored", res is not None and res.get("type") == "pca")
    if res:
        evr_sum = sum(res["explained_variance_ratio"])
        check("PCA explained variance ratio <= 1.0", evr_sum <= 1.0001, str(evr_sum))

# 10. Compare, all algorithms with correct target types
algo_targets = {
    "Linear Regression": "sepal_length",
    "Logistic Regression": "species",
    "KNN": "species",
    "Decision Tree": "species",
    "Random Forest": "species",
}
for algo, target in algo_targets.items():
    at = fresh(page="Compare", target=target)
    at.selectbox[0].set_value(algo).run(timeout=30)
    tb = [b for b in at.button if b.label == "🚀 Run Comparison"]
    check(f"Compare [{algo}] Run button present", len(tb) == 1)
    if tb:
        at.button(key=tb[0].key).click().run(timeout=60)
        check(f"Compare [{algo}] executes", not at.exception, str(at.exception))
        cr = ss_get(at, "compare_results")
        check(f"Compare [{algo}] results stored", cr is not None and cr["algo"] == algo)

# 11. Metrics + Reports pages after training a classification model
at = fresh(page="Classification", target="species")
tb = [b for b in at.button if b.label == "🚀 Train Model"]
at.button(key=tb[0].key).click().run(timeout=60)
trained_results = at.session_state["results"]

at2 = fresh(page="Metrics", target="species")
at2.session_state["results"] = trained_results
at2.run(timeout=30)
check("Metrics page renders trained results", not at2.exception, str(at2.exception))

at3 = fresh(page="Reports", target="species")
at3.session_state["results"] = trained_results
at3.run(timeout=30)
check("Reports page renders trained results", not at3.exception, str(at3.exception))

# 12. Actually exercise the report-generating functions directly (CSV/JSON/model/PDF)
from pages_content.reports import _predictions_csv, _metrics_report, _build_pdf
try:
    csv_out = _predictions_csv(trained_results)
    check("Reports: predictions CSV generated", isinstance(csv_out, str) and len(csv_out) > 0)
except Exception as e:
    check("Reports: predictions CSV generated", False, str(e))

try:
    json_out = _metrics_report(trained_results)
    check("Reports: metrics JSON generated", isinstance(json_out, str) and "algorithm" in json_out)
except Exception as e:
    check("Reports: metrics JSON generated", False, str(e))

try:
    pdf_bytes = _build_pdf(trained_results)
    check("Reports: PDF generated", isinstance(pdf_bytes, bytes) and pdf_bytes[:4] == b"%PDF")
except Exception as e:
    check("Reports: PDF generated", False, str(e))

try:
    import pickle
    model_bytes = pickle.dumps(trained_results)
    reloaded = pickle.loads(model_bytes)
    check("Reports: pickled model round-trips", reloaded["algorithm"] == trained_results["algorithm"])
except Exception as e:
    check("Reports: pickled model round-trips", False, str(e))

# ---------------------------------------------------------------
print("\n" + "=" * 60)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"TOTAL: {n_pass} passed, {n_fail} failed, {len(results)} checks")
if n_fail:
    print("\nFAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f" - {name}: {detail}")
sys.exit(1 if n_fail else 0)
