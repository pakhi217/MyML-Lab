# 🧠 MyML Lab

**A machine learning library and interactive dashboard, built entirely from scratch with NumPy.**

MyML Lab implements classic ML algorithms — regression, classification, clustering, and
dimensionality reduction — using only NumPy for the math. Scikit-learn is **not** used inside
any algorithm; it only appears on the *Compare Models* page as a benchmark reference.

The frontend is a custom-themed Streamlit dashboard (dark / light, neon pink + purple accents)
inspired by modern SaaS products like Linear, Vercel, and Stripe.

---

## ✨ Features

- **7 algorithms from scratch**: Linear Regression, Logistic Regression, KNN, Decision Tree,
  Random Forest, K-Means, PCA — all implemented with just NumPy.
- **Premium dashboard UI** with a custom dark/light theme, glowing active nav state, animated
  cards, and a consistent design language across every widget.
- **Full ML workflow**: upload → preview → feature selection → train → metrics → visualize →
  compare vs. scikit-learn → export.
- **Interactive Plotly charts**: correlation heatmap, confusion matrix, ROC curve, feature
  importance, actual vs predicted, loss curves, PCA projection, cluster plots, elbow method.
- **Exportable reports**: predictions CSV, metrics JSON, pickled model, and a generated PDF summary.

## 📁 Project Structure

```
myml_lab/
├── app.py                  # Main entry point / page router
├── requirements.txt
├── core/                   # NumPy-only ML algorithms
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── kmeans.py
│   └── pca.py
├── components/              # Reusable UI building blocks
│   ├── theme.py             # Dark/light palettes + global CSS injection
│   ├── sidebar.py           # Custom nav with glow-on-active
│   ├── navbar.py            # Top bar: search, theme toggle, avatar
│   ├── cards.py             # Metric cards, info cards, section headers
│   └── charts.py            # Theme-aware Plotly chart builders
├── pages_content/           # One module per dashboard page
│   ├── dashboard.py
│   ├── dataset.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   ├── pca_page.py
│   ├── metrics_page.py
│   ├── compare.py           # Only file that imports scikit-learn
│   ├── reports.py
│   └── about.py
└── utils/
    ├── metrics.py            # Accuracy, precision, recall, F1, RMSE, R², ROC/AUC — all from scratch
    └── data_utils.py         # Train/test split, StandardScaler, label encoding
```

## ✅ Testing

Two independent test suites are included in `tests/`:

- `tests/test_algorithms.py` — numerical correctness checks for each scratch algorithm
  (e.g. Linear Regression recovers known coefficients, K-Means separates known clusters,
  PCA's first component captures most variance on correlated features). Runs in isolation,
  no Streamlit required.
- `tests/test_app_workflow.py` — end-to-end checks using Streamlit's `AppTest` harness:
  every page render, every algorithm's train button, theme toggle, sidebar navigation, and
  the Reports page's CSV/JSON/PDF/model exports.

```bash
python tests/test_algorithms.py
python tests/test_app_workflow.py
```

Both should print `... passed, 0 failed`.

## 🚀 Getting Started

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

No dataset handy? Click **"Use demo dataset instead"** on the Dataset page to generate a small
synthetic flower-measurement dataset on the fly.

## 🧩 How training works

1. Upload a CSV on the **Dataset** page and save a feature/target selection.
2. Open an algorithm page (**Regression**, **Classification**, **Clustering**, or **PCA**),
   tune hyperparameters, and click **Train**.
3. Results are stored in session state and immediately available on the **Metrics** page.
4. Use **Compare** to benchmark the same algorithm against scikit-learn on your data.
5. Export everything from **Reports**.

## 🛠️ Tech Stack

| Layer          | Tools |
|----------------|-------|
| Frontend       | Streamlit, custom CSS, Plotly |
| ML Core        | NumPy, Pandas |
| Reports        | ReportLab (PDF), JSON, CSV |
| Benchmarking   | scikit-learn (Compare page only) |

## 📌 Notes

- The scratch algorithms only accept **numeric** feature columns; non-numeric selections are
  dropped with a visible warning rather than silently encoded.
- Session state resets on page refresh — this is a demo/portfolio app, not a persistence layer.

---

Built as a portfolio project to demonstrate ML fundamentals and product-quality UI/UX.
