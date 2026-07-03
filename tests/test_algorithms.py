import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.linear_regression import LinearRegressionScratch
from core.logistic_regression import LogisticRegressionScratch
from core.knn import KNNScratch
from core.decision_tree import DecisionTreeScratch
from core.random_forest import RandomForestScratch
from core.kmeans import KMeansScratch
from core.pca import PCAScratch
from utils.metrics import r2_score, accuracy_score
from utils.data_utils import train_test_split, StandardScaler

rng = np.random.RandomState(0)
results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# --- Linear Regression: recover known coefficients on a noiseless line ---
X = rng.uniform(-5, 5, size=(200, 1))
y = 3.5 * X[:, 0] + 7.0 + rng.normal(0, 0.05, size=200)
model = LinearRegressionScratch(learning_rate=0.05, n_iterations=1000)
model.fit(X, y)
check("Linear Regression recovers slope ~3.5", abs(model.weights[0] - 3.5) < 0.1, f"got {model.weights[0]:.3f}")
check("Linear Regression recovers intercept ~7.0", abs(model.bias - 7.0) < 0.2, f"got {model.bias:.3f}")
pred = model.predict(X)
check("Linear Regression R2 > 0.99 on clean line", r2_score(y, pred) > 0.99, f"R2={r2_score(y, pred):.4f}")

# --- Logistic Regression: separable 2-class blobs ---
Xc = np.vstack([rng.normal(-2, 0.5, size=(100, 2)), rng.normal(2, 0.5, size=(100, 2))])
yc = np.array([0] * 100 + [1] * 100)
scaler = StandardScaler()
Xc_s = scaler.fit_transform(Xc)
logit = LogisticRegressionScratch(learning_rate=0.3, n_iterations=500)
logit.fit(Xc_s, yc)
acc = accuracy_score(yc, logit.predict(Xc_s))
check("Logistic Regression accuracy > 0.95 on separable blobs", acc > 0.95, f"acc={acc:.3f}")

# --- KNN on the same separable blobs ---
knn = KNNScratch(k=5)
knn.fit(Xc_s, yc)
acc = accuracy_score(yc, knn.predict(Xc_s))
check("KNN accuracy > 0.95 on separable blobs", acc > 0.95, f"acc={acc:.3f}")

# --- Decision Tree on separable blobs ---
tree = DecisionTreeScratch(max_depth=5)
tree.fit(Xc_s, yc)
acc = accuracy_score(yc, tree.predict(Xc_s))
check("Decision Tree accuracy > 0.95 on separable blobs", acc > 0.95, f"acc={acc:.3f}")
check("Decision Tree feature_importances_ sums to ~1", abs(tree.feature_importances_.sum() - 1.0) < 1e-6)

# --- Random Forest on separable blobs ---
rf = RandomForestScratch(n_estimators=15, max_depth=5)
rf.fit(Xc_s, yc)
acc = accuracy_score(yc, rf.predict(Xc_s))
check("Random Forest accuracy > 0.95 on separable blobs", acc > 0.95, f"acc={acc:.3f}")

# --- K-Means recovers 3 well-separated clusters ---
Xk = np.vstack([
    rng.normal([0, 0], 0.3, size=(50, 2)),
    rng.normal([6, 0], 0.3, size=(50, 2)),
    rng.normal([3, 6], 0.3, size=(50, 2)),
])
km = KMeansScratch(n_clusters=3, random_state=1)
km.fit(Xk)
# Each true cluster's 50 points should map to a single dominant label
labels = km.labels_
homogeneous = all(
    np.max(np.bincount(labels[i * 50:(i + 1) * 50])) / 50 > 0.9
    for i in range(3)
)
check("K-Means recovers 3 separated clusters", homogeneous)

# --- PCA on correlated data: first component should dominate ---
base = rng.normal(0, 1, size=(300, 1))
Xp = np.hstack([base, base * 2 + rng.normal(0, 0.05, size=(300, 1)), rng.normal(0, 1, size=(300, 1))])
pca = PCAScratch(n_components=2)
pca.fit(Xp)
check("PCA: PC1 explains most variance for correlated features",
      pca.explained_variance_ratio_[0] > 0.6, f"ratio={pca.explained_variance_ratio_[0]:.3f}")

# --- train_test_split sanity ---
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=1)
check("train_test_split proportions correct", len(Xte) == 50 and len(Xtr) == 150)
check("train_test_split no overlap", len(set(map(tuple, Xtr)) & set(map(tuple, Xte))) == 0)

print("\n" + "=" * 60)
n_pass = sum(1 for _, s in results if s == "PASS")
n_fail = sum(1 for _, s in results if s == "FAIL")
print(f"TOTAL: {n_pass} passed, {n_fail} failed, {len(results)} checks")
import sys
sys.exit(1 if n_fail else 0)
