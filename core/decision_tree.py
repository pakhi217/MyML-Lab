"""
Decision Tree Classifier implemented from scratch using NumPy.
Splitting criterion: Gini impurity.
"""

import numpy as np
import time


class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


def _gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features_ = None
        self.train_time_ = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        self.n_features_ = X.shape[1]
        self._importance_accum = np.zeros(self.n_features_)
        self.root = self._grow_tree(X, y, depth=0)
        total = self._importance_accum.sum()
        if total > 0:
            self.feature_importances_ = self._importance_accum / total
        else:
            self.feature_importances_ = self._importance_accum
        self.train_time_ = time.time() - start
        return self

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None

        parent_gini = _gini(y)
        best_gain = 0.0
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(5, 95, 20))
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                n_left, n_right = left_mask.sum(), right_mask.sum()
                gini_left = _gini(y[left_mask])
                gini_right = _gini(y[right_mask])
                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _grow_tree(self, X, y, depth):
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)

        if depth >= self.max_depth or len(classes) == 1 or n_samples < self.min_samples_split:
            return _Node(value=classes[np.argmax(counts)])

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            return _Node(value=classes[np.argmax(counts)])

        self._importance_accum[feature] += gain * n_samples

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_single(x, self.root) for x in X])
