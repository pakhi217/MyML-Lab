"""
Random Forest Classifier implemented from scratch using NumPy.
An ensemble of DecisionTreeScratch trees trained on bootstrap samples,
with random feature subsampling at each split handled implicitly via
bootstrapped feature subsets per tree.
"""

import numpy as np
import time
from collections import Counter

from core.decision_tree import DecisionTreeScratch


class RandomForestScratch:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_subsets = []
        self.train_time_ = 0.0
        self.feature_importances_ = None

    def _sample_features(self, n_features):
        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            size = max(1, int(np.log2(n_features)))
        else:
            size = n_features
        return np.random.choice(n_features, size=size, replace=False)

    def fit(self, X, y):
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape

        self.trees = []
        self.feature_subsets = []
        importance_accum = np.zeros(n_features)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feat_idx = self._sample_features(n_features)

            X_sample = X[indices][:, feat_idx]
            y_sample = y[indices]

            tree = DecisionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

            if tree.feature_importances_ is not None:
                importance_accum[feat_idx] += tree.feature_importances_

        total = importance_accum.sum()
        self.feature_importances_ = importance_accum / total if total > 0 else importance_accum
        self.train_time_ = time.time() - start
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_preds = []
        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            all_preds.append(tree.predict(X[:, feat_idx]))
        all_preds = np.array(all_preds)  # shape (n_estimators, n_samples)

        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            final_preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_preds)
