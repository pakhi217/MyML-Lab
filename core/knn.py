"""
K-Nearest Neighbors Classifier implemented from scratch using NumPy.
"""

import numpy as np
import time
from collections import Counter


class KNNScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.train_time_ = 0.0

    def fit(self, X, y):
        start = time.time()
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y).reshape(-1)
        self.train_time_ = time.time() - start
        return self

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """Returns probability of the positive class (binary only)."""
        X = np.asarray(X, dtype=np.float64)
        probs = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_indices]
            probs.append(np.mean(k_labels == 1))
        return np.array(probs)
