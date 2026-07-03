"""
Principal Component Analysis implemented from scratch using NumPy,
via eigendecomposition of the covariance matrix.
"""

import numpy as np
import time


class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.train_time_ = 0.0

    def fit(self, X):
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance if total_variance > 0 else self.explained_variance_
        )
        self.train_time_ = time.time() - start
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
