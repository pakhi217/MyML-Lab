"""
K-Means Clustering implemented from scratch using NumPy.

Uses k-means++ initialization (spreads out initial centroids based on
distance, rather than picking them uniformly at random) plus multiple
random restarts, keeping the run with the lowest final inertia. Plain
uniform-random initialization with a single run can converge to a poor
local optimum when two initial centroids happen to land in the same
true cluster; k-means++ combined with restarts makes that failure mode
rare, which is exactly what production implementations (including
scikit-learn's) do by default.
"""

import numpy as np
import time


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iterations=100, tol=1e-4, random_state=42, n_init=8):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids = None
        self.labels_ = None
        self.inertia_history = []
        self.train_time_ = 0.0
        self.n_iter_ = 0

    def _kmeans_plusplus_init(self, X, rng):
        n_samples = X.shape[0]
        centroids = [X[rng.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum() if dist_sq.sum() > 0 else None
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _single_run(self, X, seed):
        rng = np.random.RandomState(seed)
        centroids = self._kmeans_plusplus_init(X, rng)
        inertia_history = []
        labels = None
        n_iter = 0

        for iteration in range(self.max_iterations):
            distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            inertia = np.sum((X - centroids[labels]) ** 2)
            inertia_history.append(inertia)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            n_iter = iteration + 1
            if shift < self.tol:
                break

        return centroids, labels, inertia_history, n_iter

    def fit(self, X):
        start = time.time()
        X = np.asarray(X, dtype=np.float64)

        best = None
        for i in range(self.n_init):
            seed = None if self.random_state is None else self.random_state + i
            centroids, labels, inertia_history, n_iter = self._single_run(X, seed)
            final_inertia = inertia_history[-1]
            if best is None or final_inertia < best[0]:
                best = (final_inertia, centroids, labels, inertia_history, n_iter)

        _, self.centroids, self.labels_, self.inertia_history, self.n_iter_ = best
        self.train_time_ = time.time() - start
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)
