"""
Logistic Regression implemented from scratch using NumPy.

Model:      p = sigmoid(X . w + b)
Loss:       Binary Cross-Entropy
Optimizer:  Batch Gradient Descent
"""

import numpy as np
import time


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=500, l2_penalty=0.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.train_time_ = 0.0

    def fit(self, X, y):
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            linear = X @ self.weights + self.bias
            y_pred = sigmoid(linear)
            error = y_pred - y

            grad_w = (1 / n_samples) * (X.T @ error) + (2 * self.l2_penalty * self.weights)
            grad_b = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            eps = 1e-12
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)

        self.train_time_ = time.time() - start
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
