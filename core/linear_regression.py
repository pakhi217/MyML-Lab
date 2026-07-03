"""
Linear Regression implemented from scratch using NumPy.

Model:      y_hat = X . w + b
Loss:       MSE = (1/n) * sum((y - y_hat)^2)
Optimizer:  Batch Gradient Descent
"""

import numpy as np
import time


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=500, l2_penalty=0.0):
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
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error) + (2 * self.l2_penalty * self.weights)
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

        self.train_time_ = time.time() - start
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias
