"""
Data utility helpers: train/test split, standard scaling, encoding,
and small helper functions shared across pages.
"""

import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def encode_labels(y):
    """Encodes categorical labels into integers 0..n_classes-1."""
    y = pd.Series(y)
    classes = sorted(y.unique())
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = y.map(mapping).values
    return encoded, mapping


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def missing_values_summary(df: pd.DataFrame):
    missing = df.isnull().sum()
    percent = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Percentage": percent.values,
        "Data Type": [str(df[col].dtype) for col in missing.index],
    })
    return summary[summary["Missing Values"] >= 0].reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, features: list, target: str):
    """
    Builds numeric X / y arrays from the user's feature & target selection.
    Returns (X, y, feature_names, warning_message_or_None).
    Only numeric feature columns are used; non-numeric features are dropped
    with a warning so the app never silently guesses an encoding scheme.
    """
    numeric_features = [f for f in features if f in get_numeric_columns(df)]
    dropped = [f for f in features if f not in numeric_features]

    warning = None
    if dropped:
        warning = f"Dropped non-numeric feature(s) not supported by scratch algorithms: {', '.join(dropped)}"

    X = df[numeric_features].to_numpy(dtype=float)
    y_raw = df[target]

    return X, y_raw, numeric_features, warning


def format_seconds(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"
