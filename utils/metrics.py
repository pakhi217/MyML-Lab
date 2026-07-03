"""
Evaluation metrics implemented from scratch using NumPy.
Covers both classification and regression tasks.
"""

import numpy as np


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t], label_to_idx[p]] += 1
    return matrix, labels


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores))


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def roc_curve(y_true, y_scores):
    """Returns fpr, tpr, thresholds for binary classification."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.sort(np.unique(y_scores))[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-6], thresholds])

    tpr_list, fpr_list = [], []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr_list.append(tp / P if P > 0 else 0.0)
        fpr_list.append(fp / N if N > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_score(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))
