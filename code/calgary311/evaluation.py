"""Evaluation metrics and validation-threshold selection."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

def weighted_threshold(y, probability, weight, metric="f1"):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 160)))
    best = (0.5, -np.inf)
    for threshold in candidates:
        predicted = probability >= threshold
        if metric == "f1":
            score = f1_score(y, predicted, sample_weight=weight, zero_division=0)
        else:
            score = balanced_accuracy_score(y, predicted, sample_weight=weight)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def classification_metrics(y, probability, threshold, weight=None):
    predicted = probability >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "pr_auc": float(average_precision_score(y, probability, sample_weight=weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=weight)),
        "precision": float(precision_score(y, predicted, sample_weight=weight, zero_division=0)),
        "recall": float(recall_score(y, predicted, sample_weight=weight, zero_division=0)),
        "f1": float(f1_score(y, predicted, sample_weight=weight, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted, sample_weight=weight)
        ),
        "threshold": float(threshold),
    }


def regression_metrics(y, prediction):
    prediction = np.maximum(np.asarray(prediction), 1e-8)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "poisson_deviance": float(mean_poisson_deviance(y, prediction)),
    }
