"""Fraud-class metrics (CLAUDE.md evaluation philosophy): precision/recall/F1 are the headline
numbers on this ~0.17%-positive dataset, never accuracy alone. Accuracy is still reported for
context, but never as the leading number.
"""
import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = expit(logits)  # numerically stable sigmoid — plain 1/(1+exp(-x)) overflows for large |x|
    y_pred = (probs >= threshold).astype(int)

    n_pos = int(y_true.sum())
    auc = float(roc_auc_score(y_true, probs)) if 0 < n_pos < len(y_true) else float("nan")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_fraud": n_pos,
    }


def macro_average(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = ["precision", "recall", "f1", "auc", "accuracy"]
    result = {}
    for key in keys:
        values = [m[key] for m in metric_dicts if not np.isnan(m[key])]
        result[key] = float(np.mean(values)) if values else float("nan")
    result["n_samples"] = sum(m["n_samples"] for m in metric_dicts)
    result["n_fraud"] = sum(m["n_fraud"] for m in metric_dicts)
    return result
