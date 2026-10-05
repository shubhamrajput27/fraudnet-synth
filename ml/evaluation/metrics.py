"""Evaluation metrics. Headline = fraud-class precision, recall, F1 and PR-AUC.
ROC-AUC and accuracy are reported as SECONDARY metrics only.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """All metrics for one threshold. The fraud class (1) is the positive class throughout."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        # --- headline (fraud class) ---
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # PR-AUC as Average Precision: the standard step-wise summary of the
        # precision-recall curve. It does not depend on the threshold.
        "pr_auc": float(average_precision_score(y_true, scores)),
        # --- secondary ---
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # --- raw counts ---
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y_true)), "n_fraud": int(y_true.sum()),
    }


def best_f1_threshold(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    """Pick the threshold that maximises fraud-class F1 on VALIDATION data.

    Never call this with test data: that would tune to the exam and inflate results.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    # precision/recall have one more entry than thresholds (the final point has no threshold).
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])
