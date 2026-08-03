"""ml/common/metrics.py: fraud-class metrics are the project's headline numbers (CLAUDE.md
evaluation philosophy) — correctness here matters more than almost anywhere else in the codebase.
"""
import numpy as np

from ml.common.metrics import compute_metrics, macro_average


def _logit(prob: float) -> float:
    return float(np.log(prob / (1 - prob)))


def test_perfect_predictions_score_1():
    y_true = np.array([0, 0, 1, 1])
    logits = np.array([_logit(0.01), _logit(0.02), _logit(0.99), _logit(0.98)])
    m = compute_metrics(y_true, logits)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_all_wrong_predictions_score_0():
    y_true = np.array([0, 0, 1, 1])
    logits = np.array([_logit(0.99), _logit(0.98), _logit(0.01), _logit(0.02)])
    m = compute_metrics(y_true, logits)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_no_positive_predictions_gives_zero_precision_not_error():
    # zero_division=0 must be honored — a naive sklearn call would warn/raise otherwise.
    y_true = np.array([0, 0, 1, 1])
    logits = np.array([_logit(0.01)] * 4)  # model predicts "legitimate" for everything
    m = compute_metrics(y_true, logits)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_extreme_logits_do_not_overflow():
    # Regression test: plain 1/(1+exp(-x)) overflows for large |x| (fixed via scipy.special.expit
    # during Phase 4's six-arm sweep — see PLAN.md).
    y_true = np.array([0, 1])
    logits = np.array([-1000.0, 1000.0])
    m = compute_metrics(y_true, logits)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_n_samples_and_n_fraud_are_counted_correctly():
    y_true = np.array([0, 0, 0, 1])
    logits = np.zeros(4)
    m = compute_metrics(y_true, logits)
    assert m["n_samples"] == 4
    assert m["n_fraud"] == 1


def test_macro_average_averages_and_sums_correctly():
    metrics = [
        {"precision": 1.0, "recall": 0.5, "f1": 0.6, "auc": 0.9, "accuracy": 0.99, "n_samples": 100, "n_fraud": 5},
        {"precision": 0.0, "recall": 0.5, "f1": 0.4, "auc": 0.7, "accuracy": 0.95, "n_samples": 50, "n_fraud": 3},
    ]
    avg = macro_average(metrics)
    assert avg["precision"] == 0.5
    assert avg["f1"] == 0.5
    assert avg["n_samples"] == 150  # summed, not averaged
    assert avg["n_fraud"] == 8


def test_macro_average_ignores_nan_auc():
    metrics = [
        {"precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": float("nan"), "accuracy": 1.0, "n_samples": 10, "n_fraud": 0},
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.8, "accuracy": 0.9, "n_samples": 10, "n_fraud": 2},
    ]
    avg = macro_average(metrics)
    assert avg["auc"] == 0.8  # the NaN entry (no positive class in that client's eval) is excluded
