"""ml/evaluation/metrics.py on a tiny example whose answers are worked out by hand.

10 transactions, 3 frauds. Scores (higher = more suspicious):
    fraud:    0.90, 0.80, 0.20
    genuine:  0.70, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01
"""
import numpy as np
import pytest

from ml.evaluation.metrics import best_f1_threshold, compute_metrics

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
S = np.array([0.90, 0.80, 0.20, 0.70, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01])


def test_confusion_matrix_and_headline_metrics_at_threshold_half():
    m = compute_metrics(Y, S, 0.5)
    # Flagged (>= 0.5): 0.90, 0.80 (frauds) and 0.70 (genuine) -> TP 2, FP 1, FN 1 (the 0.20 fraud), TN 6.
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)        # 2 of 3 alarms were fraud
    assert m["recall"] == pytest.approx(2 / 3)           # 2 of 3 frauds caught
    assert m["f1"] == pytest.approx(2 / 3)               # harmonic mean of 2/3 and 2/3
    assert m["accuracy"] == pytest.approx(8 / 10)        # secondary metric
    assert (m["n"], m["n_fraud"], m["threshold"]) == (10, 3, 0.5)


def test_pr_auc_is_average_precision():
    # Ranked by score: F(0.9) F(0.8) G(0.7) F(0.2) ...  precision at each fraud = 1/1, 2/2, 3/4.
    assert compute_metrics(Y, S, 0.5)["pr_auc"] == pytest.approx((1 + 1 + 0.75) / 3)


def test_roc_auc_counts_correctly_ordered_pairs():
    # 3 frauds x 7 genuine = 21 pairs; only (fraud 0.20 vs genuine 0.70) is ordered wrongly -> 20/21.
    assert compute_metrics(Y, S, 0.5)["roc_auc"] == pytest.approx(20 / 21)


def test_pr_auc_does_not_depend_on_threshold():
    assert compute_metrics(Y, S, 0.1)["pr_auc"] == compute_metrics(Y, S, 0.85)["pr_auc"]


def test_accuracy_paradox_always_genuine():
    # A model that never raises an alarm: high accuracy, zero fraud caught, no crash on 0/0 precision.
    m = compute_metrics(Y, np.zeros(len(Y)), 0.5)
    assert m["accuracy"] == pytest.approx(0.7)
    assert (m["precision"], m["recall"], m["f1"], m["tp"]) == (0.0, 0.0, 0.0, 0)


def test_best_f1_threshold_on_validation():
    # Cut-off 0.90 -> P 1, R 1/3, F1 0.5;  0.80 -> P 1, R 2/3, F1 0.8;
    # 0.70 -> P 2/3, R 2/3, F1 0.667;      0.20 -> P 3/4, R 1, F1 0.857 (best).
    assert best_f1_threshold(Y, S) == pytest.approx(0.20)


def test_threshold_works_on_logits_too():
    # Since D-028 scores are logits (any real number); a monotone transform must give the same decisions.
    logits = np.log(S / (1 - S))
    t = best_f1_threshold(Y, logits)
    assert t == pytest.approx(np.log(0.2 / 0.8))
    assert compute_metrics(Y, logits, t)["f1"] == pytest.approx(compute_metrics(Y, S, 0.2)["f1"])
