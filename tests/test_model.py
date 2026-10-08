"""Step 3 tests: features, weight helpers, metrics, threshold tuning, deterministic training."""
import numpy as np
import pandas as pd
import pytest

from ml.data.load import V_COLUMNS, load_config
from ml.evaluation.metrics import best_f1_threshold, compute_metrics
from ml.models.classifier import build_model, get_weights, set_weights
from ml.models.features import N_FEATURES, to_features, to_xy
from ml.models.train import balanced_pos_weight, set_seed, train_model

CFG = load_config("model")


def _toy_df(n=200, n_fraud=20, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172800, n))
    df["Amount"] = rng.exponential(80, n)
    df["Class"] = 0
    df.loc[: n_fraud - 1, "Class"] = 1
    df.loc[: n_fraud - 1, "V14"] -= 4  # make fraud learnable
    return df


def test_features_shape_and_finite():
    x = to_features(_toy_df())
    assert x.shape == (200, N_FEATURES) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_hour_encoding_is_daily_periodic():
    df = _toy_df(n=2, n_fraud=1)
    df["Time"] = [3600.0, 3600.0 + 86400]  # 01:00 on day 1 and 01:00 on day 2
    x = to_features(df)
    assert np.allclose(x[0, -2:], x[1, -2:], atol=1e-6)


def test_weights_roundtrip():
    a, b = build_model(CFG), build_model(CFG)
    set_weights(b, get_weights(a))
    for wa, wb in zip(get_weights(a), get_weights(b)):
        assert np.array_equal(wa, wb)


def test_get_weights_are_plain_arrays():
    assert all(isinstance(w, np.ndarray) for w in get_weights(build_model(CFG)))


def test_metrics_match_hand_worked_example():
    # Step 1 doc example: TP=120, FP=30, FN=53, TN=99,797.
    y = np.array([1] * 173 + [0] * 99_827)
    s = np.array([1] * 120 + [0] * 53 + [1] * 30 + [0] * 99_797, dtype=float)
    m = compute_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (120, 30, 53, 99_797)
    assert m["precision"] == pytest.approx(0.80)
    assert m["recall"] == pytest.approx(120 / 173)
    assert m["f1"] == pytest.approx(2 * 0.8 * (120 / 173) / (0.8 + 120 / 173))


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert compute_metrics(y, s, t)["f1"] == 1.0


def test_balanced_pos_weight():
    assert balanced_pos_weight(np.array([0, 0, 0, 1], dtype=float)) == 3.0


def test_training_is_deterministic():
    x, y = to_xy(_toy_df())
    weights = []
    for _ in range(2):
        set_seed(7)
        m = build_model(CFG)
        train_model(m, x, y, CFG["training"], seed=7, epochs=2)
        weights.append(get_weights(m))
    for a, b in zip(*weights):
        assert np.array_equal(a, b)


def test_scores_are_untied_logits():
    """D-028: very confident predictions must keep distinct scores (no float32 saturation at 1.0)."""
    from ml.models.train import predict_proba, predict_scores
    x, y = to_xy(_toy_df())
    set_seed(1)
    m = build_model(CFG)
    with __import__("torch").no_grad():
        m.net[-1].bias.fill_(30.0)  # push every logit far into sigmoid saturation
    s = predict_scores(m, x)
    assert s.dtype == np.float64 and len(np.unique(s)) == len(s)
    p = predict_proba(m, x)
    assert ((p >= 0) & (p <= 1)).all()
