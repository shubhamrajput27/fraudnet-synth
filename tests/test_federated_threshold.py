"""D-031: choosing the federated threshold from SUMMED per-bank counts.

Checks that summing TP/FP counts across banks gives exactly the same confusion counts as
pooling all banks' scores, so the federated threshold equals pooled-validation tuning on the
same grid, with no rows or scores leaving any bank.
"""
import numpy as np

from ml.evaluation.metrics import compute_metrics
from ml.federated.common import best_threshold_from_counts, counts_at_thresholds, threshold_grid


def _bank(rng, n, n_fraud):
    y = np.r_[np.ones(n_fraud), np.zeros(n - n_fraud)]
    s = np.where(y == 1, rng.normal(12, 6, n), rng.normal(-4, 5, n))
    return y, s


def test_grid_is_fixed_and_data_free():
    g = threshold_grid()
    assert len(g) == 241 and g[0] == -10.0 and g[-1] == 50.0
    assert np.allclose(np.diff(g), 0.25)


def test_counts_match_direct_confusion_matrix():
    rng = np.random.default_rng(0)
    y, s = _bank(rng, 500, 20)
    g = threshold_grid()
    tp, fp = counts_at_thresholds(y, s, g)
    for i in (0, 60, 120, 200):
        m = compute_metrics(y, s, g[i])
        assert (tp[i], fp[i]) == (m["tp"], m["fp"])


def test_summed_bank_counts_equal_pooled_counts():
    rng = np.random.default_rng(1)
    banks = [_bank(rng, n, f) for n, f in ((3000, 22), (2500, 17), (1500, 6), (1200, 4))]
    g = threshold_grid()
    tp = np.sum([counts_at_thresholds(y, s, g)[0] for y, s in banks], axis=0)
    fp = np.sum([counts_at_thresholds(y, s, g)[1] for y, s in banks], axis=0)
    y_all = np.concatenate([y for y, _ in banks])
    s_all = np.concatenate([s for _, s in banks])
    tp_pool, fp_pool = counts_at_thresholds(y_all, s_all, g)
    assert list(tp) == tp_pool and list(fp) == fp_pool

    t, stats = best_threshold_from_counts(g, tp, fp, int(y_all.sum()))
    # The chosen cut-off reproduces its reported F1 when applied to the pooled scores directly...
    assert abs(compute_metrics(y_all, s_all, t)["f1"] - stats["val_f1"]) < 1e-12
    # ...and no other grid cut-off does better on the pooled data.
    best_direct = max(compute_metrics(y_all, s_all, x)["f1"] for x in g)
    assert abs(best_direct - stats["val_f1"]) < 1e-12
