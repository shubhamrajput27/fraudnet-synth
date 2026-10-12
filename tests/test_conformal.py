"""Extension E2: the conformal risk control rule, on cases checked by hand."""
import numpy as np

from ml.evaluation.conformal import FLAG_ALL, crc_threshold, min_feasible_alpha, misses_on_grid
from ml.federated.common import counts_at_thresholds

GRID = np.arange(0, 11, 1.0)                       # cut-offs 0, 1, ..., 10


def test_misses_on_grid_counts_frauds_below_each_cutoff():
    assert misses_on_grid(np.array([2.5, 5.0, 7.5]), np.array([0, 3, 5, 8, 10])).tolist() == [0, 1, 1, 3, 3]


def test_misses_agree_with_federated_counts():
    # Banks send TP counts (D-031); misses = frauds - TP must equal the direct count.
    s = np.array([1.2, 4.0, 4.0, 9.9, -3.0, 6.5])
    y = np.array([1, 1, 0, 1, 0, 1])
    tp, _ = counts_at_thresholds(y, s, GRID)
    assert (y.sum() - np.array(tp)).tolist() == misses_on_grid(s[y == 1], GRID).tolist()


def test_rule_by_hand():
    # 9 frauds scoring 1..9. Missing k frauds gives a bound of (k + 1) / 10.
    fraud = np.arange(1, 10, 1.0)
    m = misses_on_grid(fraud, GRID)                # cut-off t misses t - 1 frauds (for t >= 1)
    assert crc_threshold(GRID, m, 9, alpha=0.2) == 2.0     # misses 1 -> (1+1)/10 = 0.2 <= 0.2; t=3 gives 0.3
    assert crc_threshold(GRID, m, 9, alpha=0.1) == 1.0     # misses 0 -> 0.1
    assert crc_threshold(GRID, m, 9, alpha=0.05) == FLAG_ALL   # even 0 misses gives 0.1 > 0.05


def test_small_banks_cannot_certify_strict_alpha():
    assert min_feasible_alpha(4) == 0.2            # Bank D: 4 validation frauds
    m = misses_on_grid(np.array([3.0, 4.0, 5.0, 6.0]), GRID)
    assert crc_threshold(GRID, m, 4, alpha=0.1) == FLAG_ALL
    assert crc_threshold(GRID, m, 4, alpha=0.2) == 3.0     # 0 misses -> 1/5 = 0.2


def test_no_calibration_frauds_means_flag_everything():
    assert crc_threshold(GRID, np.zeros(len(GRID)), 0, alpha=0.5) == FLAG_ALL


def test_guarantee_holds_on_average_in_simulation():
    # Exchangeable scores: average miss rate on new frauds must stay <= alpha.
    rng = np.random.default_rng(0)
    grid = np.linspace(-4, 4, 161)
    alpha, n, rates = 0.1, 30, []
    for _ in range(3000):
        cal, new = rng.normal(size=n), rng.normal(size=200)
        t = crc_threshold(grid, misses_on_grid(cal, grid), n, alpha)
        rates.append(np.mean(new < t))
    assert np.mean(rates) <= alpha + 0.005
