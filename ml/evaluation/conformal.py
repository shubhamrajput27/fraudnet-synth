"""Extension E2: conformal risk control (CRC) for the false-negative rate (decision D-062).

Loss for a calibration fraud = 1 if it is NOT flagged (score < t). With n calibration frauds, CRC picks
the strictest cut-off t with
        (n / (n + 1)) * (misses(t) / n) + 1 / (n + 1)  =  (misses(t) + 1) / (n + 1)  <=  alpha.
If calibration and future frauds are exchangeable, the expected miss rate of a new fraud is then <= alpha
(a MARGINAL guarantee, averaged over calibration sets; Angelopoulos et al. 2022, verify at source).
If no cut-off qualifies (e.g. n = 4 and alpha < 1/5), the only safe choice is to flag everything.

We search the same fixed logit grid as the federated threshold (D-031): banks only need to send counts.
Restricting to a grid can only pick a LOWER cut-off than the exact rule, so the guarantee still holds.
"""
import numpy as np

FLAG_ALL = float("-inf")


def crc_threshold(grid: np.ndarray, misses: np.ndarray, n_fraud: int, alpha: float) -> float:
    """Largest grid cut-off whose CRC bound is <= alpha; FLAG_ALL if none qualifies.

    grid ascending; misses[i] = calibration frauds scoring below grid[i] (non-decreasing in i).
    """
    if n_fraud == 0:
        return FLAG_ALL
    ok = (np.asarray(misses) + 1) / (n_fraud + 1) <= alpha
    return float(np.asarray(grid)[ok].max()) if ok.any() else FLAG_ALL


def misses_on_grid(fraud_scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """For each cut-off, how many of these frauds score below it (would be missed)."""
    return np.searchsorted(np.sort(np.asarray(fraud_scores)), np.asarray(grid), side="left")


def min_feasible_alpha(n_fraud: int) -> float:
    """The smallest alpha any cut-off can certify with n calibration frauds: 1 / (n + 1)."""
    return 1.0 / (n_fraud + 1)
