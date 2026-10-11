"""Step 9: the paired-difference statistics behind every 'clear difference' claim."""
import numpy as np
import pytest
from scipy import stats

from ml.evaluation.six_arm_report import paired


def test_paired_ci_matches_textbook_formula():
    a = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    b = np.array([0.78, 0.80, 0.78, 0.80, 0.80])
    r = paired(a, b, 0.95)
    d = a - b
    half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["ci_low"] == pytest.approx(d.mean() - half) and r["ci_high"] == pytest.approx(d.mean() + half)
    assert r["improved_in"] == 5 and r["verdict"] == "higher"


def test_ci_crossing_zero_is_no_clear_difference():
    r = paired(np.array([0.80, 0.70, 0.85, 0.72, 0.79]), np.array([0.78, 0.74, 0.80, 0.76, 0.79]), 0.95)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["verdict"] == "no clear difference"


def test_consistently_worse_is_lower():
    r = paired(np.array([0.70, 0.71, 0.69, 0.70, 0.72]), np.array([0.80, 0.80, 0.79, 0.81, 0.80]), 0.95)
    assert r["verdict"] == "lower" and r["improved_in"] == 0


def test_pairing_matters():
    """Same values, but pairing removes shared seed luck: unpaired overlap would hide this effect."""
    base = np.array([0.60, 0.75, 0.68, 0.82, 0.71])
    r = paired(base + 0.01, base, 0.95)
    assert r["verdict"] == "higher"
