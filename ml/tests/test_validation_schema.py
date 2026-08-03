"""ml/validation/schema.py: the per-row range check that caught CTGAN's real Bank B failure in
Phase 3 (Time values ~100x the real training range) — verify it actually rejects that shape of
failure, not just accepts everything."""
import numpy as np
import pandas as pd

from ml.common.data import FEATURE_COLUMNS
from ml.validation.schema import validate_schema


def _real_reference(n=50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"Time": rng.uniform(1000, 170000, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1.5, n)
    data["Amount"] = rng.exponential(50, n)
    data["Class"] = 1
    return pd.DataFrame(data)


def test_valid_candidates_all_pass():
    real = _real_reference()
    candidates = real.copy()  # within-range by construction
    passing, errors = validate_schema(candidates, real)
    assert len(passing) == len(candidates)
    assert errors == []


def test_wildly_out_of_range_value_is_rejected():
    # Mirrors the real Phase 3 finding: CTGAN generated Time values ~100x the real max.
    real = _real_reference()
    candidates = real.copy()
    candidates.loc[0, "Time"] = real["Time"].max() * 100
    passing, errors = validate_schema(candidates, real)
    assert len(passing) == len(candidates) - 1
    assert len(errors) == 1
    assert "Time" in errors[0]


def test_one_bad_row_does_not_reject_the_whole_batch():
    real = _real_reference()
    candidates = real.copy()
    candidates.loc[0, "Amount"] = -999999  # clearly invalid
    passing, _ = validate_schema(candidates, real)
    assert len(passing) == len(candidates) - 1  # the other rows still pass


def test_empty_candidates_returns_empty_with_no_errors():
    real = _real_reference()
    empty = real.iloc[0:0]
    passing, errors = validate_schema(empty, real)
    assert len(passing) == 0
    assert errors == []


def test_passing_rows_have_expected_columns():
    real = _real_reference()
    passing, _ = validate_schema(real.copy(), real)
    assert list(passing.columns) == FEATURE_COLUMNS + ["Class"]
