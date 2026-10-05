"""Step 4 tests: the CTGAN engine reads only a bank's own TRAINING fraud rows and
produces well-formed, reproducible candidates. Uses a tiny fake bank and few epochs.
"""
import numpy as np
import pandas as pd

from ml.augmentation.ctgan_engine import compare, fit_ctgan, load_bank_fraud
from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, V_COLUMNS

FAST_CFG = dict(epochs=3, batch_size=50, pac=10, enforce_min_max_values=True, enable_gpu=False)


def _fake_bank(tmp_path, n=120, n_fraud=30):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.integers(0, 172800, n).astype(float))
    df["Amount"] = rng.exponential(80, n).round(2)
    df[LABEL_COLUMN] = [1] * n_fraud + [0] * (n - n_fraud)
    df.to_csv(tmp_path / "train.csv", index=False)
    # A val/test file with a marker value: it must never be read by the engine.
    poison = df.head(5).copy()
    poison["Amount"] = -999.0
    poison.to_csv(tmp_path / "test.csv", index=False)
    return df


def test_loads_only_training_fraud_rows(tmp_path):
    _fake_bank(tmp_path)
    real = load_bank_fraud(tmp_path)
    assert len(real) == 30
    assert list(real.columns) == FEATURE_COLUMNS  # label dropped; generator models features only
    assert (real["Amount"] >= 0).all()  # nothing from test.csv leaked in


def test_candidates_are_well_formed_and_reproducible(tmp_path):
    _fake_bank(tmp_path)
    real = load_bank_fraud(tmp_path)
    samples = []
    for _ in range(2):
        model, _ = fit_ctgan(real, FAST_CFG, seed=1)
        samples.append(model.sample(num_rows=40))
    s = samples[0]
    assert list(s.columns) == FEATURE_COLUMNS and len(s) == 40
    assert not s.isna().any().any()
    # enforce_min_max_values keeps every column inside the real rows' range
    assert ((s >= real.min()) & (s <= real.max())).all().all()
    assert samples[0].equals(samples[1])


def test_compare_reports_identical_data_as_perfect(tmp_path):
    _fake_bank(tmp_path)
    real = load_bank_fraud(tmp_path)
    c = compare(real, real.copy())
    assert c["max_ks_stat"] == 0.0 and c["mean_abs_corr_diff"] == 0.0
    assert c["exact_copies_of_real_rows"] == len(real)
