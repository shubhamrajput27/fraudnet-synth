"""Shared fixtures for the ml/ test suite. Deliberately use small synthetic data, not the real
ULB dataset (data/ is gitignored — a fresh clone shouldn't need it downloaded just to run unit
tests)."""
import numpy as np
import pandas as pd
import pytest

from ml.common.data import FEATURE_COLUMNS


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_shard(rng) -> pd.DataFrame:
    """200 legit + 20 fraud rows, roughly matching the real schema's shape/scale."""
    n_legit, n_fraud = 200, 20
    n = n_legit + n_fraud

    data = {"Time": rng.uniform(0, 172792, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1.5, n)
    data["Amount"] = rng.exponential(50, n)
    data["Class"] = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(data)
    assert list(df.columns[:-1]) == FEATURE_COLUMNS  # sanity: fixture matches the real schema
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
