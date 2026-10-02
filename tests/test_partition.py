"""Checks for Step 2: no leakage between splits, shards are disjoint and complete,
quotas are respected, and stratification kept the fraud ratio.

Run after `python -m ml.data.prepare`:
    pytest tests/test_partition.py -v
"""
import pandas as pd
import pytest

from ml.data.load import LABEL_COLUMN, PROJECT_ROOT, load_config
from ml.data.partition import quota_counts

CFG = load_config("partition")
PROCESSED = PROJECT_ROOT / CFG["processed_dir"]
CLIENTS = PROJECT_ROOT / CFG["clients_dir"]
BANKS = list(CFG["banks"])

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "global_train.csv").exists(), reason="run `python -m ml.data.prepare` first"
)


@pytest.fixture(scope="module")
def data():
    g = {p: pd.read_csv(PROCESSED / f"global_{p}.csv") for p in ("train", "val", "test")}
    b = {bank: {p: pd.read_csv(CLIENTS / bank / f"{p}.csv") for p in ("train", "val", "test")}
         for bank in BANKS}
    return g, b


def _row_keys(df: pd.DataFrame) -> set:
    # Duplicates were dropped before splitting, so a full row is a unique key.
    return set(map(tuple, df.round(10).to_numpy()))


def test_global_splits_disjoint(data):
    g, _ = data
    tr, va, te = (_row_keys(g[p]) for p in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_no_duplicates_left(data):
    g, _ = data
    allrows = pd.concat(g.values())
    assert not allrows.duplicated().any()


def test_bank_files_partition_global_train_exactly(data):
    g, b = data
    bank_rows = [df for bank in BANKS for df in b[bank].values()]
    combined = pd.concat(bank_rows)
    # Every bank row comes from global train, and no row is in two places.
    assert len(combined) == len(g["train"])
    assert _row_keys(combined) == _row_keys(g["train"])


def test_bank_test_sets_never_touch_global_test(data):
    g, b = data
    global_test = _row_keys(g["test"]) | _row_keys(g["val"])
    for bank in BANKS:
        for part in ("train", "val", "test"):
            assert not (_row_keys(b[bank][part]) & global_test), f"{bank}/{part} leaks"


def test_fraud_quotas(data):
    g, b = data
    n_fraud = int(g["train"][LABEL_COLUMN].sum())
    expected = quota_counts(n_fraud, [CFG["banks"][k]["fraud_share"] for k in BANKS])
    for bank, exp in zip(BANKS, expected):
        got = sum(int(df[LABEL_COLUMN].sum()) for df in b[bank].values())
        assert got == exp, f"{bank}: {got} fraud rows, expected {exp}"


def test_rich_banks_have_more_fraud_than_poor(data):
    _, b = data
    fraud = {bank: sum(int(df[LABEL_COLUMN].sum()) for df in b[bank].values()) for bank in BANKS}
    assert min(fraud["bank_a"], fraud["bank_b"]) > max(fraud["bank_c"], fraud["bank_d"])


def test_global_stratification(data):
    g, _ = data
    rates = [g[p][LABEL_COLUMN].mean() for p in ("train", "val", "test")]
    # Stratification keeps each split's fraud rate within 0.01 percentage points.
    assert max(rates) - min(rates) < 1e-4


def test_quota_counts_sum_exactly():
    assert sum(quota_counts(331, [0.45, 0.35, 0.12, 0.08])) == 331
    assert quota_counts(10, [0.5, 0.5]) == [5, 5]
