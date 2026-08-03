"""ml/partition/partition.py: D5's non-IID partition — no row lost/duplicated, poor-bank fraud
cap actually enforced, and deterministic given a fixed seed (the property the whole project's
reproducibility claim rests on)."""
import numpy as np
import pandas as pd

from ml.partition.partition import PartitionConfig, partition_non_iid


def _make_pool(rng, n_legit=2000, n_fraud=200) -> pd.DataFrame:
    n = n_legit + n_fraud
    data = {"Time": rng.uniform(0, 172792, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1.5, n)
    data["Amount"] = rng.exponential(50, n)
    data["Class"] = np.array([0] * n_legit + [1] * n_fraud)
    return pd.DataFrame(data)


def test_partition_conserves_every_row_exactly_once():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    shards = partition_non_iid(pool, PartitionConfig(seed=42))
    assert sum(len(df) for df in shards.values()) == len(pool)


def test_poor_bank_fraud_cap_is_enforced():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    cfg = PartitionConfig(seed=42, poor_fraud_cap=15)
    shards = partition_non_iid(pool, cfg)
    assert (shards["C"]["Class"] == 1).sum() <= 15
    assert (shards["D"]["Class"] == 1).sum() <= 15


def test_all_four_banks_present_and_nonempty():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    shards = partition_non_iid(pool, PartitionConfig(seed=42))
    assert set(shards.keys()) == {"A", "B", "C", "D"}
    assert all(len(df) > 0 for df in shards.values())


def test_same_seed_is_bit_for_bit_reproducible():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    shards1 = partition_non_iid(pool, PartitionConfig(seed=42))
    shards2 = partition_non_iid(pool, PartitionConfig(seed=42))
    for bank in shards1:
        pd.testing.assert_frame_equal(shards1[bank], shards2[bank])


def test_different_seeds_produce_different_partitions():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    shards1 = partition_non_iid(pool, PartitionConfig(seed=42))
    shards2 = partition_non_iid(pool, PartitionConfig(seed=1))
    assert len(shards1["A"]) != len(shards2["A"]) or not shards1["A"].equals(shards2["A"])


def test_rich_banks_get_more_legit_rows_than_poor_banks():
    rng = np.random.default_rng(0)
    pool = _make_pool(rng)
    shards = partition_non_iid(pool, PartitionConfig(seed=42))
    legit_counts = {b: (df["Class"] == 0).sum() for b, df in shards.items()}
    assert legit_counts["A"] > legit_counts["C"]
    assert legit_counts["B"] > legit_counts["D"]
