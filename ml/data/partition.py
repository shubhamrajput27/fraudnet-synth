"""Non-IID partitioning of the global training pool into Banks A-D.

Method (decision D-007): explicit per-class quotas. Fraud rows and genuine rows are
shuffled separately and dealt out by fixed shares, so A and B are guaranteed to be
fraud-rich and C and D fraud-poor (label skew), and the banks differ in size
(quantity skew). Each row goes to exactly one bank: the shards never overlap.
"""
import numpy as np
import pandas as pd

from ml.data.load import LABEL_COLUMN
from ml.data.split import stratified_three_way


def quota_counts(n: int, shares: list[float]) -> list[int]:
    """Turn fractional shares into whole row counts that sum exactly to n.

    Largest-remainder rounding: floor every share, then hand the leftover rows to
    the banks with the biggest fractional parts (ties go to the earlier bank).
    """
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError("shares must sum to 1")
    raw = np.array(shares) * n
    counts = np.floor(raw).astype(int)
    leftover = n - counts.sum()
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:leftover]] += 1
    return counts.tolist()


def partition_by_quota(train_df: pd.DataFrame, banks: dict, seed: int) -> dict[str, pd.DataFrame]:
    """Deal each class's rows out to the banks according to their shares."""
    names = list(banks)
    rng = np.random.default_rng(seed)
    pieces = {name: [] for name in names}

    for label, share_key in ((1, "fraud_share"), (0, "genuine_share")):
        rows = train_df[train_df[LABEL_COLUMN] == label]
        rows = rows.iloc[rng.permutation(len(rows))]  # seeded shuffle, then contiguous slices
        counts = quota_counts(len(rows), [banks[b][share_key] for b in names])
        start = 0
        for name, c in zip(names, counts):
            pieces[name].append(rows.iloc[start:start + c])
            start += c

    shards = {}
    for i, name in enumerate(names):
        shard = pd.concat(pieces[name])
        # Shuffle within the shard so fraud rows aren't all at the top of the file.
        shards[name] = shard.sample(frac=1.0, random_state=seed + i)
    return shards


def local_splits(shards: dict[str, pd.DataFrame], cfg: dict, seed: int):
    """Each bank splits its own shard into local train/val/test (stratified)."""
    s = cfg["local_split"]
    out = {}
    for i, (name, shard) in enumerate(shards.items()):
        tr, va, te = stratified_three_way(shard, s["train"], s["val"], s["test"], seed + i)
        out[name] = {"train": tr, "val": va, "test": te}
    return out
