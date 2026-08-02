"""Non-IID partitioning of the training pool into 4 simulated bank shards (CLAUDE.md D5).

Banks A/B are data-rich; Banks C/D are data-poor and hold only a handful of real fraud rows
each — this scarcity is what makes Schema Mode (LLM few-shot generation) necessary for them.

Mechanism (deliberate, not literature-standard pure-Dirichlet, because D5 requires a *guaranteed*
poor-bank fraud scarcity, and unconstrained Dirichlet sampling gives no such guarantee):
  - Legit rows: split by fixed quantity-skew weights across the 4 banks (drives overall
    data-rich vs. data-poor volume).
  - Fraud rows: split via a Dirichlet-sampled proportion per bank (drives non-IID label skew),
    then Banks C/D's fraud allocation is hard-capped at `poor_fraud_cap` rows each, with the
    excess redistributed back to Banks A/B proportionally to their Dirichlet shares.

`legit_weights`, `fraud_alpha`, and `poor_fraud_cap` are proposed defaults only. Per CLAUDE.md
D5, exact shard sizes are a Phase 1 deliverable requiring explicit sign-off against the real
dataset's row/fraud counts before being treated as final anywhere else in the pipeline.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANK_NAMES = ["A", "B", "C", "D"]
RICH_BANKS = ["A", "B"]
POOR_BANKS = ["C", "D"]


@dataclass(frozen=True)
class PartitionConfig:
    seed: int = 42
    legit_weights: dict[str, float] = field(
        default_factory=lambda: {"A": 0.35, "B": 0.35, "C": 0.15, "D": 0.15}
    )
    fraud_alpha: float = 0.3
    poor_fraud_cap: int = 15


def _split_by_weights(df: pd.DataFrame, weights: dict[str, float], rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    idx = rng.permutation(df.index.to_numpy())
    n = len(idx)
    counts = {bank: int(round(w * n)) for bank, w in weights.items()}
    drift = n - sum(counts.values())
    counts[BANK_NAMES[-1]] += drift  # fix rounding drift on the last bank

    shards: dict[str, pd.DataFrame] = {}
    start = 0
    for bank in BANK_NAMES:
        end = start + counts[bank]
        shards[bank] = df.loc[idx[start:end]]
        start = end
    return shards


def _split_fraud_rows(fraud_df: pd.DataFrame, cfg: PartitionConfig, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    alpha = np.full(len(BANK_NAMES), cfg.fraud_alpha)
    proportions = rng.dirichlet(alpha)
    idx = rng.permutation(fraud_df.index.to_numpy())
    n = len(idx)
    counts = {bank: int(round(p * n)) for bank, p in zip(BANK_NAMES, proportions)}
    drift = n - sum(counts.values())
    counts[BANK_NAMES[np.argmax(proportions)]] += drift

    # Enforce the poor-bank fraud cap, redistributing excess to rich banks proportionally.
    excess = 0
    for bank in POOR_BANKS:
        if counts[bank] > cfg.poor_fraud_cap:
            excess += counts[bank] - cfg.poor_fraud_cap
            counts[bank] = cfg.poor_fraud_cap

    if excess:
        rich_props = proportions[[BANK_NAMES.index(b) for b in RICH_BANKS]]
        rich_props = rich_props / rich_props.sum()
        for bank, p in zip(RICH_BANKS, rich_props):
            counts[bank] += int(round(p * excess))
        drift = n - sum(counts.values())
        counts[RICH_BANKS[0]] += drift

    shards: dict[str, pd.DataFrame] = {}
    start = 0
    for bank in BANK_NAMES:
        end = start + counts[bank]
        shards[bank] = fraud_df.loc[idx[start:end]]
        start = end
    return shards


def partition_non_iid(train_pool: pd.DataFrame, cfg: PartitionConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    legit_df = train_pool[train_pool["Class"] == 0]
    fraud_df = train_pool[train_pool["Class"] == 1]

    legit_shards = _split_by_weights(legit_df, cfg.legit_weights, rng)
    fraud_shards = _split_fraud_rows(fraud_df, cfg, rng)

    return {
        bank: pd.concat([legit_shards[bank], fraud_shards[bank]]).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
        for bank in BANK_NAMES
    }
