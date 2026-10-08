"""Aggregate statistics computed INSIDE a data-poor bank (decision D-016).

This is the only information about real fraud that Schema Mode lets out of the bank.
Per column: mean, std, and 10th/50th/90th percentiles, rounded. Min and max are
deliberately excluded: with only 18-28 rows they are exact values of a single real
transaction. The exact bounds are computed separately by `local_bounds()` and are
used inside the bank only (Step 6 range checks), never sent anywhere.
"""
import pandas as pd

from ml.data.load import FEATURE_COLUMNS

_STAT_FUNCS = {
    "mean": lambda s: s.mean(),
    "std": lambda s: s.std(),
    "p10": lambda s: s.quantile(0.10),
    "p50": lambda s: s.quantile(0.50),
    "p90": lambda s: s.quantile(0.90),
}


def aggregate_stats(real_fraud: pd.DataFrame, include: list[str], decimals: int) -> dict:
    """{"n_rows": int, "columns": {col: {stat: value}}}. Only names in `include` are produced."""
    unknown = set(include) - set(_STAT_FUNCS)
    if unknown:  # e.g. someone adds "min" to the config: refuse rather than leak
        raise ValueError(f"statistic(s) not allowed to leave the bank: {sorted(unknown)}")
    cols = {
        c: {k: round(float(_STAT_FUNCS[k](real_fraud[c])), decimals) for k in include}
        for c in FEATURE_COLUMNS
    }
    return {"n_rows": int(len(real_fraud)), "columns": cols}


def local_bounds(real_fraud: pd.DataFrame) -> dict:
    """Exact per-column min/max. LOCAL USE ONLY: never put these in a prompt."""
    return {c: (float(real_fraud[c].min()), float(real_fraud[c].max())) for c in FEATURE_COLUMNS}
