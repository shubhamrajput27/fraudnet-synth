"""Privacy/novelty and diversity checks.

Two tools, as decided in D-021 and D-024:

1. Numeric distance to closest record (DCR): the decision-making check.
   Rows are standardised with the bank's REAL fraud mean/std (computed locally), and
   Euclidean distance is measured. A synthetic row closer to a real row than real frauds
   usually are to each other is treated as a possible near-copy (memorisation, a privacy risk).
   The same distance is used to drop near-duplicate synthetic rows (mode collapse).

2. sentence-transformers embeddings: kept because the project specifies it, but DIAGNOSTIC.
   Each row is written as text ("Time=..., V1=..., ...") and embedded. We measured that
   numeric rows written as text all look alike to a sentence model (cosine 0.96-0.995 for
   every pair type), so it can only catch essentially identical rows.
"""
import numpy as np
import pandas as pd

from ml.data.load import FEATURE_COLUMNS


# ------------------------------------------------------------ numeric DCR ---

def standardise(real: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = real[FEATURE_COLUMNS].mean()
    sd = real[FEATURE_COLUMNS].std().replace(0, 1.0)
    return (((real[FEATURE_COLUMNS] - mu) / sd).to_numpy(float),
            ((other[FEATURE_COLUMNS] - mu) / sd).to_numpy(float))


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def real_nn_distances(r: np.ndarray) -> np.ndarray:
    """Each real row's distance to its nearest OTHER real row (the natural spacing of real fraud)."""
    d = _pairwise(r, r)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def dcr(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Distance from each synthetic row to its closest real row."""
    return _pairwise(s, r).min(axis=1)


def greedy_dedupe(s: np.ndarray, min_dist: float) -> np.ndarray:
    """Keep a row only if it is at least min_dist from every row kept before it (in order).

    Returns a boolean mask. Deterministic: the candidate file's order decides which of two
    near-duplicates survives.
    """
    keep = np.zeros(len(s), dtype=bool)
    kept: list[int] = []
    for i in range(len(s)):
        if not kept or np.sqrt(((s[kept] - s[i]) ** 2).sum(1)).min() >= min_dist:
            keep[i] = True
            kept.append(i)
    return keep


# ------------------------------------------------- embeddings (diagnostic) ---

def rows_as_text(df: pd.DataFrame, decimals: int) -> list[str]:
    return [", ".join(f"{c}={v:.{decimals}f}" for c, v in zip(FEATURE_COLUMNS, row))
            for row in df[FEATURE_COLUMNS].to_numpy(float)]


def embed(texts: list[str], model) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def max_cosine(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    sims = a @ b.T
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    return sims.max(axis=1)
