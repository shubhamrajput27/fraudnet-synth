"""Turn a raw ULB dataframe into model inputs.

Decision D-011: only FIXED transforms, nothing fitted on data. This means
  * no leakage (no statistic is learnt from any split),
  * every arm and every bank feeds the model identically-prepared inputs, and
  * federated banks need not share scaler statistics with each other.
"""
import numpy as np
import pandas as pd

from ml.data.load import LABEL_COLUMN, V_COLUMNS

SECONDS_PER_DAY = 24 * 3600
FEATURE_NAMES = [*V_COLUMNS, "log_amount", "hour_sin", "hour_cos"]
N_FEATURES = len(FEATURE_NAMES)  # 31


def to_features(df: pd.DataFrame) -> np.ndarray:
    """Return a float32 matrix (n_rows x 31)."""
    # V1-V28 are PCA outputs: already centred near 0 with spreads 0.33-1.96, so used as-is.
    v = df[V_COLUMNS].to_numpy(dtype=np.float64)
    # Amount is heavily skewed (median ~22, max ~25,691); log1p squeezes it to roughly 0-10.
    log_amount = np.log1p(df["Amount"].to_numpy(dtype=np.float64))
    # Time counts seconds from the file's start over 2 days. Only the daily rhythm generalises,
    # so we place it on a 24h circle. sin+cos keep 23:59 next to 00:00. The real clock time of
    # second 0 is unknown, but a fixed offset only rotates the circle, which the model can absorb.
    angle = 2 * np.pi * (df["Time"].to_numpy(dtype=np.float64) % SECONDS_PER_DAY) / SECONDS_PER_DAY
    x = np.column_stack([v, log_amount, np.sin(angle), np.cos(angle)])
    return x.astype(np.float32)


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return to_features(df), df[LABEL_COLUMN].to_numpy(dtype=np.float32)
