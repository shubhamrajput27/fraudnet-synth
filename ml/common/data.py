"""Per-client and pooled data loading for Phase 4 (CLAUDE.md D3, D4).

D4: every StandardScaler is fit on real+validated-synthetic TRAINING rows only, per client (or
once on the pooled training set for the centralized arm) — never on the holdout, never pooled
across clients for isolated/federated. The holdout set (D3) is loaded raw and scaled at
evaluation time using whichever scaler the caller supplies (per-client scaler for isolated/
federated evaluation, the pooled scaler for centralized — see ml/federated and ml/baselines).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.common.config import CONFIG

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
BANKS = ["A", "B", "C", "D"]


def load_client_training_data(bank: str, augmented: bool) -> pd.DataFrame:
    real = pd.read_csv(CONFIG.shards_dir / f"bank_{bank}.csv")
    if not augmented:
        return real

    validated_path = CONFIG.validated_dir / f"bank_{bank}_validated.csv"
    if not validated_path.exists():
        return real
    synthetic = pd.read_csv(validated_path)
    if synthetic.empty:
        return real  # e.g. Bank B/C: validation rejected the whole batch (Phase 3 finding)
    return pd.concat([real, synthetic], ignore_index=True)


def load_pooled_training_data(augmented: bool) -> pd.DataFrame:
    frames = [load_client_training_data(bank, augmented) for bank in BANKS]
    return pd.concat(frames, ignore_index=True)


def load_holdout() -> pd.DataFrame:
    return pd.read_csv(CONFIG.holdout_path)


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].to_numpy())
    return scaler


def to_arrays(df: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    X = scaler.transform(df[FEATURE_COLUMNS].to_numpy()).astype(np.float32)
    y = df["Class"].to_numpy().astype(np.float32)
    return X, y
