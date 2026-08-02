"""Global stratified test holdout, carved out before any partitioning (CLAUDE.md D3).

Never sharded, never augmented, identical across all six experimental arms.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def global_holdout_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pool, holdout = train_test_split(
        df, test_size=test_size, stratify=df["Class"], random_state=seed
    )
    return train_pool.reset_index(drop=True), holdout.reset_index(drop=True)
