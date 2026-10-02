"""Global hold-out split: done once, BEFORE any bank sees data.

The global test set is the shared "final exam" for all six arms, so it must be
carved off first and never touched by partitioning, generators, or scalers.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.load import LABEL_COLUMN


def stratified_three_way(df: pd.DataFrame, train: float, val: float, test: float, seed: int):
    """Split df into train/val/test with the same fraud ratio in each part.

    sklearn only splits in two, so we split twice: first test off the full data,
    then val off the remainder (rescaling val's fraction to the remainder's size).
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rest, test_df = train_test_split(df, test_size=test, stratify=df[LABEL_COLUMN], random_state=seed)
    train_df, val_df = train_test_split(
        rest, test_size=val / (train + val), stratify=rest[LABEL_COLUMN], random_state=seed
    )
    return train_df, val_df, test_df


def global_split(df: pd.DataFrame, cfg: dict, seed: int):
    """Optionally drop exact duplicates, then make the global train/val/test split."""
    n_before = len(df)
    if cfg["drop_duplicates"]:
        # Duplicates removed before splitting, so no transaction can appear in both train and test.
        df = df.drop_duplicates(keep="first")
    s = cfg["global_split"]
    train_df, val_df, test_df = stratified_three_way(df, s["train"], s["val"], s["test"], seed)
    info = {"rows_before_dedup": n_before, "rows_after_dedup": len(df)}
    return train_df, val_df, test_df, info
