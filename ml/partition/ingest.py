"""Load and validate the raw ULB Credit Card Fraud dataset (CLAUDE.md Phase 1)."""
from pathlib import Path

import pandas as pd

RAW_SCHEMA_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )
    df = pd.read_csv(path)

    missing = set(RAW_SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset missing expected columns: {sorted(missing)}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Raw dataset contains nulls in columns {null_cols}; the published ULB dataset has "
            "none — investigate the source file before proceeding."
        )

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        df = df.drop_duplicates().reset_index(drop=True)

    df["Class"] = df["Class"].astype(int)
    return df
