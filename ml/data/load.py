"""Loading the raw ULB Credit Card Fraud dataset.

Every later step reads the raw file through `load_raw`, so the schema check here
catches a wrong or corrupted file once, early, instead of failing mysteriously later.
"""
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The ULB dataset's fixed schema: Time, 28 PCA components, Amount, and the label.
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
LABEL_COLUMN = "Class"
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, LABEL_COLUMN]


def load_config(name: str = "data") -> dict:
    """Read configs/<name>.yaml."""
    with open(PROJECT_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load creditcard.csv and verify it has exactly the expected columns and labels."""
    if path is None:
        path = PROJECT_ROOT / load_config("data")["raw_path"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from Kaggle and place it in data/raw/."
        )

    df = pd.read_csv(path)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    # Class must be a clean 0/1 label; anything else means a wrong file.
    if not set(df[LABEL_COLUMN].unique()) <= {0, 1}:
        raise ValueError("Class column contains values other than 0 and 1")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df
