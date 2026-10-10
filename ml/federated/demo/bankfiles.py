"""Helpers shared by the Step 8B export and check scripts: manifests, checksums, row fingerprints."""
import hashlib
from pathlib import Path

import pandas as pd

from ml.data.load import LABEL_COLUMN, load_config

BANKS = list(load_config("partition")["banks"])
EXPORT_FILES = load_config("demo")["export"]["files"]
REAL_FILES = ["train.csv", "val.csv", "test.csv"]
MANIFEST = "manifest.json"
ALLOWED_IN_BANK_FOLDER = set(EXPORT_FILES) | {MANIFEST, ".gitkeep"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(df: pd.DataFrame) -> set[int]:
    """One 64-bit hash per row over ALL column values (the CSVs have no row-ID column).
    Rows are unique after D-005, so a fingerprint identifies a row."""
    return set(pd.util.hash_pandas_object(df, index=False).astype("uint64").tolist())


def file_summary(path: Path) -> dict:
    df = pd.read_csv(path)
    return {"rows": int(len(df)), "fraud": int(df[LABEL_COLUMN].sum()), "sha256": sha256(path)}


def data_files(folder: Path) -> list[Path]:
    """Every file in a folder tree except Git's placeholder `.gitkeep`."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep")
