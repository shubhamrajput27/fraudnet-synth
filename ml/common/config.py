"""Shared configuration: paths and the global random seed (CLAUDE.md D5 / .env.example)."""
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class Config:
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    raw_data_path: Path = DATA_DIR / "raw" / "creditcard.csv"
    processed_dir: Path = DATA_DIR / "processed"
    shards_dir: Path = DATA_DIR / "shards"
    holdout_path: Path = DATA_DIR / "processed" / "global_test_holdout.csv"


CONFIG = Config()
