"""Reproducibility helper (CLAUDE.md: every random operation seeded, RANDOM_SEED default 42)."""
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
