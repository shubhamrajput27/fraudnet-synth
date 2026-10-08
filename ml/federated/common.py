"""Shared helpers for the Flower client and server (flwr 1.39 Message API)."""
from functools import lru_cache

import numpy as np
import pandas as pd

from ml.baselines.arms import load_local, sample_synthetic
from ml.data.load import LABEL_COLUMN, load_config
from ml.models.features import to_xy

BANKS = list(load_config("partition")["banks"])  # partition-id 0..3 -> bank_a..bank_d


def threshold_grid() -> np.ndarray:
    g = load_config("fl")["threshold_grid"]
    n = int(round((g["logit_max"] - g["logit_min"]) / g["step"])) + 1
    return np.round(np.linspace(g["logit_min"], g["logit_max"], n), 6)


@lru_cache(maxsize=8)
def bank_data(bank: str, augmented: bool, seed: int):
    """THIS bank's data only: local train (+ its own validated synthetic rows), val and test.

    Called inside a client with that client's own bank name. It reads nothing but
    data/clients/<bank>/. Synthetic rows are sampled exactly as in Arm 2 (same ratio, same seed),
    so Arm 4 is directly comparable with Arm 2.
    """
    local = load_local(bank)
    train = local["train"]
    if augmented:
        n_real = int(train[LABEL_COLUMN].sum())
        train = pd.concat([train, sample_synthetic(bank, n_real, load_config("experiments"), seed)])
    return {"train": to_xy(train), "val": to_xy(local["val"]), "test": to_xy(local["test"])}


def counts_at_thresholds(y: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> tuple[list[int], list[int]]:
    """For each cut-off t: frauds with score >= t (TP) and genuine rows with score >= t (FP)."""
    fraud = np.sort(scores[y == 1])
    genuine = np.sort(scores[y == 0])
    tp = len(fraud) - np.searchsorted(fraud, grid, side="left")
    fp = len(genuine) - np.searchsorted(genuine, grid, side="left")
    return tp.astype(int).tolist(), fp.astype(int).tolist()


def best_threshold_from_counts(grid: np.ndarray, tp: np.ndarray, fp: np.ndarray, n_fraud: int) -> tuple[float, dict]:
    """Server side: choose the cut-off with the best F1 from SUMMED counts only."""
    tp, fp = np.asarray(tp, float), np.asarray(fp, float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / n_fraud
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp),
                   where=(precision + recall) > 0)
    i = int(np.argmax(f1))  # ties -> the lowest cut-off reaching the best F1
    return float(grid[i]), {"val_f1": float(f1[i]), "val_precision": float(precision[i]),
                            "val_recall": float(recall[i]), "val_tp": int(tp[i]), "val_fp": int(fp[i])}
