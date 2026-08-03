"""Shared local-training and evaluation loop (CLAUDE.md: no duplicated training logic across the
six arms — isolated, federated clients, and centralized all call these same two functions).
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.common.metrics import compute_metrics
from ml.common.model import FraudMLP


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = max(1, int(y.sum()))
    n_neg = len(y) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_local(
    model: FraudMLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
) -> float:
    """Trains `model` in place for `epochs` local epochs. Returns the mean training loss over
    the final epoch."""
    torch.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    final_epoch_losses: list[float] = []
    for _ in range(epochs):
        final_epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            final_epoch_losses.append(loss.item())

    return float(np.mean(final_epoch_losses)) if final_epoch_losses else float("nan")


@torch.no_grad()
def evaluate(model: FraudMLP, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    logits = model(torch.from_numpy(X)).numpy()
    return compute_metrics(y, logits)
