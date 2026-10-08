"""Training and scoring for the shared classifier. Reused by isolated, centralized and federated arms."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Fix every random number generator we touch, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def balanced_pos_weight(y: np.ndarray) -> float:
    """Decision D-010: fraud weight = #genuine / #fraud of THIS training set.

    Computed from the model's own training labels only, so in FL it never needs
    another bank's data. With this weight both classes carry equal total loss,
    so adding synthetic fraud rows lowers the weight, rather than adding extra emphasis.
    """
    n_pos = float(y.sum())
    if n_pos == 0:
        raise ValueError("training data contains no fraud rows")
    return (len(y) - n_pos) / n_pos


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, train_cfg: dict, seed: int,
                epochs: int | None = None, on_epoch_end=None) -> dict:
    """Train in place with a class-weighted BCE loss and Adam. Returns a small log.

    on_epoch_end(epoch, model) is an optional callback (e.g. to track validation scores).
    """
    epochs = train_cfg["epochs"] if epochs is None else epochs
    pos_weight = balanced_pos_weight(y)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    generator = torch.Generator().manual_seed(seed)  # seeded shuffling of batches
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=train_cfg["batch_size"], shuffle=True, generator=generator)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        entry = {"epoch": epoch, "train_loss": total / n}
        if on_epoch_end is not None:
            entry.update(on_epoch_end(epoch, model) or {})
        history.append(entry)
    return {"pos_weight": pos_weight, "history": history}


@torch.no_grad()
def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Fraud SCORES = raw logits (float64), used for ranking, thresholds and PR/ROC-AUC.

    Decision D-028: with the large fraud weight, confident predictions have logits of 17-46.
    sigmoid() of those rounds to exactly 1.0 in float32, so many transactions tied at the top
    score and their ranking was lost (49 tied test rows, 41 of them fraud, in one Step 7 model).
    The sigmoid preserves order, so ranking by logit gives the same ordering, with no ties.
    Dropout is off in eval mode.
    """
    model.eval()
    out = [model(torch.from_numpy(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]
    return torch.cat(out).double().numpy()


def predict_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Fraud probability in [0, 1] for display only (e.g. the demo). Not used for metrics."""
    return 1.0 / (1.0 + np.exp(-predict_scores(model, x)))
