"""The ONE classifier shared by all six arms (decision D-009), plus weight helpers for Flower."""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from ml.models.features import N_FEATURES


class FraudMLP(nn.Module):
    """Small multi-layer perceptron: 31 inputs -> 64 -> 32 -> 1 fraud logit.

    It outputs a raw score (logit), not a probability: the loss function applies the
    sigmoid internally, which is numerically more stable. predict_scores() returns logits for
    ranking and thresholds; predict_proba() gives probabilities for display.
    """

    def __init__(self, hidden_sizes=(64, 32), dropout=0.1, n_features=N_FEATURES):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(cfg: dict) -> FraudMLP:
    m = cfg["mlp"]
    return FraudMLP(hidden_sizes=tuple(m["hidden_sizes"]), dropout=m["dropout"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- Weight helpers: in federated learning these arrays are ALL a bank ever sends. ---

def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Model weights as a list of plain NumPy arrays (a fixed order, from state_dict)."""
    return [t.detach().cpu().numpy().copy() for t in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    """Load a list of NumPy arrays (e.g. the server's averaged weights) into the model."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"expected {len(keys)} arrays, got {len(weights)}")
    state = OrderedDict((k, torch.tensor(w)) for k, w in zip(keys, weights))
    model.load_state_dict(state, strict=True)
