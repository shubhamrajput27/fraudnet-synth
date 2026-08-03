"""Fraud classifier MLP (CLAUDE.md D2): input(30) -> 64 -> 32 -> 1, ReLU, dropout,
BCEWithLogitsLoss with positive-class weighting. Shared by all six experimental arms — the only
model definition in the project (CLAUDE.md: "never fork per-arm training logic").

Also provides state_dict <-> list-of-NumPy-arrays conversion, the exact form Flower's
NumPyClient.get_parameters/set_parameters and FedAvg's aggregate_fit expect.
"""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

N_FEATURES = 30


class FraudMLP(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits, no sigmoid — BCEWithLogitsLoss applies it


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)


def new_model(seed: int, dropout: float = 0.2) -> FraudMLP:
    torch.manual_seed(seed)
    return FraudMLP(dropout=dropout)
