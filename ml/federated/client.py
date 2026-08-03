"""Per-bank Flower NumPyClient (CLAUDE.md D2/D4). Real `flwr.client.NumPyClient` subclass, driven
by a hand-rolled sequential loop rather than Flower's ray-based simulation harness (ray has no
Windows wheel for Python 3.13 as of 2026-08-03 — see PLAN.md Phase 4 progress log; confirmed with
the user before building this way).

Each client only ever sees its own data and its own locally-fit StandardScaler (D4) — fit(),
get_parameters(), and evaluate() never touch another bank's rows. Only `parameters` (weight
arrays) cross the client/server boundary, preserving the privacy invariant.
"""
from flwr.client import NumPyClient

from ml.common.model import FraudMLP, get_parameters, set_parameters
from ml.common.train import evaluate, train_local


class FedClient(NumPyClient):
    def __init__(self, bank: str, model: FraudMLP, X_train, y_train, X_eval, y_eval, seed: int, scaler=None):
        self.bank = bank
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_eval, self.y_eval = X_eval, y_eval
        self.seed = seed
        self.scaler = scaler  # kept only so callers can persist it (Phase 6 /predict) — never
        # used inside fit()/evaluate() itself, which only ever touch this client's own arrays.

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        loss = train_local(self.model, self.X_train, self.y_train, epochs=local_epochs, seed=self.seed)
        return get_parameters(self.model), len(self.X_train), {"train_loss": loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.X_eval, self.y_eval)
        loss = 1.0 - metrics["f1"]  # Flower requires a scalar loss; F1-based so lower is better
        return loss, len(self.X_eval), metrics
