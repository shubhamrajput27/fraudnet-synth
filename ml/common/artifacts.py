"""Model/scaler persistence for trained runs (Phase 6: the dashboard's "test a transaction" demo
and /predict endpoint need a real trained model to load, which nothing before Phase 6 saved to
disk — Phase 4's CLI runs train/evaluate purely in-memory).

Layout: data/models/<run_id>/
    model.pt              - isolated_real/augmented, centralized_real/augmented: the one model
    model_<BANK>.pt        - isolated_real/augmented only: one model per bank (no shared model)
    scaler.joblib          - centralized_real/augmented: the one pooled scaler
    scaler_<BANK>.joblib   - isolated/federated: each bank's own locally-fit scaler (D4 — no
                             pooled scaler exists for these arms, so predicting from one of them
                             requires picking a bank's scaler, same as D13's evaluation design)
"""
import json
from pathlib import Path

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from ml.common.config import CONFIG
from ml.common.model import FraudMLP

def run_artifacts_dir(run_id: str) -> Path:
    return CONFIG.models_dir / run_id


def save_model(run_id: str, model: FraudMLP, bank: str | None = None) -> None:
    run_dir = run_artifacts_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = f"model_{bank}.pt" if bank else "model.pt"
    torch.save(model.state_dict(), run_dir / filename)


def save_scaler(run_id: str, scaler: StandardScaler, bank: str | None = None) -> None:
    run_dir = run_artifacts_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = f"scaler_{bank}.joblib" if bank else "scaler.joblib"
    joblib.dump(scaler, run_dir / filename)


def save_manifest(run_id: str, arm: str, banks: list[str] | None) -> None:
    run_dir = run_artifacts_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "arm": arm, "banks": banks}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(run_id: str) -> dict | None:
    path = run_artifacts_dir(run_id) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_model(run_id: str, bank: str | None = None) -> FraudMLP:
    filename = f"model_{bank}.pt" if bank else "model.pt"
    path = run_artifacts_dir(run_id) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    model = FraudMLP()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_scaler(run_id: str, bank: str | None = None) -> StandardScaler:
    filename = f"scaler_{bank}.joblib" if bank else "scaler.joblib"
    path = run_artifacts_dir(run_id) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved scaler at {path}")
    return joblib.load(path)
