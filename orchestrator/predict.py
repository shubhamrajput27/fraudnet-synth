"""Inference for the dashboard's "test a transaction" demo (CLAUDE.md Tier 1) — the /predict
endpoint the pipeline-order comment in CLAUDE.md refers to ("...federated training -> per-round
evaluation & logging -> serve via /predict"). Loads whichever model+scaler a completed run saved
(ml/common/artifacts.py) and scores one transaction.

Bank selection: isolated has a separate model *and* scaler per bank (no shared model at all);
federated has one shared model but per-client scalers (D4 — no pooled scaler exists); centralized
has one model and one pooled scaler, so no bank is needed. This mirrors D13's evaluation design
rather than inventing a new rule.
"""
import numpy as np
import torch

from ml.common.artifacts import load_manifest, load_model, load_scaler
from ml.common.data import FEATURE_COLUMNS


class PredictionError(ValueError):
    pass


def get_predict_manifest(run_id: str) -> dict | None:
    """Tells a caller (the dashboard) whether this run needs a `bank` for /predict, and which
    banks are valid, without it having to know the isolated/federated/centralized distinction."""
    manifest = load_manifest(run_id)
    if manifest is None:
        return None
    arm = manifest["arm"]
    return {
        "run_id": run_id,
        "arm": arm,
        "needs_bank": arm.startswith("isolated") or arm.startswith("federated"),
        "banks": manifest["banks"],
    }


def predict(run_id: str, bank: str | None, features: dict[str, float]) -> tuple[float, str]:
    manifest = load_manifest(run_id)
    if manifest is None:
        raise PredictionError(f"No saved model artifacts for run_id '{run_id}'")

    arm = manifest["arm"]
    needs_bank = arm.startswith("isolated") or arm.startswith("federated")
    if needs_bank and not bank:
        raise PredictionError(f"arm '{arm}' requires a 'bank' (one of {manifest['banks']})")
    if bank and manifest["banks"] and bank not in manifest["banks"]:
        raise PredictionError(f"Unknown bank '{bank}', expected one of {manifest['banks']}")

    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise PredictionError(f"Missing feature values: {missing}")

    # isolated: per-bank model file. federated/centralized: one shared model file.
    model = load_model(run_id, bank=bank if arm.startswith("isolated") else None)
    scaler = load_scaler(run_id, bank=bank if needs_bank else None)

    row = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
    X = scaler.transform(row).astype(np.float32)

    with torch.no_grad():
        logit = model(torch.from_numpy(X)).item()
    probability = float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
    label = "fraud" if probability >= 0.5 else "legitimate"
    return probability, label
