"""/predict: score one transaction with the federated Arm 3 model (D-040)."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from ml.data.load import FEATURE_COLUMNS, load_config
from ml.models.classifier import build_model
from ml.models.features import to_features
from ml.models.train import predict_scores
from services.orchestrator.settings import MODEL_PATH


@lru_cache(maxsize=1)
def load_predictor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.name} not found: run the 'prepare_model' job once "
                                "(python -m services.orchestrator.tasks prepare_model <dir>, or POST /admin/prepare-model)")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_model(load_config("model"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta_path = MODEL_PATH.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    info = {"arm": ckpt["arm"], "seed": ckpt["seed"], "num_rounds": ckpt["num_rounds"],
            "global_test_f1": meta.get("global_test", {}).get("f1"),
            "global_test_pr_auc": meta.get("global_test", {}).get("pr_auc")}
    return model, float(ckpt["threshold_logit"]), info


def predict(transactions: list[dict]) -> list[dict]:
    model, thr, info = load_predictor()
    df = pd.DataFrame(transactions)[FEATURE_COLUMNS]
    logits = predict_scores(model, to_features(df))
    out = []
    for z in logits:
        out.append({"score_logit": float(z), "fraud_probability": float(1 / (1 + np.exp(-z))),
                    "threshold_logit": thr, "is_fraud": bool(z >= thr), "model": info})
    return out
