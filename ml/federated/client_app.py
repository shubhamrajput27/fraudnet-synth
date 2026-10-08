"""Flower ClientApp: one simulated bank (flwr 1.39 Message API).

PRIVACY INVARIANT (CLAUDE.md 2.4): a client loads ONLY its own bank folder (chosen by its
partition-id) and replies ONLY with
  * train:    model weights (ArrayRecord) + scalar metrics (MetricRecord)
  * evaluate: scalar metrics
  * query:    integer counts / scalar metrics
Never a data row, never per-transaction scores. tests/test_privacy_invariant.py checks this.
"""
import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import average_precision_score

from ml.data.load import load_config
from ml.evaluation.metrics import compute_metrics
from ml.federated.common import BANKS, bank_data, counts_at_thresholds
from ml.models.classifier import build_model
from ml.models.train import balanced_pos_weight, predict_scores, set_seed, train_model

MODEL_CFG = load_config("model")
FL_CFG = load_config("fl")


def _setup(msg: Message, context: Context):
    """Which bank am I, which arm is this, and the current global model."""
    idx = int(context.node_config["partition-id"])
    bank = BANKS[idx]
    cfg = msg.content["config"]
    augmented, seed = bool(cfg["augmented"]), int(cfg["seed"])
    model = build_model(MODEL_CFG)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return idx, bank, augmented, seed, model, cfg


def train_fn(msg: Message, context: Context) -> Message:
    idx, bank, augmented, seed, model, cfg = _setup(msg, context)
    x, y = bank_data(bank, augmented, seed)["train"]
    rnd = int(cfg["server-round"])
    local_seed = seed + 1000 * rnd + idx           # different per round and bank, but reproducible
    set_seed(local_seed)
    log = train_model(model, x, y, MODEL_CFG["training"], local_seed, epochs=FL_CFG["local_epochs"])
    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "num-examples": int(len(y)),               # FedAvg weights the average by this
        "bank-idx": idx,
        "train-loss": float(log["history"][-1]["train_loss"]),
        "train-fraud": int(y.sum()),
        "upload-bytes": int(arrays.count_bytes()),
    })
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)


def evaluate_fn(msg: Message, context: Context) -> Message:
    """The averaged global model, scored on THIS bank's own local validation set (scalars only)."""
    idx, bank, augmented, seed, model, _ = _setup(msg, context)
    x, y = bank_data(bank, augmented, seed)["val"]
    s = predict_scores(model, x)
    pos_w = balanced_pos_weight(bank_data(bank, augmented, seed)["train"][1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(s), torch.from_numpy(y.astype(np.float64)),
        pos_weight=torch.tensor(pos_w, dtype=torch.float64)).item()
    metrics = MetricRecord({"num-examples": int(len(y)), "bank-idx": idx, "val-loss": float(loss),
                            "val-pr-auc": float(average_precision_score(y, s)), "val-fraud": int(y.sum())})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


def query_fn(msg: Message, context: Context) -> Message:
    """Two post-training queries, answered with counts or scalar metrics only.

    task = "threshold-counts": TP/FP counts of local VAL rows at each grid cut-off (D-031).
    task = "local-test":       metrics of the final model on this bank's LOCAL TEST at the chosen threshold.
    """
    idx, bank, augmented, seed, model, cfg = _setup(msg, context)
    data = bank_data(bank, augmented, seed)
    if cfg["task"] == "threshold-counts":
        x, y = data["val"]
        grid = np.asarray(cfg["grid"], dtype=float)
        tp, fp = counts_at_thresholds(y, predict_scores(model, x), grid)
        out = MetricRecord({"bank-idx": idx, "tp": tp, "fp": fp,
                            "n-fraud": int(y.sum()), "n-genuine": int(len(y) - y.sum())})
    elif cfg["task"] == "local-test":
        x, y = data["test"]
        m = compute_metrics(y, predict_scores(model, x), float(cfg["threshold"]))
        out = MetricRecord({"bank-idx": idx, **{k: float(m[k]) for k in
                            ("precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy")},
                            **{k: int(m[k]) for k in ("tp", "fp", "fn", "tn", "n_fraud")}})
    else:
        raise ValueError(f"unknown query task {cfg['task']}")
    return Message(content=RecordDict({"metrics": out}), reply_to=msg)


app = ClientApp()
app.train()(train_fn)
app.evaluate()(evaluate_fn)
app.query()(query_fn)
