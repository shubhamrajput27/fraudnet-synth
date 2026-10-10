"""Flower ServerApp: FedAvg over the four banks, then a count-only threshold step (flwr 1.39).

The server never receives data rows or per-transaction scores. It receives model weights,
scalar metrics, and (after training) integer TP/FP counts per threshold from each bank.

The global test set is held by the EXPERIMENTER, not by any bank. Per-round global-test
PR-AUC is logged for the convergence plot only; the final model is always the last round's
(choosing a round by test score would be leakage).
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.data.load import DATA_ROOT, PROJECT_ROOT, load_config
from ml.evaluation.metrics import compute_metrics
from ml.federated.common import BANKS, best_threshold_from_counts, threshold_grid
from ml.models.classifier import build_model
from ml.models.features import to_xy
from ml.models.train import predict_scores, set_seed

MODEL_CFG = load_config("model")
FL_CFG = load_config("fl")
ROUND_LOG: dict = {"train": [], "evaluate": []}   # per-bank metrics captured every round
DEMO = {"on": False}                              # Step 8B: per-message logging for the live demo


def demo_log(line: str) -> None:
    """Print and append to results/demo_8b/server.log (in the server's own data folder)."""
    if not DEMO["on"]:
        return
    print(line, flush=True)
    log_dir = DATA_ROOT / "results" / "demo_8b"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "server.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  {line}\n")


def describe_message(msg: Message) -> str:
    """What a reply physically contains: record types, names and sizes (proof that no rows travel)."""
    c = msg.content
    parts = [f"ArrayRecord '{k}' ({len(list(v.keys()))} weight tensors, {v.count_bytes():,} B)"
             for k, v in c.array_records.items()]
    parts += [f"MetricRecord '{k}' ({len(list(v.keys()))} scalars)" for k, v in c.metric_records.items()]
    parts += [f"ConfigRecord '{k}' (UNEXPECTED)" for k in c.config_records.keys()]
    return ", ".join(parts) or "empty"


def _bank_idx(msg: Message) -> int:
    return int(next(iter(msg.content.metric_records.values()))["bank-idx"])


class OrderedFedAvg(FedAvg):
    """Plain FedAvg, but replies are aggregated in a FIXED bank order (A, B, C, D).

    Ray trains banks in parallel, so replies arrive in varying order. Floating-point addition is
    not associative, so a different order gave ~1e-8 different averages, which training then
    amplified (two identical-seed runs drifted apart from the round where the order changed).
    Sorting makes runs bit-reproducible without changing the FedAvg maths (decision D-032).
    """

    def aggregate_train(self, server_round, replies):
        replies = sorted(replies, key=_bank_idx)
        for r in replies:
            demo_log(f"Round {server_round}: update from {BANKS[_bank_idx(r)]} <- {describe_message(r)}")
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            demo_log(f"Round {server_round}: FedAvg of {len(replies)} bank updates (weighted by rows) -> "
                     f"new global model ({arrays.count_bytes():,} B) sent to every bank")
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        return super().aggregate_evaluate(server_round, sorted(replies, key=_bank_idx))


def _per_bank(kind: str):
    """Metric aggregation that ALSO records each bank's own scalars for the convergence plots."""
    def aggr(records: list[RecordDict], weighting_key: str) -> MetricRecord:
        rows, total = [], 0
        for rd in records:
            m = next(iter(rd.metric_records.values()))
            rows.append({k: m[k] for k in m.keys()})
            total += m[weighting_key]
        ROUND_LOG[kind].append(rows)
        keys = [k for k in rows[0] if k not in ("bank-idx", weighting_key) and isinstance(rows[0][k], (int, float))]
        return MetricRecord({k: float(sum(r[k] * r[weighting_key] for r in rows) / total) for k in keys})
    return aggr


def _global_eval_fn(global_test: pd.DataFrame):
    x, y = to_xy(global_test)
    history = []

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(MODEL_CFG)
        model.load_state_dict(arrays.to_torch_state_dict())
        s = predict_scores(model, x)
        rec = {"round": server_round, "global_test_pr_auc": float(average_precision_score(y, s)),
               "global_test_roc_auc": float(roc_auc_score(y, s))}
        history.append(rec)
        return MetricRecord({k: v for k, v in rec.items() if k != "round"})
    return evaluate, history


def _query_all(grid: Grid, arrays: ArrayRecord, config: dict) -> list[MetricRecord]:
    """Send the final model + a task to every bank; collect their (count/scalar-only) replies."""
    msgs = [Message(content=RecordDict({"arrays": arrays, "config": ConfigRecord(config)}),
                    dst_node_id=nid, message_type=MessageType.QUERY)
            for nid in grid.get_node_ids()]
    replies = list(grid.send_and_receive(msgs))
    errors = [r.error for r in replies if r.has_error()]
    if errors:
        raise RuntimeError(f"query failed: {errors}")
    return [next(iter(r.content.metric_records.values())) for r in replies]


def run_arm(grid: Grid, arm: str, augmented: bool, seed: int, num_rounds: int, expected_banks: int = 4) -> dict:
    ROUND_LOG["train"].clear()
    ROUND_LOG["evaluate"].clear()
    processed = DATA_ROOT / load_config("partition")["processed_dir"]
    global_test = pd.read_csv(processed / "global_test.csv")

    set_seed(seed)  # same initial weights as every Step 7 model (paired comparison)
    initial = ArrayRecord(build_model(MODEL_CFG).state_dict())
    strategy = OrderedFedAvg(fraction_train=FL_CFG["fraction_train"], fraction_evaluate=FL_CFG["fraction_evaluate"],
                      min_train_nodes=expected_banks, min_evaluate_nodes=expected_banks,
                      min_available_nodes=expected_banks,
                      train_metrics_aggr_fn=_per_bank("train"), evaluate_metrics_aggr_fn=_per_bank("evaluate"))
    eval_fn, global_history = _global_eval_fn(global_test)
    base_cfg = {"augmented": augmented, "seed": seed}

    t0 = time.perf_counter()
    result = strategy.start(grid=grid, initial_arrays=initial, num_rounds=num_rounds,
                            train_config=ConfigRecord(base_cfg), evaluate_config=ConfigRecord(base_cfg),
                            evaluate_fn=eval_fn)
    train_seconds = time.perf_counter() - t0
    final = result.arrays

    # Threshold (D-031): banks send TP/FP COUNTS on their local val at fixed cut-offs; the server sums them.
    grid_vals = threshold_grid()
    count_replies = _query_all(grid, final, {**base_cfg, "task": "threshold-counts", "grid": grid_vals.tolist()})
    tp = np.sum([np.asarray(r["tp"]) for r in count_replies], axis=0)
    fp = np.sum([np.asarray(r["fp"]) for r in count_replies], axis=0)
    n_fraud = int(sum(r["n-fraud"] for r in count_replies))
    threshold, val_stats = best_threshold_from_counts(grid_vals, tp, fp, n_fraud)
    for r in count_replies:
        demo_log(f"Threshold step: {BANKS[int(r['bank-idx'])]} sent {len(r['tp'])} TP + {len(r['fp'])} FP "
                 f"integer counts (no scores, no rows)")
    demo_log(f"Threshold step: summed counts -> cut-off logit {threshold} (pooled val F1 {val_stats['val_f1']:.4f})")

    # Local test: each bank scores the final model on its OWN local test and returns scalars.
    local_replies = _query_all(grid, final, {**base_cfg, "task": "local-test", "threshold": threshold})
    local = {BANKS[int(r["bank-idx"])]: {k: r[k] for k in r.keys() if k != "bank-idx"} for r in local_replies}

    # Global test: held by the experimenter (not a bank), scored once with the final model.
    model = build_model(MODEL_CFG)
    model.load_state_dict(final.to_torch_state_dict())
    xg, yg = to_xy(global_test)
    global_metrics = compute_metrics(yg, predict_scores(model, xg), threshold)

    per_round = []
    for rnd, (tr, ev) in enumerate(zip(ROUND_LOG["train"], ROUND_LOG["evaluate"]), start=1):
        for t in tr:
            e = next(x for x in ev if x["bank-idx"] == t["bank-idx"])
            per_round.append({"round": rnd, "bank": BANKS[int(t["bank-idx"])], "train_loss": t["train-loss"],
                              "num_examples": t["num-examples"], "upload_bytes": t["upload-bytes"],
                              "val_loss": e["val-loss"], "val_pr_auc": e["val-pr-auc"], "val_fraud": e["val-fraud"]})
    return {"arm": arm, "augmented": augmented, "seed": seed, "num_rounds": num_rounds,
            "local_epochs": FL_CFG["local_epochs"], "train_seconds": round(train_seconds, 1),
            "threshold_logit": threshold, "threshold_val_stats": {**val_stats, "val_fraud_total": n_fraud},
            "global_test": global_metrics, "local_test": local,
            "global_test_per_round": [h for h in global_history if h["round"] > 0],
            "global_test_round0": next((h for h in global_history if h["round"] == 0), None),
            "per_bank_per_round": per_round}


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config  # deployment (`flwr run`, Step 8B): values from pyproject.toml / --run-config
    # Simulation (ml/federated/run.py) passes the arm via environment variables instead.
    arm = str(rc.get("arm", os.environ.get("FRAUDNET_FL_ARM", "")))
    augmented = arm == "arm4_federated_aug"
    seed = int(rc.get("seed", os.environ.get("FRAUDNET_SEED", load_config("data")["seed"])))
    # FRAUDNET_FL_ROUNDS is only for quick smoke tests; reported runs use configs/fl.yaml (30).
    num_rounds = int(rc.get("num-rounds", os.environ.get("FRAUDNET_FL_ROUNDS", FL_CFG["num_rounds"])))
    expected = int(rc.get("expected-banks", 4))
    DEMO["on"] = bool(rc.get("demo-log", False))
    demo_log(f"=== FraudNet-Synth demo run: {arm}, {num_rounds} rounds, waiting for {expected} banks; "
             f"server data folder holds only the global test set ===")
    t0 = time.perf_counter()
    res = run_arm(grid, arm, augmented, seed, num_rounds, expected)
    if DEMO["on"]:
        g = res["global_test"]
        demo_log(f"=== Done in {time.perf_counter() - t0:.0f}s. Global test (95 fraud): precision {g['precision']:.4f} "
                 f"recall {g['recall']:.4f} F1 {g['f1']:.4f} PR-AUC {g['pr_auc']:.4f}. DEMO ONLY: reported results "
                 f"come from the single-machine runs. ===")
        out_dir = DATA_ROOT / "results" / "demo_8b"      # the demo never overwrites experiment results
    else:
        out_dir = PROJECT_ROOT / FL_CFG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{arm}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
