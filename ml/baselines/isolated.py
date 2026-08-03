"""Isolated arm (CLAUDE.md six-arm grid): each bank trains alone on its own shard, real or
real+validated-synthetic. Privacy-preserving lower bound — no weights or data ever leave a bank.

Evaluation: each bank's model is evaluated on the shared global holdout using that bank's own
locally-fit scaler (never a pooled scaler — D4), then macro-averaged for a single headline
number, alongside the full per-client breakdown. This mirrors how the federated arm is evaluated
(ml/federated/run_federated.py) so the two are comparable.
"""
from ml.common.data import BANKS, fit_scaler, load_client_training_data, load_holdout, to_arrays
from ml.common.metric_logger import RunLogger
from ml.common.metrics import macro_average
from ml.common.model import new_model
from ml.common.train import evaluate, train_local


def run_isolated(augmented: bool, seed: int, epochs: int = 20, run_id: str | None = None) -> dict:
    arm = "isolated_augmented" if augmented else "isolated_real"
    logger = RunLogger(arm=arm, seed=seed, config={"augmented": augmented, "epochs": epochs}, run_id=run_id)

    holdout = load_holdout()
    per_client_metrics = []

    for bank in BANKS:
        train_df = load_client_training_data(bank, augmented)
        scaler = fit_scaler(train_df)
        X_train, y_train = to_arrays(train_df, scaler)
        X_holdout, y_holdout = to_arrays(holdout, scaler)

        model = new_model(seed)
        train_local(model, X_train, y_train, epochs=epochs, seed=seed)
        metrics = evaluate(model, X_holdout, y_holdout)
        metrics["n_train_rows"] = len(train_df)

        logger.log_client_metric(round_num=None, bank=bank, metrics=metrics)
        per_client_metrics.append(metrics)
        print(f"  Bank {bank}: n_train={len(train_df)} f1={metrics['f1']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")

    final = macro_average(per_client_metrics)
    logger.finalize(final)
    logger.log_round_metric(round_num=None, metrics=final)
    return {"run_id": logger.run_id, "arm": arm, "final_metrics": final, "per_client": per_client_metrics}
