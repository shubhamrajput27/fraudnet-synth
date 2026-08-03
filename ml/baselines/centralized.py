"""Centralized arm (CLAUDE.md six-arm grid): all four banks' data pooled into one training set,
one model, one pooled StandardScaler. Non-privacy-preserving upper bound — deliberately violates
the privacy invariant by design (that's the point of this arm: an upper-bound reference), so
pooling here is not a defect the way it would be for the isolated/federated arms.
"""
from ml.common.data import fit_scaler, load_holdout, load_pooled_training_data, to_arrays
from ml.common.metric_logger import RunLogger
from ml.common.model import new_model
from ml.common.train import evaluate, train_local


def run_centralized(augmented: bool, seed: int, epochs: int = 20) -> dict:
    arm = "centralized_augmented" if augmented else "centralized_real"
    logger = RunLogger(arm=arm, seed=seed, config={"augmented": augmented, "epochs": epochs})

    train_df = load_pooled_training_data(augmented)
    holdout = load_holdout()

    scaler = fit_scaler(train_df)
    X_train, y_train = to_arrays(train_df, scaler)
    X_holdout, y_holdout = to_arrays(holdout, scaler)

    model = new_model(seed)
    train_local(model, X_train, y_train, epochs=epochs, seed=seed)
    metrics = evaluate(model, X_holdout, y_holdout)
    metrics["n_train_rows"] = len(train_df)

    print(f"  Pooled: n_train={len(train_df)} f1={metrics['f1']:.4f} "
          f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")

    logger.log_round_metric(round_num=None, metrics=metrics)
    logger.finalize(metrics)
    return {"run_id": logger.run_id, "arm": arm, "final_metrics": metrics}
