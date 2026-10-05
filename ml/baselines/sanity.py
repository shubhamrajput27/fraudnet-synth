"""Step 3 sanity baseline: prove the pipeline works end to end.

Train the shared MLP on the GLOBAL train set, tune the threshold on the GLOBAL val set,
and score once on the GLOBAL test set. This is NOT one of the six arms. It only
checks that features, model, training, thresholding and metrics fit together.

Run from the project root:
    python -m ml.baselines.sanity
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data.load import PROJECT_ROOT, load_config
from ml.evaluation.metrics import best_f1_threshold, compute_metrics
from ml.models.classifier import build_model, count_parameters
from ml.models.features import FEATURE_NAMES, to_xy
from ml.models.train import predict_scores, set_seed, train_model

LINE, INK, INK_MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plot_history(history, out):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    # Two panels rather than one chart with two y-axes: the measures have different scales.
    for ax, key, title in ((axes[0], "train_loss", "Training loss (weighted BCE)"),
                           (axes[1], "val_pr_auc", "Validation PR-AUC (global val set)")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.plot(epochs, [h[key] for h in history], color=LINE, linewidth=2, marker="o", markersize=4)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Epoch", color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    seed = load_config("data")["seed"]
    cfg = load_config("model")
    processed = PROJECT_ROOT / load_config("partition")["processed_dir"]
    out_dir = PROJECT_ROOT / "results" / "sanity_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = to_xy(pd.read_csv(processed / "global_train.csv"))
    x_va, y_va = to_xy(pd.read_csv(processed / "global_val.csv"))
    x_te, y_te = to_xy(pd.read_csv(processed / "global_test.csv"))

    set_seed(seed)
    model = build_model(cfg)

    def track(epoch, m):
        from sklearn.metrics import average_precision_score
        return {"val_pr_auc": float(average_precision_score(y_va, predict_scores(m, x_va)))}

    t0 = time.perf_counter()
    log = train_model(model, x_tr, y_tr, cfg["training"], seed, on_epoch_end=track)
    train_seconds = time.perf_counter() - t0

    # Threshold chosen on VALIDATION scores only; the test set is scored once at the end.
    val_scores, test_scores = predict_scores(model, x_va), predict_scores(model, x_te)
    tuned = best_f1_threshold(y_va, val_scores)
    results = {
        "description": "Step 3 sanity baseline: MLP on global train, threshold on global val, scored on global test. NOT a six-arm result.",
        "seed": seed,
        "n_parameters": count_parameters(model),
        "features": FEATURE_NAMES,
        "train_rows": int(len(y_tr)), "train_fraud": int(y_tr.sum()),
        "pos_weight": round(log["pos_weight"], 2),
        "train_seconds": round(train_seconds, 1),
        "history": log["history"],
        "tuned_threshold_from_val": tuned,
        "val_at_tuned": compute_metrics(y_va, val_scores, tuned),
        "test_at_0.5": compute_metrics(y_te, test_scores, 0.5),
        "test_at_tuned": compute_metrics(y_te, test_scores, tuned),
        "test_always_genuine": compute_metrics(y_te, np.zeros_like(y_te), 0.5),
    }
    with open(out_dir / "sanity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _plot_history(log["history"], out_dir / "training_curve.png")

    print(f"Model: FraudMLP, {results['n_parameters']:,} parameters, {len(FEATURE_NAMES)} input features")
    print(f"Train: {results['train_rows']:,} rows, {results['train_fraud']} fraud | "
          f"pos_weight = {results['pos_weight']} | {cfg['training']['epochs']} epochs in {results['train_seconds']}s")
    for h in log["history"]:
        print(f"  epoch {h['epoch']:2d}  loss {h['train_loss']:.4f}  val PR-AUC {h['val_pr_auc']:.4f}")
    print(f"Threshold tuned on global VAL (max F1): {tuned:.4f}")
    cols = ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn", "tn"]
    table = pd.DataFrame({k: {c: results[k][c] for c in cols}
                          for k in ("test_always_genuine", "test_at_0.5", "test_at_tuned")}).T
    print("\nGLOBAL TEST set (95 fraud / 56,746 rows):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
