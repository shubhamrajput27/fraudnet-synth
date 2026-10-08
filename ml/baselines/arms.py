"""Step 7: isolated and centralized arms (Arms 1, 2, 5, 6).

  Arm 1  isolated, real only        each bank trains on its own local train set
  Arm 2  isolated, augmented        + validated synthetic fraud (ratio 1:1 to real training fraud)
  Arm 5  centralized, real only     one model on the union of all banks' local train sets
  Arm 6  centralized, augmented     + every bank's sampled synthetic rows

Every model: the shared FraudMLP and training settings (configs/model.yaml), the same seed
(so real-only vs augmented differ only in data), a threshold tuned on the owner's own
validation data, and evaluation on the SAME global test set plus each bank's local test set.
Synthetic rows only ever enter TRAINING data, never validation or test.

Run from the project root:
    python -m ml.baselines.arms
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config
from ml.evaluation.metrics import best_f1_threshold, compute_metrics
from ml.models.classifier import build_model
from ml.models.features import to_xy
from ml.models.train import predict_scores, set_seed, train_model

REAL_COLOR, AUG_COLOR = "#2a78d6", "#1baf7a"   # validated palette slots 1 (blue) and 3 (aqua)
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HEADLINE = ["precision", "recall", "f1", "pr_auc"]
SECONDARY = ["roc_auc", "accuracy"]


# ------------------------------------------------------------ data ---

def bank_dir(bank: str):
    return PROJECT_ROOT / load_config("partition")["clients_dir"] / bank


def load_local(bank: str) -> dict[str, pd.DataFrame]:
    return {p: pd.read_csv(bank_dir(bank) / f"{p}.csv") for p in ("train", "val", "test")}


def sample_synthetic(bank: str, n_real_fraud: int, exp_cfg: dict, seed: int) -> pd.DataFrame:
    """Validated synthetic fraud rows for this bank, ratio x its real training fraud (D-025)."""
    pool = pd.read_csv(bank_dir(bank) / exp_cfg["augmentation"]["source_file"])
    n = min(len(pool), int(round(exp_cfg["augmentation"]["ratio_to_real_fraud"] * n_real_fraud)))
    out = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    assert (out[LABEL_COLUMN] == 1).all()
    return out[FEATURE_COLUMNS + [LABEL_COLUMN]]


# -------------------------------------------------------- one model ---

def fit_and_evaluate(train_df, val_df, eval_sets: dict, model_cfg: dict, seed: int) -> dict:
    """Train one model, tune its threshold on val_df, score it on every evaluation set."""
    x_tr, y_tr = to_xy(train_df)
    x_va, y_va = to_xy(val_df)
    set_seed(seed)
    model = build_model(model_cfg)
    t0 = time.perf_counter()
    log = train_model(model, x_tr, y_tr, model_cfg["training"], seed)
    seconds = time.perf_counter() - t0
    threshold = best_f1_threshold(y_va, predict_scores(model, x_va))
    evals = {}
    for name, df in eval_sets.items():
        x, y = to_xy(df)
        evals[name] = compute_metrics(y, predict_scores(model, x), threshold)
    return {"train_rows": int(len(y_tr)), "train_fraud": int(y_tr.sum()),
            "val_fraud": int(y_va.sum()), "pos_weight": round(log["pos_weight"], 2),
            "threshold": threshold, "train_seconds": round(seconds, 1),
            "final_train_loss": round(log["history"][-1]["train_loss"], 4), "eval": evals}


# ------------------------------------------------------------ arms ---

def run(seed: int) -> dict:
    model_cfg = load_config("model")
    exp_cfg = load_config("experiments")
    banks = list(load_config("partition")["banks"])
    processed = PROJECT_ROOT / load_config("partition")["processed_dir"]
    global_test = pd.read_csv(processed / "global_test.csv")

    local = {b: load_local(b) for b in banks}
    synth = {b: sample_synthetic(b, int(local[b]["train"][LABEL_COLUMN].sum()), exp_cfg, seed) for b in banks}
    eval_sets = {"global_test": global_test, **{f"{b}_test": local[b]["test"] for b in banks}}

    results = {"seed": seed, "synthetic_rows_added": {b: len(synth[b]) for b in banks}, "arms": {}}

    # Arms 1 and 2: each bank alone. Evaluated on the global test and its OWN local test.
    for arm, augmented in (("arm1_isolated_real", False), ("arm2_isolated_aug", True)):
        results["arms"][arm] = {}
        for b in banks:
            train = pd.concat([local[b]["train"], synth[b]]) if augmented else local[b]["train"]
            print(f"  {arm} {b}: {len(train):,} rows, {int(train[LABEL_COLUMN].sum())} fraud", flush=True)
            results["arms"][arm][b] = fit_and_evaluate(
                train, local[b]["val"], {"global_test": global_test, f"{b}_test": local[b]["test"]}, model_cfg, seed)

    # Arms 5 and 6: pooled local TRAIN parts (D-027); threshold on pooled local val (D-026).
    pooled_train = pd.concat([local[b]["train"] for b in banks])
    pooled_val = pd.concat([local[b]["val"] for b in banks])
    for arm, augmented in (("arm5_central_real", False), ("arm6_central_aug", True)):
        train = pd.concat([pooled_train, *synth.values()]) if augmented else pooled_train
        print(f"  {arm}: {len(train):,} rows, {int(train[LABEL_COLUMN].sum())} fraud", flush=True)
        results["arms"][arm] = {"pooled": fit_and_evaluate(train, pooled_val, eval_sets, model_cfg, seed)}
    return results


# ---------------------------------------------------------- tables ---

def global_table(res: dict) -> pd.DataFrame:
    rows = []
    for arm, per in res["arms"].items():
        for who, r in per.items():
            m = r["eval"]["global_test"]
            rows.append({"arm": arm, "model": who, "train_fraud": r["train_fraud"], "val_fraud": r["val_fraud"],
                         "threshold": round(r["threshold"], 4),
                         **{k: round(m[k], 4) for k in HEADLINE + SECONDARY}, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    return pd.DataFrame(rows)


def local_table(res: dict) -> pd.DataFrame:
    """Each bank's local test: its isolated models vs the centralized models."""
    rows = []
    for arm, per in res["arms"].items():
        for who, r in per.items():
            for name, m in r["eval"].items():
                if name == "global_test" or (who != "pooled" and name != f"{who}_test"):
                    continue
                rows.append({"local_test_of": name.replace("_test", ""), "arm": arm, "model": who,
                             "n_fraud": m["n_fraud"], **{k: round(m[k], 4) for k in HEADLINE},
                             "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    return pd.DataFrame(rows).sort_values(["local_test_of", "arm"]).reset_index(drop=True)


def delta_table(g: pd.DataFrame) -> pd.DataFrame:
    """Augmented minus real-only, on the global test, per bank (and for the pooled model)."""
    pairs = [("arm1_isolated_real", "arm2_isolated_aug"), ("arm5_central_real", "arm6_central_aug")]
    rows = []
    for real_arm, aug_arm in pairs:
        r = g[g.arm == real_arm].set_index("model")
        a = g[g.arm == aug_arm].set_index("model")
        for who in r.index:
            rows.append({"comparison": f"{aug_arm} - {real_arm}", "model": who,
                         **{f"d_{k}": round(a.loc[who, k] - r.loc[who, k], 4) for k in HEADLINE}})
    return pd.DataFrame(rows)


def plot(g: pd.DataFrame, out):
    who = ["bank_a", "bank_b", "bank_c", "bank_d", "pooled"]
    names = ["Bank A\n(CTGAN)", "Bank B\n(CTGAN)", "Bank C\n(LLM)", "Bank D\n(LLM)", "Centralized\n(pooled)"]
    real_arm = {w: ("arm5_central_real" if w == "pooled" else "arm1_isolated_real") for w in who}
    aug_arm = {w: ("arm6_central_aug" if w == "pooled" else "arm2_isolated_aug") for w in who}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURFACE)
    x = np.arange(len(who))
    for ax, metric, title in ((axes[0], "f1", "Fraud F1 on the global test set"),
                              (axes[1], "pr_auc", "PR-AUC on the global test set")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for off, arms, color, label in ((-0.2, real_arm, REAL_COLOR, "Real only"),
                                        (0.2, aug_arm, AUG_COLOR, "Real + validated synthetic")):
            vals = [g[(g.arm == arms[w]) & (g.model == w)][metric].item() for w in who]
            bars = ax.bar(x + off, vals, width=0.38, color=color, label=label, edgecolor=SURFACE, linewidth=2)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    exp_cfg = load_config("experiments")
    out_dir = PROJECT_ROOT / exp_cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = exp_cfg["seeds"][0]
    t0 = time.perf_counter()
    res = run(seed)
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    with open(out_dir / f"results_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    g, loc = global_table(res), local_table(res)
    d = delta_table(g)
    g.to_csv(out_dir / "global_test_table.csv", index=False)
    loc.to_csv(out_dir / "local_test_table.csv", index=False)
    d.to_csv(out_dir / "augmentation_delta_table.csv", index=False)
    plot(g, out_dir / "f1_prauc_global_test.png")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\nSynthetic rows added (ratio 1:1): {res['synthetic_rows_added']} | total {res['total_seconds']}s")
        print("\nGLOBAL TEST (95 fraud / 56,746 rows) - headline: precision, recall, f1, pr_auc")
        print(g.to_string(index=False))
        print("\nAUGMENTATION DELTA on global test (augmented - real only)")
        print(d.to_string(index=False))
        print("\nLOCAL TEST SETS (isolated model of that bank vs pooled models)")
        print(loc.to_string(index=False))


if __name__ == "__main__":
    main()
