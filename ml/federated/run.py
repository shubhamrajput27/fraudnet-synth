"""Step 8 runner: single-machine Flower simulation of Arms 3 and 4, then tables and plots.

Run from the project root:
    python -m ml.federated.run                  # both arms
    python -m ml.federated.run --arms arm3_federated_real
"""
import argparse
import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.data.load import PROJECT_ROOT, load_config

ARMS = ["arm3_federated_real", "arm4_federated_aug"]
BANK_COLORS = {"bank_a": "#2a78d6", "bank_b": "#eb6834", "bank_c": "#1baf7a", "bank_d": "#eda100"}
ARM_COLORS = {"arm3_federated_real": "#2a78d6", "arm4_federated_aug": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def simulate(arm: str, seed: int):
    from flwr.simulation import run_simulation

    from ml.federated.client_app import app as client_app
    from ml.federated.server_app import app as server_app

    fl = load_config("fl")
    os.environ["FRAUDNET_FL_ARM"] = arm
    os.environ["FRAUDNET_SEED"] = str(seed)
    # Ray worker processes must be able to import the `ml` package.
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=4,
                   backend_name=fl["simulation"]["backend"],
                   backend_config={"client_resources": {"num_cpus": fl["simulation"]["num_cpus_per_client"],
                                                        "num_gpus": 0.0}})


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_convergence(results: dict, out_dir):
    # 1) Global test PR-AUC per round, both arms (plot only; never used to pick a round).
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    _style(ax)
    for arm, r in results.items():
        h = pd.DataFrame(r["global_test_per_round"])
        ax.plot(h["round"], h["global_test_pr_auc"], color=ARM_COLORS[arm], linewidth=2, marker="o", markersize=3,
                label="Arm 3: federated, real only" if arm == ARMS[0] else "Arm 4: federated, real + synthetic")
    ax.set_xlabel("FL round", color=INK_MUTED)
    ax.set_title("Global test PR-AUC after each FedAvg round", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_global_pr_auc.png", dpi=150)
    plt.close(fig)

    # 2) Per-bank curves for each arm: training loss and local-val PR-AUC (separate panels, no twin axes).
    for arm, r in results.items():
        d = pd.DataFrame(r["per_bank_per_round"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
        for ax, col, title in ((axes[0], "train_loss", "Local training loss (weighted BCE)"),
                               (axes[1], "val_pr_auc", "Global model's PR-AUC on each bank's local val")):
            _style(ax)
            for bank, g in d.groupby("bank"):
                ax.plot(g["round"], g[col], color=BANK_COLORS[bank], linewidth=2,
                        label=f"{bank.replace('bank_', 'Bank ').title()} (val fraud {int(g['val_fraud'].iloc[0])})")
            ax.set_xlabel("FL round", color=INK_MUTED)
            ax.set_title(title, loc="left", color=INK, fontsize=10)
        axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.suptitle(arm, x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"convergence_per_bank_{arm}.png", dpi=150)
        plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=ARMS)
    args = parser.parse_args()
    seed = load_config("experiments")["seeds"][0]
    out_dir = PROJECT_ROOT / load_config("fl")["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        t0 = time.perf_counter()
        print(f"=== {arm}: simulating {load_config('fl')['num_rounds']} rounds ===", flush=True)
        simulate(arm, seed)
        print(f"=== {arm} done in {time.perf_counter() - t0:.0f}s (wall clock incl. Ray start-up) ===", flush=True)

    results = {}
    for arm in ARMS:
        p = out_dir / f"{arm}_seed{seed}.json"
        if p.exists():
            results[arm] = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for arm, r in results.items():
        g = r["global_test"]
        rows.append({"arm": arm, "threshold_logit": r["threshold_logit"],
                     **{k: round(g[k], 4) for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy")},
                     "tp": g["tp"], "fp": g["fp"], "fn": g["fn"], "train_s": r["train_seconds"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "global_test_table.csv", index=False)
    local_rows = [{"arm": arm, "local_test_of": b, **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items() if k in ("n_fraud", "precision", "recall", "f1", "pr_auc", "tp", "fp", "fn")}}
                  for arm, r in results.items() for b, m in sorted(r["local_test"].items())]
    pd.DataFrame(local_rows).to_csv(out_dir / "local_test_table.csv", index=False)
    plot_convergence(results, out_dir)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nGLOBAL TEST (95 fraud / 56,746 rows), final round, threshold from summed bank counts")
        print(table.to_string(index=False))
        for arm, r in results.items():
            print(f"\n[{arm}] threshold logit {r['threshold_logit']} | pooled-val stats {r['threshold_val_stats']}")
            h = r["global_test_per_round"]
            print(f"[{arm}] global PR-AUC by round: " + ", ".join(f"r{x['round']}={x['global_test_pr_auc']:.3f}"
                                                              for x in h if x["round"] in (1, 5, 10, 15, 20, 25, 30)))
            up = pd.DataFrame(r["per_bank_per_round"]).groupby("bank")["upload_bytes"].first().to_dict()
            print(f"[{arm}] bytes uploaded per bank per round: {up}")
        print("\nLOCAL TEST SETS (each bank scores the final global model itself)")
        print(pd.DataFrame(local_rows).to_string(index=False))


if __name__ == "__main__":
    main()
