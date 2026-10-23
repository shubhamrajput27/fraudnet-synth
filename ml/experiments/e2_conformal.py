"""Optional extension E2: conformal risk control of the false-negative (miss) rate, global vs per-bank.

    python -m ml.experiments.e2_conformal               # (re)creates the 5 Arm 3 models if missing, then analyses
    python -m ml.experiments.e2_conformal --report-only

Models: federated Arm 3 (real data), seeds 42-46, re-run with the final model saved (bit-identical to Step 9).
Calibration (D-063):
  global   = all four banks' local VALIDATION frauds (49), combined from per-bank miss counts on the fixed grid;
  per-bank = each bank's own local validation frauds (22 / 17 / 6 / 4).
Reported results use the fixed test sets. The separately labelled 'guarantee check' re-splits val+test frauds
at random only to show the average miss rate stays <= alpha; nothing reported is chosen on test data.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ml.data.load import PROJECT_ROOT, load_config
from ml.evaluation.conformal import FLAG_ALL, crc_threshold, min_feasible_alpha, misses_on_grid
from ml.federated.common import BANKS, bank_data, threshold_grid
from ml.models.classifier import build_model
from ml.models.features import to_xy
from ml.models.train import predict_scores
from ml.evaluation.plot_style import INK, INK_MUTED, GRID, SURFACE

ARM = "arm3_federated_real"
CFG = load_config("e2_conformal")
ROOT = PROJECT_ROOT / CFG["results_dir"]
COLORS = {"global": "#2a78d6", "per-bank": "#eb6834", "F1-tuned": "#898781"}


def ensure_model(seed: int):
    """Re-run the federated simulation with the final model saved (skipped if already there)."""
    run_dir, mdir = ROOT / "runs", ROOT / "models"
    model_path = mdir / f"{ARM}_seed{seed}.pt"
    res_path = run_dir / f"{ARM}_seed{seed}.json"
    if not (model_path.exists() and res_path.exists()):
        from ml.federated.run import simulate
        run_dir.mkdir(parents=True, exist_ok=True)
        mdir.mkdir(parents=True, exist_ok=True)
        os.environ.pop("FRAUDNET_STRATEGY", None)          # plain FedAvg
        os.environ.update({"FRAUDNET_FL_OUT_DIR": str(run_dir), "FRAUDNET_SAVE_MODEL": str(model_path)})
        simulate(ARM, seed)
        os.environ.pop("FRAUDNET_SAVE_MODEL", None)
    run = json.loads(res_path.read_text(encoding="utf-8"))
    ref = json.loads((PROJECT_ROOT / "results/six_arms" / f"seed{seed}" / f"{ARM}_seed{seed}.json").read_text(encoding="utf-8"))
    if run["global_test"] != ref["global_test"]:
        raise SystemExit(f"seed {seed}: re-run model differs from Step 9; refusing to analyse it")
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(load_config("model"))
    model.load_state_dict(ck["state_dict"])
    return model, float(run["threshold_logit"])


def scores_for(model, seed):
    """Scores of one model on every bank's val/test and the global test. (Single-machine simulation: in a
    deployment each bank scores its own data; only the global step needs counts from the banks.)"""
    out = {}
    for b in BANKS:
        d = bank_data(b, False, seed)
        for part in ("val", "test"):
            x, y = d[part]
            out[(b, part)] = (predict_scores(model, x), y)
    xg, yg = to_xy(pd.read_csv(PROJECT_ROOT / load_config("partition")["processed_dir"] / "global_test.csv"))
    out[("global", "test")] = (predict_scores(model, xg), yg)
    return out


def outcome(scores, y, t):
    flagged = scores >= t
    n_fraud = int(y.sum())
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {"miss_rate": (n_fraud - tp) / n_fraud if n_fraud else float("nan"), "caught": f"{tp}/{n_fraud}",
            "precision": tp / (tp + fp) if tp + fp else 0.0, "alert_rate": float(flagged.mean()),
            "alerts": int(flagged.sum())}


def analyse_seed(seed):
    model, f1_thr = ensure_model(seed)
    S = scores_for(model, seed)
    grid = threshold_grid()
    rows, guar = [], []
    n_val = {b: int(S[(b, "val")][1].sum()) for b in BANKS}
    # Each bank: misses of ITS validation frauds at every grid cut-off (all it would send for the global rule).
    bank_misses = {b: misses_on_grid(S[(b, "val")][0][S[(b, "val")][1] == 1], grid) for b in BANKS}
    for alpha in CFG["alphas"]:
        t_global = crc_threshold(grid, sum(bank_misses.values()), sum(n_val.values()), alpha)
        for scope, thr_of in (("global", lambda b: t_global),
                              ("per-bank", lambda b: crc_threshold(grid, bank_misses[b], n_val[b], alpha))):
            for b in BANKS:
                t = thr_of(b)
                rows.append({"seed": seed, "alpha": alpha, "scope": scope, "eval": f"{b} local test",
                             "threshold": "flag all" if t == FLAG_ALL else t,
                             "cal_frauds": sum(n_val.values()) if scope == "global" else n_val[b],
                             **outcome(*S[(b, "test")], t)})
        rows.append({"seed": seed, "alpha": alpha, "scope": "global", "eval": "global test",
                     "threshold": "flag all" if t_global == FLAG_ALL else t_global,
                     "cal_frauds": sum(n_val.values()), **outcome(*S[("global", "test")], t_global)})
    for b in BANKS + ["global"]:
        ev = S[(b, "test")] if b != "global" else S[("global", "test")]
        rows.append({"seed": seed, "alpha": None, "scope": "F1-tuned", "eval": "global test" if b == "global" else f"{b} local test",
                     "threshold": f1_thr, "cal_frauds": sum(n_val.values()), **outcome(*ev, f1_thr)})

    # Guarantee check: random re-splits of val+test frauds into calibration / evaluation.
    rng = np.random.default_rng(seed)
    pools = {b: np.concatenate([S[(b, p)][0][S[(b, p)][1] == 1] for p in ("val", "test")]) for b in BANKS}
    pools["all banks"] = np.concatenate(list(pools.values()))
    cal_n = {**n_val, "all banks": sum(n_val.values())}
    for name, fr in pools.items():
        for alpha in CFG["alphas"]:
            misses = []
            for _ in range(CFG["guarantee_check"]["splits"]):
                perm = rng.permutation(len(fr))
                cal, ev = fr[perm[:cal_n[name]]], fr[perm[cal_n[name]:]]
                t = crc_threshold(grid, misses_on_grid(cal, grid), len(cal), alpha)
                misses.append(float(np.mean(ev < t)))
            m = np.array(misses)
            guar.append({"seed": seed, "pool": name, "alpha": alpha, "cal_frauds": cal_n[name], "eval_frauds": len(fr) - cal_n[name],
                         "mean_miss_rate": float(m.mean()), "split_sd": float(m.std(ddof=1)), "n_splits": len(m),
                         "share_of_splits_above_alpha": float((m > alpha).mean())})
    return rows, guar


def summarise(long: pd.DataFrame, guar: pd.DataFrame):
    pm = lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
    long["alpha_label"] = long["alpha"].map(lambda a: "F1-tuned" if pd.isna(a) else f"{a:.2f}")
    keys = ["alpha_label", "scope", "eval"]
    tab = long.groupby(keys, sort=False).agg(miss_rate=("miss_rate", pm), precision=("precision", pm),
                                              alert_rate=("alert_rate", lambda s: f"{100 * s.mean():.2f}%"),
                                              caught=("caught", " ".join),
                                              threshold=("threshold", lambda s: " ".join(str(v) for v in s))).reset_index()
    g = guar.groupby(["pool", "alpha", "cal_frauds", "eval_frauds"], sort=False).agg(
        mean_miss_rate=("mean_miss_rate", "mean"), share_of_splits_above_alpha=("share_of_splits_above_alpha", "mean"),
        split_sd=("split_sd", "mean"), n_total=("n_splits", "sum")).reset_index()
    # Monte Carlo standard error of the average miss rate (splits re-use the same frauds, so this is approximate).
    g["mc_se"] = (g["split_sd"] / np.sqrt(g["n_total"])).round(4)
    g["within_alpha"] = g["mean_miss_rate"] <= g["alpha"] + 2 * g["mc_se"]
    g = g.drop(columns=["split_sd", "n_total"])
    g["min_feasible_alpha"] = g["cal_frauds"].map(lambda n: round(min_feasible_alpha(n), 3))
    for c in ("mean_miss_rate", "share_of_splits_above_alpha"):
        g[c] = g[c].round(4)
    return tab, g


def plot(long):
    alphas = CFG["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    # Left: realised miss rate on the global test vs target alpha (global calibration).
    ax = axes[0]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    g = long[(long["scope"] == "global") & (long["eval"] == "global test")]
    m = [g[g["alpha"] == a]["miss_rate"].mean() for a in alphas]
    s = [g[g["alpha"] == a]["miss_rate"].std(ddof=1) for a in alphas]
    ax.plot([0, 0.25], [0, 0.25], color=INK_MUTED, linestyle="--", linewidth=1, label="target: miss rate = α")
    ax.errorbar(alphas, m, yerr=s, fmt="o-", color=COLORS["global"], capsize=3, linewidth=2, label="CRC, global calibration")
    f1 = long[(long["scope"] == "F1-tuned") & (long["eval"] == "global test")]["miss_rate"].mean()
    ax.axhline(f1, color=COLORS["F1-tuned"], linestyle=":", linewidth=1.5, label=f"F1-tuned threshold ({f1:.2f})")
    ax.set_xlabel("target miss rate α", color=INK_MUTED)
    ax.set_ylabel("realised miss rate, global test (95 frauds)", color=INK_MUTED)
    ax.set_title("Miss rate is held near or below α", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    # Right: the price, alert rate per bank at alpha = 0.10, global vs per-bank calibration (log scale).
    ax = axes[1]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    a = 0.10
    x = np.arange(len(BANKS))
    for i, scope in enumerate(("global", "per-bank")):
        d = long[(long["alpha"] == a) & (long["scope"] == scope)]
        vals = [100 * d[d["eval"] == f"{b} local test"]["alert_rate"].mean() for b in BANKS]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=COLORS[scope], label=f"{scope} calibration")
        for xi, v in zip(x + (i - 0.5) * 0.38, vals):
            ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=7, color=INK)
    ax.set_ylim(0, 110)
    ax.set_xticks(x, [f"Bank {b[-1].upper()}" for b in BANKS])
    ax.set_ylabel("% of the bank's transactions flagged", color=INK_MUTED)
    ax.set_title("The price at α = 0.10: alerts per bank", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "e2_conformal.png", dpi=150)
    plt.close(fig)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, guar = [], []
    for seed in CFG["seeds"]:
        r, g = analyse_seed(seed)
        rows += r
        guar += g
        print(f"seed {seed}: analysed", flush=True)
    long, guar = pd.DataFrame(rows), pd.DataFrame(guar)
    long.to_csv(ROOT / "long_results.csv", index=False)
    tab, g = summarise(long, guar)
    tab.to_csv(ROOT / "summary_table.csv", index=False)
    g.to_csv(ROOT / "guarantee_check.csv", index=False)
    plot(long)
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 60):
        print("\nE2 SUMMARY (fixed test sets, mean ± std over seeds)\n" + tab.drop(columns=["threshold"]).to_string(index=False))
        print("\nGUARANTEE CHECK (random re-splits of val+test frauds; average over seeds)\n" + g.to_string(index=False))


if __name__ == "__main__":
    main()
