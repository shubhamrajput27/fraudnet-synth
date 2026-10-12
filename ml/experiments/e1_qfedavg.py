"""Optional extension E1: q-FedAvg vs FedAvg, per-bank fairness of fraud-class F1.

    python -m ml.experiments.e1_qfedavg              # tune (seed 42, validation) -> final 5 seeds -> report
    python -m ml.experiments.e1_qfedavg --report-only

Every simulation writes its own JSON, so an interrupted run resumes. FedAvg (the reference) is re-run
through the same code path (it is bit-identical to Step 9's Arm 3).
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data.load import PROJECT_ROOT, load_config
from ml.evaluation.six_arm_report import paired

ARM = "arm3_federated_real"
BANKS = ["bank_a", "bank_b", "bank_c", "bank_d"]
CFG = load_config("e1_qfedavg")
ROOT = PROJECT_ROOT / CFG["results_dir"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # validated palette, fixed order
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def run_one(name: str, seed: int, strategy: str, q=None, L=None) -> dict:
    out = ROOT / "runs" / name
    path = out / f"{ARM}_seed{seed}.json"
    if not path.exists():
        from ml.federated.run import simulate
        out.mkdir(parents=True, exist_ok=True)
        os.environ.update({"FRAUDNET_FL_OUT_DIR": str(out), "FRAUDNET_STRATEGY": strategy})
        if strategy == "qfedavg":
            os.environ.update({"FRAUDNET_Q": str(q), "FRAUDNET_QFFL_L": str(L)})
        t0 = time.perf_counter()
        simulate(ARM, seed)
        print(f"  {name} seed {seed}: {time.perf_counter() - t0:.0f}s", flush=True)
    return json.loads(path.read_text(encoding="utf-8"))


def tune() -> dict:
    """For each q, pick L by pooled VALIDATION F1 (summed bank counts, D-031); ties -> mean bank val PR-AUC."""
    t = CFG["tune"]
    rows = []
    for q in t["q_grid"]:
        for L in t["L_grid"]:
            r = run_one(f"tune_q{q}_L{L}", t["seed"], "qfedavg", q, L)
            last = [x for x in r["per_bank_per_round"] if x["round"] == r["num_rounds"]]
            rows.append({"q": q, "L": L, "val_f1_pooled": round(r["threshold_val_stats"]["val_f1"], 4),
                         "mean_bank_val_pr_auc": round(float(np.mean([x["val_pr_auc"] for x in last])), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "tuning_validation.csv", index=False)
    chosen = {}
    for q, g in df.groupby("q"):
        best = g.sort_values(["val_f1_pooled", "mean_bank_val_pr_auc"], ascending=False).iloc[0]
        chosen[float(q)] = float(best["L"])
    (ROOT / "chosen_L.json").write_text(json.dumps({str(k): v for k, v in chosen.items()}, indent=2))
    print("Tuning (validation only):\n" + df.to_string(index=False) + f"\nChosen L per q: {chosen}", flush=True)
    return chosen


def configs(chosen: dict) -> list[dict]:
    out = [{"name": "FedAvg", "strategy": "fedavg"},
           {"name": "q-FedAvg q=0", "strategy": "qfedavg", "q": 0.0, "L": float(CFG["final"]["q0_L"])}]
    out += [{"name": f"q-FedAvg q={q:g}", "strategy": "qfedavg", "q": q, "L": L} for q, L in sorted(chosen.items())]
    return out


def collect(cfgs) -> pd.DataFrame:
    rows = []
    for c in cfgs:
        tag = "fedavg" if c["strategy"] == "fedavg" else f"q{c['q']:g}_L{c['L']:g}"
        for seed in CFG["final"]["seeds"]:
            r = run_one(f"final_{tag}", seed, c["strategy"], c.get("q"), c.get("L"))
            f1 = {b: r["local_test"][b]["f1"] for b in BANKS}
            row = {"config": c["name"], "seed": seed, "global_f1": r["global_test"]["f1"],
                   "global_pr_auc": r["global_test"]["pr_auc"],
                   "worst_bank_f1": min(f1.values()), "bank_f1_std": float(np.std(list(f1.values()), ddof=0))}
            for b in BANKS:
                lt = r["local_test"][b]
                row[f"{b}_f1"] = lt["f1"]
                row[f"{b}_recall"] = lt["recall"]
                row[f"{b}_pr_auc"] = lt["pr_auc"]
                row[f"{b}_caught"] = f"{int(lt['tp'])}/{int(lt['n_fraud'])}"
            rows.append(row)
    return pd.DataFrame(rows)


def report(long: pd.DataFrame, cfgs) -> None:
    order = [c["name"] for c in cfgs]
    metrics = ["global_f1", "global_pr_auc", "worst_bank_f1", "bank_f1_std"] + [f"{b}_f1" for b in BANKS]
    pm = lambda s: f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"
    summary = long.groupby("config", sort=False)[metrics].agg(pm).reindex(order).reset_index()
    caught = long.groupby("config", sort=False)[[f"{b}_caught" for b in BANKS]].agg(" ".join).reindex(order).reset_index()
    summary.to_csv(ROOT / "summary_table.csv", index=False)
    caught.to_csv(ROOT / "frauds_caught_per_seed.csv", index=False)

    ref = long[long["config"] == "FedAvg"].set_index("seed")
    comp = []
    for name in order[1:]:
        cur = long[long["config"] == name].set_index("seed").loc[ref.index]
        for m in metrics:
            res = paired(cur[m].to_numpy(), ref[m].to_numpy(), CFG["confidence"])
            if m == "bank_f1_std":   # LOWER spread = fairer, so 'higher' would be bad: say it plainly
                res["verdict"] = {"higher": "less fair (spread up)", "lower": "fairer (spread down)"}.get(res["verdict"], res["verdict"])
            comp.append({"comparison": f"{name} - FedAvg", "metric": m, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                          for k, v in res.items()}})
    comp = pd.DataFrame(comp)
    comp.to_csv(ROOT / "paired_vs_fedavg.csv", index=False)
    long.to_csv(ROOT / "long_results.csv", index=False)
    plot(long, order)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nE1 SUMMARY (mean ± std over seeds; per-bank F1 on each bank's OWN local test)\n" + summary.to_string(index=False))
        print("\nFrauds caught per seed on local tests\n" + caught.to_string(index=False))
        print("\nPAIRED vs FedAvg (95% CI)\n" + comp.to_string(index=False))


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)


def plot(long, order):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF, gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    _style(ax)
    w = 0.8 / len(order)
    x = np.arange(len(BANKS))
    for i, name in enumerate(order):
        g = long[long["config"] == name]
        means = [g[f"{b}_f1"].mean() for b in BANKS]
        sds = [g[f"{b}_f1"].std(ddof=1) for b in BANKS]
        ax.bar(x + (i - (len(order) - 1) / 2) * w, means, w * 0.9, yerr=sds, color=COLORS[i], label=name,
               error_kw={"ecolor": MUTED, "capsize": 2, "elinewidth": 1})
    ax.set_xticks(x, [f"Bank {b[-1].upper()}" for b in BANKS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraud F1 on the bank's own local test", color=MUTED)
    ax.set_title("Per-bank fraud F1 (mean ± std, 5 seeds)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(order), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax = axes[1]
    _style(ax)
    ax.grid(color=GRID, linewidth=0.8)
    for i, name in enumerate(order):
        g = long[long["config"] == name]
        ax.errorbar(g["global_f1"].mean(), g["worst_bank_f1"].mean(), xerr=g["global_f1"].std(ddof=1),
                    yerr=g["worst_bank_f1"].std(ddof=1), fmt="o", ms=8, color=COLORS[i], ecolor=MUTED, capsize=2)
        ax.annotate(name, (g["global_f1"].mean(), g["worst_bank_f1"].mean()), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=INK)
    ax.set_xlabel("global test fraud F1", color=MUTED)
    ax.set_ylabel("worst bank's local F1", color=MUTED)
    ax.set_title("Fairness vs overall performance", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "e1_per_bank_f1.png", dpi=150)
    plt.close(fig)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        chosen = {float(k): v for k, v in json.loads((ROOT / "chosen_L.json").read_text()).items()}
    else:
        chosen = tune()
    cfgs = configs(chosen)
    report(collect(cfgs), cfgs)


if __name__ == "__main__":
    main()
