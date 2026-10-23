"""Step 9: aggregate the multi-seed six-arm results into report tables, paired statistics and charts.

    python -m ml.experiments.run_all_arms --report-only

Outputs in results/six_arms/:
    long_results.csv            every (seed, arm, model, evaluation set) row with all metrics
    six_arm_global_table.csv    mean +/- std on the GLOBAL test set (headline: P, R, F1, PR-AUC)
    local_test_table.csv        each bank's local test, all six arms
    paired_comparisons.csv      paired differences with t-based 95% CI and "improved in k of N seeds"
    six_arm_f1_prauc.png, paired_deltas.png, convergence_seeds.png, summary.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ml.data.load import PROJECT_ROOT, load_config
from ml.evaluation.plot_style import INK, INK_MUTED, GRID, SURFACE, style_axes

METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"]
HEADLINE = ["precision", "recall", "f1", "pr_auc"]
BANKS = list(load_config("partition")["banks"])
ARM_LABEL = {1: "Isolated, real", 2: "Isolated, augmented", 3: "Federated, real",
             4: "Federated, augmented", 5: "Centralized, real", 6: "Centralized, augmented"}
REAL_COLOR, AUG_COLOR = "#2a78d6", "#1baf7a"


# ------------------------------------------------------------------ load ---

def _row(seed, arm, model, eval_set, m):
    return {"seed": seed, "arm": arm, "model": model, "eval": eval_set, **{k: float(m[k]) for k in METRICS},
            **{k: int(m[k]) for k in ("tp", "fp", "fn")}}


def load_long(root, seeds=None) -> tuple[pd.DataFrame, dict]:
    seeds = seeds or load_config("experiments")["six_arms"]["seeds"]
    rows, curves = [], {"arm3": [], "arm4": []}
    for s in seeds:
        d = root / f"seed{s}"
        ic = json.loads((d / "isolated_centralized.json").read_text(encoding="utf-8"))
        for arm_no, key in ((1, "arm1_isolated_real"), (2, "arm2_isolated_aug")):
            for b, r in ic["arms"][key].items():
                rows.append(_row(s, arm_no, b, "global", r["eval"]["global_test"]))
                rows.append(_row(s, arm_no, b, b, r["eval"][f"{b}_test"]))
        for arm_no, key in ((5, "arm5_central_real"), (6, "arm6_central_aug")):
            r = ic["arms"][key]["pooled"]
            rows.append(_row(s, arm_no, "centralized", "global", r["eval"]["global_test"]))
            for b in BANKS:
                rows.append(_row(s, arm_no, "centralized", b, r["eval"][f"{b}_test"]))
        for arm_no, key in ((3, "arm3_federated_real"), (4, "arm4_federated_aug")):
            r = json.loads((d / f"{key}_seed{s}.json").read_text(encoding="utf-8"))
            rows.append(_row(s, arm_no, "federated", "global", r["global_test"]))
            for b, m in r["local_test"].items():
                rows.append(_row(s, arm_no, "federated", b, m))
            curves[f"arm{arm_no}"].append([h["global_test_pr_auc"] for h in r["global_test_per_round"]])
    return pd.DataFrame(rows), curves


# ----------------------------------------------------------- statistics ---

def paired(a: np.ndarray, b: np.ndarray, conf: float) -> dict:
    """Paired difference a - b over seeds: mean, std, t-based CI, and how many seeds improved."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean - half, mean + half
    verdict = "higher" if lo > 0 else ("lower" if hi < 0 else "no clear difference")
    return {"mean_diff": mean, "std_diff": sd, "ci_low": lo, "ci_high": hi, "n": n,
            "improved_in": int((d > 0).sum()), "verdict": verdict}


def comparisons(long: pd.DataFrame, conf: float) -> pd.DataFrame:
    def series(arm, model, ev, metric):
        x = long[(long.arm == arm) & (long.model == model) & (long["eval"] == ev)].sort_values("seed")
        return x[metric].to_numpy()

    specs = []
    for b in BANKS:  # augmentation effect, per bank and setting
        specs += [("augmentation", "Arm 2 - Arm 1", b, (2, b), (1, b), "global"),
                  ("augmentation", "Arm 2 - Arm 1", b, (2, b), (1, b), b),
                  ("augmentation", "Arm 4 - Arm 3", b, (4, "federated"), (3, "federated"), b),
                  ("augmentation", "Arm 6 - Arm 5", b, (6, "centralized"), (5, "centralized"), b),
                  ("federation", "Arm 3 - Arm 1", b, (3, "federated"), (1, b), "global"),
                  ("federation", "Arm 3 - Arm 1", b, (3, "federated"), (1, b), b)]
    specs += [("augmentation", "Arm 4 - Arm 3", "all banks", (4, "federated"), (3, "federated"), "global"),
              ("augmentation", "Arm 6 - Arm 5", "all banks", (6, "centralized"), (5, "centralized"), "global"),
              ("fed_vs_central", "Arm 3 - Arm 5", "all banks", (3, "federated"), (5, "centralized"), "global"),
              ("fed_vs_central", "Arm 4 - Arm 6", "all banks", (4, "federated"), (6, "centralized"), "global")]
    rows = []
    for family, comp, who, (a_arm, a_m), (b_arm, b_m), ev in specs:
        for metric in HEADLINE:
            r = paired(series(a_arm, a_m, ev, metric), series(b_arm, b_m, ev, metric), conf)
            rows.append({"family": family, "comparison": comp, "bank": who,
                         "eval_set": "global test" if ev == "global" else f"{ev} local test",
                         "metric": metric, **r})
    return pd.DataFrame(rows)


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def global_table(long: pd.DataFrame) -> pd.DataFrame:
    g = long[long["eval"] == "global"]
    rows = []
    for arm in range(1, 7):
        sub = g[g.arm == arm]
        models = BANKS if arm in (1, 2) else sorted(sub.model.unique())
        for model in models:
            x = sub[sub.model == model]
            rows.append({"arm": arm, "setting": ARM_LABEL[arm], "model": model,
                         **{k: fmt(x[k].mean(), x[k].std(ddof=1)) for k in METRICS}})
        if arm in (1, 2):  # the "average bank" view: mean over banks within a seed, then over seeds
            per_seed = sub.groupby("seed")[METRICS].mean()
            rows.append({"arm": arm, "setting": ARM_LABEL[arm], "model": "average of 4 banks",
                         **{k: fmt(per_seed[k].mean(), per_seed[k].std(ddof=1)) for k in METRICS}})
    return pd.DataFrame(rows)


def local_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BANKS:
        x = long[long["eval"] == b]
        for arm in range(1, 7):
            model = b if arm in (1, 2) else ("federated" if arm in (3, 4) else "centralized")
            y = x[(x.arm == arm) & (x.model == model)]
            rows.append({"local_test_of": b, "arm": arm, "setting": ARM_LABEL[arm],
                         "f1": fmt(y.f1.mean(), y.f1.std(ddof=1)), "pr_auc": fmt(y.pr_auc.mean(), y.pr_auc.std(ddof=1)),
                         "recall": fmt(y.recall.mean(), y.recall.std(ddof=1)),
                         "frauds_caught_per_seed": "/".join(str(v) for v in y.sort_values("seed").tp)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts ---

def plot_six_arms(long: pd.DataFrame, out):
    g = long[long["eval"] == "global"]
    groups = [("Bank A\nisolated", 1, 2, "bank_a"), ("Bank B\nisolated", 1, 2, "bank_b"),
              ("Bank C\nisolated", 1, 2, "bank_c"), ("Bank D\nisolated", 1, 2, "bank_d"),
              ("Federated\n(FedAvg)", 3, 4, "federated"), ("Centralized\n(upper bound)", 5, 6, "centralized")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    x = np.arange(len(groups))
    for ax, metric, title in ((axes[0], "f1", "Fraud F1, global test (mean ± std, 5 seeds)"),
                              (axes[1], "pr_auc", "PR-AUC, global test (mean ± std, 5 seeds)")):
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for off, idx, color, label in ((-0.2, 1, REAL_COLOR, "Real only"), (0.2, 2, AUG_COLOR, "Real + validated synthetic")):
            means, sds = [], []
            for grp in groups:
                v = g[(g.arm == grp[idx]) & (g.model == grp[3])][metric]
                means.append(v.mean())
                sds.append(v.std(ddof=1))
            bars = ax.bar(x + off, means, 0.38, yerr=sds, capsize=3, color=color, label=label,
                          edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
            for bar, m, sd in zip(bars, means, sds):
                ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.015, f"{m:.2f}", ha="center",
                        fontsize=7.5, color=INK)
        ax.set_xticks(x, [g_[0] for g_ in groups])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_paired(comp: pd.DataFrame, out):
    sel = comp[(comp.metric.isin(["f1", "pr_auc"])) & (
        ((comp.family == "augmentation") & (comp.eval_set == "global test")) |
        (comp.family == "fed_vs_central") |
        ((comp.family == "federation") & (comp.eval_set == "global test")))]
    labels = sel[sel.metric == "f1"].apply(lambda r: f"{r.comparison}  {r.bank}", axis=1).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(labels) + 1.4), facecolor=SURFACE, sharey=True)
    for ax, metric, title in ((axes[0], "f1", "Δ F1 (paired, 95% CI)"), (axes[1], "pr_auc", "Δ PR-AUC (paired, 95% CI)")):
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        s = sel[sel.metric == metric].reset_index(drop=True)
        y = np.arange(len(s))
        colors = [REAL_COLOR if v != "no clear difference" else INK_MUTED for v in s.verdict]
        ax.errorbar(s.mean_diff, y, xerr=[s.mean_diff - s.ci_low, s.ci_high - s.mean_diff], fmt="none",
                    ecolor=INK_MUTED, elinewidth=1.2, capsize=3)
        ax.scatter(s.mean_diff, y, color=colors, s=36, zorder=3)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    fig.text(0.01, 0.005, "Blue dot = 95% CI excludes 0 (a clear difference); grey = no clear difference. Global test set.",
             color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_convergence(curves: dict, central_mean: float, out):
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(color=GRID, linewidth=0.8)
    for key, color, label in (("arm3", REAL_COLOR, "Arm 3: federated, real"), ("arm4", AUG_COLOR, "Arm 4: federated, augmented")):
        c = np.array(curves[key])
        r = np.arange(1, c.shape[1] + 1)
        ax.fill_between(r, c.mean(0) - c.std(0, ddof=1), c.mean(0) + c.std(0, ddof=1), color=color, alpha=0.18)
        ax.plot(r, c.mean(0), color=color, linewidth=2, label=label)
    ax.axhline(central_mean, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(c.shape[1], central_mean, " centralized real (mean)", color=INK_MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("FL round", color=INK_MUTED)
    ax.set_title("Global test PR-AUC per round (mean ± std over 5 seeds)", loc="left", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main ---

def build_report(root=None, seeds=None):
    """Default: the Step 9 results. The API passes its own job folder and seeds."""
    cfg = dict(load_config("experiments")["six_arms"])
    if seeds:
        cfg["seeds"] = list(seeds)
    root = Path(root) if root else PROJECT_ROOT / cfg["results_dir"]
    long, curves = load_long(root, cfg["seeds"])
    long.to_csv(root / "long_results.csv", index=False)
    gt, lt = global_table(long), local_table(long)
    comp = comparisons(long, cfg["confidence"])
    gt.to_csv(root / "six_arm_global_table.csv", index=False)
    lt.to_csv(root / "local_test_table.csv", index=False)
    comp.round(4).to_csv(root / "paired_comparisons.csv", index=False)
    plot_six_arms(long, root / "six_arm_f1_prauc.png")
    plot_paired(comp, root / "paired_deltas.png")
    central = long[(long.arm == 5) & (long["eval"] == "global")].pr_auc.mean()
    plot_convergence(curves, central, root / "convergence_seeds.png")
    summary = {"seeds": cfg["seeds"], "n_rows": len(long),
               "verdict_counts": comp.groupby(["family", "verdict"]).size().to_dict()}
    (root / "summary.json").write_text(json.dumps({k: (str(v) if k == "verdict_counts" else v)
                                                    for k, v in summary.items()}, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 40):
        print("\nSIX-ARM TABLE: global test (95 fraud), mean ± std over seeds", cfg["seeds"])
        print(gt[["arm", "setting", "model"] + HEADLINE + ["roc_auc", "accuracy"]].to_string(index=False))
        print("\nLOCAL TEST SETS (F1 / PR-AUC mean ± std; frauds caught per seed)")
        print(lt.to_string(index=False))
        show = comp[comp.metric.isin(["f1", "pr_auc"])].copy()
        for c in ("mean_diff", "std_diff", "ci_low", "ci_high"):
            show[c] = show[c].map(lambda v: f"{v:+.4f}")
        print("\nPAIRED COMPARISONS (95% CI; 'improved_in' = seeds where the first arm was higher)")
        print(show[["family", "comparison", "bank", "eval_set", "metric", "mean_diff", "ci_low", "ci_high",
                    "improved_in", "verdict"]].to_string(index=False))
