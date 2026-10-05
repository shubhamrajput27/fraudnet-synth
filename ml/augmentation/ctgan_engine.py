"""Augment Mode: a CTGAN trained INSIDE one data-rich bank, on that bank's own fraud rows only.

Privacy: `run_bank()` is given one bank folder. It reads only <bank>/train.csv and
writes only into that same folder (synthesizer + candidates). Only summary
statistics and plots go to results/.

Run from the project root (all banks with mode: augment):
    python -m ml.augmentation.ctgan_engine
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config

REAL_COLOR, SYNTH_COLOR = "#2a78d6", "#1baf7a"  # validated palette slots 1 (blue) and 3 (aqua)
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_bank_fraud(bank_dir: Path) -> pd.DataFrame:
    """The bank's own TRAINING fraud rows (never its val/test), without the constant label."""
    train = pd.read_csv(bank_dir / "train.csv")
    return train[train[LABEL_COLUMN] == 1][FEATURE_COLUMNS].reset_index(drop=True)


def fit_ctgan(real: pd.DataFrame, syn_cfg: dict, seed: int) -> tuple[CTGANSynthesizer, float]:
    # infer_keys=None: every column is a feature. Otherwise SDV might treat the
    # unique-valued Time column as an ID and refuse to model it.
    metadata = Metadata.detect_from_dataframe(real, table_name="fraud", infer_keys=None)
    synth = CTGANSynthesizer(metadata, **syn_cfg)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CTGAN's networks and noise come from torch
    t0 = time.perf_counter()
    synth.fit(real)
    return synth, time.perf_counter() - t0


def compare(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Quick real-vs-synthetic summary. The formal quality gate is Step 6 (SDMetrics etc.)."""
    per_col = {}
    for c in FEATURE_COLUMNS:
        ks = ks_2samp(real[c], synth[c])
        per_col[c] = {"real_mean": round(float(real[c].mean()), 3), "synth_mean": round(float(synth[c].mean()), 3),
                      "real_std": round(float(real[c].std()), 3), "synth_std": round(float(synth[c].std()), 3),
                      "ks_stat": round(float(ks.statistic), 3)}
    corr_gap = (real[FEATURE_COLUMNS].corr() - synth[FEATURE_COLUMNS].corr()).abs().to_numpy()
    off_diag = corr_gap[~np.eye(len(FEATURE_COLUMNS), dtype=bool)]
    ks_vals = [v["ks_stat"] for v in per_col.values()]
    # enforce_min_max_values clamps out-of-range samples ONTO the real min/max, which piles
    # synthetic values up at the edges. Measure how often that happens.
    at_edge = pd.DataFrame(
        np.isclose(synth[FEATURE_COLUMNS], real[FEATURE_COLUMNS].min().values, atol=1e-6)
        | np.isclose(synth[FEATURE_COLUMNS], real[FEATURE_COLUMNS].max().values, atol=1e-6),
        columns=FEATURE_COLUMNS)
    edge_by_col = at_edge.mean().sort_values(ascending=False)
    return {
        "mean_ks_stat": round(float(np.mean(ks_vals)), 3),
        "max_ks_stat": round(float(np.max(ks_vals)), 3),
        "mean_abs_corr_diff": round(float(off_diag.mean()), 3),
        "exact_copies_of_real_rows": int(synth.merge(real, how="inner").shape[0]),
        "duplicate_synthetic_rows": int(synth.duplicated().sum()),
        "edge_clamped": {
            "pct_cells_at_real_min_or_max": round(100 * float(at_edge.to_numpy().mean()), 1),
            "pct_rows_with_any_edge_value": round(100 * float(at_edge.any(axis=1).mean()), 1),
            "worst_columns_pct": {c: round(100 * float(v), 1) for c, v in edge_by_col.head(5).items()},
        },
        "per_column": per_col,
    }


def plot_losses(losses: pd.DataFrame, bank: str, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    # Separate panels (not twin axes): the two losses are opponents with different scales.
    for ax, col in zip(axes, ("Generator Loss", "Discriminator Loss")):
        _style(ax)
        vals = losses[col].astype(float)
        ax.plot(losses["Epoch"], vals, color=REAL_COLOR, linewidth=0.6, alpha=0.35)
        ax.plot(losses["Epoch"], vals.rolling(50, min_periods=1).mean(), color=REAL_COLOR, linewidth=2)
        ax.set_title(f"{bank}: {col.lower()} (line = 50-epoch rolling mean)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Epoch", color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distributions(real: pd.DataFrame, synth: pd.DataFrame, cols, bank: str, out: Path):
    fig = plt.figure(figsize=(11, 6.2), facecolor=SURFACE)
    for i, c in enumerate(cols):
        ax = fig.add_subplot(2, 3, i + 1)
        _style(ax)
        r, s = real[c], synth[c]
        if c == "Amount":  # same log view as Step 1
            r, s = np.log10(r + 1), np.log10(s + 1)
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        bins = np.linspace(lo, hi, 30)
        for vals, color, label in ((r, REAL_COLOR, f"Real fraud (n={len(real)})"),
                                   (s, SYNTH_COLOR, f"CTGAN (n={len(synth)})")):
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.3, color=color)
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=label)
        ax.set_title("log10(Amount+1)" if c == "Amount" else c, loc="left", color=INK, fontsize=10)
        ax.set_yticks([])
        if i == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"{bank}: real training fraud vs CTGAN candidates (density)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_bank(bank: str, cfg: dict, seed: int) -> dict:
    bank_dir = PROJECT_ROOT / load_config("partition")["clients_dir"] / bank
    results_dir = PROJECT_ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    real = load_bank_fraud(bank_dir)
    print(f"[{bank}] training CTGAN on {len(real)} real fraud rows "
          f"({cfg['synthesizer']['epochs']} epochs, CPU)...", flush=True)
    synth_model, fit_s = fit_ctgan(real, cfg["synthesizer"], seed)

    t0 = time.perf_counter()
    candidates = synth_model.sample(num_rows=cfg["n_candidates"])
    sample_s = time.perf_counter() - t0

    # Everything derived from real rows stays INSIDE the bank folder.
    synth_model.save(bank_dir / "ctgan_synthesizer.pkl")
    out = candidates[FEATURE_COLUMNS].copy()
    out[LABEL_COLUMN] = 1
    out.to_csv(bank_dir / "synthetic_candidates.csv", index=False)

    losses = synth_model.get_loss_values()
    plot_losses(losses, bank, results_dir / f"{bank}_losses.png")
    plot_distributions(real, candidates, cfg["compare_columns"], bank, results_dir / f"{bank}_real_vs_synthetic.png")

    last = losses.tail(100)
    summary = {
        "bank": bank, "seed": seed, "n_real_fraud": len(real), "n_candidates": len(candidates),
        "fit_seconds": round(fit_s, 1), "sample_seconds": round(sample_s, 2),
        "final_100_epochs_mean_loss": {"generator": round(float(last["Generator Loss"].astype(float).mean()), 3),
                                       "discriminator": round(float(last["Discriminator Loss"].astype(float).mean()), 3)},
        **compare(real, candidates),
    }
    with open(results_dir / f"{bank}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["synthesizer"], **summary}, f, indent=2)
    return summary


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", nargs="*", help="default: every bank with mode: augment")
    args = parser.parse_args()

    seed = load_config("data")["seed"]
    cfg = load_config("ctgan")
    banks_cfg = load_config("partition")["banks"]
    banks = args.banks or [b for b, v in banks_cfg.items() if v["mode"] == "augment"]

    rows = []
    for i, bank in enumerate(banks):
        s = run_bank(bank, cfg, seed + i)
        rows.append(s)
        print(f"[{bank}] fit {s['fit_seconds']}s | sampled {s['n_candidates']} rows in {s['sample_seconds']}s | "
              f"mean KS {s['mean_ks_stat']} (max {s['max_ks_stat']}) | mean |corr diff| {s['mean_abs_corr_diff']} | "
              f"exact copies of real rows {s['exact_copies_of_real_rows']} | duplicate synthetic rows {s['duplicate_synthetic_rows']}")
        e = s["edge_clamped"]
        print(f"[{bank}] edge-clamped: {e['pct_cells_at_real_min_or_max']}% of cells, "
              f"{e['pct_rows_with_any_edge_value']}% of rows | worst: {e['worst_columns_pct']}")

    print("\nPer-column KS statistic (0 = identical distributions, 1 = completely different):")
    table = pd.DataFrame({r["bank"]: {c: v["ks_stat"] for c, v in r["per_column"].items()} for r in rows})
    print(table.T.to_string())
    print("\nMean real vs synthetic for plotted columns:")
    for r in rows:
        for c in cfg["compare_columns"]:
            v = r["per_column"][c]
            print(f"  {r['bank']} {c:6s} real {v['real_mean']:>8} +/- {v['real_std']:<8} synth {v['synth_mean']:>8} +/- {v['synth_std']}")


if __name__ == "__main__":
    main()
