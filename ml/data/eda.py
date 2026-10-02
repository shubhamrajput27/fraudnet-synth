"""Step 1: Exploratory Data Analysis of the ULB Credit Card Fraud dataset.

Run from the project root:
    python -m ml.data.eda

Writes results/eda/eda_summary.json and five PNG charts. The analysis is purely
descriptive: no modelling choice is fitted here, so looking at the full dataset
does not leak test information into any model.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only backend; no window needed
import matplotlib.pyplot as plt
import numpy as np

from ml.data.load import LABEL_COLUMN, PROJECT_ROOT, V_COLUMNS, load_config, load_raw

# Validated categorical palette (dataviz reference, slots 1-2). Color follows the
# class, never the rank, so the same two colors mean the same thing in every chart.
GENUINE_COLOR = "#2a78d6"  # blue
FRAUD_COLOR = "#eb6834"    # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    """Recessive axes: light grid, no top/right spines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference: how many pooled standard deviations apart two groups are."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled)


def summarize(df) -> dict:
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    n_genuine, n_fraud = int(counts[0]), int(counts[1])
    n = len(df)
    dup_mask = df.duplicated(keep="first")
    genuine, fraud = df[df[LABEL_COLUMN] == 0], df[df[LABEL_COLUMN] == 1]

    effect = {c: cohens_d(genuine[c].to_numpy(), fraud[c].to_numpy()) for c in V_COLUMNS}

    return {
        "n_rows": n,
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "class_counts": {"genuine_0": n_genuine, "fraud_1": n_fraud},
        "fraud_percent": round(100 * n_fraud / n, 4),
        "imbalance_ratio_genuine_per_fraud": round(n_genuine / n_fraud, 1),
        # Accuracy of a "model" that always answers "genuine": the accuracy paradox.
        "always_genuine_accuracy_percent": round(100 * n_genuine / n, 4),
        "duplicate_rows": {
            "total": int(dup_mask.sum()),
            "genuine": int((dup_mask & (df[LABEL_COLUMN] == 0)).sum()),
            "fraud": int((dup_mask & (df[LABEL_COLUMN] == 1)).sum()),
        },
        "time_span_hours": round(float(df["Time"].max() - df["Time"].min()) / 3600, 2),
        "amount_stats_by_class": {
            name: {k: round(float(v), 2) for k, v in part["Amount"].describe().items()}
            for name, part in (("genuine", genuine), ("fraud", fraud))
        },
        "amount_zero_count_by_class": {
            "genuine": int((genuine["Amount"] == 0).sum()),
            "fraud": int((fraud["Amount"] == 0).sum()),
        },
        "v_feature_std_range": [
            round(float(df[V_COLUMNS].std().min()), 3),
            round(float(df[V_COLUMNS].std().max()), 3),
        ],
        "v_feature_mean_abs_max": round(float(df[V_COLUMNS].mean().abs().max()), 6),
        "cohens_d_fraud_vs_genuine": {k: round(v, 3) for k, v in effect.items()},
    }


def plot_class_balance(summary, out: Path):
    g, f = summary["class_counts"]["genuine_0"], summary["class_counts"]["fraud_1"]
    fig = _new_fig(8, 2.8)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    bars = ax.barh(["Fraud (1)", "Genuine (0)"], [f, g], color=[FRAUD_COLOR, GENUINE_COLOR], height=0.55)
    total = g + f
    for bar, val in zip(bars, [f, g]):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({100 * val / total:.3f}%)", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, total * 1.25)
    ax.set_xlabel("Number of transactions", color=INK_MUTED)
    ax.set_title("Class balance: fraud is a sliver of the data", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _class_hist(ax, genuine_vals, fraud_vals, bins):
    """Overlaid density histograms. Density (not counts) so the 492-row class is visible."""
    ax.hist(genuine_vals, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color=GENUINE_COLOR, label="Genuine")
    ax.hist(genuine_vals, bins=bins, density=True, histtype="step", linewidth=2, color=GENUINE_COLOR)
    ax.hist(fraud_vals, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color=FRAUD_COLOR, label="Fraud")
    ax.hist(fraud_vals, bins=bins, density=True, histtype="step", linewidth=2, color=FRAUD_COLOR)


def plot_amount(df, out: Path):
    genuine = df.loc[df[LABEL_COLUMN] == 0, "Amount"]
    fraud = df.loc[df[LABEL_COLUMN] == 1, "Amount"]
    fig = _new_fig(8, 4)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    # Amounts are heavily right-skewed, so plot log10(Amount + 1); +1 keeps zero amounts.
    bins = np.linspace(0, np.log10(df["Amount"].max() + 1), 60)
    _class_hist(ax, np.log10(genuine + 1), np.log10(fraud + 1), bins)
    amounts = [0, 1, 10, 100, 1000, 10000]
    ax.set_xticks(np.log10(np.array(amounts) + 1), [f"{a:,}" for a in amounts])
    ax.set_xlabel("Amount (log scale)", color=INK_MUTED)
    ax.set_ylabel("Density (each class normalized)", color=INK_MUTED)
    ax.set_title("Transaction amount by class", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_time(df, out: Path):
    hours = df["Time"] / 3600
    fig = _new_fig(8, 4)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    bins = np.linspace(0, hours.max(), 49)  # roughly one-hour bins over ~48 hours
    _class_hist(ax, hours[df[LABEL_COLUMN] == 0], hours[df[LABEL_COLUMN] == 1], bins)
    ax.set_xlabel("Hours since first transaction", color=INK_MUTED)
    ax.set_ylabel("Density (each class normalized)", color=INK_MUTED)
    ax.set_title("Transaction time by class", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_separation(summary, out: Path):
    d = summary["cohens_d_fraud_vs_genuine"]
    items = sorted(d.items(), key=lambda kv: abs(kv[1]))
    names = [k for k, _ in items]
    vals = [abs(v) for _, v in items]
    fig = _new_fig(8, 7)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(names, vals, color=GENUINE_COLOR, height=0.7)
    ax.set_xlabel("|Cohen's d|: gap between fraud and genuine means, in standard deviations",
                  color=INK_MUTED)
    ax.set_title("How strongly each V-feature separates fraud from genuine", loc="left",
                 color=INK, fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_v_features(df, summary, out: Path, n_strong=4, n_weak=2):
    d = summary["cohens_d_fraud_vs_genuine"]
    ranked = sorted(d, key=lambda c: abs(d[c]), reverse=True)
    chosen = ranked[:n_strong] + ranked[-n_weak:]  # strongest separators + weakest, for contrast
    fig = _new_fig(11, 6.5)
    for i, col in enumerate(chosen):
        ax = fig.add_subplot(2, 3, i + 1)
        _style_axes(ax)
        g_vals = df.loc[df[LABEL_COLUMN] == 0, col]
        f_vals = df.loc[df[LABEL_COLUMN] == 1, col]
        # Show the central range of both classes; rows outside it are left out of the
        # plot (not clipped, which would pile them into fake spikes at the edges).
        lo = min(g_vals.quantile(0.005), f_vals.quantile(0.01))
        hi = max(g_vals.quantile(0.995), f_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 50)
        _class_hist(ax, g_vals, f_vals, bins)
        kind = "strong" if i < n_strong else "weak"
        ax.set_title(f"{col}  (|d| = {abs(d[col]):.2f}, {kind})", loc="left", color=INK, fontsize=10)
        ax.set_yticks([])
        if i == 0:
            ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle("V-feature distributions by class (density; extreme 1% tails not shown)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return chosen


def main():
    cfg = load_config("data")
    out_dir = PROJECT_ROOT / cfg["eda_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    summary = summarize(df)

    plot_class_balance(summary, out_dir / "class_balance.png")
    plot_amount(df, out_dir / "amount_by_class.png")
    plot_time(df, out_dir / "time_by_class.png")
    plot_separation(summary, out_dir / "v_feature_separation.png")
    summary["v_features_plotted"] = plot_v_features(df, summary, out_dir / "v_features_by_class.png")

    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- console report ----
    cc = summary["class_counts"]
    print(f"Shape: {summary['n_rows']:,} rows x {summary['n_columns']} columns")
    print(f"Dtypes: {summary['dtypes']}")
    print(f"Missing values (all cells): {summary['missing_values_total']}")
    print(f"Genuine (0): {cc['genuine_0']:,} | Fraud (1): {cc['fraud_1']:,}")
    print(f"Fraud percentage: {summary['fraud_percent']}%  "
          f"(1 fraud per {summary['imbalance_ratio_genuine_per_fraud']} genuine)")
    print(f"'Always genuine' accuracy: {summary['always_genuine_accuracy_percent']}%  <- catches 0 frauds")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Time span: {summary['time_span_hours']} hours")
    for cls, s in summary["amount_stats_by_class"].items():
        print(f"Amount [{cls}]: mean={s['mean']} median={s['50%']} max={s['max']}")
    print(f"Zero-amount transactions: {summary['amount_zero_count_by_class']}")
    print(f"V1-V28 std range: {summary['v_feature_std_range']}, "
          f"max |mean|: {summary['v_feature_mean_abs_max']}")
    top = sorted(summary["cohens_d_fraud_vs_genuine"].items(), key=lambda kv: -abs(kv[1]))
    print("Top 6 separating V-features (Cohen's d): " + ", ".join(f"{k}={v}" for k, v in top[:6]))
    print("Weakest 3: " + ", ".join(f"{k}={v}" for k, v in top[-3:]))
    print(f"Saved charts + eda_summary.json to {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
