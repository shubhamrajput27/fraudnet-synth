"""Step 2 entry point: global split + non-IID partition + statistics.

Run from the project root:
    python -m ml.data.prepare

Writes:
    data/processed/global_{train,val,test}.csv
    data/clients/bank_x/{train,val,test}.csv      (each bank's PRIVATE shard)
    results/partition/shard_stats.csv, partition_summary.json, manifest.json, shard_stats.png
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.data.load import LABEL_COLUMN, PROJECT_ROOT, load_config, load_raw
from ml.data.partition import local_splits, partition_by_quota
from ml.data.split import global_split

# Same palette and styling conventions as the Step 1 EDA charts.
BAR_COLOR = "#2a78d6"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _stats(name: str, part: str, df: pd.DataFrame) -> dict:
    fraud = int(df[LABEL_COLUMN].sum())
    return {"set": name, "part": part, "rows": len(df), "fraud": fraud,
            "genuine": len(df) - fraud, "fraud_pct": round(100 * fraud / len(df), 4)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(df: pd.DataFrame, path: Path, manifest: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    manifest[str(path.relative_to(PROJECT_ROOT).as_posix())] = _sha256(path)


def _plot(stats: pd.DataFrame, global_pct: float, banks: dict, out: Path):
    shard = stats[stats["part"] == "shard"].set_index("set")
    labels = [f"{b.replace('bank_', 'Bank ').title()}\n({banks[b]['mode'].title()})" for b in shard.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [("fraud", "Fraud rows per bank (whole shard)", "{:,.0f}"),
              ("fraud_pct", "Fraud % per bank", "{:.3f}%")]
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        vals = shard[col].to_numpy()
        bars = ax.bar(labels, vals, color=BAR_COLOR, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v),
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylim(0, vals.max() * 1.18)
    # Reference line: fraud % of the global train pool, which an IID split would give every bank.
    axes[1].axhline(global_pct, color=INK_MUTED, linestyle="--", linewidth=1)
    axes[1].text(3.45, global_pct, f"global train {global_pct:.3f}%", color=INK_MUTED,
                 fontsize=8, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    seed = load_config("data")["seed"]
    cfg = load_config("partition")
    processed = PROJECT_ROOT / cfg["processed_dir"]
    clients = PROJECT_ROOT / cfg["clients_dir"]
    results = PROJECT_ROOT / cfg["results_dir"]
    results.mkdir(parents=True, exist_ok=True)

    manifest, rows = {}, []

    # 1) Global hold-out FIRST: the test set is fixed before any bank gets data.
    g_train, g_val, g_test, info = global_split(load_raw(), cfg, seed)
    for part, df in (("train", g_train), ("val", g_val), ("test", g_test)):
        _save(df, processed / f"global_{part}.csv", manifest)
        rows.append(_stats("global", part, df))

    # 2) Partition ONLY the global train pool into the four banks.
    shards = partition_by_quota(g_train, cfg["banks"], seed)
    # 3) Each bank makes its own local train/val/test split.
    splits = local_splits(shards, cfg, seed)
    for bank, shard in shards.items():
        rows.append(_stats(bank, "shard", shard))
        for part, df in splits[bank].items():
            _save(df, clients / bank / f"{part}.csv", manifest)
            rows.append(_stats(bank, part, df))

    stats = pd.DataFrame(rows)
    stats.to_csv(results / "shard_stats.csv", index=False)
    global_train_pct = stats.query("set == 'global' and part == 'train'")["fraud_pct"].item()
    _plot(stats, global_train_pct, cfg["banks"], results / "shard_stats.png")

    with open(results / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(results / "partition_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, **info, "config": cfg, "stats": rows}, f, indent=2)

    print(f"Rows before dedup: {info['rows_before_dedup']:,} | after: {info['rows_after_dedup']:,}")
    print(stats.to_string(index=False))
    combined = hashlib.sha256("".join(manifest[k] for k in sorted(manifest)).encode()).hexdigest()
    print(f"\nFiles written: {len(manifest)} | combined SHA-256 of all CSVs: {combined}")


if __name__ == "__main__":
    main()
