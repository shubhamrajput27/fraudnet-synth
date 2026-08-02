"""CLI entrypoint: ingest -> global holdout -> non-IID partition -> shard statistics report.

Usage:
    python -m ml.partition.run_partition --seed 42

Exit criterion (PLAN.md Phase 1): reproducible partition script; shard statistics documented.
Shard sizes printed here are a *proposal* — CLAUDE.md D5 requires explicit approval before
they're treated as final anywhere downstream.
"""
import argparse

from ml.common.config import CONFIG, REPO_ROOT
from ml.common.seeding import set_seed
from ml.partition.holdout import global_holdout_split
from ml.partition.ingest import load_raw
from ml.partition.partition import PartitionConfig, partition_non_iid
from ml.partition.stats import build_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--test-size", type=float, default=0.2, help="Global holdout fraction")
    parser.add_argument("--fraud-alpha", type=float, default=0.3, help="Dirichlet concentration for fraud-row skew")
    parser.add_argument("--poor-fraud-cap", type=int, default=15, help="Max fraud rows for Banks C/D")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading raw dataset from {CONFIG.raw_data_path} ...")
    df = load_raw(CONFIG.raw_data_path)
    print(f"Loaded {len(df)} rows, {int(df['Class'].sum())} fraud ({100 * df['Class'].mean():.4f}%).")

    train_pool, holdout = global_holdout_split(df, test_size=args.test_size, seed=args.seed)
    print(f"Global holdout: {len(holdout)} rows ({int(holdout['Class'].sum())} fraud). Train pool: {len(train_pool)} rows.")

    pcfg = PartitionConfig(seed=args.seed, fraud_alpha=args.fraud_alpha, poor_fraud_cap=args.poor_fraud_cap)
    shards = partition_non_iid(train_pool, pcfg)

    CONFIG.shards_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(CONFIG.holdout_path, index=False)
    for bank, shard_df in shards.items():
        shard_df.to_csv(CONFIG.shards_dir / f"bank_{bank}.csv", index=False)

    report = build_report(shards, holdout, seed=args.seed)
    write_report(
        report,
        json_path=CONFIG.processed_dir / "shard_stats.json",
        markdown_path=REPO_ROOT / "docs" / "phase1_shard_stats.md",
    )

    print("\nShard statistics (PROPOSED - pending D5 approval):")
    for bank, s in report["banks"].items():
        print(f"  Bank {bank}: {s['n_rows']} rows, {s['n_fraud']} fraud ({s['fraud_rate_pct']}%)")
    print(f"  Global holdout: {report['global_holdout']['n_rows']} rows, "
          f"{report['global_holdout']['n_fraud']} fraud ({report['global_holdout']['fraud_rate_pct']}%)")
    print(f"\nWrote shards to {CONFIG.shards_dir}, holdout to {CONFIG.holdout_path}, "
          f"and report to {CONFIG.processed_dir / 'shard_stats.json'} + docs/phase1_shard_stats.md")


if __name__ == "__main__":
    main()
