"""CLI entrypoint for all six experimental arms (CLAUDE.md six-arm grid).

Usage:
    python -m ml.run_experiment --arm federated_augmented --seed 42
    python -m ml.run_experiment --arm all --seed 42

Exit criterion (PLAN.md Phase 4): all six arms runnable end-to-end from CLI. All six share
ml/common/{model,data,train,metrics}.py — no per-arm training logic is duplicated here.
"""
import argparse
import json

from ml.baselines.centralized import run_centralized
from ml.baselines.isolated import run_isolated
from ml.common.config import CONFIG
from ml.common.seeding import set_seed
from ml.federated.run_federated import run_federated

ARMS = [
    "isolated_real", "isolated_augmented",
    "federated_real", "federated_augmented",
    "centralized_real", "centralized_augmented",
]


def run_arm(
    arm: str,
    seed: int,
    epochs: int,
    num_rounds: int,
    local_epochs: int,
    run_id: str | None = None,
    save_artifacts: bool = False,
) -> dict:
    set_seed(seed)
    print(f"\n=== {arm} (seed={seed}) ===")
    if arm == "isolated_real":
        return run_isolated(augmented=False, seed=seed, epochs=epochs, run_id=run_id, save_artifacts=save_artifacts)
    if arm == "isolated_augmented":
        return run_isolated(augmented=True, seed=seed, epochs=epochs, run_id=run_id, save_artifacts=save_artifacts)
    if arm == "centralized_real":
        return run_centralized(augmented=False, seed=seed, epochs=epochs, run_id=run_id, save_artifacts=save_artifacts)
    if arm == "centralized_augmented":
        return run_centralized(augmented=True, seed=seed, epochs=epochs, run_id=run_id, save_artifacts=save_artifacts)
    if arm == "federated_real":
        return run_federated(augmented=False, seed=seed, num_rounds=num_rounds, local_epochs=local_epochs, run_id=run_id, save_artifacts=save_artifacts)
    if arm == "federated_augmented":
        return run_federated(augmented=True, seed=seed, num_rounds=num_rounds, local_epochs=local_epochs, run_id=run_id, save_artifacts=save_artifacts)
    raise ValueError(f"Unknown arm '{arm}', expected one of {ARMS + ['all']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--epochs", type=int, default=20, help="Local training epochs, isolated/centralized")
    parser.add_argument("--num-rounds", type=int, default=10, help="FedAvg rounds, federated arms")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per FedAvg round")
    args = parser.parse_args()

    arms_to_run = ARMS if args.arm == "all" else [args.arm]
    results = {}
    for arm in arms_to_run:
        results[arm] = run_arm(arm, args.seed, args.epochs, args.num_rounds, args.local_epochs)

    if len(arms_to_run) > 1:
        print("\n=== Six-arm summary (fraud-class metrics on the global holdout) ===")
        print(f"{'Arm':<22}{'F1':>8}{'Precision':>12}{'Recall':>10}{'AUC':>8}")
        for arm, result in results.items():
            m = result["final_metrics"]
            print(f"{arm:<22}{m['f1']:>8.4f}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['auc']:>8.4f}")

        summary_path = CONFIG.processed_dir / "phase4_six_arm_summary.json"
        summary_path.write_text(json.dumps({a: r["final_metrics"] for a, r in results.items()}, indent=2), encoding="utf-8")
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
