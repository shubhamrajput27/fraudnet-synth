"""Step 9: run ALL six arms over several seeds from one command, then build the report tables.

    python -m ml.experiments.run_all_arms              # runs what's missing, then aggregates
    python -m ml.experiments.run_all_arms --report-only

Per seed it writes results/six_arms/seed<S>/:
    isolated_centralized.json        Arms 1, 2, 5, 6  (ml.baselines.arms.run)
    arm3_federated_real_seed<S>.json Arm 3            (Flower simulation, 30 rounds)
    arm4_federated_aug_seed<S>.json  Arm 4
Existing files are reused, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import time

from ml.data.load import PROJECT_ROOT, load_config

FED_ARMS = ["arm3_federated_real", "arm4_federated_aug"]


def run_seed(seed: int, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / "isolated_centralized.json"
    if not p.exists():
        from ml.baselines.arms import run
        t0 = time.perf_counter()
        res = run(seed)
        res["total_seconds"] = round(time.perf_counter() - t0, 1)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[seed {seed}] arms 1,2,5,6 done in {res['total_seconds']}s", flush=True)
    else:
        print(f"[seed {seed}] arms 1,2,5,6: cached", flush=True)

    from ml.federated.run import simulate
    os.environ["FRAUDNET_FL_OUT_DIR"] = str(out_dir)   # keep Step 8's own result files untouched
    for arm in FED_ARMS:
        if (out_dir / f"{arm}_seed{seed}.json").exists():
            print(f"[seed {seed}] {arm}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        simulate(arm, seed)
        print(f"[seed {seed}] {arm} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    cfg = load_config("experiments")["six_arms"]
    root = PROJECT_ROOT / cfg["results_dir"]
    if not args.report_only:
        t0 = time.perf_counter()
        for seed in cfg["seeds"]:
            run_seed(seed, root / f"seed{seed}")
        print(f"All seeds done in {time.perf_counter() - t0:.0f}s", flush=True)
    from ml.evaluation.six_arm_report import build_report
    build_report()


if __name__ == "__main__":
    main()
