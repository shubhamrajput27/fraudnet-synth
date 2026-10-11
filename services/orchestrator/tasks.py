"""Long-running work, executed in a SEPARATE Python process per job (started by jobs.py), so heavy
CPU training never blocks the API.

    python -m services.orchestrator.tasks <run|generate|prepare_model|selftest> <job_dir>

Reads <job_dir>/params.json, appends live events to <job_dir>/events.jsonl, and writes
<job_dir>/result.json (aggregate metrics / checksums only, never data rows).
"""
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

from ml.data.load import PROJECT_ROOT, load_config

FED = {3: "arm3_federated_real", 4: "arm4_federated_aug"}
NON_FED = {1: "arm1_isolated_real", 2: "arm2_isolated_aug", 5: "arm5_central_real", 6: "arm6_central_aug"}


def emit(job_dir: Path, event: dict) -> None:
    with open(job_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------- run ---

def task_run(job_dir: Path, params: dict) -> dict:
    """D-043: any subset of the six arms over the requested seeds. Results go to the JOB folder,
    never over the Step 7-9 results."""
    arms = list(range(1, 7)) if params["arms"] == "all" else sorted(params["arms"])
    seeds = params["seeds"]
    os.environ["FRAUDNET_EVENTS_FILE"] = str(job_dir / "events.jsonl")   # federated rounds stream live
    summary = {}
    for seed in seeds:
        seed_dir = job_dir / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        wanted = [NON_FED[a] for a in arms if a in NON_FED]
        if wanted:
            from ml.baselines.arms import run
            emit(job_dir, {"type": "status", "message": f"seed {seed}: training {', '.join(wanted)}"})
            res = run(seed, arms=wanted)
            (seed_dir / "isolated_centralized.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            for arm, per in res["arms"].items():
                for model, r in per.items():
                    m = r["eval"]["global_test"]
                    emit(job_dir, {"type": "model_done", "seed": seed, "arm": arm, "model": model,
                                   **{k: m[k] for k in ("precision", "recall", "f1", "pr_auc")}})
                    summary[f"seed{seed}/{arm}/{model}"] = {k: m[k] for k in ("precision", "recall", "f1", "pr_auc")}
        for a in arms:
            if a in FED:
                from ml.federated.run import simulate
                emit(job_dir, {"type": "status", "message": f"seed {seed}: federated simulation {FED[a]}"})
                os.environ["FRAUDNET_FL_OUT_DIR"] = str(seed_dir)
                simulate(FED[a], seed)
                r = json.loads((seed_dir / f"{FED[a]}_seed{seed}.json").read_text(encoding="utf-8"))
                m = r["global_test"]
                emit(job_dir, {"type": "model_done", "seed": seed, "arm": FED[a], "model": "federated",
                               **{k: m[k] for k in ("precision", "recall", "f1", "pr_auc")}})
                summary[f"seed{seed}/{FED[a]}/federated"] = {k: m[k] for k in ("precision", "recall", "f1", "pr_auc")}
    if arms == list(range(1, 7)) and len(seeds) >= 2:
        from ml.evaluation.six_arm_report import build_report
        build_report(root=job_dir, seeds=seeds)
        summary["report"] = "six_arm_global_table.csv, paired_comparisons.csv and charts in the job folder"
    return {"arms": arms, "seeds": seeds, "global_test": summary}


# -------------------------------------------------------------- generate ---

def _next_version(bank_dir: Path, sub: str) -> str:
    gen = bank_dir / sub
    n = 1
    while (gen / f"v{n}").exists():
        n += 1
    return f"v{n}"


def task_generate(job_dir: Path, params: dict) -> dict:
    """D-041. replay: CTGAN re-trained with its original seed, LLM rebuilt from the response cache
    (0 API calls), then the same gate, all into generations/replay_<job>/ inside the bank folder, then
    compared by SHA-256 with the files the experiments used. regenerate: new data into generations/vN/."""
    warnings.filterwarnings("ignore")
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    from ml.augmentation.ctgan_engine import run_bank as ctgan_run
    from ml.augmentation.llm_engine import generate_for_bank as llm_run
    from ml.validation.gate import validate_bank

    load_dotenv(PROJECT_ROOT / ".env")
    seed = load_config("data")["seed"]
    banks_cfg = load_config("partition")["banks"]
    order = list(banks_cfg)
    augment = [b for b in order if banks_cfg[b]["mode"] == "augment"]
    sub = load_config("orchestrator")["generations_subdir"]
    vcfg = load_config("validation")
    try:
        st = SentenceTransformer(vcfg["embedding"]["model"], device="cpu", local_files_only=True)
    except OSError:
        st = SentenceTransformer(vcfg["embedding"]["model"], device="cpu")

    out = {}
    for bank in params["banks"]:
        bank_dir = PROJECT_ROOT / load_config("partition")["clients_dir"] / bank
        mode = banks_cfg[bank]["mode"]
        label = f"replay_{job_dir.name}" if params["mode"] == "replay" else _next_version(bank_dir, sub)
        art = bank_dir / sub / label                      # bank-private outputs stay in the bank folder
        report = job_dir / bank                           # aggregate reports only
        emit(job_dir, {"type": "status", "message": f"{bank}: {params['mode']} ({mode} mode) -> {sub}/{label}"})
        t0 = time.perf_counter()
        if mode == "augment":
            ctgan_run(bank, load_config("ctgan"), seed + augment.index(bank), art_dir=art, report_dir=report)
        else:
            cache = bank_dir / "llm_cache" if params["mode"] == "replay" else art / "llm_cache"
            llm_run(bank, load_config("llm"), seed + order.index(bank), art_dir=art, cache_dir=cache, report_dir=report)
        gate = validate_bank(bank, mode, vcfg, st, work_dir=art)
        (report / "validation_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        entry = {"mode": mode, "folder": f"data/clients/{bank}/{sub}/{label}", "seconds": round(time.perf_counter() - t0, 1),
                 "candidates": gate["n_candidates"], "validated": gate["n_validated"], "pass_rate_pct": gate["pass_rate_pct"]}
        if params["mode"] == "replay":
            entry["identical_to_experiment_files"] = {
                f: _sha(art / f) == _sha(bank_dir / f) for f in ("synthetic_candidates.csv", "synthetic_validated.csv")}
        out[bank] = entry
        emit(job_dir, {"type": "bank_done", "bank": bank, **entry})
    return {"mode": params["mode"], "banks": out}


# ------------------------------------------------------------ predictor ---

def task_prepare_model(job_dir: Path, params: dict) -> dict:
    """D-040: re-run the federated Arm 3 simulation (seed 42, 30 rounds; bit-reproducible) and save
    its final global model + threshold for /predict. Checks it matches the Step 9 result."""
    from ml.federated.run import simulate
    model_path = PROJECT_ROOT / load_config("orchestrator")["predict_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["FRAUDNET_SAVE_MODEL"] = str(model_path)
    os.environ["FRAUDNET_FL_OUT_DIR"] = str(job_dir)
    os.environ["FRAUDNET_EVENTS_FILE"] = str(job_dir / "events.jsonl")
    simulate("arm3_federated_real", 42)
    new = json.loads((job_dir / "arm3_federated_real_seed42.json").read_text(encoding="utf-8"))
    ref_path = PROJECT_ROOT / "results/six_arms/seed42/arm3_federated_real_seed42.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    meta = {"arm": "arm3_federated_real", "seed": 42, "num_rounds": new["num_rounds"],
            "threshold_logit": new["threshold_logit"], "global_test": new["global_test"],
            "matches_step9_seed42": bool(ref) and ref["global_test"] == new["global_test"],
            "model_file": str(model_path.relative_to(PROJECT_ROOT).as_posix()), "model_sha256": _sha(model_path)}
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def task_selftest(job_dir: Path, params: dict) -> dict:
    """A 2-second job that exercises the job system and the event stream (used by the API tests)."""
    for i in range(1, 4):
        emit(job_dir, {"type": "round", "round": i, "global_test_pr_auc": 0.5 + i / 10})
        time.sleep(0.5)
    return {"ok": True}


TASKS = {"run": task_run, "generate": task_generate, "prepare_model": task_prepare_model, "selftest": task_selftest}


def main():
    task, job_dir = sys.argv[1], Path(sys.argv[2])
    params = json.loads((job_dir / "params.json").read_text(encoding="utf-8"))
    result = TASKS[task](job_dir, params)
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
