"""Viva-day pre-flight check (Step 14, decision D-057). Needs NO internet.

    python -m demo.preflight

Checks everything the offline demo depends on and prints PASS / WARN / FAIL with a fix for each failure.
Exit code 0 = ready (warnings allowed), 1 = something must be fixed first.
"""
import os
import socket
import sys

from dotenv import dotenv_values

from ml.data.load import PROJECT_ROOT, load_config

# Offline flags BEFORE any Hugging Face import: a hidden download can never hang the demo.
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
os.environ.update(OFFLINE_ENV)

BANKS = ["bank_a", "bank_b", "bank_c", "bank_d"]
results = []


def check(name, ok, detail="", fix="", warn=False):
    level = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(level)
    print(f"[{level}] {name}{(': ' + detail) if detail else ''}")
    if not ok and fix:
        print(f"       fix: {fix}")


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def run():
    env = dotenv_values(PROJECT_ROOT / ".env")
    R = PROJECT_ROOT / "results"
    print(f"FraudNet-Synth pre-flight check  ({PROJECT_ROOT})\n")

    # 1. Secrets the services need (values never printed).
    need = ["ORCHESTRATOR_INTERNAL_KEY", "GATEWAY_JWT_SECRET"] + [f"GATEWAY_PW_{u}" for u in
            ("ADMIN", "BANK_A", "BANK_B", "BANK_C", "BANK_D", "GUIDE")]
    missing = [k for k in need if not env.get(k)]
    check(".env secrets", not missing, f"{len(need) - len(missing)}/{len(need)} set",
          f"add {', '.join(missing)} to .env (see .env.example)")

    # 2. Data each bank and the server need.
    clients = PROJECT_ROOT / "data" / "clients"
    miss = [f"{b}/{f}" for b in BANKS for f in ("train.csv", "val.csv", "test.csv", "synthetic_validated.csv")
            if not (clients / b / f).exists()]
    check("bank data + validated synthetic rows", not miss, f"{4 * 4 - len(miss)}/16 files",
          "run python -m ml.data.prepare, then Steps 4-6 (or restore data/clients)")
    gt = PROJECT_ROOT / "data" / "processed" / "global_test.csv"
    check("global test set", gt.exists(), fix="run python -m ml.data.prepare")

    # 3. Cached generation, so replay needs no API calls.
    for b in ("bank_c", "bank_d"):
        n = len(list((clients / b / "llm_cache").glob("batch_*.json")))
        check(f"LLM response cache {b}", n >= 25, f"{n} batches (replay makes 0 Groq calls)",
              "the cache is not in Git: copy data/clients/<bank>/llm_cache from the original laptop")
    for b in ("bank_a", "bank_b"):
        check(f"CTGAN synthesizer {b}", (clients / b / "ctgan_synthesizer.pkl").exists(),
              fix="run python -m ml.augmentation.ctgan_engine")

    # 4. The /predict model.
    mp = PROJECT_ROOT / load_config("orchestrator")["predict_model"]
    if mp.exists():
        import torch
        ck = torch.load(mp, map_location="cpu", weights_only=False)
        check("predict model (federated Arm 3)", True, f"{mp.name}, threshold logit {ck['threshold_logit']}")
    else:
        check("predict model (federated Arm 3)", False, "missing",
              "python -m services.orchestrator.tasks prepare_model results/api_jobs/manual_prepare_model (~1 min)")

    # 5. Embedding model in the local cache (validation layer), loaded strictly offline.
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(load_config("validation")["embedding"]["model"], device="cpu", local_files_only=True)
        check("embedding model cached (offline load)", True, load_config("validation")["embedding"]["model"])
    except Exception as e:  # noqa: BLE001 - report any failure plainly
        check("embedding model cached (offline load)", False, type(e).__name__,
              "connect to the internet once and run python -m ml.validation.gate", warn=True)

    # 6. Results the dashboard shows.
    for p in (R / "six_arms" / "six_arm_global_table.csv", R / "validation" / "validation_summary.csv",
              PROJECT_ROOT / "docs" / "RESULTS_ANNEXURE.md"):
        check(f"result file {p.relative_to(PROJECT_ROOT).as_posix()}", p.exists(),
              fix="run python -m ml.experiments.run_all_arms (or the step that creates it)")

    # 7. Node side: gateway packages and built dashboard.
    check("gateway packages installed", (PROJECT_ROOT / "services/gateway/node_modules/express").exists(),
          fix="cd services/gateway; npm install")
    dist = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    src_newest = max(p.stat().st_mtime for p in (PROJECT_ROOT / "frontend" / "src").rglob("*") if p.is_file())
    check("dashboard built", dist.exists(), fix="cd frontend; npm install; npm run build")
    if dist.exists():
        check("dashboard build up to date", dist.stat().st_mtime >= src_newest, "built after the latest source edit",
              "cd frontend; npm run build", warn=True)

    # 8. MongoDB with the demo users.
    uri = env.get("GATEWAY_MONGO_URI") or "mongodb://localhost:27017/fraudnet_gateway"
    try:
        from pymongo import MongoClient
        c = MongoClient(uri, serverSelectionTimeoutMS=2000)
        db = c.get_default_database()
        users = sorted(u["username"] for u in db.users.find({}, {"username": 1}))
        check("MongoDB reachable", True, f"database {db.name}")
        check("demo users seeded", len(users) == 6, ", ".join(users) or "none", "cd services/gateway; npm run seed")
    except Exception as e:  # noqa: BLE001
        check("MongoDB reachable", False, type(e).__name__,
              "start the 'MongoDB' Windows service (services.msc) or run mongod")

    # 9. Ports (free, or already used by our own running services).
    for port, what in ((8100, "FastAPI"), (8200, "gateway")):
        check(f"port {port} ({what}) free", port_free(port), "",
              f"something already listens on {port}: close the old demo window, or it's already running",
              warn=True)

    n_fail, n_warn = results.count("FAIL"), results.count("WARN")
    print(f"\n{results.count('PASS')} passed, {n_warn} warnings, {n_fail} failed -> "
          + ("READY for the demo" if not n_fail else "NOT READY: fix the FAIL lines above"))
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
