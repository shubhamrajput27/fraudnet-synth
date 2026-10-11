"""FraudNet-Synth orchestration API (Tier 3, FastAPI).

Start:  python -m services.orchestrator        (listens on 127.0.0.1:8100; interactive docs at /docs)

Every endpoint except /health requires the header  X-Internal-Key: <ORCHESTRATOR_INTERNAL_KEY>
(only the Node gateway knows it; users authenticate at the gateway, Step 11).
"""
import asyncio
import json
import secrets
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config
from services.orchestrator.jobs import JobManager
from services.orchestrator.schemas import ARM_NAMES, GenerationRequest, Job, Prediction, RunRequest, Transaction
from services.orchestrator.settings import AUDIT_DIR, BANK_MODES, BANKS, CFG, JOBS_DIR, MODEL_PATH, internal_key

KEY = internal_key()
app = FastAPI(title="FraudNet-Synth Orchestrator", version="1.0",
              description="Tier 3: synthetic generation, validation, six-arm runs, live FL rounds, prediction.")
jobs = JobManager(JOBS_DIR)


def require_key(x_internal_key: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_internal_key, KEY):
        raise HTTPException(401, "missing or wrong X-Internal-Key")


AUTH = [Depends(require_key)]


def _read_json(path):
    if not path.exists():
        raise HTTPException(404, f"{path.relative_to(PROJECT_ROOT).as_posix()} not found (run that step first)")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ status ---

@app.get("/health")
def health():
    """Open (no key): is the service up, is a model loaded, how many jobs are queued/running."""
    counts = {}
    for j in jobs.list():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {"status": "ok", "predict_model_present": MODEL_PATH.exists(), "jobs": counts,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/banks", dependencies=AUTH)
def banks():
    """Per bank: generation mode, shard size, validated synthetic pool (aggregate counts only)."""
    stats = pd.read_csv(PROJECT_ROOT / "results/partition/shard_stats.csv")
    out = []
    for b in BANKS:
        shard = stats[(stats.set == b) & (stats.part == "shard")].iloc[0]
        train = stats[(stats.set == b) & (stats.part == "train")].iloc[0]
        rep_path = PROJECT_ROOT / f"results/validation/{b}_report.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
        out.append({"bank": b, "mode": {"augment": "Augment (CTGAN)", "schema": "Schema (LLM)"}[BANK_MODES[b]],
                    "rows": int(shard.rows), "fraud": int(shard.fraud), "fraud_pct": float(shard.fraud_pct),
                    "train_fraud": int(train.fraud), "synthetic_candidates": rep.get("n_candidates"),
                    "synthetic_validated": rep.get("n_validated"), "pass_rate_pct": rep.get("pass_rate_pct"),
                    "fidelity_validated": (rep.get("fidelity_validated") or {}).get("overall")})
    return out


@app.get("/validation", dependencies=AUTH)
def validation_summary():
    return pd.read_csv(PROJECT_ROOT / "results/validation/validation_summary.csv").to_dict(orient="records")


@app.get("/validation/{bank}", dependencies=AUTH)
def validation_report(bank: str):
    if bank not in BANKS:
        raise HTTPException(404, "unknown bank")
    return _read_json(PROJECT_ROOT / f"results/validation/{bank}_report.json")


# ----------------------------------------------------------------- results ---

@app.get("/results/six-arms", dependencies=AUTH)
def six_arm_results():
    """The Step 9 multi-seed tables (mean ± std, paired 95% CIs)."""
    root = PROJECT_ROOT / load_config("experiments")["six_arms"]["results_dir"]
    read = lambda n: pd.read_csv(root / n).to_dict(orient="records")  # noqa: E731
    return {"arm_names": ARM_NAMES, "seeds": load_config("experiments")["six_arms"]["seeds"],
            "global_table": read("six_arm_global_table.csv"), "local_table": read("local_test_table.csv"),
            "paired_comparisons": read("paired_comparisons.csv")}


@app.get("/results/six-arms/convergence", dependencies=AUTH)
def six_arm_convergence():
    """Per-round global PR-AUC of the federated arms for every seed (for the dashboard chart)."""
    cfg = load_config("experiments")["six_arms"]
    root = PROJECT_ROOT / cfg["results_dir"]
    out = {}
    for arm in ("arm3_federated_real", "arm4_federated_aug"):
        out[arm] = {s: [r["global_test_pr_auc"] for r in
                        _read_json(root / f"seed{s}" / f"{arm}_seed{s}.json")["global_test_per_round"]]
                    for s in cfg["seeds"]}
    return out


# -------------------------------------------------------------------- jobs ---

@app.post("/runs", response_model=Job, status_code=202, dependencies=AUTH)
def start_run(req: RunRequest):
    """Start arms 1-6 (or 'all') over seeds. Returns at once with a job id; watch it via
    GET /jobs/{id} or the WebSocket /ws/jobs/{id}."""
    return jobs.submit("run", req.model_dump())


@app.post("/generation", response_model=Job, status_code=202, dependencies=AUTH)
def start_generation(req: GenerationRequest):
    """'replay' (default) or 'regenerate'. Never overwrites the files the experiments used (D-041)."""
    return jobs.submit("generate", req.model_dump())


@app.post("/admin/prepare-model", response_model=Job, status_code=202, dependencies=AUTH)
def prepare_model():
    return jobs.submit("prepare_model", {})


@app.post("/admin/selftest", response_model=Job, status_code=202, dependencies=AUTH)
def selftest():
    return jobs.submit("selftest", {})


@app.get("/jobs", response_model=list[Job], dependencies=AUTH)
def list_jobs():
    return jobs.list()


@app.get("/jobs/{jid}", response_model=Job, dependencies=AUTH)
def get_job(jid: str):
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.delete("/jobs/{jid}", response_model=Job, dependencies=AUTH)
def cancel_job(jid: str):
    job = jobs.cancel(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/jobs/{jid}/log", dependencies=AUTH)
def job_log(jid: str, tail: int = Query(200, ge=1, le=5000)):
    p = JOBS_DIR / jid / "log.txt"
    if jobs.get(jid) is None:
        raise HTTPException(404, "unknown job")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
    return {"lines": lines[-tail:]}


@app.get("/jobs/{jid}/events", dependencies=AUTH)
def job_events(jid: str):
    if jobs.get(jid) is None:
        raise HTTPException(404, "unknown job")
    p = JOBS_DIR / jid / "events.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


@app.websocket("/ws/jobs/{jid}")
async def job_stream(ws: WebSocket, jid: str):
    """Live stream: every new event line (e.g. one per FL round) plus status changes, then 'end'.
    Browsers can't set headers on WebSockets, so the key may also come as ?key=... (the gateway
    adds it; users never see it)."""
    key = ws.headers.get("x-internal-key") or ws.query_params.get("key", "")
    if not secrets.compare_digest(key, KEY):
        await ws.close(code=4401)
        return
    if jobs.get(jid) is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    events = JOBS_DIR / jid / "events.jsonl"
    sent, last_status = 0, None
    try:
        while True:
            job = jobs.get(jid)
            if job["status"] != last_status:
                last_status = job["status"]
                await ws.send_json({"type": "job_status", "status": last_status})
            if events.exists():
                lines = events.read_text(encoding="utf-8").splitlines()
                for line in lines[sent:]:
                    await ws.send_json(json.loads(line))
                sent = len(lines)
            if last_status in ("succeeded", "failed", "cancelled", "interrupted"):
                await ws.send_json({"type": "end", "status": last_status, "result": job["result"]})
                await ws.close()
                return
            await asyncio.sleep(CFG["ws_poll_seconds"])
    except WebSocketDisconnect:
        return


# ----------------------------------------------------------------- predict ---

@app.post("/predict", response_model=Prediction, dependencies=AUTH)
def predict_one(tx: Transaction):
    from services.orchestrator.predictor import predict
    try:
        return predict([tx.model_dump()])[0]
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.get("/predict/examples", dependencies=AUTH)
def predict_examples():
    """One fraud and one genuine transaction from the GLOBAL TEST set, which is held by the
    experimenter (server side), not by any bank, to try /predict with. Disclosed design choice."""
    gt = pd.read_csv(PROJECT_ROOT / load_config("partition")["processed_dir"] / "global_test.csv")
    pick = {"fraud": gt[gt[LABEL_COLUMN] == 1].iloc[0], "genuine": gt[gt[LABEL_COLUMN] == 0].iloc[0]}
    return {k: {"transaction": {c: float(r[c]) for c in FEATURE_COLUMNS}, "true_label": int(r[LABEL_COLUMN])}
            for k, r in pick.items()}


# ------------------------------------------------------------------ export ---

@app.get("/banks/{bank}/synthetic/export", dependencies=AUTH)
def export_synthetic(bank: str, x_bank_scope: str = Header(default=""), x_user: str = Header(default="unknown")):
    """D-045: a bank's VALIDATED SYNTHETIC rows only, and only to that bank's own operator. The
    gateway sets X-Bank-Scope from the logged-in user's role; any mismatch is refused. Real rows are
    never exported. Every attempt is written to results/api_audit/exports.log."""
    if bank not in BANKS:
        raise HTTPException(404, "unknown bank")
    allowed = secrets.compare_digest(x_bank_scope, bank)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = PROJECT_ROOT / load_config("partition")["clients_dir"] / bank / "synthetic_validated.csv"
    with open(AUDIT_DIR / "exports.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "user": x_user,
                            "scope": x_bank_scope, "bank": bank, "allowed": allowed}) + "\n")
    if not allowed:
        raise HTTPException(403, f"only {bank}'s own operator may export {bank}'s synthetic data")
    if not path.exists():
        raise HTTPException(404, "no validated synthetic data for this bank")
    return FileResponse(path, media_type="text/csv", filename=f"{bank}_synthetic_validated.csv")
