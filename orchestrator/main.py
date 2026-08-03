"""FastAPI orchestration service (CLAUDE.md Tier 3): unifies the augmentation and federated-
training layers behind internal HTTP endpoints called by gateway/ (Tier 2). Internal-only — no
auth here; the gateway is the public-facing, authenticated boundary (D9).

Run with (from repo root, so both `ml` and `orchestrator` resolve as packages):
    ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000
"""
from fastapi import FastAPI, HTTPException

from orchestrator.run_manager import get_run_status, start_run
from orchestrator.schemas import RunRequest, RunStatusResponse

app = FastAPI(title="FraudNet-Synth Orchestrator", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs", status_code=202)
def create_run(request: RunRequest) -> dict:
    run_id = start_run(
        arm=request.arm,
        seed=request.seed,
        epochs=request.epochs,
        num_rounds=request.num_rounds,
        local_epochs=request.local_epochs,
    )
    return {"run_id": run_id, "arm": request.arm, "status": "running"}


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def read_run(run_id: str) -> RunStatusResponse:
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'")
    return RunStatusResponse(**status)
