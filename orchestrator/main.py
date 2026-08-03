"""FastAPI orchestration service (CLAUDE.md Tier 3): unifies the augmentation and federated-
training layers behind internal HTTP endpoints called by gateway/ (Tier 2). Internal-only — no
auth here; the gateway is the public-facing, authenticated boundary (D9).

Run with (from repo root, so both `ml` and `orchestrator` resolve as packages):
    ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000
"""
from fastapi import FastAPI, HTTPException

from orchestrator.predict import PredictionError, get_predict_manifest, predict
from orchestrator.run_manager import get_run_status, start_run
from orchestrator.schemas import PredictRequest, PredictResponse, RunRequest, RunStatusResponse

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


@app.get("/predict/manifest/{run_id}")
def read_predict_manifest(run_id: str) -> dict:
    manifest = get_predict_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No saved model artifacts for run_id '{run_id}'")
    return manifest


@app.post("/predict", response_model=PredictResponse)
def create_prediction(request: PredictRequest) -> PredictResponse:
    try:
        probability, label = predict(request.run_id, request.bank, request.features)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return PredictResponse(run_id=request.run_id, bank=request.bank, probability=probability, prediction=label)
