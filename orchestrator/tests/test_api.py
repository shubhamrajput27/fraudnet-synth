"""Orchestrator API smoke tests (Phase 8 hardening). Doesn't spin up a real training run (slow,
CPU-bound) — covers request validation, 404s, and the /predict error paths, which are exactly the
places a REST API silently does the wrong thing when nobody's watching."""
from fastapi.testclient import TestClient

from orchestrator.main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_run_rejects_unknown_arm():
    res = client.post("/runs", json={"arm": "not_a_real_arm"})
    assert res.status_code == 422  # pydantic Literal validation


def test_create_run_accepts_a_valid_arm_and_returns_a_run_id():
    res = client.post("/runs", json={"arm": "isolated_real", "seed": 42, "epochs": 1})
    assert res.status_code == 202
    body = res.json()
    assert body["arm"] == "isolated_real"
    assert body["status"] == "running"
    assert body["run_id"].startswith("isolated_real_")


def test_read_run_404s_for_unknown_run_id():
    res = client.get("/runs/definitely_not_a_real_run_id")
    assert res.status_code == 404


def test_predict_404s_for_run_with_no_saved_artifacts():
    res = client.post(
        "/predict",
        json={"run_id": "definitely_not_a_real_run_id", "bank": "A", "features": {}},
    )
    assert res.status_code == 404


def test_predict_manifest_404s_for_unknown_run():
    res = client.get("/predict/manifest/definitely_not_a_real_run_id")
    assert res.status_code == 404
