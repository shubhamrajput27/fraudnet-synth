"""Step 10: API tests with FastAPI's TestClient (no real network, own temp jobs/audit folders)."""
import os
import time

import pytest

os.environ.setdefault("ORCHESTRATOR_INTERNAL_KEY", "test-key-0123456789abcdef")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["FRAUDNET_JOBS_DIR"] = str(tmp_path_factory.mktemp("jobs"))
    os.environ["FRAUDNET_AUDIT_DIR"] = str(tmp_path_factory.mktemp("audit"))
    import importlib

    import services.orchestrator.settings as settings
    importlib.reload(settings)
    import services.orchestrator.app as app_mod
    importlib.reload(app_mod)
    from fastapi.testclient import TestClient
    c = TestClient(app_mod.app)
    c.key = app_mod.KEY
    c.audit_dir = settings.AUDIT_DIR
    return c


def H(c, **extra):
    return {"X-Internal-Key": c.key, **extra}


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/banks", "/validation", "/jobs", "/results/six-arms", "/predict/examples"])
def test_everything_else_needs_the_key(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Internal-Key": "wrong-key-wrong-key"}).status_code == 401


def test_banks_returns_aggregates_only(client):
    r = client.get("/banks", headers=H(client))
    assert r.status_code == 200
    banks = r.json()
    assert [b["bank"] for b in banks] == ["bank_a", "bank_b", "bank_c", "bank_d"]
    allowed = {"bank", "mode", "rows", "fraud", "fraud_pct", "train_fraud", "synthetic_candidates",
               "synthetic_validated", "pass_rate_pct", "fidelity_validated"}
    assert all(set(b) == allowed for b in banks)  # counts and scores, no rows


def test_six_arm_results(client):
    r = client.get("/results/six-arms", headers=H(client)).json()
    assert len(r["seeds"]) == 5 and r["global_table"] and r["paired_comparisons"]


def test_run_request_validation(client):
    assert client.post("/runs", json={"arms": [7]}, headers=H(client)).status_code == 422
    assert client.post("/runs", json={"arms": [3], "seeds": []}, headers=H(client)).status_code == 422
    assert client.post("/generation", json={"banks": ["bank_z"]}, headers=H(client)).status_code == 422


def test_predict_rejects_malformed(client):
    assert client.post("/predict", json={"Time": 1, "Amount": -1}, headers=H(client)).status_code == 422


def test_predict_examples_and_prediction(client):
    from services.orchestrator.settings import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("predict model not prepared")
    ex = client.get("/predict/examples", headers=H(client)).json()
    for kind in ("fraud", "genuine"):
        p = client.post("/predict", json=ex[kind]["transaction"], headers=H(client))
        assert p.status_code == 200
        body = p.json()
        assert 0 <= body["fraud_probability"] <= 1 and body["model"]["arm"] == "arm3_federated_real"


def test_export_only_to_own_bank_and_audited(client):
    no_scope = client.get("/banks/bank_b/synthetic/export", headers=H(client))
    other = client.get("/banks/bank_b/synthetic/export", headers=H(client, **{"X-Bank-Scope": "bank_c"}))
    own = client.get("/banks/bank_b/synthetic/export", headers=H(client, **{"X-Bank-Scope": "bank_b"}))
    assert no_scope.status_code == 403 and other.status_code == 403
    assert own.status_code == 200 and own.text.splitlines()[0].endswith("Class")
    log = (client.audit_dir / "exports.log").read_text(encoding="utf-8").splitlines()
    assert len(log) == 3 and '"allowed": true' in log[-1]


def test_selftest_job_and_websocket_stream(client):
    job = client.post("/admin/selftest", headers=H(client))
    assert job.status_code == 202
    jid = job.json()["id"]
    received = []
    with client.websocket_connect(f"/ws/jobs/{jid}", headers={"X-Internal-Key": client.key}) as ws:
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "end":
                break
    rounds = [m for m in received if m["type"] == "round"]
    assert [m["round"] for m in rounds] == [1, 2, 3]
    assert received[-1]["status"] == "succeeded" and received[-1]["result"] == {"ok": True}
    assert client.get(f"/jobs/{jid}", headers=H(client)).json()["status"] == "succeeded"


def test_websocket_rejects_wrong_key(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/jobs/whatever", headers={"X-Internal-Key": "nope"}) as ws:
            ws.receive_json()
