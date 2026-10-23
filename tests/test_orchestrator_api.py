"""Step 10: API tests with FastAPI's TestClient (no real network, own temp jobs/audit folders)."""
import os

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


def test_predict_examples_are_random_rows_with_correct_labels(client):
    fixed = client.get("/predict/examples?random=false", headers=H(client)).json()
    again = client.get("/predict/examples?random=false", headers=H(client)).json()
    assert fixed == again                                  # random=false: always the first rows
    fraud_rows = set()
    for _ in range(8):                                     # default: a new random row each call
        ex = client.get("/predict/examples", headers=H(client)).json()
        assert ex["fraud"]["true_label"] == 1 and ex["genuine"]["true_label"] == 0
        assert 0 <= ex["fraud"]["row"] < ex["fraud"]["rows_in_test_set"]
        fraud_rows.add(ex["fraud"]["row"])
    assert len(fraud_rows) > 1                             # 70 caught frauds: 8 identical picks is ~1e-13


def test_predict_examples_match_the_model_verdict(client):
    """'fraud'/'genuine' come from rows the model gets right, 'missed_fraud' from frauds it misses,
    so the demo button and the verdict agree (the bug: a random fraud was 'Looks genuine' ~1 in 4)."""
    from services.orchestrator.settings import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("predict model not prepared")
    for _ in range(5):
        ex = client.get("/predict/examples", headers=H(client)).json()
        for kind, label, flagged in (("fraud", 1, True), ("genuine", 0, False), ("missed_fraud", 1, False)):
            assert ex[kind]["true_label"] == label and ex[kind]["filtered_by_model"]
            assert ex[kind]["candidates"] <= ex[kind]["class_total"]
            p = client.post("/predict", json=ex[kind]["transaction"], headers=H(client)).json()
            assert p["is_fraud"] is flagged, kind
    assert ex["fraud"]["candidates"] + ex["missed_fraud"]["candidates"] == ex["fraud"]["class_total"]


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
