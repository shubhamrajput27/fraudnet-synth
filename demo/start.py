"""One-command offline demo (Step 14, decision D-057).

    python -m demo.start             # pre-flight, start FastAPI + gateway, open http://127.0.0.1:8200
    python -m demo.start --smoke     # same, then a quick automatic tour via the gateway, then stop

Both services run with Hugging Face offline flags; synthetic data comes from saved files and caches, so
nothing needs the internet (no Groq calls are possible from the demo). Press Ctrl+C to stop everything.
Service logs: demo/logs/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

from dotenv import dotenv_values

from demo.preflight import OFFLINE_ENV, run as preflight
from ml.data.load import PROJECT_ROOT

URL = "http://127.0.0.1:8200"
LOGS = PROJECT_ROOT / "demo" / "logs"


def http(method, path, body=None, token=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def wait_healthy(timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            h = http("GET", "/api/health")
            if all(v == "ok" for v in h.values()):
                return h
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(1)
    raise SystemExit("services did not become healthy; see demo/logs/")


def smoke():
    """A short tour through the gateway, the same way the dashboard talks to it."""
    pw = dotenv_values(PROJECT_ROOT / ".env")
    t = http("POST", "/api/auth/login", {"username": "guide", "password": pw["GATEWAY_PW_GUIDE"]})["token"]
    banks = http("GET", "/api/banks", token=t)
    res = http("GET", "/api/results/six-arms", token=t)
    fed = next(r for r in res["global_table"] if r["arm"] == 3)
    ex = http("GET", "/api/predict/examples", token=t)
    pred = http("POST", "/api/predict", ex["fraud"]["transaction"], token=t)
    print(f"  login as guide: OK | banks: {', '.join(b['bank'] for b in banks)}")
    print(f"  six-arm results: Arm 3 (federated, real) F1 {fed['f1']}, PR-AUC {fed['pr_auc']}")
    print(f"  predict fraud example (true label {ex['fraud']['true_label']}): is_fraud={pred['is_fraud']}, "
          f"logit {pred['score_logit']:.4f} vs threshold {pred['threshold_logit']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run a quick automatic tour, then stop")
    args = ap.parse_args()
    if not preflight():
        raise SystemExit(1)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **OFFLINE_ENV, "PYTHONIOENCODING": "utf-8"}
    node = "node.exe" if sys.platform == "win32" else "node"
    procs = []
    try:
        print("\nStarting FastAPI orchestrator (127.0.0.1:8100) and gateway + dashboard (127.0.0.1:8200)...")
        procs.append(subprocess.Popen([sys.executable, "-W", "ignore", "-m", "services.orchestrator"], cwd=PROJECT_ROOT,
                                      env=env, stdout=open(LOGS / "orchestrator.log", "w"), stderr=subprocess.STDOUT))
        procs.append(subprocess.Popen([node, "src/server.js"], cwd=PROJECT_ROOT / "services" / "gateway", env=env,
                                      stdout=open(LOGS / "gateway.log", "w"), stderr=subprocess.STDOUT))
        print(f"Health: {wait_healthy()}")
        if args.smoke:
            smoke()
            print("Smoke tour passed.")
            return
        print(f"\nDashboard: {URL}   (log in with your account from .env; Ctrl+C here stops the demo)")
        webbrowser.open(URL)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A service stopped unexpectedly; see demo/logs/.")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Demo stopped.")


if __name__ == "__main__":
    main()
