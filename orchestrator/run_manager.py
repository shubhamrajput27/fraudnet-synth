"""Background execution + progress polling for ML runs (CLAUDE.md Phase 5: "triggers ML runs and
streams round-by-round progress"). Runs execute in a background thread so POST /runs returns
immediately with a run_id; GET /runs/{run_id} reads whatever ml/common/metric_logger.py has
written to experiments/results/<run_id>/ so far — this is the poll-based "streaming" this phase
provides. Push-based streaming (WebSocket) to a browser is Tier 1's job (Phase 6), not this one.
"""
import json
import threading
import traceback
import uuid
from pathlib import Path

from ml.common.metric_logger import RESULTS_DIR
from ml.run_experiment import run_arm

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def start_run(arm: str, seed: int, epochs: int, num_rounds: int, local_epochs: int) -> str:
    run_id = f"{arm}_{uuid.uuid4().hex[:8]}"
    with _lock:
        _jobs[run_id] = {"status": "running", "arm": arm, "error": None}

    def _worker() -> None:
        try:
            run_arm(arm, seed, epochs, num_rounds, local_epochs, run_id=run_id, save_artifacts=True)
            with _lock:
                _jobs[run_id]["status"] = "complete"
        except Exception:
            with _lock:
                _jobs[run_id]["status"] = "failed"
                _jobs[run_id]["error"] = traceback.format_exc()

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


def get_run_status(run_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return None
        job = dict(job)

    run_dir = RESULTS_DIR / run_id
    final_path = run_dir / "final_metrics.json"

    return {
        "run_id": run_id,
        "arm": job["arm"],
        "status": job["status"],
        "round_metrics": _read_jsonl(run_dir / "round_metrics.jsonl"),
        "client_metrics": _read_jsonl(run_dir / "client_metrics.jsonl"),
        "final_metrics": json.loads(final_path.read_text()) if final_path.exists() else None,
        "error": job["error"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
