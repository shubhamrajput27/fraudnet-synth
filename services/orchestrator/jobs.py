"""Background job manager: a queue plus one worker thread that runs each job as a separate process.

Why a separate process (not a thread)? Training uses all CPU cores for minutes; in a separate
process it can't freeze the API, and a crash can't take the API down. Why one at a time?
Two trainings on one laptop would just slow each other down (D-042).
Job state is saved to results/api_jobs/<id>/job.json, so it survives an API restart.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ml.data.load import PROJECT_ROOT


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.dir = jobs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self._recover()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- persistence ----
    def _path(self, jid: str) -> Path:
        return self.dir / jid / "job.json"

    def get(self, jid: str) -> dict | None:
        p = self._path(jid)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _save(self, job: dict) -> None:
        tmp = self._path(job["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(tmp, self._path(job["id"]))     # atomic: readers never see half a file

    def _update(self, jid: str, **fields) -> dict:
        with self.lock:
            job = self.get(jid)
            job.update(fields)
            self._save(job)
            return job

    def list(self) -> list[dict]:
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in self.dir.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def _recover(self) -> None:
        """Jobs left 'queued'/'running' by a previous API process can't be resumed: mark them."""
        for job in self.list():
            if job["status"] in ("queued", "running"):
                job.update(status="interrupted", finished=_now())
                self._save(job)

    # ---- API ----
    def submit(self, jtype: str, params: dict) -> dict:
        jid = f"{datetime.now():%Y%m%d-%H%M%S}-{jtype}-{uuid.uuid4().hex[:6]}"
        (self.dir / jid).mkdir()
        (self.dir / jid / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        job = {"id": jid, "type": jtype, "params": params, "status": "queued", "created": _now(),
               "started": None, "finished": None, "return_code": None, "result": None}
        with self.lock:
            self._save(job)
        self.q.put(jid)
        return job

    def cancel(self, jid: str) -> dict | None:
        job = self.get(jid)
        if job is None:
            return None
        if job["status"] == "queued":
            return self._update(jid, status="cancelled", finished=_now())
        if job["status"] == "running" and jid in self.procs:
            self.procs[jid].terminate()
            return self._update(jid, status="cancelled")
        return job

    # ---- worker ----
    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            job = self.get(jid)
            if job is None or job["status"] != "queued":
                continue
            jdir = self.dir / jid
            self._update(jid, status="running", started=_now())
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            with open(jdir / "log.txt", "w", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "services.orchestrator.tasks",
                                         job["type"], str(jdir)], cwd=str(PROJECT_ROOT), env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                self.procs[jid] = proc
                code = proc.wait()
                self.procs.pop(jid, None)
            result_file = jdir / "result.json"
            result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
            status = self.get(jid)["status"]
            if status != "cancelled":
                status = "succeeded" if code == 0 and result is not None else "failed"
            self._update(jid, status=status, finished=_now(), return_code=code, result=result)
