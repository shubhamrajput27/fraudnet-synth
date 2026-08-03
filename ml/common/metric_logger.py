"""Local metric logging in the document shape CLAUDE.md D8 assigns to MongoDB collections
`runs`, `round_metrics`, `client_metrics` — the Phase 5 API will read/write these same shapes
against a real MongoDB, but Phase 4 has no gateway/DB yet, so it writes them as local JSON/JSONL
files under experiments/results/<run_id>/ (gitignored, regenerable). Every document carries
`run_id` and `arm`, per D8.
"""
import json
import time
import uuid
from pathlib import Path

from ml.common.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "experiments" / "results"


class RunLogger:
    def __init__(self, arm: str, seed: int, config: dict, run_id: str | None = None):
        """`run_id`: pass a pre-generated id (e.g. from orchestrator/run_manager.py) so a caller
        can know the id — and start polling experiments/results/<run_id>/ — before this run
        finishes. Defaults to auto-generating one, as before."""
        self.run_id = run_id or f"{arm}_{uuid.uuid4().hex[:8]}"
        self.arm = arm
        self.run_dir = RESULTS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        run_doc = {
            "run_id": self.run_id,
            "arm": arm,
            "seed": seed,
            "config": config,
            "created_at": time.time(),
        }
        (self.run_dir / "run.json").write_text(json.dumps(run_doc, indent=2))

    def log_round_metric(self, round_num: int | None, metrics: dict) -> None:
        doc = {"run_id": self.run_id, "arm": self.arm, "round": round_num, "metrics": metrics}
        self._append_jsonl(self.run_dir / "round_metrics.jsonl", doc)

    def log_client_metric(self, round_num: int | None, bank: str, metrics: dict) -> None:
        doc = {"run_id": self.run_id, "arm": self.arm, "round": round_num, "bank": bank, "metrics": metrics}
        self._append_jsonl(self.run_dir / "client_metrics.jsonl", doc)

    def finalize(self, final_metrics: dict) -> None:
        (self.run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    @staticmethod
    def _append_jsonl(path: Path, doc: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(doc) + "\n")
