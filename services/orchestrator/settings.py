"""Settings for the orchestrator: config file + the internal key shared only with the gateway."""
import os
from pathlib import Path

from dotenv import load_dotenv

from ml.data.load import PROJECT_ROOT, load_config

load_dotenv(PROJECT_ROOT / ".env")
CFG = load_config("orchestrator")
# Overridable so tests use their own folders and never touch the live service's jobs/audit log.
JOBS_DIR = Path(os.environ.get("FRAUDNET_JOBS_DIR", PROJECT_ROOT / CFG["jobs_dir"]))
AUDIT_DIR = Path(os.environ.get("FRAUDNET_AUDIT_DIR", PROJECT_ROOT / CFG["audit_dir"]))
MODEL_PATH = PROJECT_ROOT / CFG["predict_model"]
BANKS = list(load_config("partition")["banks"])
BANK_MODES = {b: v["mode"] for b, v in load_config("partition")["banks"].items()}


def internal_key() -> str:
    """D-044: every request (except /health) must carry this key in X-Internal-Key. Only the
    gateway knows it; users never see it. Refuse to run without one rather than run open."""
    key = os.environ.get("ORCHESTRATOR_INTERNAL_KEY", "")
    if len(key) < 16:
        raise RuntimeError("Set ORCHESTRATOR_INTERNAL_KEY in .env (at least 16 characters).")
    return key
