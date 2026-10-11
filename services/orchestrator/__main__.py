"""python -m services.orchestrator  ->  starts the API on 127.0.0.1:8100 (docs at /docs)."""
import uvicorn

from services.orchestrator.settings import CFG

if __name__ == "__main__":
    uvicorn.run("services.orchestrator.app:app", host=CFG["host"], port=CFG["port"], log_level="info")
