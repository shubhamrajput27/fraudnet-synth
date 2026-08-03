"""Pydantic request/response models for the orchestration API (CLAUDE.md Phase 5)."""
from typing import Literal

from pydantic import BaseModel

Arm = Literal[
    "isolated_real", "isolated_augmented",
    "federated_real", "federated_augmented",
    "centralized_real", "centralized_augmented",
]


class RunRequest(BaseModel):
    arm: Arm
    seed: int = 42
    epochs: int = 20  # isolated/centralized
    num_rounds: int = 10  # federated
    local_epochs: int = 2  # federated


class RunStatusResponse(BaseModel):
    run_id: str
    arm: str
    status: Literal["running", "complete", "failed"]
    round_metrics: list[dict]
    client_metrics: list[dict]
    final_metrics: dict | None = None
    error: str | None = None
