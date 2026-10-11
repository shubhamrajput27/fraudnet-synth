"""Request/response shapes (Pydantic v2). FastAPI validates every request against these and
rejects malformed ones with HTTP 422 before our code runs."""
from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

from ml.data.load import V_COLUMNS
from services.orchestrator.settings import BANKS

ARM_NAMES = {1: "Isolated, real", 2: "Isolated, augmented", 3: "Federated, real",
             4: "Federated, augmented", 5: "Centralized, real", 6: "Centralized, augmented"}

# One transaction: Time, V1..V28, Amount (the 30 raw ULB feature columns).
Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0, le=172792, description="seconds since the start of the 48-hour window")),
    **{v: (float, Field(description="anonymised PCA component")) for v in V_COLUMNS},
    Amount=(float, Field(ge=0, description="transaction amount")),
)


class Prediction(BaseModel):
    score_logit: float
    fraud_probability: float = Field(description="display only; the model's scores are not calibrated")
    threshold_logit: float
    is_fraud: bool
    model: dict


class RunRequest(BaseModel):
    """D-043: one or more arms (1-6) or 'all', over chosen seeds."""
    arms: list[int] | Literal["all"] = "all"
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=10)

    @field_validator("arms")
    @classmethod
    def _valid_arms(cls, v):
        if v != "all" and (not v or any(a not in ARM_NAMES for a in v)):
            raise ValueError("arms must be 'all' or a list of numbers 1-6")
        return v


class GenerationRequest(BaseModel):
    """D-041: 'replay' rebuilds deterministically into a separate folder and compares checksums;
    'regenerate' makes genuinely new data into a NEW versioned folder. Neither touches the files
    the experiments used."""
    banks: list[str] = Field(default_factory=lambda: list(BANKS))
    mode: Literal["replay", "regenerate"] = "replay"

    @field_validator("banks")
    @classmethod
    def _valid_banks(cls, v):
        if not v or any(b not in BANKS for b in v):
            raise ValueError(f"banks must be a non-empty subset of {BANKS}")
        return v


class Job(BaseModel):
    id: str
    type: str
    params: dict
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
    created: str
    started: str | None = None
    finished: str | None = None
    return_code: int | None = None
    result: dict | None = None
