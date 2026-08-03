"""Shared validation orchestrator (CLAUDE.md Phase 3 / D6): schema -> fidelity -> novelty ->
mode-collapse, per client per mode. Produces a validation report and the surviving row set.

Schema is a per-row filter (a structurally bad row doesn't sink an otherwise-fine batch).
Fidelity/novelty/mode-collapse are batch-level judgments on the schema-surviving rows — CLAUDE.md
D6 treats "validation pass rate" itself as a headline CTGAN-vs-LLM comparison, which only makes
sense as a per-batch verdict, not a per-row one.
"""
from dataclasses import dataclass

import pandas as pd

from ml.validation.diversity import compute_mode_collapse_score
from ml.validation.fidelity import compute_fidelity_score
from ml.validation.novelty import compute_novelty_score
from ml.validation.schema import validate_schema
from ml.validation.thresholds import FROZEN_THRESHOLDS, ValidationThresholds


@dataclass
class ValidationReport:
    bank: str
    mode: str
    n_candidates: int
    n_schema_pass: int
    n_schema_errors: int
    batch_verdict: str  # "PASS", "REJECT_TOO_FEW_ROWS", "REJECT_FIDELITY", "REJECT_NOVELTY", "REJECT_MODE_COLLAPSE"
    fidelity_score: float | None
    novelty_score: float | None
    mode_collapse_mean: float | None
    mode_collapse_max: float | None
    n_validated_rows: int

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def validate_batch(
    bank: str,
    mode: str,
    candidates: pd.DataFrame,
    real_reference: pd.DataFrame,
    real_fraud: pd.DataFrame,
    thresholds: ValidationThresholds = FROZEN_THRESHOLDS,
) -> tuple[pd.DataFrame, ValidationReport]:
    n_candidates = len(candidates)
    schema_passing, schema_errors = validate_schema(candidates, real_reference)
    n_schema_pass = len(schema_passing)

    base_kwargs = dict(
        bank=bank,
        mode=mode,
        n_candidates=n_candidates,
        n_schema_pass=n_schema_pass,
        n_schema_errors=len(schema_errors),
    )

    if n_schema_pass < thresholds.min_valid_rows_for_distributional_checks:
        report = ValidationReport(
            **base_kwargs,
            batch_verdict="REJECT_TOO_FEW_ROWS",
            fidelity_score=None,
            novelty_score=None,
            mode_collapse_mean=None,
            mode_collapse_max=None,
            n_validated_rows=0,
        )
        return schema_passing.iloc[0:0], report

    fidelity_score, _ = compute_fidelity_score(real_fraud, schema_passing)
    novelty_score = compute_novelty_score(real_fraud, schema_passing)
    mode_collapse_mean, mode_collapse_max = compute_mode_collapse_score(schema_passing)

    if fidelity_score < thresholds.fidelity_floor:
        verdict = "REJECT_FIDELITY"
    elif novelty_score < thresholds.novelty_floor:
        verdict = "REJECT_NOVELTY"
    elif mode_collapse_max > thresholds.mode_collapse_ceiling:
        verdict = "REJECT_MODE_COLLAPSE"
    else:
        verdict = "PASS"

    validated_rows = schema_passing if verdict == "PASS" else schema_passing.iloc[0:0]

    report = ValidationReport(
        **base_kwargs,
        batch_verdict=verdict,
        fidelity_score=fidelity_score,
        novelty_score=novelty_score,
        mode_collapse_mean=mode_collapse_mean,
        mode_collapse_max=mode_collapse_max,
        n_validated_rows=len(validated_rows),
    )
    return validated_rows, report
