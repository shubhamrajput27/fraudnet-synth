"""Frozen validation thresholds (CLAUDE.md D6).

Set 2026-08-03 during Phase 3, based on real computed fidelity/novelty/mode-collapse values
across all four banks' actual candidate batches (see PLAN.md progress log for the numbers that
informed these values). Approved explicitly by the user 2026-08-03.

NEVER change these after any experimental arm has been evaluated — that would let results be
tuned post-hoc, which CLAUDE.md's evaluation philosophy explicitly forbids. If a threshold turns
out to be wrong, that's a finding to report, not a value to quietly edit.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationThresholds:
    fidelity_floor: float = 0.5
    novelty_floor: float = 0.95
    mode_collapse_ceiling: float = 0.998  # applied to max pairwise nearest-neighbor similarity
    min_valid_rows_for_distributional_checks: int = 5


FROZEN_THRESHOLDS = ValidationThresholds()
