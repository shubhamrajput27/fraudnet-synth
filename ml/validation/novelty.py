"""SDMetrics NewRowSynthesis: novelty check that doubles as the PII/leakage guard.

A synthetic row that exactly (or near-exactly, within `numerical_match_tolerance`) matches a
real row means real client data leaked into the "synthetic" batch — a privacy-invariant defect,
not just a quality issue. Verified empirically (see Phase 3 progress log / PLAN.md) that the
score drops proportionally when real rows are injected as duplicates, confirming it measures
what CLAUDE.md's D6 calls the "novelty/nearest-neighbour distance floor".
"""
import pandas as pd
from sdmetrics.single_table import NewRowSynthesis
from sdv.metadata import Metadata

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TABLE_NAME = "fraud"


def compute_novelty_score(
    real_fraud: pd.DataFrame, synthetic_fraud: pd.DataFrame, numerical_match_tolerance: float = 0.01
) -> float:
    """Fraction of synthetic rows that do NOT exactly/near-exactly match a real row. 1.0 = fully
    novel, 0.0 = every synthetic row is a leaked copy of a real one."""
    real = real_fraud[FEATURE_COLUMNS].reset_index(drop=True)
    synth = synthetic_fraud[FEATURE_COLUMNS].reset_index(drop=True)

    metadata = Metadata.detect_from_dataframe(real, table_name=TABLE_NAME)
    table_md = metadata.to_dict()["tables"][TABLE_NAME]

    return NewRowSynthesis.compute(real, synth, table_md, numerical_match_tolerance=numerical_match_tolerance)
