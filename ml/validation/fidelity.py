"""SDMetrics-based fidelity scoring: how well synthetic rows match the real fraud distribution
they're meant to augment (CLAUDE.md Phase 3 / validation stack).

Uses the unified `sdmetrics.reports.QualityReport`, not the deprecated
`sdmetrics.reports.single_table.QualityReport` (verified against installed sdmetrics==0.28.2:
the unified report requires `real_data`/`synthetic_data` as `{table_name: df}` dicts and the
full `Metadata.to_dict()` output including the `tables` key, not just the per-table sub-dict —
confirmed empirically, not documented clearly).
"""
import pandas as pd
from sdmetrics.reports import QualityReport
from sdv.metadata import Metadata

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TABLE_NAME = "fraud"


def compute_fidelity_score(real_fraud: pd.DataFrame, synthetic_fraud: pd.DataFrame) -> tuple[float, dict]:
    """Returns (overall_score, property_breakdown). Both inputs must contain only
    FEATURE_COLUMNS (no Class column — it's constant and uninformative for fidelity)."""
    real = real_fraud[FEATURE_COLUMNS].reset_index(drop=True)
    synth = synthetic_fraud[FEATURE_COLUMNS].reset_index(drop=True)

    metadata = Metadata.detect_from_dataframe(real, table_name=TABLE_NAME)
    report = QualityReport()
    report.generate({TABLE_NAME: real}, {TABLE_NAME: synth}, metadata.to_dict(), verbose=False)

    score = report.get_score()
    properties = report.get_properties().set_index("Property")["Score"].to_dict()
    return score, properties
