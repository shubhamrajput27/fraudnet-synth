"""Fidelity with SDMetrics (sdmetrics 0.32): how statistically similar a synthetic set is to real data.

QualityReport gives two property scores (0-1, higher is better):
  * Column Shapes: does each column's distribution match? (per-column KS-based similarity)
  * Column Pair Trends: do pairs of columns move together the same way? (correlation similarity)
The overall score is their average. It scores a whole SET, not individual rows.
"""
import pandas as pd
from sdmetrics.reports.single_table import QualityReport

from ml.data.load import FEATURE_COLUMNS

METADATA = {"columns": {c: {"sdtype": "numerical"} for c in FEATURE_COLUMNS}}


def quality(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    if len(synth) < 2:
        return {"overall": None, "column_shapes": None, "column_pair_trends": None}
    report = QualityReport()
    report.generate(real[FEATURE_COLUMNS], synth[FEATURE_COLUMNS], METADATA, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]
    return {"overall": round(float(report.get_score()), 4),
            "column_shapes": round(float(props["Column Shapes"]), 4),
            "column_pair_trends": round(float(props["Column Pair Trends"]), 4)}
