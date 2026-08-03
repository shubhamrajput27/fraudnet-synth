"""Pandera structural schema validation for synthetic candidate rows (CLAUDE.md Phase 3).

Purely structural: correct columns, numeric dtypes, no nulls/infinities, Class is always 1
(these are synthetic *fraud* candidates), and each PCA/Amount/Time column falls within a
generous multiple of the real dataset's observed range (catches generation gone wildly wrong,
e.g. NaNs coerced to absurd floats) without being a fidelity check (that's fidelity.py).
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def build_schema(real_reference: pd.DataFrame, range_slack: float = 3.0) -> DataFrameSchema:
    """`real_reference` is the client's own real data (any class) — used only to derive generous
    per-column bounds, never compared row-by-row here (that's fidelity/novelty)."""
    columns = {}
    for col in FEATURE_COLUMNS:
        lo, hi = real_reference[col].min(), real_reference[col].max()
        span = hi - lo if hi > lo else 1.0
        columns[col] = Column(
            float,
            checks=[
                Check(lambda s: np.isfinite(s), element_wise=False, error=f"{col} has non-finite values"),
                Check.ge(lo - range_slack * span),
                Check.le(hi + range_slack * span),
            ],
            nullable=False,
            coerce=True,
        )
    columns["Class"] = Column(int, checks=Check.eq(1), nullable=False, coerce=True)
    return DataFrameSchema(columns)


def validate_schema(candidates: pd.DataFrame, real_reference: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (rows_passing_schema, list_of_error_messages_for_dropped_rows).
    Validates row-by-row so one bad row doesn't reject an otherwise-fine batch."""
    if candidates.empty:
        return candidates, []

    schema = build_schema(real_reference)
    passing_rows = []
    errors: list[str] = []
    for idx, row in candidates.iterrows():
        row_df = pd.DataFrame([row])
        try:
            schema.validate(row_df, lazy=False)
            passing_rows.append(idx)
        except pa.errors.SchemaError as exc:
            errors.append(f"row {idx}: {exc}")

    return candidates.loc[passing_rows].reset_index(drop=True), errors
