"""Schema and validity checks with Pandera (decision D-022).

ULB is already anonymised (no names, card numbers or addresses), so there is no real
PII to detect. The "PII/schema" check here is therefore a schema and range check:
the exact column set, numeric types, no missing values, Class == 1, values within a
plausible range, and not piled up on the real min/max (CTGAN edge clamping).
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN


def edge_clamped_counts(df: pd.DataFrame, bounds: dict) -> pd.Series:
    """Per row: how many values sit exactly on the bank's real min or max."""
    lo = np.array([bounds[c][0] for c in FEATURE_COLUMNS])
    hi = np.array([bounds[c][1] for c in FEATURE_COLUMNS])
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    hits = np.isclose(x, lo, atol=1e-6) | np.isclose(x, hi, atol=1e-6)
    return pd.Series(hits.sum(axis=1), index=df.index)


def build_schema(bounds: dict, cfg: dict) -> pa.DataFrameSchema:
    """bounds = the bank's real training-fraud min/max per column (computed locally, never shared)."""
    margin = cfg["range_margin_fraction"]
    t_lo, t_hi = cfg["time_range_seconds"]
    columns = {}
    for c in FEATURE_COLUMNS:
        lo, hi = bounds[c]
        pad = margin * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if c == "Time":  # also respect the public schema limits
            lo, hi = max(lo, t_lo), min(hi, t_hi)
        if c == "Amount":
            lo = max(lo, cfg["amount_min"])
        columns[c] = pa.Column(float, pa.Check.in_range(lo, hi), nullable=False, coerce=True,
                               title=f"{c} within real range +/- {int(margin * 100)}%")
    columns[LABEL_COLUMN] = pa.Column(int, pa.Check.isin([1]), nullable=False, coerce=True)

    max_clamped = cfg["max_edge_clamped_values"]
    clamp_check = pa.Check(lambda df: edge_clamped_counts(df, bounds) <= max_clamped,
                           name=f"edge_clamped_at_most_{max_clamped}")
    # strict=True: no missing or extra columns allowed.
    return pa.DataFrameSchema(columns, checks=[clamp_check], strict=True, ordered=False)


def schema_failures(df: pd.DataFrame, schema: pa.DataFrameSchema) -> dict[int, list[str]]:
    """Validate all rows at once (lazy) and return {row_index: [failure reasons]}."""
    try:
        schema.validate(df, lazy=True)
        return {}
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        reasons: dict[int, list[str]] = {}
        for _, r in fc.iterrows():
            if pd.isna(r["index"]):  # whole-table problem (e.g. missing column): fail every row
                for i in df.index:
                    reasons.setdefault(int(i), []).append(f"schema:{r['check']}")
                continue
            check = str(r["check"])
            if check.startswith("in_range"):
                label = f"out_of_range:{r['column']}"
            elif check.startswith("edge_clamped"):
                label = "edge_clamped"
            elif check == "not_nullable":
                label = "missing_value"
            else:
                label = f"schema:{check}"
            row = reasons.setdefault(int(r["index"]), [])
            if label not in row:
                row.append(label)
        return reasons
