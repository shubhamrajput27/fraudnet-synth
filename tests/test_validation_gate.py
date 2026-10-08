"""Step 6 tests: schema/range/clamp checks, DCR privacy and diversity helpers. No network, no real data."""
import numpy as np
import pandas as pd

from ml.augmentation.schema_stats import local_bounds
from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, V_COLUMNS, load_config
from ml.validation import diversity
from ml.validation.schema_checks import build_schema, edge_clamped_counts, schema_failures

CFG = load_config("validation")


def _real(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(-2, 3, size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(1000, 170000, n))
    df["Amount"] = rng.exponential(100, n)
    return df[FEATURE_COLUMNS]


def _interior(real, n=4):
    """Rows built from column medians: safely inside the range and touching no min/max."""
    return pd.DataFrame([real.median()] * n).reset_index(drop=True)


def _with_label(df):
    out = df.copy()
    out[LABEL_COLUMN] = 1
    return out


def test_valid_rows_pass_schema():
    real = _real()
    assert schema_failures(_with_label(_interior(real)), build_schema(local_bounds(real), CFG["schema"])) == {}


def test_out_of_range_missing_and_wrong_class_are_caught():
    real = _real()
    bounds = local_bounds(real)
    cand = _with_label(_interior(real, 4))
    lo, hi = bounds["V3"]
    cand.loc[0, "V3"] = hi + 0.5 * (hi - lo)   # beyond the 10% margin
    cand.loc[1, "V5"] = np.nan                 # missing value
    cand.loc[2, LABEL_COLUMN] = 0              # not fraud
    fails = schema_failures(cand, build_schema(bounds, CFG["schema"]))
    assert "out_of_range:V3" in fails[0]
    assert "missing_value" in fails[1]
    assert any("isin" in r for r in fails[2])
    assert 3 not in fails


def test_value_inside_margin_is_allowed():
    real = _real()
    bounds = local_bounds(real)
    cand = _with_label(_interior(real, 1))
    lo, hi = bounds["V3"]
    cand.loc[0, "V3"] = hi + 0.05 * (hi - lo)  # within the 10% margin
    assert schema_failures(cand, build_schema(bounds, CFG["schema"])) == {}


def test_edge_clamp_rule():
    real = _real()
    bounds = local_bounds(real)
    cand = _with_label(_interior(real, 2))
    for c in ["V1", "V2", "V3"]:
        cand.loc[0, c] = bounds[c][1]          # 3 values stuck on the real max
    cand.loc[1, "V1"] = bounds["V1"][0]        # only 1 stuck value: tolerated
    assert list(edge_clamped_counts(cand, bounds)) == [3, 1]
    fails = schema_failures(cand, build_schema(bounds, CFG["schema"]))
    assert fails.get(0) == ["edge_clamped"] and 1 not in fails


def test_dcr_flags_near_copies():
    real = _real()
    near_copy = real.head(1) + 1e-4
    far = real.head(1) + 50
    r, s = diversity.standardise(real, pd.concat([near_copy, far]))
    d = diversity.dcr(s, r)
    thr = np.percentile(diversity.real_nn_distances(r), 5)
    assert d[0] < thr < d[1]


def test_greedy_dedupe_keeps_first_of_close_pair():
    s = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    assert list(diversity.greedy_dedupe(s, min_dist=1.0)) == [True, False, True]


def test_rows_as_text_format():
    t = diversity.rows_as_text(_real(n=1), 3)[0]
    assert t.startswith("Time=") and "V28=" in t and t.count("=") == 30
