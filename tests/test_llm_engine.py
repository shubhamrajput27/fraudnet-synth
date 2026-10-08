"""Step 5 tests: Schema Mode privacy (no real value in the prompt), parsing, and stats rules.
No network calls: these tests never contact Groq.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from ml.augmentation.llm_engine import build_prompt, parse_rows
from ml.augmentation.schema_stats import aggregate_stats, local_bounds
from ml.data.load import FEATURE_COLUMNS, V_COLUMNS, load_config

CFG = load_config("llm")


def _fake_fraud(n=18, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(-3, 4, size=(n, 28)).round(6), columns=V_COLUMNS)
    df.insert(0, "Time", rng.integers(0, 172792, n).astype(float))
    df["Amount"] = rng.exponential(100, n).round(2)
    return df[FEATURE_COLUMNS]


def _stats(df):
    return aggregate_stats(df, CFG["stats"]["include"], CFG["stats"]["decimals"])


def test_stats_exclude_min_and_max():
    s = _stats(_fake_fraud())
    for col_stats in s["columns"].values():
        assert set(col_stats) == {"mean", "std", "p10", "p50", "p90"}


def test_disallowed_statistic_is_refused():
    with pytest.raises(ValueError):
        aggregate_stats(_fake_fraud(), ["mean", "min"], 2)


def test_prompt_contains_no_real_value():
    """Privacy: no real cell value (at any precision the LLM could see) appears in the prompt."""
    real = _fake_fraud()
    prompt = build_prompt(_stats(real), CFG["schema"], 25, 3, 1)
    prompt_numbers = {float(x) for x in re.findall(r"-?\d+\.\d+", prompt)}
    for col in FEATURE_COLUMNS:
        for v in real[col]:
            if v == 0:
                continue
            # A real value counts as leaked if it appears at full or 2-decimal precision.
            assert float(v) not in prompt_numbers, f"{col} value {v} leaked"
            assert round(float(v), 2) not in prompt_numbers or round(float(v), 2) in {
                s for st in _stats(real)["columns"].values() for s in st.values()
            }, f"{col} value {v} leaked"


def test_prompt_never_contains_bounds():
    """A column's exact min/max must not appear in that column's statistics line.
    (Compared as whole numbers per column: a different column's median can coincidentally
    share digits, e.g. V23 median -3.53 vs V5 max 3.53, without leaking anything.)"""
    real = _fake_fraud()
    prompt = build_prompt(_stats(real), CFG["schema"], 25, 3, 1)
    stat_lines = {ln.split(" | ")[0]: ln for ln in prompt.splitlines() if " | " in ln}
    for col, (lo, hi) in local_bounds(real).items():
        numbers = {float(x) for x in stat_lines[col].split(" | ")[1:]}
        for b in (lo, hi):
            assert round(b, 2) not in numbers, f"{col} bound {b} leaked"


def test_parse_keeps_valid_rows_and_counts_rejects():
    good = [5000] + [0.1] * 28 + [12.5]
    named = dict(zip(FEATURE_COLUMNS, good))
    missing = {k: v for k, v in named.items() if k != "V7"}
    content = json.dumps({"rows": [named, good, missing, good[:-1], [5000] + ["x"] * 29,
                                   [-5] + [0.1] * 29, good[:-1] + [-1]]})
    rows, rej = parse_rows(content, CFG["schema"])
    assert rows == [[float(v) for v in good]] * 2
    assert rej == {"bad_json": 0, "wrong_shape": 2, "non_numeric": 1, "time_out_of_range": 1, "negative_amount": 1}


def test_parse_bad_json():
    rows, rej = parse_rows("not json at all", CFG["schema"])
    assert rows == [] and rej["bad_json"] == 1


def test_strict_schema_fixes_row_count_and_length():
    from ml.augmentation.llm_engine import response_format_for
    rf = response_format_for(25, "json_schema_strict")
    rows = rf["json_schema"]["schema"]["properties"]["rows"]
    assert rf["json_schema"]["strict"] is True
    assert rows["minItems"] == 1 and rows["maxItems"] == 25
    assert rows["items"]["required"] == FEATURE_COLUMNS
    assert rows["items"]["additionalProperties"] is False


def test_patterned_decimal_detector():
    from ml.augmentation.llm_engine import patterned_decimal_pct
    df = _fake_fraud(n=50)
    df[V_COLUMNS] = -3.210  # every value a "321" run
    assert patterned_decimal_pct(df) == 100.0
    df[V_COLUMNS] = 1.507   # no run
    assert patterned_decimal_pct(df) == 0.0
