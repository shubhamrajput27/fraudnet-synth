"""Schema Mode: LLM-generated candidate fraud rows for data-poor banks (Banks C, D).

Privacy (decision D-015): the prompt is built ONLY from (a) public schema facts and
(b) aggregate statistics computed inside the bank (schema_stats.py). No real row,
and no single real value such as a min or max, ever goes to Groq.
`build_prompt()` does not even receive the bank's dataframe, only the stats dict.

Every API response is cached inside the bank folder, so:
  * a rerun resumes from the cache instead of calling the API again,
  * demos never depend on a live API call.

Run from the project root (all banks with mode: schema):
    python -m ml.augmentation.llm_engine
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.augmentation.ctgan_engine import compare, load_bank_fraud, plot_distributions
from ml.augmentation.schema_stats import aggregate_stats, local_bounds
from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config

SYSTEM_PROMPT = (
    "You generate synthetic tabular data for academic fraud-detection research. "
    "You always answer with a single JSON object and nothing else."
)


# ---------------------------------------------------------------- prompt ---

def build_prompt(stats: dict, schema: dict, n_rows: int, decimals: int, batch_no: int) -> str:
    """User prompt from schema + aggregate statistics only. Deterministic for given inputs."""
    header = "column | " + " | ".join(next(iter(stats["columns"].values())).keys())
    stat_lines = [header] + [
        f"{col} | " + " | ".join(str(v) for v in s.values()) for col, s in stats["columns"].items()
    ]
    t_lo, t_hi = schema["time_range_seconds"]
    return f"""Generate {n_rows} NEW synthetic credit-card FRAUD transactions (batch {batch_no}).

DATASET SCHEMA (30 numeric columns, in this exact order):
{", ".join(FEATURE_COLUMNS)}
- Time: seconds elapsed since the start of a 48-hour window, a whole number from {t_lo} to {t_hi}.
- V1..V28: anonymised PCA components (unitless real numbers, no human meaning).
- Amount: transaction amount, >= {schema['amount_min']}, two decimals.
Every row you generate is a fraud case.

STATISTICS of the bank's real fraud transactions (n = {stats['n_rows']}).
p10/p50/p90 are the 10th, 50th (median) and 90th percentiles:
{chr(10).join(stat_lines)}

INSTRUCTIONS
1. Each row must follow these statistics as a distribution: about 10% of values below p10,
   about 10% above p90, centred near p50, with the given spread (std). Do not make every row average.
   Values outside the p10-p90 range are expected in the tails.
2. Make rows varied: no two rows may be identical or near-identical, and do not reuse a value
   pattern from one row in another.
3. Use {decimals} decimals for V1..V28, 2 decimals for Amount, whole numbers for Time.
   Use irregular digits, like real measurements; avoid patterned sequences such as 1.234, 5.432, 9.876, 3.210.
4. Output ONLY this JSON shape, with {n_rows} rows and all 30 named fields in every row:
   {{"rows": [{{"Time": ..., "V1": ..., ..., "V28": ..., "Amount": ...}}, ...]}}. No comments."""


# ------------------------------------------------------------- parsing ---

def parse_rows(content: str, schema: dict) -> tuple[list[list[float]], dict]:
    """Parse the JSON reply; keep only structurally valid rows. Returns (rows, reject_counts).

    These are basic sanity checks only. The full validation layer is Step 6.
    """
    rejects = {"bad_json": 0, "wrong_shape": 0, "non_numeric": 0, "time_out_of_range": 0, "negative_amount": 0}
    try:
        raw_rows = json.loads(content)["rows"]
        if not isinstance(raw_rows, list):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError):
        rejects["bad_json"] += 1
        return [], rejects

    t_lo, t_hi = schema["time_range_seconds"]
    rows = []
    for r in raw_rows:
        # Rows arrive as {"Time": .., "V1": .., ...} objects (D-019); plain lists are accepted too.
        if isinstance(r, dict):
            if set(r) != set(FEATURE_COLUMNS):
                rejects["wrong_shape"] += 1
                continue
            r = [r[c] for c in FEATURE_COLUMNS]
        if not isinstance(r, list) or len(r) != len(FEATURE_COLUMNS):
            rejects["wrong_shape"] += 1
            continue
        try:
            vals = [float(v) for v in r]
        except (TypeError, ValueError):
            rejects["non_numeric"] += 1
            continue
        if not all(np.isfinite(vals)):
            rejects["non_numeric"] += 1
        elif not (t_lo <= vals[0] <= t_hi):
            rejects["time_out_of_range"] += 1
        elif vals[-1] < schema["amount_min"]:
            rejects["negative_amount"] += 1
        else:
            rows.append(vals)
    return rows, rejects


# ------------------------------------------------------- quality check ---

# 3-digit runs like 123, 234, ... and 321, 432, ...: by chance ~16 of 1000 decimal endings (1.6%).
_SEQ_RUNS = {100 * a + 10 * (a + 1) + (a + 2) for a in range(8)} | {100 * a + 10 * (a - 1) + (a - 2) for a in range(2, 10)}


def patterned_decimal_pct(df: pd.DataFrame) -> float:
    """% of V1-V28 values whose first 3 decimals form a consecutive-digit run (e.g. -3.210, 5.432).

    LLMs tend to write number-shaped text rather than truly irregular values. Real data shows
    roughly the chance rate (~1.6%), so a much higher share signals patterned, low-entropy output.
    """
    v = df[[c for c in FEATURE_COLUMNS if c.startswith("V")]].abs().to_numpy()
    decimals = np.floor(np.round(v * 1000, 6)).astype(np.int64) % 1000
    return round(100 * float(np.isin(decimals, list(_SEQ_RUNS)).mean()), 1)


# ----------------------------------------------------------- API calls ---

def response_format_for(n_rows: int, mode: str) -> dict:
    """Decision D-019: strict JSON schema with NAMED fields per row.

    Attempt 1 (plain JSON mode) returned 30/22 rows instead of 25 and rows of 29/31 numbers.
    Attempt 2 (strict schema, rows as bare 30-number arrays) was rejected by Groq because
    18 of 27 rows had 31 numbers: the model loses count in long unlabelled arrays.
    Named fields let the model fill labelled slots instead of counting positions; Groq checks the
    reply against this schema and returns HTTP 400 'json_validate_failed' if it doesn't match.
    Attempt 3 (named fields, exactly n_rows required) still failed: 12 rows instead of 15.
    So each row's shape is enforced strictly, but the row COUNT may be anything from 1 to n_rows:
    we simply keep calling until enough valid rows are collected.
    """
    if mode == "json_object":
        return {"type": "json_object"}
    row = {"type": "object", "properties": {c: {"type": "number"} for c in FEATURE_COLUMNS},
           "required": list(FEATURE_COLUMNS), "additionalProperties": False}
    schema = {"type": "object", "properties": {"rows": {"type": "array", "minItems": 1,
                                                        "maxItems": n_rows, "items": row}},
              "required": ["rows"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "fraud_rows", "strict": True, "schema": schema}}


def call_with_backoff(client, request: dict, rl: dict) -> tuple[object, int]:
    """Call Groq; on rate-limit/connection errors wait 2, 4, 8, ... s (or the server's retry-after)."""
    import groq

    for attempt in range(rl["max_retries"] + 1):
        try:
            return client.chat.completions.create(**request), attempt
        except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as e:
            if attempt == rl["max_retries"]:
                raise
            wait = min(rl["backoff_base_seconds"] * 2 ** attempt, rl["backoff_cap_seconds"])
            retry_after = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
            if retry_after:
                wait = max(wait, float(retry_after))
            print(f"    {type(e).__name__}; retry {attempt + 1}/{rl['max_retries']} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def generate_for_bank(bank: str, cfg: dict, seed: int, client=None) -> dict:
    bank_dir = PROJECT_ROOT / load_config("partition")["clients_dir"] / bank
    cache_dir = bank_dir / "llm_cache"
    cache_dir.mkdir(exist_ok=True)
    results_dir = PROJECT_ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    gen, rl, schema = cfg["generation"], cfg["rate_limit"], cfg["schema"]
    model = os.environ["GROQ_MODEL"]

    # 1) Inside the bank: compute what may leave (aggregates) and what may not (bounds).
    real = load_bank_fraud(bank_dir)
    stats = aggregate_stats(real, cfg["stats"]["include"], cfg["stats"]["decimals"])
    bounds = local_bounds(real)
    with open(results_dir / f"{bank}_stats_sent.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)  # exactly what Groq sees: auditable evidence
    (results_dir / f"{bank}_prompt_example.txt").write_text(
        "SYSTEM:\n" + SYSTEM_PROMPT + "\n\nUSER:\n"
        + build_prompt(stats, schema, gen["rows_per_call"], gen["value_decimals"], 1), encoding="utf-8")

    # 2) Collect rows: cached batches first, then new API calls until we have enough.
    rows, rejects_total, calls_log = [], {}, []
    batch_no, api_calls, tokens_used = 0, 0, 0
    while len(rows) < gen["n_candidates"]:
        batch_no += 1
        if batch_no > rl["max_calls_per_bank"]:
            print(f"  [{bank}] stopping: reached max_calls_per_bank={rl['max_calls_per_bank']}")
            break
        cache_file = cache_dir / f"batch_{batch_no:03d}.json"
        if cache_file.exists():
            entry = json.loads(cache_file.read_text(encoding="utf-8"))
            source = "cache"
        else:
            if client is None:
                from groq import Groq
                client = Groq(max_retries=0)  # we do our own, visible backoff
            request = dict(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_prompt(stats, schema, gen["rows_per_call"],
                                                                   gen["value_decimals"], batch_no)}],
                response_format=response_format_for(gen["rows_per_call"], gen["response_format"]),
                temperature=gen["temperature"],
                reasoning_effort=gen["reasoning_effort"],
                max_completion_tokens=gen["max_completion_tokens"],
                seed=seed * 1000 + batch_no,  # best effort; LLM output is not guaranteed reproducible
            )
            import groq

            t0 = time.perf_counter()
            base = {"batch": batch_no, "model": model, "created_utc": datetime.now(timezone.utc).isoformat()}
            try:
                resp, retries = call_with_backoff(client, request, rl)
                usage = resp.usage
                spent = usage.total_tokens
                entry = {**base, "seconds": round(time.perf_counter() - t0, 1), "retries": retries,
                         "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                         "finish_reason": resp.choices[0].finish_reason,
                         "content": resp.choices[0].message.content or ""}
            except groq.BadRequestError as e:
                # The reply broke the JSON schema. Record a failed batch and move on;
                # the generated text is discarded and Groq doesn't report usage for it.
                if "json_validate_failed" not in str(e):
                    raise
                spent = gen["max_completion_tokens"]  # unknown, so assume the worst for pacing
                entry = {**base, "seconds": round(time.perf_counter() - t0, 1), "retries": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "finish_reason": "json_validate_failed", "content": ""}
            elapsed = time.perf_counter() - t0
            api_calls += 1
            cache_file.write_text(json.dumps(entry, indent=1), encoding="utf-8")
            source = "api"
            tokens_used += spent
            # Pace: a call using T tokens "costs" T/budget minutes of the per-minute budget.
            min_gap = 60 * spent / rl["tokens_per_minute_budget"]
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)

        batch_rows, rej = parse_rows(entry["content"], schema)
        rows.extend(batch_rows)
        for k, v in rej.items():
            rejects_total[k] = rejects_total.get(k, 0) + v
        calls_log.append({"batch": batch_no, "source": source, "valid_rows": len(batch_rows),
                          "rejected": sum(rej.values()), "finish_reason": entry.get("finish_reason"),
                          "tokens": entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0),
                          "seconds": entry.get("seconds")})
        print(f"  [{bank}] batch {batch_no:2d} ({source}): +{len(batch_rows)} valid, "
              f"{sum(rej.values())} rejected -> {len(rows)}/{gen['n_candidates']}", flush=True)

    # 3) Save candidates INSIDE the bank, then summarise.
    cand = pd.DataFrame(rows[: gen["n_candidates"]], columns=FEATURE_COLUMNS)
    out = cand.copy()
    out[LABEL_COLUMN] = 1
    out.to_csv(bank_dir / "synthetic_candidates.csv", index=False)

    lo = np.array([bounds[c][0] for c in FEATURE_COLUMNS])
    hi = np.array([bounds[c][1] for c in FEATURE_COLUMNS])
    outside = (cand.to_numpy() < lo) | (cand.to_numpy() > hi)
    plot_distributions(real, cand, cfg["compare_columns"], bank,
                       results_dir / f"{bank}_real_vs_synthetic.png", synth_label="LLM")
    summary = {
        "bank": bank, "model": model, "n_real_fraud": len(real), "n_candidates": len(cand),
        "batches_used": batch_no, "api_calls_this_run": api_calls, "tokens_this_run": tokens_used,
        "total_tokens_all_batches": sum(c["tokens"] for c in calls_log),
        "total_api_seconds_all_batches": round(sum(c["seconds"] or 0 for c in calls_log), 1),
        "parse_rejects": rejects_total,
        "pct_cells_outside_real_min_max": round(100 * float(outside.mean()), 1),
        "pct_rows_with_any_value_outside_real_min_max": round(100 * float(outside.any(axis=1).mean()), 1),
        "pct_patterned_decimals": {"llm_candidates": patterned_decimal_pct(cand),
                                   "real_fraud_rows": patterned_decimal_pct(real),
                                   "chance_rate": 1.6},
        **compare(real, cand),
        "calls": calls_log,
    }
    summary.pop("edge_clamped")  # LLM output is not clamped; the outside-range metrics above replace it
    with open(results_dir / f"{bank}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, **summary}, f, indent=2)
    return summary


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("GROQ_MODEL"):
        raise SystemExit("Set GROQ_API_KEY and GROQ_MODEL in .env (see .env.example).")
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", nargs="*", help="default: every bank with mode: schema")
    args = parser.parse_args()

    seed = load_config("data")["seed"]
    cfg = load_config("llm")
    banks_cfg = load_config("partition")["banks"]
    banks = args.banks or [b for b, v in banks_cfg.items() if v["mode"] == "schema"]
    order = list(banks_cfg)

    rows = []
    for bank in banks:
        print(f"[{bank}] Schema Mode with {os.environ['GROQ_MODEL']}", flush=True)
        s = generate_for_bank(bank, cfg, seed + order.index(bank))
        rows.append(s)
        print(f"[{bank}] {s['n_candidates']} candidates from {s['batches_used']} batches | "
              f"tokens {s['total_tokens_all_batches']:,} | API time {s['total_api_seconds_all_batches']}s | "
              f"parse rejects {s['parse_rejects']}")
        print(f"[{bank}] mean KS {s['mean_ks_stat']} (max {s['max_ks_stat']}) | mean |corr diff| "
              f"{s['mean_abs_corr_diff']} | exact copies {s['exact_copies_of_real_rows']} | duplicate rows "
              f"{s['duplicate_synthetic_rows']} | rows outside real min/max "
              f"{s['pct_rows_with_any_value_outside_real_min_max']}% (cells {s['pct_cells_outside_real_min_max']}%)")
        pd_ = s["pct_patterned_decimals"]
        print(f"[{bank}] patterned decimals (e.g. .123/.432): LLM {pd_['llm_candidates']}% vs real "
              f"{pd_['real_fraud_rows']}% (chance ~{pd_['chance_rate']}%)")

    print("\nPer-column KS statistic (0 = identical distributions, 1 = completely different):")
    print(pd.DataFrame({r["bank"]: {c: v["ks_stat"] for c, v in r["per_column"].items()} for r in rows}).T.to_string())
    print("\nMean real vs synthetic for plotted columns:")
    for r in rows:
        for c in cfg["compare_columns"]:
            v = r["per_column"][c]
            print(f"  {r['bank']} {c:6s} real {v['real_mean']:>10} +/- {v['real_std']:<10} "
                  f"synth {v['synth_mean']:>10} +/- {v['synth_std']}")


if __name__ == "__main__":
    main()
