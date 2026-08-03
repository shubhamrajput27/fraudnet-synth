"""Groq LLM-based synthetic fraud-row generator for data-poor clients (CLAUDE.md D1: Schema Mode).

V1-V28 are anonymized PCA components with no semantic meaning, so the prompt cannot ground on
column meaning. Mitigation: probability-aware prompting — inject per-column mean/std/min/max/
quartiles and top-k pairwise correlations computed locally from the client's own real fraud rows,
plus a small number of few-shot examples.

Few-shot examples are synthetic prototype rows built FROM the aggregate statistics (typical /
low-tail / high-tail, using per-column median/25th/75th percentile) — never real transaction
rows. This was an explicit design call (D1 says both "include few-shot examples" and "never send
raw rows to the API"; literal real rows would violate the second). Confirmed with the user
2026-08-03; record kept in sync with PLAN.md.

A low Schema Mode validation pass rate is an expected, reportable finding, not a bug — this
module only generates *candidate* rows. Validation happens in ml/validation/ (Phase 3).
"""
import json
import random
import time
from dataclasses import dataclass

import pandas as pd
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    RateLimitError,
)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@dataclass
class LLMEngineConfig:
    model: str
    api_key: str
    top_k_correlations: int = 5
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    temperature: float = 0.9
    request_timeout: float = 60.0
    # Batched high: the stats/correlations block (~1.8k tokens) is resent on every request, so
    # small batches waste tokens fast — verified empirically: rows_per_request=10 burned ~97k of
    # a 100k Groq free-tier daily-token budget generating just 100 rows for one bank.
    rows_per_request: int = 50
    max_tokens_per_row: int = 150
    seed: int = 42


def compute_column_stats(fraud_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        s = fraud_rows[col]
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()) if len(s) > 1 else 0.0, 4),
            "min": round(float(s.min()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "q50": round(float(s.quantile(0.50)), 4),
            "q75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        }
    return stats


def compute_top_k_correlations(fraud_rows: pd.DataFrame, k: int) -> list[tuple[str, str, float]]:
    corr = fraud_rows[FEATURE_COLUMNS].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(FEATURE_COLUMNS):
        for col_b in FEATURE_COLUMNS[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value):
                pairs.append((col_a, col_b, round(float(value), 4)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:k]


def build_prototype_rows(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Statistically-derived few-shot examples — no real row is ever included (see module docstring)."""
    return {
        "typical": {col: s["q50"] for col, s in stats.items()},
        "low_tail": {col: s["q25"] for col, s in stats.items()},
        "high_tail": {col: s["q75"] for col, s in stats.items()},
    }


def build_prompt(
    bank_name: str,
    n_rows: int,
    stats: dict[str, dict[str, float]],
    correlations: list[tuple[str, str, float]],
    prototypes: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    system = (
        "You generate synthetic tabular data for a fraud-detection research pipeline. "
        "The columns are anonymized PCA components (V1-V28) plus Time and Amount, so they have "
        "no semantic meaning — treat this purely as a numeric distribution-matching task. "
        "Respond ONLY with a JSON object of the form {\"rows\": [{...}, ...]} containing exactly "
        f"{n_rows} rows, each with keys {FEATURE_COLUMNS}. No prose, no markdown fences."
    )
    user = (
        f"Simulated bank client '{bank_name}' holds very few real fraud transactions. "
        "Generate new synthetic fraud transactions that plausibly belong to the same "
        "distribution as this client's real fraud rows, described below only via aggregate "
        "statistics and derived prototypes — no real row is shown to you.\n\n"
        f"Per-column statistics (mean, std, min, 25th/50th/75th percentile, max):\n"
        f"{json.dumps(stats, indent=2)}\n\n"
        f"Top pairwise correlations observed among these columns (col_a, col_b, pearson_r):\n"
        f"{json.dumps(correlations, indent=2)}\n\n"
        f"Prototype example rows derived from the statistics above (not real transactions):\n"
        f"{json.dumps(prototypes, indent=2)}\n\n"
        f"Generate {n_rows} new rows that vary plausibly around these statistics and respect the "
        "given correlations. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_groq_with_retry(
    client: Groq, messages: list[dict[str, str]], config: LLMEngineConfig, max_completion_tokens: int
) -> str:
    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"},
                timeout=config.request_timeout,
                max_completion_tokens=max_completion_tokens,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError:
            raise  # bad key — retrying won't help
        except RETRYABLE_ERRORS as exc:
            last_error = exc
            sleep_s = config.backoff_base_seconds * (2**attempt) + random.uniform(0, 1)
            time.sleep(sleep_s)
    raise RuntimeError(f"Groq API call failed after {config.max_retries} retries: {last_error}") from last_error


def _parse_rows(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    valid_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not all(col in row for col in FEATURE_COLUMNS):
            continue
        try:
            valid_rows.append({col: float(row[col]) for col in FEATURE_COLUMNS})
        except (TypeError, ValueError):
            continue
    return valid_rows


def generate_synthetic_fraud_llm(
    fraud_rows: pd.DataFrame, n_rows: int, bank_name: str, config: LLMEngineConfig
) -> tuple[pd.DataFrame, dict]:
    """Returns (candidate_rows_df, generation_log). candidate_rows_df may have fewer than
    n_rows if the LLM returns malformed rows on some batches — that's logged, not hidden."""
    if fraud_rows.empty:
        raise ValueError("generate_synthetic_fraud_llm called with zero real fraud rows.")

    stats = compute_column_stats(fraud_rows)
    correlations = compute_top_k_correlations(fraud_rows, config.top_k_correlations)
    prototypes = build_prototype_rows(stats)

    client = Groq(api_key=config.api_key)
    all_rows: list[dict] = []
    n_requests = 0
    n_malformed_rows = 0
    remaining = n_rows

    while remaining > 0:
        batch_n = min(config.rows_per_request, remaining)
        messages = build_prompt(bank_name, batch_n, stats, correlations, prototypes)
        raw_text = _call_groq_with_retry(client, messages, config, batch_n * config.max_tokens_per_row)
        n_requests += 1
        parsed = _parse_rows(raw_text)
        n_malformed_rows += max(0, batch_n - len(parsed))
        all_rows.extend(parsed[:batch_n])
        remaining -= batch_n

    candidate_df = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)
    if not candidate_df.empty:
        candidate_df["Class"] = 1

    log = {
        "bank": bank_name,
        "n_real_fraud_used": len(fraud_rows),
        "n_requested": n_rows,
        "n_generated": len(candidate_df),
        "n_requests": n_requests,
        "n_malformed_rows_dropped": n_malformed_rows,
        "model": config.model,
    }
    return candidate_df, log
