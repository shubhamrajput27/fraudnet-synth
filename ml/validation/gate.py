"""Step 6: the shared validation gate. The SAME rules for Augment Mode (CTGAN) and Schema Mode (LLM).

For each bank, inside its own folder:
  candidates --> 1. Pandera schema/range/edge-clamp checks
             --> 2. privacy: numeric distance to closest real record (DCR)
             --> 3. embedding check (diagnostic; only essentially identical text is rejected)
             --> 4. diversity: drop near-duplicates of already-kept synthetic rows
             --> 5. fidelity (SDMetrics) of the admitted set; refuse the batch if below the minimum
             --> synthetic_validated.csv (stays in the bank folder)
Only aggregate counts and scores go to results/validation/.

Run from the project root:
    python -m ml.validation.gate
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.augmentation.ctgan_engine import load_bank_fraud
from ml.augmentation.llm_engine import patterned_decimal_pct
from ml.augmentation.schema_stats import local_bounds
from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config
from ml.validation import diversity, fidelity
from ml.validation.schema_checks import build_schema, edge_clamped_counts, schema_failures

# Outcome categories in the order checks run; a row's PRIMARY reason is its first failure.
OUTCOMES = ["passed", "schema_or_range", "edge_clamped", "too_close_to_real",
            "embedding_identical", "near_duplicate_synthetic", "batch_fidelity_too_low"]
LABELS = {"passed": "Passed", "schema_or_range": "Out of range / schema", "edge_clamped": "Edge-clamped (3+ values)",
          "too_close_to_real": "Too close to a real row", "embedding_identical": "Identical text embedding",
          "near_duplicate_synthetic": "Near-duplicate of a kept synthetic row",
          "batch_fidelity_too_low": "Batch fidelity below minimum"}
# Validated categorical palette, fixed order (dataviz reference slots 1-7).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _bucket(reason: str) -> str:
    return "edge_clamped" if reason == "edge_clamped" else "schema_or_range"


def validate_bank(bank: str, mode: str, cfg: dict, st_model) -> dict:
    bank_dir = PROJECT_ROOT / load_config("partition")["clients_dir"] / bank
    real = load_bank_fraud(bank_dir)  # the bank's own training fraud rows only
    cand = pd.read_csv(bank_dir / "synthetic_candidates.csv").reset_index(drop=True)
    n = len(cand)
    reasons: dict[int, list[str]] = {i: [] for i in range(n)}

    # 1) Schema, range and edge clamping (Pandera). Bounds are computed locally and never leave.
    bounds = local_bounds(real)
    for i, rs in schema_failures(cand, build_schema(bounds, cfg["schema"])).items():
        reasons[i].extend(rs)
    alive = np.array([not reasons[i] for i in range(n)])

    # 2) Privacy: DCR against the bank's real fraud, threshold from real-to-real spacing.
    r_std, s_std = diversity.standardise(real, cand)
    real_nn = diversity.real_nn_distances(r_std)
    threshold = float(np.percentile(real_nn, cfg["privacy"]["real_nn_percentile"]))
    dcr = diversity.dcr(s_std, r_std)
    for i in np.where(alive & (dcr < threshold))[0]:
        reasons[i].append("too_close_to_real")
    alive = np.array([not reasons[i] for i in range(n)])

    # 3) Embeddings (diagnostic): rows written as text, compared with real rows' text.
    dec = cfg["embedding"]["decimals_in_text"]
    e_real = diversity.embed(diversity.rows_as_text(real, dec), st_model)
    e_syn = diversity.embed(diversity.rows_as_text(cand, dec), st_model)
    cos_to_real = diversity.max_cosine(e_syn, e_real)
    for i in np.where(alive & (cos_to_real >= cfg["embedding"]["reject_cosine_at_or_above"]))[0]:
        reasons[i].append("embedding_identical")
    alive = np.array([not reasons[i] for i in range(n)])

    # 4) Diversity: among surviving rows, drop near-duplicates of rows already kept.
    idx = np.where(alive)[0]
    keep_mask = diversity.greedy_dedupe(s_std[idx], threshold) if len(idx) else np.array([], bool)
    for i in idx[~keep_mask]:
        reasons[i].append("near_duplicate_synthetic")
    alive = np.array([not reasons[i] for i in range(n)])

    # 5) Fidelity of the admitted set (SDMetrics). A set-level property, so it can only refuse the whole batch.
    admitted = cand[alive]
    fid_admitted = fidelity.quality(real, admitted)
    fid_candidates = fidelity.quality(real, cand)
    batch_ok = fid_admitted["overall"] is not None and fid_admitted["overall"] >= cfg["fidelity"]["min_overall_score"]
    if not batch_ok:
        for i in np.where(alive)[0]:
            reasons[i].append("batch_fidelity_too_low")
        admitted = cand.iloc[0:0]

    out = admitted[FEATURE_COLUMNS].copy()
    out[LABEL_COLUMN] = 1
    out.to_csv(bank_dir / "synthetic_validated.csv", index=False)  # stays inside the bank

    primary = {k: 0 for k in OUTCOMES}
    all_reasons: dict[str, int] = {}
    for rs in reasons.values():
        if not rs:
            primary["passed"] += 1
            continue
        first = rs[0]
        primary[first if first in OUTCOMES else _bucket(first)] += 1
        for r in rs:
            all_reasons[r] = all_reasons.get(r, 0) + 1

    syn_nn = diversity.real_nn_distances(s_std)  # synthetic-to-synthetic spacing, for the report
    e_rr = diversity.max_cosine(e_real, e_real, exclude_self=True)
    clamped = edge_clamped_counts(cand, bounds)
    return {
        "bank": bank, "mode": mode, "n_real_fraud": len(real), "n_candidates": n,
        "n_validated": len(admitted), "pass_rate_pct": round(100 * len(admitted) / n, 1),
        "primary_outcome_counts": primary, "all_reason_counts": dict(sorted(all_reasons.items())),
        "thresholds": {"dcr_privacy_and_diversity": round(threshold, 4),
                       "embedding_reject_cosine": cfg["embedding"]["reject_cosine_at_or_above"],
                       "fidelity_min_overall": cfg["fidelity"]["min_overall_score"]},
        "dcr_synth_to_real": {f"p{q}": round(float(np.percentile(dcr, q)), 3) for q in (1, 5, 50)},
        "nn_real_to_real": {f"p{q}": round(float(np.percentile(real_nn, q)), 3) for q in (1, 5, 50)},
        "nn_synth_to_synth_candidates": {f"p{q}": round(float(np.percentile(syn_nn, q)), 3) for q in (1, 5, 50)},
        "embedding_cosine": {"synth_to_real_median": round(float(np.median(cos_to_real)), 4),
                             "synth_to_real_max": round(float(cos_to_real.max()), 4),
                             "real_to_real_median": round(float(np.median(e_rr)), 4)},
        "fidelity_all_candidates": fid_candidates, "fidelity_validated": fid_admitted,
        "batch_fidelity_ok": bool(batch_ok),
        "edge_clamped_rows_pct": {"any": round(100 * float((clamped > 0).mean()), 1),
                                  "3_or_more": round(100 * float((clamped >= 3).mean()), 1)},
        "report_only_patterned_decimals_pct": {"candidates": patterned_decimal_pct(cand),
                                               "validated": patterned_decimal_pct(admitted) if len(admitted) else None,
                                               "real": patterned_decimal_pct(real)},
    }


def plot_outcomes(reports: list[dict], out):
    fig, ax = plt.subplots(figsize=(11, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    labels = [f"{r['bank'].replace('bank_', 'Bank ').title()} ({'CTGAN' if r['mode'] == 'augment' else 'LLM'}, "
              f"n={r['n_candidates']})" for r in reports]
    used = [k for k in OUTCOMES if any(r["primary_outcome_counts"][k] for r in reports)]
    left = np.zeros(len(reports))
    for k in used:
        vals = np.array([100 * r["primary_outcome_counts"][k] / r["n_candidates"] for r in reports])
        ax.barh(labels, vals, left=left, color=COLORS[OUTCOMES.index(k)], height=0.6,
                edgecolor=SURFACE, linewidth=2, label=LABELS[k])
        if k == "passed":
            for y, v in enumerate(vals):
                ax.text(v / 2, y, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of candidate rows (primary outcome)", color=INK_MUTED)
    ax.set_title("Validation gate outcome per bank", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer

    cfg = load_config("validation")
    results_dir = PROJECT_ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    try:  # prefer the local cache so the gate (and the viva demo) works without internet
        st_model = SentenceTransformer(cfg["embedding"]["model"], device="cpu", local_files_only=True)
    except OSError:  # first run on a new machine: download once
        st_model = SentenceTransformer(cfg["embedding"]["model"], device="cpu")
    banks_cfg = load_config("partition")["banks"]

    reports = []
    for bank, bcfg in banks_cfg.items():
        r = validate_bank(bank, bcfg["mode"], cfg, st_model)
        reports.append(r)
        with open(results_dir / f"{bank}_report.json", "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)

    rows = [{"bank": r["bank"], "mode": "Augment (CTGAN)" if r["mode"] == "augment" else "Schema (LLM)",
             "candidates": r["n_candidates"], "validated": r["n_validated"], "pass_%": r["pass_rate_pct"],
             **{LABELS[k]: r["primary_outcome_counts"][k] for k in OUTCOMES[1:]},
             "fidelity_all": r["fidelity_all_candidates"]["overall"],
             "fidelity_validated": r["fidelity_validated"]["overall"],
             "shapes_validated": r["fidelity_validated"]["column_shapes"],
             "pairs_validated": r["fidelity_validated"]["column_pair_trends"],
             "dcr_threshold": r["thresholds"]["dcr_privacy_and_diversity"]} for r in reports]
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "validation_summary.csv", index=False)
    plot_outcomes(reports, results_dir / "gate_outcomes.png")

    mode_tbl = table.groupby("mode")[["candidates", "validated"]].sum()
    mode_tbl["pass_%"] = (100 * mode_tbl["validated"] / mode_tbl["candidates"]).round(1)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.T.to_string(header=False))
        print("\nPass rate by mode:\n" + mode_tbl.to_string())
    for r in reports:
        print(f"\n[{r['bank']}] all rejection reasons (a row can have several): {r['all_reason_counts']}")
        print(f"[{r['bank']}] DCR synth->real {r['dcr_synth_to_real']} | real->real {r['nn_real_to_real']} | "
              f"synth->synth {r['nn_synth_to_synth_candidates']}")
        print(f"[{r['bank']}] embedding cosine {r['embedding_cosine']} | edge-clamped rows {r['edge_clamped_rows_pct']} | "
              f"patterned decimals (report only) {r['report_only_patterned_decimals_pct']}")


if __name__ == "__main__":
    main()
