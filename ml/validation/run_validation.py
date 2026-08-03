"""CLI entrypoint: shared validation layer over each bank's synthetic candidate batch.

Usage:
    python -m ml.validation.run_validation

Exit criterion (PLAN.md Phase 3): validation report per synthetic batch; rejection stats logged.
Reads data/synthetic/bank_{X}_candidates.csv (Phase 2 output), writes validated rows to
data/validated/bank_{X}_validated.csv and a consolidated report to
data/validated/validation_report.json + docs/phase3_validation_report.md.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ml.common.config import CONFIG, REPO_ROOT
from ml.validation.thresholds import FROZEN_THRESHOLDS
from ml.validation.validate import validate_batch

BANK_MODE = {"A": "augment_ctgan", "B": "augment_ctgan", "C": "schema_llm", "D": "schema_llm"}


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _write_markdown_report(reports: list[dict], path: Path) -> None:
    lines = [
        "# Phase 3 validation report",
        "",
        f"Frozen thresholds (D6): fidelity_floor={FROZEN_THRESHOLDS.fidelity_floor}, "
        f"novelty_floor={FROZEN_THRESHOLDS.novelty_floor}, "
        f"mode_collapse_ceiling={FROZEN_THRESHOLDS.mode_collapse_ceiling}, "
        f"min_valid_rows={FROZEN_THRESHOLDS.min_valid_rows_for_distributional_checks}.",
        "",
        "| Bank | Mode | Candidates | Schema pass | Fidelity | Novelty | Mode-collapse (max) | Verdict | Validated rows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['bank']} | {r['mode']} | {r['n_candidates']} | {r['n_schema_pass']} | "
            f"{_fmt(r['fidelity_score'])} | {_fmt(r['novelty_score'])} | {_fmt(r['mode_collapse_max'])} | "
            f"{r['batch_verdict']} | {r['n_validated_rows']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", default=list(BANK_MODE.keys()))
    args = parser.parse_args()

    CONFIG.validated_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for bank in args.banks:
        candidates_path = CONFIG.synthetic_dir / f"bank_{bank}_candidates.csv"
        if not candidates_path.exists():
            raise FileNotFoundError(
                f"{candidates_path} not found. Run ml.augmentation.run_augmentation for bank {bank} first."
            )
        candidates = pd.read_csv(candidates_path)
        real_shard = pd.read_csv(CONFIG.shards_dir / f"bank_{bank}.csv")
        real_fraud = real_shard[real_shard["Class"] == 1].reset_index(drop=True)

        print(f"Bank {bank} ({BANK_MODE[bank]}): validating {len(candidates)} candidates ...")
        validated_rows, report = validate_batch(
            bank=bank,
            mode=BANK_MODE[bank],
            candidates=candidates,
            real_reference=real_shard,
            real_fraud=real_fraud,
        )
        print(f"  -> {report.batch_verdict}: {report.n_validated_rows}/{report.n_candidates} rows validated")

        validated_rows.to_csv(CONFIG.validated_dir / f"bank_{bank}_validated.csv", index=False)
        reports.append(report.to_dict())

    report_json_path = CONFIG.validated_dir / "validation_report.json"
    report_json_path.write_text(json.dumps(reports, indent=2))
    _write_markdown_report(reports, REPO_ROOT / "docs" / "phase3_validation_report.md")

    print(f"\nValidation report written to {report_json_path} and docs/phase3_validation_report.md")


if __name__ == "__main__":
    main()
