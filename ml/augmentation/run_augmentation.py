"""CLI entrypoint: client-adaptive synthetic generation (CLAUDE.md architecture).

Banks A/B (data-rich) -> CTGAN Augment Mode. Banks C/D (data-poor) -> Groq LLM Schema Mode.
Produces *candidate* rows only — nothing here is validated yet (Phase 3 does that). Exit
criterion (PLAN.md Phase 2): both modes produce candidate rows for their assigned clients.

Usage:
    python -m ml.augmentation.run_augmentation --seed 42

Augmentation volume is not specified anywhere in CLAUDE.md/PLAN.md, so the defaults below
(--ctgan-augment-ratio, --llm-target-rows) are proposed starting points, not locked decisions.
"""
import argparse
import json

import pandas as pd

from ml.common.config import CONFIG
from ml.common.seeding import set_seed
from ml.augmentation.ctgan_engine import CTGANEngine, CTGANEngineConfig
from ml.augmentation.llm_engine import LLMEngineConfig, generate_synthetic_fraud_llm

CTGAN_BANKS = ["A", "B"]
LLM_BANKS = ["C", "D"]


def _load_bank_fraud(bank: str) -> pd.DataFrame:
    df = pd.read_csv(CONFIG.shards_dir / f"bank_{bank}.csv")
    return df[df["Class"] == 1].reset_index(drop=True)


def run_ctgan_bank(bank: str, seed: int, augment_ratio: float) -> tuple[pd.DataFrame, dict]:
    fraud_rows = _load_bank_fraud(bank)
    n_target = max(1, round(len(fraud_rows) * augment_ratio))

    engine = CTGANEngine(CTGANEngineConfig(seed=seed))
    engine.fit(fraud_rows)
    candidates = engine.generate(n_target)

    log = {
        "bank": bank,
        "mode": "augment_ctgan",
        "n_real_fraud_used": len(fraud_rows),
        "n_requested": n_target,
        "n_generated": len(candidates),
    }
    return candidates, log


def run_llm_bank(bank: str, seed: int, target_rows: int) -> tuple[pd.DataFrame, dict]:
    if not CONFIG.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and fill in a Groq free-tier key "
            "(https://console.groq.com) before running Schema Mode banks."
        )
    fraud_rows = _load_bank_fraud(bank)
    llm_config = LLMEngineConfig(model=CONFIG.groq_model, api_key=CONFIG.groq_api_key, seed=seed)
    candidates, log = generate_synthetic_fraud_llm(fraud_rows, target_rows, bank, llm_config)
    log["mode"] = "schema_llm"
    return candidates, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--ctgan-augment-ratio", type=float, default=1.0,
                         help="Synthetic fraud rows generated per real fraud row, Banks A/B")
    parser.add_argument("--llm-target-rows", type=int, default=100,
                         help="Absolute synthetic fraud row target, Banks C/D (real counts are tiny by design)")
    parser.add_argument("--banks", nargs="+", default=CTGAN_BANKS + LLM_BANKS,
                         help="Subset of banks to run, e.g. --banks A B")
    args = parser.parse_args()

    set_seed(args.seed)
    CONFIG.synthetic_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for bank in args.banks:
        if bank in CTGAN_BANKS:
            print(f"Bank {bank}: Augment Mode (CTGAN) ...")
            candidates, log = run_ctgan_bank(bank, args.seed, args.ctgan_augment_ratio)
        elif bank in LLM_BANKS:
            print(f"Bank {bank}: Schema Mode (Groq LLM) ...")
            candidates, log = run_llm_bank(bank, args.seed, args.llm_target_rows)
        else:
            raise ValueError(f"Unknown bank '{bank}', expected one of {CTGAN_BANKS + LLM_BANKS}")

        out_path = CONFIG.synthetic_dir / f"bank_{bank}_candidates.csv"
        candidates.to_csv(out_path, index=False)
        print(f"  -> {log['n_generated']} candidate rows written to {out_path}")
        reports.append(log)

    report_path = CONFIG.synthetic_dir / "generation_report.json"
    report_path.write_text(json.dumps(reports, indent=2))
    print(f"\nGeneration report written to {report_path}")


if __name__ == "__main__":
    main()
