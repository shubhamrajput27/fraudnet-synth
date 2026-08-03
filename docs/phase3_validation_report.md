# Phase 3 validation report

Frozen thresholds (D6): fidelity_floor=0.5, novelty_floor=0.95, mode_collapse_ceiling=0.998, min_valid_rows=5.

| Bank | Mode | Candidates | Schema pass | Fidelity | Novelty | Mode-collapse (max) | Verdict | Validated rows |
|---|---|---|---|---|---|---|---|---|
| A | augment_ctgan | 310 | 310 | 0.6725 | 1.0000 | 0.9921 | PASS | 310 |
| B | augment_ctgan | 38 | 2 | None | None | None | REJECT_TOO_FEW_ROWS | 0 |
| C | schema_llm | 100 | 100 | 0.6660 | 1.0000 | 0.9999 | REJECT_MODE_COLLAPSE | 0 |
| D | schema_llm | 7 | 7 | 0.6308 | 1.0000 | 0.9877 | PASS | 7 |
