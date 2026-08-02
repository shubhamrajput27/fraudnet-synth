# ml/validation/

Phase 3. Shared validation layer gating every synthetic batch regardless of which engine produced it: SDMetrics fidelity, diversity/novelty (sentence-transformers nearest-neighbour distance), Pandera schema checks, and PII/bias screening. Thresholds are frozen in a committed config before any arm is evaluated (D6) — never adjusted after seeing results.
