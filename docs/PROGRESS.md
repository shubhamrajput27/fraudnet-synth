# Progress

| Step | Name | Phase | Status | Doc |
|---|---|---|---|---|
| 0 | Environment setup & project orientation | Prereq | ✅ Done (2026-10-02). Dataset placed | [STEP_00](steps/STEP_00_environment_setup.md) |
| 1 | Data ingestion & EDA | 1 | ✅ Done (2026-10-02) | [STEP_01](steps/STEP_01_data_eda.md) |
| 2 | Global split & non-IID partitioning | 1 | ✅ Done (2026-10-02) | [STEP_02](steps/STEP_02_split_partition.md) |
| 3 | Shared classifier & sanity baseline | 1→4 | ✅ Done (2026-10-05) | [STEP_03](steps/STEP_03_classifier_baseline.md) |
| 4 | Augment Mode: CTGAN (Banks A, B) | 2 | ✅ Done (2026-10-05) | [STEP_04](steps/STEP_04_ctgan_augment_mode.md) |
| 5 | Schema Mode: LLM (Banks C, D) | 2 | ✅ Done (2026-10-08). Privacy option (b): aggregate stats only | [STEP_05](steps/STEP_05_llm_schema_mode.md) |
| 6 | Shared validation layer | 3 | ✅ Done (2026-10-08) | [STEP_06](steps/STEP_06_validation_gate.md) |
| 7 | Isolated & centralized arms | 4 | Not started | |
| 8 | Federated arms (Flower) | 4 | Not started | |
| 8B | Multi-device federated demo (live demo only, D-020) | 4 → demo | Not started. Planned 2026-10-07 at guide's request | |
| 9 | Six-arm runner & results | 4→7 | Not started | |
| 10 | FastAPI orchestrator | 5 | Not started | |
| 11 | Express gateway + MongoDB | 5 | Not started | |
| 12 | React dashboard | 6 | Not started | |
| 13 | Integration & full evaluation | 7 | Not started | |
| 14 | Testing, docs, demo hardening | 8 | Not started | |
| E1 | q-FedAvg (optional extension) | — | Only after Step 14, if confirmed | |
| E2 | Conformal risk control (optional extension) | — | Only after Step 14, if confirmed | |
