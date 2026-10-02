# Decisions Log

Each entry records the decision, the alternatives considered, the reason and the date. Values marked as decisions are ours, not specifications from the project documents.

---

## D-000: Python 3.12 instead of 3.11
- **Date:** 2026-10-02 (Step 0)
- **Decision:** the project virtual environment uses **Python 3.12.10**.
- **Alternatives:** (a) install Python 3.11 to match CLAUDE.md exactly; (b) use the system default Python 3.13.7.
- **Reason:** 3.11 is not installed on the development laptop, but 3.12 is. All required libraries (Flower, SDV, SDMetrics, Pandera, PyTorch-CPU, sentence-transformers) installed and imported successfully on 3.12 (see Step 0 evidence). 3.13 was avoided because some ML packages have lagged in shipping 3.13 Windows wheels.
- **Approved by:** Shubham (2026-10-02).

## D-001: CPU-only PyTorch wheel
- **Date:** 2026-10-02 (Step 0)
- **Decision:** install PyTorch from the official CPU-only index (`https://download.pytorch.org/whl/cpu`).
- **Alternatives:** the default PyPI wheel.
- **Reason:** hard constraint #1 (CPU-only). The CPU wheel is also much smaller. PyTorch is needed by SDV's CTGAN and sentence-transformers, and possibly by our classifier (decided in Step 3).

## D-002: Use the existing local MongoDB 8.0 service
- **Date:** 2026-10-02 (Step 0)
- **Decision:** use the MongoDB Community Server 8.0.13 already running as a Windows service at `mongodb://localhost:27017`.
- **Reason:** zero cost, single machine, already installed.

## D-003: Global random seed = 42
- **Date:** 2026-10-02 (Step 1)
- **Decision:** a single base seed of 42 in `configs/data.yaml`, used for all randomness.
- **Alternatives:** any other fixed integer.
- **Reason:** the value itself is arbitrary. What matters is that it is fixed and recorded. Multi-seed repeats for the final results are decided in Step 9.

## D-004: EDA is descriptive only and runs on the full dataset
- **Date:** 2026-10-02 (Step 1)
- **Decision:** Step 1 statistics and plots use all 284,807 rows. Nothing computed here (e.g. the Cohen's d ranking) is used to select features or fit any transformation.
- **Alternatives:** run EDA only on the training portion after the Step 2 split.
- **Reason:** describing the raw dataset is standard and needed for the report. Leakage happens only if test information shapes the model, and nothing here does. All fitted components (scalers, generators, thresholds) are fitted on training/validation data only.

## D-005: Drop exact duplicate rows before splitting
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** `drop_duplicates: true`. Keep the first copy of each of the 1,081 duplicate rows, leaving 283,726 rows / 473 fraud.
- **Alternatives:** keep all rows but force copies into the same split (keeps 19 more fraud rows, but double-weights them); keep as-is (leakage risk).
- **Reason:** guarantees no identical transaction is in both train and test. Simple, standard, easy to explain.

## D-006: Global split 70 / 10 / 20 (train / val / test), stratified by Class
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** test is carved off first, then val, before any partitioning. Result: 331 / 47 / 95 fraud.
- **Alternatives:** 60/20/20 (fewer fraud rows for banks); 80/10/10 (only ~47 test fraud, so arm comparisons are too noisy).
- **Reason:** a test set of 95 fraud rows keeps the six-arm comparison reasonably stable while leaving 331 fraud rows for the banks.

## D-007: Non-IID partition by explicit per-class quotas
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** fraud shares A 45% / B 35% / C 12% / D 8%; genuine shares 35 / 30 / 20 / 15%, with largest-remainder rounding. Result: fraud 149 / 116 / 40 / 26.
- **Alternatives:** Dirichlet label skew (α e.g. 0.5), with the two largest draws assigned to A/B; more extreme quotas (50/38/8/4).
- **Reason:** deterministic, guarantees the rich (A, B) vs poor (C, D) contrast the dual-mode design needs, easy to justify. The shares themselves are our choice, not a project specification, and are configurable in `configs/partition.yaml`.

## D-008: Local per-bank split 70 / 15 / 15, stratified
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** each bank splits its own shard. Local fraud train/val/test: A 105/22/22, B 82/17/17, C 28/6/6, D 18/4/4.
- **Alternatives:** 60/20/20; 80/10/10.
- **Reason:** balance between training fraud rows and having any local val/test fraud at all for C and D. Known limitation: C/D local metrics are very noisy.
