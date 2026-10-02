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
