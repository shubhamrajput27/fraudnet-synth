# FraudNet-Synth

Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation.

Final-year B.E. CSE project (AY 2025–26), PES Institute of Technology & Management, Shivamogga — VTU, Belagavi.
Guide: Dr. Chethan L S. Team: Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096), Shubham Kumar Singh (4PM23CS101).

Four simulated banks holding non-IID shards of the ULB Credit Card Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive synthetic data augmentation — CTGAN for data-rich banks, LLM schema-and-few-shot generation for data-poor banks — gated by a shared fidelity/diversity/PII validation layer.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief (locked stack, architecture, privacy invariant, evaluation philosophy) and [`PLAN.md`](PLAN.md) for the phased implementation plan and progress log.

**Status:** Phase 5 (FastAPI orchestrator + Express gateway + MongoDB) complete. A full training
run can be triggered and persisted end-to-end via API alone — `POST /auth/login`, `POST
/api/runs`, `GET /api/runs/:id` — verified against a real local MongoDB instance. See `PLAN.md`'s
progress log for the full six-arm results table (Phase 4) and the Phase 5 API test.

**Running it locally:**
```
# Terminal 1 — orchestrator (from repo root)
ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000

# Terminal 2 — gateway
cd gateway && npm install && npm start
```
