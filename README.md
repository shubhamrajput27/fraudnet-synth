# FraudNet-Synth

Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation.

Final-year B.E. CSE project (AY 2025–26), PES Institute of Technology & Management, Shivamogga — VTU, Belagavi.
Guide: Dr. Chethan L S. Team: Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096), Shubham Kumar Singh (4PM23CS101).

Four simulated banks holding non-IID shards of the ULB Credit Card Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive synthetic data augmentation — CTGAN for data-rich banks, LLM schema-and-few-shot generation for data-poor banks — gated by a shared fidelity/diversity/PII validation layer.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief (locked stack, architecture, privacy invariant, evaluation philosophy) and [`PLAN.md`](PLAN.md) for the phased implementation plan and progress log.

**Status:** Phase 6 (React dashboard) complete. The full stack — login, trigger a run, watch it
train live via WebSocket, browse history, compare all six arms, inspect synthetic data quality,
export synthetic CSVs, and test a transaction against a trained model — is drivable end-to-end
from the browser. Browser-tested with Playwright against real running services (zero console
errors). See `PLAN.md`'s progress log for the full six-arm results table (Phase 4) and the
Phase 6 browser-test writeup.

**Running it locally:**
```
# Terminal 1 — orchestrator (from repo root)
ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000

# Terminal 2 — gateway
cd gateway && npm install && npm start

# Terminal 3 — dashboard
cd dashboard && npm install && npm run dev
```
Then open the dashboard's printed local URL (default `http://localhost:5173`) and sign in with
the `ADMIN_USERNAME`/`ADMIN_PASSWORD` set in `.env`.
