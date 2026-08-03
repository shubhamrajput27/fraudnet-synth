# FraudNet-Synth

Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation.

Final-year B.E. CSE project (AY 2025–26), PES Institute of Technology & Management, Shivamogga — VTU, Belagavi.
Guide: Dr. Chethan L S. Team: Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096), Shubham Kumar Singh (4PM23CS101).

Four simulated banks holding non-IID shards of the ULB Credit Card Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive synthetic data augmentation — CTGAN for data-rich banks, LLM schema-and-few-shot generation for data-poor banks — gated by a shared fidelity/diversity/PII validation layer.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief (locked stack, architecture, privacy invariant, evaluation philosophy) and [`PLAN.md`](PLAN.md) for the phased implementation plan and progress log.

**Status:** Phase 7 (integration & evaluation) complete. Final six-arm results table and
convergence plots are in [`docs/phase7_results.md`](docs/phase7_results.md) (regenerate via
`ml/.venv/Scripts/python.exe -m experiments.generate_report`). Headline finding:
**federated_augmented** scores best (F1 0.6146), beating federated_real, both isolated arms, and
both centralized arms. A fresh re-run reproduced Phase 4's numbers bit-for-bit, confirming the
whole pipeline is deterministic under its fixed seed. See `PLAN.md`'s progress log for the full
write-up, including an honest comparison to literature reference points.

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
