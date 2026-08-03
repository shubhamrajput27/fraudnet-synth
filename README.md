# FraudNet-Synth

Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation.

Final-year B.E. CSE project (AY 2025–26), PES Institute of Technology & Management, Shivamogga — VTU, Belagavi.
Guide: Dr. Chethan L S. Team: Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096), Shubham Kumar Singh (4PM23CS101).

Four simulated banks holding non-IID shards of the ULB Credit Card Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive synthetic data augmentation — CTGAN for data-rich banks, LLM schema-and-few-shot generation for data-poor banks — gated by a shared fidelity/diversity/PII validation layer.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief (locked stack, architecture, privacy invariant, evaluation philosophy) and [`PLAN.md`](PLAN.md) for the phased implementation plan and progress log.

**Status:** All 8 phases complete. Headline finding: **federated_augmented** scores best (F1
0.6146), beating federated_real, both isolated arms, and both centralized arms — see
[`docs/final_report.md`](docs/final_report.md) for the full write-up (architecture, methodology,
results, honest comparison to literature, limitations) and
[`docs/phase7_results.md`](docs/phase7_results.md) for the results table and convergence plots
(regenerate via `ml/.venv/Scripts/python.exe -m experiments.generate_report`).
[`docs/presentation_deck.md`](docs/presentation_deck.md) has a slide-by-slide outline and
[`docs/demo_script.md`](docs/demo_script.md) a click-by-click live-demo checklist. `PLAN.md` has
the full dated progress log across all eight phases.

**Running it locally:**
```
# One command (Windows, PowerShell) — opens three labeled windows:
.\start_demo.ps1

# ...or manually, one per terminal:
ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000   # orchestrator
cd gateway && npm install && npm start                                     # gateway
cd dashboard && npm install && npm run dev                                 # dashboard
```
Then open the dashboard's printed local URL (default `http://localhost:5173`) and sign in with
the `ADMIN_USERNAME`/`ADMIN_PASSWORD` set in `.env`.

**Running the test suite:**
```
ml/.venv/Scripts/python.exe -m pytest      # 40 Python tests (ml/, orchestrator/)
cd gateway && npm test                      # 13 Node tests
```
