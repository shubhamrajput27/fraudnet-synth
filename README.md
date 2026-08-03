# FraudNet-Synth

Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation.

Final-year B.E. CSE project (AY 2025–26), PES Institute of Technology & Management, Shivamogga — VTU, Belagavi.
Guide: Dr. Chethan L S. Team: Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096), Shubham Kumar Singh (4PM23CS101).

Four simulated banks holding non-IID shards of the ULB Credit Card Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive synthetic data augmentation — CTGAN for data-rich banks, LLM schema-and-few-shot generation for data-poor banks — gated by a shared fidelity/diversity/PII validation layer.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief (locked stack, architecture, privacy invariant, evaluation philosophy) and [`PLAN.md`](PLAN.md) for the phased implementation plan and progress log.

**Status:** Phase 3 (shared validation layer) complete. Schema, fidelity, novelty, and
mode-collapse checks run against real Phase 2 output; thresholds frozen per D6. Validation
correctly rejected 2 of 4 banks' synthetic batches (Bank B: CTGAN failure on tiny data; Bank C:
mode collapse) — see `docs/phase3_validation_report.md` and `PLAN.md`'s progress log.
