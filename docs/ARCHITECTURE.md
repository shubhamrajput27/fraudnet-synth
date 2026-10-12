# FraudNet-Synth: Architecture

Four tiers on one laptop (CPU only, zero cost). Users only ever talk to Tier 2; Tier 3 listens on localhost only; each bank in Tier 4 reads only its own folder.

```mermaid
flowchart TB
  subgraph T1["Tier 1 · React dashboard (frontend/, served by the gateway)"]
    UI["Overview · Six-arm results · Runs (live) · Test a transaction · Export · Audit"]
  end
  subgraph T2["Tier 2 · Node.js + Express gateway (services/gateway/, 127.0.0.1:8200)"]
    AUTH["JWT login · roles (admin / bank operator / viewer) · rate limit · audit"]
    WATCH["job watcher → MongoDB · WebSocket relay to browsers"]
    MDB[("MongoDB fraudnet_gateway<br/>users · runs · rounds · modelresults<br/>banks · validationreports · auditevents")]
  end
  subgraph T3["Tier 3 · FastAPI orchestrator (services/orchestrator/, 127.0.0.1:8100)"]
    API["REST + WebSocket · internal key · one background job at a time"]
    AUG["Data augmentation layer<br/>CTGAN (SDV) · Groq LLM · validation gate"]
    FL["Federated training layer<br/>Flower ServerApp · FedAvg (fixed order) · count-based threshold"]
    PRED["/predict · federated Arm 3 model"]
  end
  subgraph T4["Tier 4 · four simulated banks (data/clients/bank_x/)"]
    A["Bank A · Augment"] & B["Bank B · Augment"] & C["Bank C · Schema"] & D["Bank D · Schema"]
  end
  UI -- "HTTPS-style JSON + Bearer token; WebSocket /ws/runs/:job" --> AUTH
  AUTH --> API
  WATCH <-- "job events" --> API
  AUTH --- MDB
  WATCH --- MDB
  API --> AUG & FL & PRED
  FL <-- "weights + scalars only" --> A & B & C & D
  AUG -. "aggregate statistics only (Schema mode)" .-> GROQ(["Groq API (external)"])
```

## What crosses each boundary
| Boundary | What travels | What never travels |
|---|---|---|
| Browser ↔ gateway | JSON, JWT (8 h), live round events | the internal key, any transaction row (except a bank operator exporting **their own** synthetic rows, D-045) |
| Gateway ↔ FastAPI | requests + `X-Internal-Key`; `X-Bank-Scope` taken from the token | user passwords |
| Bank ↔ federated server | 4,161 weights (17,478 B/round), row count, scalar metrics; after training 241 TP/FP count pairs and scalar local-test metrics | rows, per-transaction scores (checked by `tests/test_privacy_invariant.py`) |
| Schema-mode bank ↔ Groq | schema + mean, std, p10/p50/p90 per column (D-015/D-016) | real rows, min/max |
| Anything ↔ MongoDB | metrics, run metadata, audit events, bcrypt password hashes | transaction rows |

## Main flows
1. **Start a run:** dashboard → `POST /api/runs` (admin only) → `POST /runs` → job process runs Arms 1–6 (`ml.baselines.arms`, `ml.federated.run`) → events → gateway watcher → MongoDB + browsers.
2. **Generation:** `POST /api/generation` → replay (CTGAN retrained with its seed, LLM rows from cache, 0 API calls) or versioned regenerate → validation gate → `generations/` inside the bank folder.
3. **Predict:** dashboard form → gateway → `/predict` → fixed feature transforms → MLP logit vs threshold 3.75.
4. **Report:** `python -m ml.experiments.run_all_arms` (or a dashboard run) → `python -m ml.evaluation.annexure` → `docs/RESULTS_ANNEXURE.md`.

## Ports and processes
| Port | Process | Exposed to |
|---|---|---|
| 8200 | gateway + built dashboard | this laptop (browser) |
| 8100 | FastAPI orchestrator | localhost only (gateway) |
| 27017 | MongoDB (Windows service) | localhost |
| 9092 / 9093 | Flower SuperLink fleet / control (Step 8B demo only) | hotspot LAN / localhost |

## Key code
| Area | Path |
|---|---|
| Data, split, partition | `ml/data/` |
| Classifier, features, training | `ml/models/` (`FraudMLP` 31→64→32→1) |
| CTGAN / LLM generation | `ml/augmentation/ctgan_engine.py`, `llm_engine.py`, `schema_stats.py` |
| Validation gate | `ml/validation/` |
| Isolated + centralized arms | `ml/baselines/arms.py` |
| Federated (simulation + deployment) | `ml/federated/` (`client_app.py`, `server_app.py`, `run.py`, `demo/`) |
| Six-arm runner, report, annexure | `ml/experiments/run_all_arms.py`, `ml/evaluation/` |
| Services | `services/orchestrator/`, `services/gateway/` |
| Dashboard | `frontend/src/` |
| Offline demo | `demo/` (`preflight.py`, `start.py`, `viva_sheet.py`) |

All design choices and their reasons are in `docs/DECISIONS.md` (D-000 onwards).
