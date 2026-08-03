# CLAUDE.md — FraudNet-Synth persistent project memory

This file is auto-loaded every session. It must be sufficient, alone, for a fresh session with
no other context to work correctly on this project. If something is not in this file or in
`PLAN.md`, it is an open decision — flag it, do not invent it.

## Standing rules (override anything said later in impatience)

1. **Do not build ahead of the current phase.** If asked for something from a later phase, say
   so and name the dependency that is missing. See the phase table in `PLAN.md`.
2. **Do not silently substitute technologies.** The stack below is fixed by documents already
   submitted to the university. If a substitution seems genuinely necessary, stop and explain
   why — do not just swap it in.

## Project identity

- **Official title:** Privacy-Preserving Fraud Detection using Federated Learning and Synthetic
  Data Augmentation
- **System name:** FraudNet-Synth
- **Institution:** PES Institute of Technology & Management, Shivamogga — VTU, Belagavi. Dept. of
  CSE, AY 2025–26.
- **Guide:** Dr. Chethan L S
- **Team:** Palleti Pradeepa (4PM23CS070), Prachi Yadav (4PM23CS074), Sathvik D (4PM23CS096),
  Shubham Kumar Singh (4PM23CS101)
- **One-line description:** Four simulated banks holding non-IID shards of the ULB Credit Card
  Fraud dataset jointly train a fraud classifier via Flower/FedAvg, with client-adaptive
  synthetic data generation — CTGAN for data-rich banks (**Augment Mode**), LLM
  schema-and-few-shot generation for data-poor banks (**Schema Mode**) — gated by a shared
  fidelity/diversity/PII validation layer.
- **Research gap filled:** dual-mode, *client-adaptive* generation (statistical vs. LLM, selected
  per client by how much real fraud data that client holds), integrated inside a federated
  pipeline, gated by a shared validation layer, evaluated across a complete
  isolated/federated/centralized grid with and without augmentation.

## Locked technology stack — do not substitute

| Layer | Technology | Why locked |
|---|---|---|
| FL framework | Flower (`flwr`) | Best rated for research prototyping; CPU-feasible |
| Statistical generator | SDV / CTGAN (`sdv`'s `CTGANSynthesizer`) | Offline, zero-cost, CPU |
| LLM generator | Groq API (free tier) | Free-tier availability; Schema Mode only |
| Classifier | Small PyTorch MLP (D2) | Clean `state_dict()` averaging for FedAvg |
| Orchestration API | FastAPI | Async; unifies augmentation + FL orchestration |
| Gateway | Node.js + Express | Auth, run management, DB persistence |
| Frontend | React + Recharts (or Chart.js) | Mode indicators, live round charts, comparison grid |
| Validation | SDMetrics, sentence-transformers, Pandera | Fidelity, diversity/novelty, schema + PII/bias |
| Database | MongoDB | Flexible schema for run/round/client/arm metadata |

Exact pinned versions live in `ml/requirements.txt`, with a verified/unverified breakdown in its
header comment (see D7).

## Hard constraints — reject anything that violates these

- **CPU-only.** No CUDA, no GPU-dependent library paths, no model that needs one.
- **Zero infrastructure cost.** No paid APIs, no cloud deployment, no managed services beyond
  free tiers.
- **Single-machine simulation.** All four "banks" are processes/objects on one laptop.
- **Privacy invariant — the core of the project:** *no raw or synthetic data row ever crosses a
  client boundary — only serialized model weight updates.* Any design that moves rows between
  clients, or through the server, is a defect. Flag it immediately.

## Architecture — four tiers

```
TIER 1 · React Dashboard
  mode indicators (Augment / Schema / Real-only) · synthetic quality panel ·
  live FL round charts (WebSocket) · six-arm comparison grid ·
  "test a transaction" demo · run history · synthetic dataset export
        v  REST / WebSocket
TIER 2 · Node.js + Express API Gateway
  auth · run management · MongoDB persistence
        v  HTTP (internal)
TIER 3 · FastAPI Orchestration Service
  |- Data Augmentation Layer: CTGAN engine · Groq LLM engine · shared validation
  `- Federated Training Layer: Flower server (FedAvg) · round orchestration · metric logging
        v  weight updates only
TIER 4 · Simulated Bank Clients (private non-IID shards)
  Bank A (rich, CTGAN) · Bank B (rich, CTGAN) · Bank C (poor, LLM) · Bank D (poor, LLM)
```

**Pipeline order:** ingest -> non-IID partition -> local generation -> shared validation ->
baseline training -> federated training -> per-round evaluation & logging -> serve via
`/predict`.

## The six experimental arms — preserve this grid always

|  | Real only | Augmented |
|---|---|---|
| **Isolated** | each bank trains alone on its real shard (privacy-preserving lower bound) | each bank alone on real + validated synthetic |
| **Federated** | FedAvg across 4 banks, real data only | **full FraudNet-Synth pipeline — headline configuration** |
| **Centralized** | all data pooled (non-privacy-preserving upper bound) | pooled + augmented |

Every results table, chart, database schema, evaluation script, and dashboard view must
accommodate all six. All six share one training codebase (`ml/common/`) — never fork per-arm
training logic.

## Evaluation philosophy — non-negotiable

- **Augmentation is a hypothesis tested per client, never an assumed improvement.** Published
  work documents CTGAN augmentation *degrading* downstream fraud models, and no augmentation
  method is universally best.
- **A client shard where augmentation underperforms is a legitimate, reportable result.** Never
  hide, spin, smooth, or overclaim. Never tune thresholds after seeing results to make
  augmentation look better.
- **Fraud-class precision, recall, and F1 are the headline metrics — not accuracy.** On a
  dataset with ~0.17% positives, predicting "legitimate" for everything scores ~99.8% accuracy
  and is worthless. Push back if asked to lead with accuracy.
- **Reference points, not pass/fail bars:** literature reports ~91% federated vs. ~95%
  centralized accuracy on this benchmark; FedFraud reports F1 0.90 / AUC 0.96 under non-IID
  conditions. These frame expectations only — never tune toward them.

## Decisions made during implementation (D1–D11)

These are not in the original design documents; they are defaults adopted during Phase 0 and
recorded here (and in `PLAN.md`) so they can be defended or revised at review.

- **D1 — Schema Mode semantics.** ULB columns `V1`–`V28` are anonymized PCA components with no
  semantic meaning, so an LLM prompt cannot ground on "V14 is usually negative for fraud".
  Mitigation: **probability-aware prompting** — inject per-column mean/std/min/max/quartiles and
  top-k pairwise correlations computed locally from that client's own fraud rows, plus a small
  number of few-shot examples. Never send raw rows to the API. Let the validation layer judge the
  result honestly — **a low Schema Mode pass rate is a publishable finding, not a bug to hide.**
  Few-shot examples are synthetic prototype rows (per-column median / 25th-percentile /
  75th-percentile combinations) — **never real transaction rows** — resolving the apparent
  tension between "include few-shot examples" and "never send raw rows to the API". See D10.
- **D2 — Classifier: small PyTorch MLP**, not scikit-learn `LogisticRegression`. Architecture:
  input(30) -> 64 -> 32 -> 1, ReLU, dropout, `BCEWithLogitsLoss` with positive-class weighting.
  Chosen for real local epochs per FedAvg round and clean `state_dict()` <-> list-of-NumPy-arrays
  serialization. CPU-trivial at this data size.
- **D3 — Global held-out test set.** One stratified split carved out of the full dataset *before*
  partitioning. Never sharded, never augmented, never seen by any client during training. All six
  arms are evaluated on this identical set. Synthetic rows must never appear in any test set.
- **D4 — Feature scaling fitted per-client only.** Each client fits its own `StandardScaler` on
  its own local rows — fitting globally before partitioning would leak dataset-wide statistics
  into every client, violating the privacy invariant in spirit, and would also erase the
  realistic feature-scale drift that strengthens the non-IID simulation.
- **D5 — Non-IID partition scheme.** Seeded and reproducible (`--seed`, default 42). Dirichlet
  label skew combined with deliberate quantity skew. Target shape: Banks A/B data-rich, Banks
  C/D data-poor with only a handful of real fraud rows each — this scarcity is what makes Schema
  Mode necessary. Emits a shard statistics report (rows, fraud count, fraud rate, feature summary
  per bank). Exact counts are a Phase 1 deliverable requiring explicit approval before use.
- **D6 — Validation thresholds frozen before results are seen.** SDMetrics fidelity floor,
  novelty/nearest-neighbour distance floor, and mode-collapse ceiling are set in a config file
  during Phase 3, committed, and never changed after any arm has been evaluated. Rejection
  statistics per client per mode are logged — validation pass rate is itself a headline
  CTGAN-vs-LLM comparison.
- **D7 — Every dependency version is pinned exactly** in `ml/requirements.txt`, resolved via
  live PyPI queries (not guessed). The Flower API changed significantly across the 1.x line
  (`start_server`/`start_numpy_client` vs. `ServerApp`/`ClientApp`/`run_simulation`); the pinned
  version (1.32.1) is well past the old API. **Never write code against a Flower API that has
  not been confirmed against the actually-installed package.** See the requirements.txt header
  for what has and has not been verified.
- **D8 — MongoDB document model sketched at Phase 4, not Phase 5.** The Phase 4 metric logger
  emits the documents the Phase 5 API will serve. Collections: `runs`, `round_metrics`,
  `client_metrics`, `synthetic_batches`, `validation_reports`. Every metric document carries
  `run_id` and `arm` (one of the six).
- **D9 — Auth is minimal and scope-trimmable.** A single JWT-issuing login in the Express gateway
  is sufficient for a single-machine demo. If Phase 6 runs short on time, trim dashboard scope
  before pipeline scope.
- **D10 — Schema Mode few-shot examples are synthetic prototype rows, not real rows.** Confirmed
  with the user 2026-08-03 (Phase 2). Built from that client's own aggregate stats: a "typical"
  row (per-column median), a "low-tail" row (25th percentile), a "high-tail" row (75th
  percentile). No real transaction row is ever included in a Groq prompt. Also: Groq free-tier
  has a shared daily token budget (100k TPD as of 2026-08-03) across all requests for a model —
  batch generation requests to reuse the stats/correlations prompt overhead instead of resending
  it per small batch, or the budget disappears fast (see `ml/augmentation/llm_engine.py`
  `rows_per_request`).
- **D11 — Validation stack role assignment.** Confirmed with the user 2026-08-03 (Phase 3), since
  the locked stack names technologies but not their exact roles on purely numeric (non-text)
  data. Pandera: per-row structural/range checks. SDMetrics `QualityReport` (use the unified
  `sdmetrics.reports.QualityReport`, not the deprecated `sdmetrics.reports.single_table.
  QualityReport` — verified against installed `sdmetrics==0.28.2`): fidelity vs. that client's
  real fraud rows. SDMetrics `NewRowSynthesis`: novelty *and* the PII/leakage guard in one
  (verified: score drops when real rows are injected as synthetic duplicates). sentence-
  transformers (`all-MiniLM-L6-v2`): embeds each row as a canonical text string, pairwise cosine
  similarity *within* a synthetic batch measures mode collapse. Known limitation: this compresses
  into a narrow high range (~0.95-0.9999 observed) since a general-purpose text encoder mostly
  reflects the shared column-name template rather than fine numeric differences — still
  discriminates at the tail, not a precise instrument in absolute terms.

## Out of scope this semester (deferred to Future Scope)

If work drifts toward any of these, flag them as out of scope:

- Graph neural networks / FinGraphFL / federated graph learning
- Differential privacy (on CTGAN or on aggregation)
- Secure / cryptographic aggregation
- Robust or trust-aware aggregation against malicious clients
- IEEE-CIS dataset replication
- More than 4 clients

## Coding conventions

- Runnable, typed, commented-only-where-non-obvious code. No pseudocode.
- Small functions, modular files, reused abstractions. No duplicated training logic across the
  six arms — they share `ml/common/`.
- Every random operation seeded and reproducible (`RANDOM_SEED` from `.env`, default 42).
- Handle real failure modes: missing files, malformed rows, empty shards, Groq rate limits/API
  failures, MongoDB connection failures, CTGAN non-convergence on tiny fraud sets.
- Never fabricate a metric, benchmark figure, citation, or DOI. If unknown, say so and say to
  verify at the source.
- Update `PLAN.md`'s progress log at the end of each phase.
- When a phase exit criterion is met, say so explicitly and wait rather than rolling into the
  next phase.

## Repository layout

```
fraudnet-synth/
|- CLAUDE.md            this file
|- PLAN.md               living implementation plan, decisions, open questions, progress log
|- README.md
|- .gitignore
|- .env.example
|- data/                 gitignored: raw dataset, shards, synthetic batches
|- ml/
|  |- partition/         Phase 1
|  |- augmentation/      Phase 2: ctgan_engine, llm_engine
|  |- validation/        Phase 3: shared validation layer
|  |- federated/         Phase 4: flower server, client, model
|  |- baselines/         Phase 4: isolated + centralized arms
|  |- common/            model def, metrics, config, seeding
|  `- requirements.txt
|- orchestrator/         Phase 5: FastAPI
|- gateway/               Phase 5: Node/Express + MongoDB
|- dashboard/            Phase 6: React
|- experiments/          run configs, results, plots
`- docs/                 architecture notes, results annexure drafts
```
