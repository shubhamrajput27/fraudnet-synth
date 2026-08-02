# PLAN.md — FraudNet-Synth living implementation plan

Companion to `CLAUDE.md` (persistent project memory). This file tracks the phase plan,
implementation-time decisions, open questions, and a dated progress log. Update the progress log
at the end of every phase; do not roll into the next phase without stating the exit criterion is
met and waiting for review.

## Phase plan (8 phases, ~14 weeks)

| Phase | Work package | Weeks | Exit criterion |
|---|---|---|---|
| 1 | Data pipeline: ingestion, cleaning, global test holdout, non-IID partition into 4 shards | 1–2 | Reproducible partition script; shard statistics documented |
| 2 | Augmentation modules: CTGAN per rich client; Groq prompt design + generation client | 3–5 | Both modes produce candidate rows for their assigned clients |
| 3 | Shared validation layer: SDMetrics fidelity, diversity/novelty, Pandera schema + PII/bias, thresholding | 5–6 | Validation report per synthetic batch; rejection stats logged |
| 4 | Federated training: Flower server + 4 clients, FedAvg, per-round metrics; isolated + centralized baselines | 6–9 | All six arms runnable end-to-end from CLI |
| 5 | Backend: FastAPI orchestration endpoints; Express gateway with auth, run management, MongoDB | 8–10 | A full run can be triggered and persisted via API alone |
| 6 | Dashboard: React UI — mode indicators, quality panel, live charts, comparison grid, prediction demo, history, export | 9–12 | Dashboard drives and displays a complete live run |
| 7 | Integration & evaluation: full runs across all arms, results collection, comparison to literature | 12–13 | Final results tables and convergence plots produced |
| 8 | Testing, documentation, presentation: hardening, report finalization, demo rehearsal | 13–14 | Submission-ready report, working demo, deck |

Phases 5 and 6 depend on Phase 4's metric-logging shape (D8) being settled first — the Phase 4
logger should emit documents the Phase 5 API can serve directly, not be redesigned later.

## Decisions made during implementation (not in the original design documents)

All nine are inference made during Phase 0 scaffolding, adopted as defaults and recorded here so
they can be defended or revised at review. Full rationale for each also lives in `CLAUDE.md`
(kept in sync — if one changes, update both).

- **D1 — Schema Mode probability-aware prompting.** V1–V28 are anonymized PCA components with no
  semantic meaning; inject per-client, per-column mean/std/min/max/quartiles + top-k pairwise
  correlations computed locally, plus a few-shot sample, into the Groq prompt. Never send raw
  rows to the API. A low Schema Mode validation pass rate is an expected, reportable finding.
- **D2 — Classifier is a small PyTorch MLP**, not scikit-learn `LogisticRegression`. Architecture
  input(30) -> 64 -> 32 -> 1, ReLU, dropout, `BCEWithLogitsLoss` with positive-class weighting.
  Rationale: real local epochs per FedAvg round, clean `state_dict()` serialization,
  round-by-round convergence curves.
- **D3 — One global stratified test holdout**, carved out before partitioning, never sharded,
  never augmented, identical across all six arms.
- **D4 — Per-client `StandardScaler`** fitted independently by each client on its own local rows
  only — never fit globally before partitioning.
- **D5 — Non-IID partitioning via Dirichlet label skew + deliberate quantity skew**, seeded
  (`--seed`, default 42). Banks A/B rich, Banks C/D poor (only a handful of real fraud rows each).
  Exact shard counts are a **Phase 1 deliverable requiring explicit approval** before being used
  anywhere — none are set yet.
- **D6 — Validation thresholds frozen before any arm is evaluated**, committed to a config file
  in Phase 3, never adjusted afterward. Rejection stats per client per mode are logged as a
  headline result in their own right.
- **D7 — Every dependency pinned exactly**, resolved 2026-08-03 via live PyPI queries. See
  `ml/requirements.txt` header for the full verified/unverified breakdown. Key resolved facts:
  - `pandas` pinned to `2.3.3`, deliberately below the current PyPI latest (`3.0.5`), because
    both `sdv==1.37.4` and `sdmetrics==0.28.2` declare `pandas<3.0.0` in their published
    metadata — pinning pandas 3.x would make the requirements file unresolvable. Verified by
    reading `requires_dist` from PyPI's JSON API, not assumed.
  - `sdv` transitively provides `ctgan`, `sdmetrics`, `rdt`, `copulas`, `deepecho` — no separate
    `ctgan` pin needed.
  - `torch==2.13.0` has a confirmed CPU-only wheel (`+cpu`) for `cp313-win_amd64`; install with
    `--index-url https://download.pytorch.org/whl/cpu` to avoid pulling a CUDA build.
  - Full N-way dependency resolution has **not** been run (no install performed this session per
    the "do not install without asking" constraint). First action of Phase 1 must be
    `pip install -r ml/requirements.txt` in a clean venv, resolving any conflict pip's resolver
    surfaces before writing code against these exact versions.
  - Flower 1.32.1 is well past the deprecated `start_server`/`start_numpy_client` API. Phase 4
    code must target `ServerApp`/`ClientApp`/`run_simulation` (or the `flwr run` CLI +
    `pyproject.toml` app layout) — but this must be confirmed against the actually-installed
    package's module contents at Phase 4 kickoff, not assumed from this note.
- **D8 — MongoDB document model sketched at Phase 4.** Collections: `runs`, `round_metrics`,
  `client_metrics`, `synthetic_batches`, `validation_reports`, each metric document carrying
  `run_id` and `arm`.
- **D9 — Minimal single-login JWT auth** in the Express gateway; trim dashboard scope before
  pipeline scope if Phase 6 runs short.

## Open questions

Flagged, not resolved. Do not guess these — surface them for explicit answers when the relevant
phase starts.

- ~~Exact shard sizes and fraud-row counts per bank (D5)~~ — **resolved 2026-08-03**, see progress
  log. Approved: Bank A 79,621 rows/310 fraud, Bank B 79,349 rows/38 fraud, Bank C 34,005 rows/15
  fraud, Bank D 34,005 rows/15 fraud, global holdout 56,746 rows/95 fraud. Banks C/D are scarce in
  fraud rows only (capped at 15 each), not in overall volume — noted and accepted as-is.
- **Validation thresholds (D6)** — SDMetrics fidelity floor, novelty/nearest-neighbour distance
  floor, mode-collapse ceiling: no numeric values exist yet. Set and frozen in Phase 3.
- **CTGAN epoch budget / convergence behavior on Banks C/D's tiny real fraud counts** — CTGAN is
  known to struggle on very small training sets; whether it's even viable there vs. relying
  entirely on Schema Mode is an empirical Phase 2 question, not assumed either way.
- **Full transitive dependency resolution for `ml/requirements.txt`** — only pairwise
  pandas-vs-(sdv, sdmetrics) conflicts were checked (D7). Must be verified with a real
  `pip install` at Phase 1 kickoff.
- **Groq free-tier rate limits** at the time of actual Phase 2 development — not looked up this
  session; must be checked against Groq's current published limits before designing the LLM
  client's retry/backoff behavior.
- **React chart library: Recharts vs Chart.js** — design documents list both as acceptable; not
  decided. Defer to Phase 6 kickoff.

## Progress log

- **2026-08-03 — Phase 0 complete.** Repository scaffold created at `fraudnet-synth/` (root:
  `C:\Users\shubh\fraudnet-synth`, git-initialized, no commits made): `CLAUDE.md`, `PLAN.md`,
  root `README.md`, `.gitignore`, `.env.example`, empty directory tree with one-line `README.md`
  per directory (`data/`, `ml/{partition,augmentation,validation,federated,baselines,common}/`,
  `orchestrator/`, `gateway/`, `dashboard/`, `experiments/`, `docs/`), and `ml/requirements.txt`
  with exact pinned versions resolved live against PyPI (see D7). No implementation code,
  dataset, or installed packages. Awaiting review before Phase 1 begins.

- **2026-08-03 — Phase 1 complete.** `ml/.venv` (Python 3.13.7) created; full
  `ml/requirements.txt` installed with zero resolver conflicts (torch 2.13.0+cpu confirmed via
  the PyTorch CPU wheel index, `torch.cuda.is_available()` verified `False`). User supplied
  `data/raw/creditcard.csv` (ULB dataset, Kaggle manual download). Built `ml/common/{config,
  seeding}.py` and `ml/partition/{ingest,holdout,partition,stats,run_partition}.py`:
  ingestion + dedup/schema/null validation -> global stratified 80/20 test holdout (D3) ->
  non-IID partition (Dirichlet fraud-row skew, `alpha=0.3`, seed 42, deliberate legit-row
  quantity skew A/B 35%/35% vs C/D 15%/15%, poor-bank fraud hard-capped at 15 rows each,
  excess redistributed to A/B) -> shard statistics report (`data/processed/shard_stats.json`,
  `docs/phase1_shard_stats.md`). Raw file had 284,807 rows / 492 fraud; after dropping 1,081
  duplicate rows (19 of them fraud, a documented quirk of this Kaggle release) the clean pool is
  283,726 rows / 473 fraud. Resulting shards, run and approved by the user 2026-08-03: Bank A
  79,621 rows/310 fraud, Bank B 79,349 rows/38 fraud, Bank C 34,005 rows/15 fraud, Bank D 34,005
  rows/15 fraud, global holdout 56,746 rows/95 fraud (D5 open question resolved). Exit criterion
  met: reproducible partition script (`python -m ml.partition.run_partition --seed 42`), shard
  statistics documented. Awaiting review before Phase 2 (augmentation modules) begins.
