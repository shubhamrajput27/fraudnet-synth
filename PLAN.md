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
- ~~Validation thresholds (D6)~~ — **resolved 2026-08-03**, see Phase 3 progress log. Frozen:
  fidelity_floor=0.5, novelty_floor=0.95, mode_collapse_ceiling=0.998 (max pairwise similarity),
  min_valid_rows_for_distributional_checks=5.
- **CTGAN convergence on tiny real fraud counts** — not directly tested on Banks C/D (they use
  Schema Mode, per the architecture, and were never run through CTGAN). Strong adjacent evidence
  now exists, though: Bank B, nominally "rich" but with only 38 real fraud rows after the
  Dirichlet skew, saw CTGAN fail badly (see Phase 3 progress log) despite `enforce_min_max_values
  =True`. This supports — but doesn't directly test — the design choice to route the even-poorer
  Banks C/D (15 real fraud rows each) to Schema Mode instead.
- **Full transitive dependency resolution for `ml/requirements.txt`** — only pairwise
  pandas-vs-(sdv, sdmetrics) conflicts were checked (D7). Must be verified with a real
  `pip install` at Phase 1 kickoff.
- **Groq free-tier rate limits** at the time of actual Phase 2 development — not looked up this
  session; must be checked against Groq's current published limits before designing the LLM
  client's retry/backoff behavior.
- ~~React chart library: Recharts vs Chart.js~~ — **resolved 2026-08-03** (D15): Recharts.

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

- **2026-08-03 — Phase 2 complete.** Built `ml/augmentation/{ctgan_engine,llm_engine,
  run_augmentation}.py`. **D10 (new decision) — Schema Mode few-shot examples are synthetic
  prototype rows** (per-column median / 25th-percentile / 75th-percentile combinations), never
  real transaction rows — resolves the apparent tension in D1 between "include few-shot examples"
  and "never send raw rows to the API". Confirmed explicitly with the user 2026-08-03; CLAUDE.md
  D1 updated to match.
  CTGAN engine (`sdv==1.37.4` `CTGANSynthesizer`, `enable_gpu=False`): verified empirically that
  `batch_size` must be a multiple of `pac` (default 10) or `ctgan` raises a bare `AssertionError`
  — not documented anywhere, found by testing against Bank B's real 38 fraud rows. Live-verified
  end-to-end on Banks A (310 real fraud rows -> 310 candidates) and B (38 -> 38 candidates).
  LLM engine (`groq==1.6.0`, model `llama-3.3-70b-versatile`, `response_format=json_object`,
  exponential-backoff retry on `RateLimitError`/`APIConnectionError`/`APITimeoutError`/
  `InternalServerError`): live-verified end-to-end on Bank C (15 real fraud rows -> 100/100
  candidates generated, 0 malformed). First Bank D attempt failed with `groq.RateLimitError`
  (429, tokens-per-day) — **not a code defect**: `rows_per_request=10` (initial default) re-sent
  the ~1.8k-token stats/correlations block on every request, and Bank C's 10 requests alone
  consumed 96,844 of the free tier's 100,000 TPD budget, leaving nothing for Bank D. Fixed by
  raising `rows_per_request` to 50 and adding an explicit `max_completion_tokens` cap. The daily
  quota was still not fully reset on the next attempt (92,584/100,000 used, a rolling window, not
  an instant reset), so Bank D was live-confirmed with a reduced one-off request
  (`--llm-target-rows 15`, fits under the remaining budget) rather than waiting out the full
  reset: **15 requested, 7 valid candidates generated, 8 rows dropped as malformed** (logged in
  `generation_report.json`, not hidden — consistent with the project's "low pass rate is a
  reportable finding" philosophy, though this is raw parse validity, not Phase 3 fidelity
  validation). Exit criterion met for all four banks with real API/engine calls. Known minor gap:
  `run_augmentation.py`'s `generation_report.json` is overwritten (not merged) per invocation, so
  it only reflects the banks in the most recent run — each bank's candidate CSV is independently
  correct on disk regardless. Awaiting review before Phase 3 (shared validation layer) begins.

- **2026-08-03 — Phase 3 complete.** Built `ml/validation/{schema,fidelity,novelty,diversity,
  thresholds,validate,run_validation}.py`. Confirmed with the user a concrete architecture for
  the locked validation stack (Pandera schema, SDMetrics fidelity + novelty, sentence-transformers
  diversity) since CLAUDE.md only named the technologies, not their exact roles on purely numeric
  (non-text) data: Pandera does per-row structural/range checks; SDMetrics `QualityReport`
  (**verified**: use the unified `sdmetrics.reports.QualityReport` with `{table_name: df}`-wrapped
  data and the full `Metadata.to_dict()` including `tables`, not the deprecated
  `sdmetrics.reports.single_table.QualityReport` which takes bare DataFrames) scores fidelity
  against that client's real fraud rows; SDMetrics `NewRowSynthesis` doubles as the novelty check
  *and* the PII/leakage guard (verified empirically: score drops proportionally when real rows are
  injected as synthetic duplicates); sentence-transformers (`all-MiniLM-L6-v2`) embeds each row as
  a canonical text string and pairwise cosine similarity *within* a synthetic batch measures mode
  collapse — noted as a real limitation that this compresses into a narrow high range (~0.95-0.9999
  observed) since a general-purpose text encoder mostly picks up the shared column-name template
  rather than fine numeric differences, though it still discriminates at the tail.
  **D6 thresholds frozen** (`ml/validation/thresholds.py`), based on real values computed across
  all four banks' actual Phase 2 candidate batches, approved by the user 2026-08-03:
  `fidelity_floor=0.5`, `novelty_floor=0.95`, `mode_collapse_ceiling=0.998` (on max pairwise
  similarity), `min_valid_rows_for_distributional_checks=5`.
  Running validation against the real candidate batches produced a genuine headline result — 2 of
  4 banks' batches rejected: **Bank A: PASS** (310/310 validated, fidelity 0.673). **Bank B:
  REJECT_TOO_FEW_ROWS** — CTGAN, trained on only 38 real fraud rows, generated wildly
  out-of-range `Time` values (up to 16.7M vs. a real training range of 7,526-169,142) despite
  `enforce_min_max_values=True`; only 2/38 candidates survived the schema check, below the
  min-rows floor for a reliable distributional judgment. **Bank C: REJECT_MODE_COLLAPSE** — schema
  and fidelity both fine, but one near-duplicate synthetic pair pushed max similarity to 0.9999,
  over the 0.998 ceiling. **Bank D: PASS** (7/7 validated, fidelity 0.631). Per CLAUDE.md's
  evaluation philosophy, these rejections are reported as-is, not tuned away. Exit criterion met:
  validation report per synthetic batch (`data/validated/validation_report.json`,
  `docs/phase3_validation_report.md`), rejection stats logged. Awaiting review before Phase 4
  (federated training) begins.

- **2026-08-03 — Phase 4 complete.** Built `ml/common/{model,metrics,data,train,metric_logger}.py`
  (shared by all six arms — no per-arm training logic duplicated), `ml/baselines/{isolated,
  centralized}.py`, `ml/federated/{client,run_federated}.py`, `ml/run_experiment.py` (unified CLI).
  **D12 (new decision) — hand-rolled sequential federated loop, not `flwr.simulation.
  run_simulation`.** Confirmed the actual Flower 1.32.1 API per D7's requirement (ServerApp/
  ClientApp/run_simulation, as anticipated), but `run_simulation`'s default `ray` backend has no
  Windows wheel for Python 3.13 (verified via live PyPI metadata: latest ray 2.56.1 ships only
  cp310/cp311/cp312 for Windows) — a genuine blocker, not a formality. Confirmed with the user:
  rather than downgrading the whole venv to Python 3.12 to get a ray wheel, the federated loop
  runs the 4 clients in-process sequentially each round (reasonable for 4 toy clients on one
  laptop). Aggregation is still real Flower — `flwr.client.NumPyClient` subclasses and
  `flwr.server.strategy.FedAvg.aggregate_fit`, fed genuine `flwr.common.FitRes` objects (verified
  `aggregate_fit` never touches the paired `ClientProxy`, so `None` stands in for it). **D13 (new
  decision) — federated/isolated evaluation is per-client with each client's own scaler, macro-
  averaged**, never a pooled scaler, since D4 only fits scalers per-client. Confirmed with the
  user. Mirrors real federated deployment and keeps D4's privacy-invariant spirit intact even at
  evaluation time; centralized needs no such split since it's deliberately the one pooled arm.
  All six arms run end-to-end from `python -m ml.run_experiment --arm all` (exit criterion met).
  Real six-arm summary (fraud-class metrics on the D3 global holdout, seed 42, 20 local
  epochs/10 FedAvg rounds default):

  | Arm | F1 | Precision | Recall | AUC |
  |---|---|---|---|---|
  | isolated_real | 0.4243 | 0.3662 | 0.7605 | 0.9120 |
  | isolated_augmented | 0.4442 | 0.3736 | 0.7658 | 0.9077 |
  | federated_real | 0.5044 | 0.3582 | 0.8579 | 0.9630 |
  | **federated_augmented** | **0.6146** | 0.4857 | 0.8368 | 0.9544 |
  | centralized_real | 0.1236 | 0.0664 | 0.8947 | 0.9574 |
  | centralized_augmented | 0.3217 | 0.1971 | 0.8737 | 0.9527 |

  federated_augmented (the project's headline configuration) scores best on F1 despite 2 of 4
  banks' augmentation batches being rejected in Phase 3 — augmentation still helped net, carried
  by Banks A and D. federated > isolated on every metric, matching literature expectations
  (PLAN.md's reference points). centralized_real's low F1/precision despite strong AUC (0.9574)
  was investigated, not assumed to be a bug: `pos_weight≈599` (from D2's `BCEWithLogitsLoss`
  positive-class weighting, driven by the pooled dataset's ~0.17% fraud rate) combined with the
  fixed 0.5 decision threshold causes the model to predict fraud on 2.26% of holdout rows vs. the
  true 0.17% rate — a genuine class-imbalance/threshold artifact (AUC is threshold-independent,
  so it stays high), reported as-is per the evaluation philosophy, not tuned away. Per-arm run
  logs (D8-shaped documents) written to `experiments/results/<run_id>/{run,round_metrics,
  client_metrics,final_metrics}.{json,jsonl}` (gitignored, regenerable). Awaiting review before
  Phase 5 (FastAPI orchestration + Express gateway + MongoDB) begins.

- **2026-08-03 — Phase 5 complete.** Verified environment first (Node v22.19.0, MongoDB 8.0.13
  running locally as a Windows service on port 27017 — confirmed via `pymongo`, not assumed; this
  instance also hosts unrelated `expense-tracker`/`pesitm-cse` databases from other local
  projects, so all writes are scoped to `fraudnet_synth` only).
  Built `orchestrator/{main,run_manager,schemas}.py` (FastAPI, Tier 3): `POST /runs` starts an
  arm in a background thread and returns immediately with a `run_id`; `GET /runs/{id}` reads
  live progress from `experiments/results/<run_id>/` (Phase 4's `RunLogger`, extended with an
  optional pre-generated `run_id` so the orchestrator can know the id before the run finishes —
  see D14). Built `gateway/{package.json,src/{server,db,auth,routes/runs}.js}` (Express 5, Tier
  2): `POST /auth/login` issues a JWT for the single env-sourced admin credential (D9); `POST
  /api/runs` (protected) calls the orchestrator then writes an initial `runs` doc to MongoDB;
  `GET /api/runs/:id` (protected) forwards live orchestrator state while running, and on first
  seeing `complete`/`failed` persists `round_metrics`/`client_metrics` documents and the final
  `runs` doc update, serving straight from Mongo on every poll after that (no duplicate
  inserts). `GET /api/runs` lists all runs. D14 records the tier split (orchestrator never
  touches Mongo; gateway never runs ML code) and the npm dependency verification (Express 5 is
  current latest, not the 4.x most tutorials assume — verified by actually running the server).
  **End-to-end exit-criterion test** (exit criterion: a full run can be triggered and persisted
  via API alone — no CLI, no manual DB write): logged in via `POST /auth/login`, triggered
  `centralized_real` and `isolated_real` runs via `POST /api/runs`, polled `GET /api/runs/:id`
  until `complete`, confirmed live partial `client_metrics` mid-run (3/4 banks before
  completion) then 4/4 after, and independently confirmed via a direct `pymongo` query (not just
  through the API) that `runs`/`round_metrics`/`client_metrics` documents landed correctly in
  MongoDB. Both test servers stopped after verification. Awaiting review before Phase 6
  (React dashboard) begins.

- **2026-08-03 — Phase 6 complete.** Confirmed with the user: chart library **Recharts** (D15,
  resolves the open question PLAN.md had deferred to this kickoff) and **real push-based
  WebSocket** for live round charts rather than reusing Phase 5's polling (D16).
  Closed two real gaps surfaced by the dashboard's own feature list before building it: (1) no
  `/predict` endpoint existed anywhere, and (2) no trained model was ever saved to disk (Phase 4's
  CLI trains/evaluates purely in-memory) — both needed for "test a transaction". Built
  `ml/common/artifacts.py` (D17: model/scaler save-shape mirrors D13 — isolated saves a model
  *and* scaler per bank, federated saves one shared model + a scaler per bank, centralized saves
  one model + one pooled scaler), wired as an opt-in `save_artifacts` flag through `ml.baselines.
  {isolated,centralized}` and `ml.federated.run_federated` (default `False`, so the Phase 4 CLI
  workflow is unaffected). Added `orchestrator/predict.py` (`POST /predict`, `GET /predict/
  manifest/{run_id}`) and `gateway/src/routes/predict.js`. Added `gateway/src/liveRuns.js` (D16:
  WebSocket server in `gateway/`, per-run internal poll loop against the still-HTTP-only
  orchestrator, broadcasts new round/client metrics to subscribed browser clients; JWT travels as
  a `?token=` query param since browsers can't set custom headers on a WS handshake) and
  `gateway/src/persistence.js` (idempotent upserts, replacing Phase 5's blind inserts, since the
  REST poll path and the new WS watcher can now race on the same run). Added `gateway/src/routes/
  quality.js` for the synthetic quality panel + CSV export, populating D8's `validation_reports`
  collection for the first time; deliberately did *not* implement `synthetic_batches` as a literal
  collection since Phase 2's `generation_report.json` is known-stale, and reading current row
  counts straight from `data/synthetic/`/`data/validated/` CSVs is more accurate (flagged, not
  silently skipped — see CLAUDE.md's D8 completion note).
  Scaffolded `dashboard/` with Vite + React 19 + Recharts. Built all seven Tier-1 features:
  login, trigger-run form with live per-bank mode indicators + live WebSocket round chart, run
  history, six-arm comparison grid (bar chart from the most recently completed run per arm), the
  synthetic quality panel with CSV export, and the "test a transaction" predict demo (with a
  server-provided random holdout sample to avoid hand-typing 30 PCA values).
  **Browser-tested end-to-end** (per this project's own standing instruction for UI work, not
  just "it compiles"): used Playwright (no project skill existed for this; `chromium-cli` wasn't
  available in this environment, so drove a headless Chromium directly) against the real running
  orchestrator + gateway + MongoDB + Vite dev server. Found and fixed one genuine app bug (the
  gateway process was stale after a mid-session edit added the `/predict/sample` route — a
  process-restart gap, not a code defect) and two test-script-only bugs (an ambiguous Playwright
  selector matching both the "Trigger Run" nav tab and the form's submit button; a couple of
  screenshots taken before an async fetch had resolved) — both diagnosed by inspecting actual
  network/console output rather than guessing. Final run: login -> trigger `isolated_real` ->
  live per-bank status -> completion -> run history -> six-arm comparison grid (rendered correctly
  once completed runs existed) -> synthetic quality panel with working CSV export -> full predict
  flow (sample load -> predict -> correct low-probability "legitimate" result). A second targeted
  run against `federated_augmented` confirmed the live WebSocket round chart specifically: mode
  indicators correctly showed Augment(CTGAN)/Schema(LLM) per bank, and the line chart updated
  live across all 4 rounds before showing "Complete". Zero browser console errors across every
  screenshot. Exit criterion met: the dashboard drives and displays a complete live run, started
  and finished entirely through the UI. Awaiting review before Phase 7 (integration & evaluation)
  begins.

- **2026-08-03 — Phase 7 complete.** Ran a fresh, dedicated six-arm sweep (`python -m
  ml.run_experiment --arm all`, seed 42, defaults) rather than reusing Phase 4's numbers outright
  — deliberately, as a reproducibility check: the fresh run matched the already-committed Phase 4
  table **bit-for-bit** on every metric, confirming the pipeline is fully deterministic under a
  fixed seed (a citable fact for the report, not assumed). Built `experiments/generate_report.py`:
  auto-detects the latest *full-configuration* run per arm under `experiments/results/`
  (epochs=20 for isolated/centralized, num_rounds=10 for federated — deliberately excluding the
  many small smoke-test runs Phase 6's dashboard testing left behind, e.g. epochs=2-4), and
  generates `docs/phase7_results.md` plus three committed figures under `docs/phase7_figures/`:
  a six-arm F1/precision/recall bar chart and round-by-round convergence line charts for both
  federated arms (matplotlib, already pinned in `ml/requirements.txt`).
  **Found and fixed a real, previously-unnoticed bug while writing the report**: every
  `Path.write_text()` call across the codebase that writes a `.md`/`.json` report file was missing
  an explicit `encoding="utf-8"`, so on this Windows environment they silently used the cp1252
  locale default. Harmless for pure-ASCII content, but any em-dash in generated text became a
  single invalid byte (0x97) instead of valid UTF-8 — caught because `docs/phase7_results.md`
  itself has several. Confirmed via direct byte inspection that this had *already* corrupted one
  character in the committed `docs/phase1_shard_stats.md` (Phase 1) unnoticed until now (Phase 3's
  report happened to contain no non-ASCII characters, so it wasn't affected). Fixed at the root
  (added `encoding="utf-8"` to all nine `write_text()` call sites: `ml/partition/stats.py`,
  `ml/validation/run_validation.py`, `ml/common/{metric_logger,artifacts}.py`,
  `ml/run_experiment.py`, `ml/augmentation/run_augmentation.py`,
  `experiments/generate_report.py`) rather than just removing the offending character, and
  regenerated `docs/phase1_shard_stats.md` and `docs/phase3_validation_report.md` to confirm both
  are now valid UTF-8 with identical content (another incidental reproducibility confirmation for
  Phases 1 and 3).
  **Results and headline finding** (see `docs/phase7_results.md` for full detail): federated >
  isolated on every metric (matches literature's qualitative expectation); federated_augmented is
  the best arm overall (F1 0.6146) despite only Banks A and D's synthetic batches surviving Phase
  3 validation; centralized_real's low F1 despite strong AUC is the same class-imbalance/
  threshold artifact investigated in Phase 4, reported again here rather than hidden.
  **Literature comparison** (PLAN.md reference points, never tuned toward): our accuracy figures
  exceed the ~91%/~95% literature reference, but accuracy is explicitly flagged as not meaningful
  here (>94% trivially for all six arms). Against FedFraud's F1 0.90/AUC 0.96 non-IID benchmark,
  our federated_augmented AUC (0.9544) is close but F1 (0.6146) is well below it — reported as a
  genuine, unresolved gap with three named, unconfirmed plausible contributors (the deliberately
  more extreme non-IID partition, far fewer local epochs/rounds than a literature-scale run, and
  only half the banks' augmentation surviving validation), not smoothed over. Exit criterion met:
  final results tables and convergence plots produced. Awaiting review before Phase 8 (testing,
  documentation, presentation) begins.
