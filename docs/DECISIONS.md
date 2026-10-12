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

## D-003: Global random seed = 42
- **Date:** 2026-10-02 (Step 1)
- **Decision:** a single base seed of 42 in `configs/data.yaml`, used for all randomness.
- **Alternatives:** any other fixed integer.
- **Reason:** the value itself is arbitrary. What matters is that it is fixed and recorded. Multi-seed repeats for the final results are decided in Step 9.

## D-004: EDA is descriptive only and runs on the full dataset
- **Date:** 2026-10-02 (Step 1)
- **Decision:** Step 1 statistics and plots use all 284,807 rows. Nothing computed here (e.g. the Cohen's d ranking) is used to select features or fit any transformation.
- **Alternatives:** run EDA only on the training portion after the Step 2 split.
- **Reason:** describing the raw dataset is standard and needed for the report. Leakage happens only if test information shapes the model, and nothing here does. All fitted components (scalers, generators, thresholds) are fitted on training/validation data only.

## D-005: Drop exact duplicate rows before splitting
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** `drop_duplicates: true`. Keep the first copy of each of the 1,081 duplicate rows, leaving 283,726 rows / 473 fraud.
- **Alternatives:** keep all rows but force copies into the same split (keeps 19 more fraud rows, but double-weights them); keep as-is (leakage risk).
- **Reason:** guarantees no identical transaction is in both train and test. Simple, standard, easy to explain.

## D-006: Global split 70 / 10 / 20 (train / val / test), stratified by Class
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** test is carved off first, then val, before any partitioning. Result: 331 / 47 / 95 fraud.
- **Alternatives:** 60/20/20 (fewer fraud rows for banks); 80/10/10 (only ~47 test fraud, so arm comparisons are too noisy).
- **Reason:** a test set of 95 fraud rows keeps the six-arm comparison reasonably stable while leaving 331 fraud rows for the banks.

## D-007: Non-IID partition by explicit per-class quotas
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** fraud shares A 45% / B 35% / C 12% / D 8%; genuine shares 35 / 30 / 20 / 15%, with largest-remainder rounding. Result: fraud 149 / 116 / 40 / 26.
- **Alternatives:** Dirichlet label skew (α e.g. 0.5), with the two largest draws assigned to A/B; more extreme quotas (50/38/8/4).
- **Reason:** deterministic, guarantees the rich (A, B) vs poor (C, D) contrast the dual-mode design needs, easy to justify. The shares themselves are our choice, not a project specification, and are configurable in `configs/partition.yaml`.

## D-008: Local per-bank split 70 / 15 / 15, stratified
- **Date:** 2026-10-02 (Step 2). **Approved by:** Shubham.
- **Decision:** each bank splits its own shard. Local fraud train/val/test: A 105/22/22, B 82/17/17, C 28/6/6, D 18/4/4.
- **Alternatives:** 60/20/20; 80/10/10.
- **Reason:** balance between training fraud rows and having any local val/test fraud at all for C and D. Known limitation: C/D local metrics are very noisy.

## D-009: Shared classifier = small PyTorch MLP
- **Date:** 2026-10-05 (Step 3). **Approved by:** Shubham.
- **Decision:** `FraudMLP`, 31 → 64 → 32 → 1, ReLU, dropout 0.1 (4,161 weights). Used by all six arms.
- **Alternatives:** scikit-learn logistic regression (SGDClassifier).
- **Reason:** non-linear capacity, weights average naturally under FedAvg, matches Flower's PyTorch examples, trains in about 26 s on CPU for 15 epochs over the full global train set.

## D-010: Class imbalance via a balanced weighted loss
- **Date:** 2026-10-05 (Step 3). **Approved by:** Shubham.
- **Decision:** BCEWithLogitsLoss with pos_weight = #genuine / #fraud of the model's own training data (computed locally), plus threshold tuning on validation data.
- **Alternatives:** random oversampling (overlaps with augmentation and muddies the comparison); no weighting with threshold tuning only.
- **Reason:** equal total weight for both classes. When synthetic fraud is added, the weight re-balances automatically, so augmentation is tested for *new examples* rather than extra emphasis. Known side effect: scores are pushed toward 1 (tuned threshold 0.9997 in the sanity run).

## D-011: Fixed feature transforms (nothing fitted)
- **Date:** 2026-10-05 (Step 3). **Approved by:** Shubham.
- **Decision:** V1–V28 as-is; Amount → log1p; Time → hour-of-day as (sin, cos). 31 inputs.
- **Alternatives:** StandardScaler fitted on training data (needs federated aggregation of statistics in FL); fixed transforms with Time dropped.
- **Reason:** no leakage, identical inputs in every arm and bank, no scaler statistics shared in FL, and keeps the night-time fraud signal seen in Step 1.

## D-012: Training settings (proposed values)
- **Date:** 2026-10-05 (Step 3).
- **Decision:** Adam, learning rate 0.001, batch 512, 15 epochs for centralized training, dropout 0.1, threshold = max F1 on validation data. Configurable in `configs/model.yaml`.
- **Alternatives:** early stopping on validation PR-AUC; more or fewer epochs.
- **Reason:** common, conservative defaults that converge in seconds on CPU. Fixed epochs keep runs simple and reproducible. These are our choices, not project specifications, and may be revisited in Step 7 (with you) if per-bank training shows problems. FL local epochs are decided in Step 8.

## D-013: CTGAN settings for Augment Mode
- **Date:** 2026-10-05 (Step 4). **Approved by:** Shubham.
- **Decision:** 2,000 epochs, batch 500, pac 10, `enable_gpu=False`. All other settings are SDV 1.38.5 defaults (including `enforce_min_max_values=True`). Seed = base seed + bank index.
- **Alternatives:** 500 epochs (likely under-trained); 5,000 epochs (more memorisation risk).
- **Reason:** with ~100 fraud rows ctgan performs 1 step per epoch, so many epochs are needed. Measured cost is ~1.5–2 min per bank on CPU. Losses settled without divergence.
- **Open issue found:** edge clamping affects 63.7% / 79.7% of candidate rows. How to handle it is to be decided with Shubham in Step 6.

## D-014: 1,000 candidate rows per data-rich bank
- **Date:** 2026-10-05 (Step 4). **Approved by:** Shubham.
- **Decision:** generate 1,000 candidates per bank before validation.
- **Alternatives:** 500; equal to the real fraud count (105 / 82).
- **Reason:** a large pool lets the Step 6 gate reject freely. How many validated rows join training (the augmentation ratio) is decided separately in Step 6/7.

## D-015: Schema Mode privacy = option (b), aggregate statistics only
- **Date:** 2026-10-07 (Step 5). Resolves CLAUDE.md Section 2, point 5. **Decided by:** Shubham (to be confirmed with guide).
- **Decision:** prompts to Groq contain only public schema facts plus per-column aggregate statistics computed inside the bank. No real row is ever placed in a prompt. `build_prompt()` receives only the stats dict, never the bank's dataframe.
- **Alternatives:** (a) the bank's own rows as few-shot examples (better fidelity, but real rows leave the client boundary); a locally hosted LLM (no data leaves, but heavy on CPU).
- **Reason:** keeps the privacy invariant intact, even toward an external API. Expected cost: weaker fidelity, which we measure and report.
- **Enforced in code:** `aggregate_stats()` refuses non-whitelisted statistics; tests check that no real value or bound appears in the prompt.

## D-016: Which statistics may leave the bank
- **Date:** 2026-10-07 (Step 5). **Approved by:** Shubham.
- **Decision:** per column: mean, std, p10, p50, p90, rounded to 2 decimals, from training fraud rows only. **No min/max.**
- **Alternatives:** include min/max; send only mean and std; add the top correlations (aggregate, but a larger prompt).
- **Reason:** with 18–28 fraud rows, a min or max is one real row's exact value, which is a leak. Percentiles at 10/50/90 still describe the shape. What was sent is saved in `results/llm/bank_x_stats_sent.json` for audit. Known cost: no correlation information reaches the LLM (mean |corr diff| ≈ 0.49 resulted).

## D-017: Schema Mode generation settings
- **Date:** 2026-10-07 (Step 5). **Candidate count approved by:** Shubham. The model is chosen by Shubham in `.env` (`GROQ_MODEL=openai/gpt-oss-120b`), not hardcoded.
- **Decision:** 300 candidates per bank, up to 15 rows per call, temperature 1.0, reasoning effort low, values to 3 decimals, max 6,000 completion tokens.
- **Alternatives:** 1,000 candidates (to match CTGAN; at the measured ~426 tokens/row that is ~425k tokens per bank, far beyond what the free tier allowed in a day); 25 rows per call (outputs got truncated or malformed).
- **Reason:** fits the free tier. 300 is about 10× the real fraud count of C/D, which leaves the Step 6 gate room to reject.

## D-018: Rate-limit handling
- **Date:** 2026-10-07 (Step 5).
- **Decision:** pace calls to 7,000 tokens/min (limit reported as 8,000). Retry up to 6 times with exponential backoff (2, 4, 8 … s, capped at 60 s), or Groq's `retry-after` if larger. Hard cap of 40 calls per bank. Every response is cached in `data/clients/bank_x/llm_cache/` (git-ignored), so reruns resume.
- **Finding (2026-10-07):** Bank C used 127,683 tokens for 300 rows. During Bank D, Groq began returning `retry-after` waits of about 13–20 minutes per batch, consistent with a daily token limit, not the per-minute one. Generation keeps going but slows to about one batch per 15 minutes. One bank per day fits comfortably.

## D-019: Structured output = strict JSON schema with named fields
- **Date:** 2026-10-07 (Step 5).
- **Decision:** `response_format` = strict `json_schema`, each row an object with named fields (Time, V1…V28, Amount), accepting 1–15 rows per reply.
- **Alternatives tried and rejected** (evidence in `results/llm/attempt*.json`):
  1. `json_object` mode: wrong row counts and row lengths (29/31 values instead of 30); all rows rejected.
  2. Strict schema with array rows: HTTP 400 `json_validate_failed`, mixed row lengths, patterned digits (-3.210, 5.432…), and 6 duplicated rows.
  3. Named fields with exactly 15 rows: HTTP 400 (`minItems: got 12, want 15`).
- **Reason:** named fields fix row shape. A flexible row count avoids the model failing a hard count. Rows still go through local parse checks (Time range, Amount ≥ 0, numeric).

## D-020: Live demo = multi-device Flower federation (replaces "four GitHub repos")
- **Date:** 2026-10-07 (recorded during Step 5; built in Step 8B). **Proposed by:** Dr. Chethan L S (guide). **Decided by:** Shubham and team.
- **Guide's proposal:** four public GitHub repos (one per bank and team member), each holding that bank's dataset, to show the model moving from one repo to another.
- **Why it was replaced:**
  1. GitHub repos only store files. They cannot run training, so the model would not actually train or move. It would be manual file uploads, not federated learning.
  2. Public repos would make each bank's data public, which contradicts the project's privacy claim and the privacy invariant (CLAUDE.md Section 2, point 4).
  3. Federated learning needs a live client process at each bank that receives the global model, trains locally, and sends back only weights over the network.
- **Chosen design:** Shubham's laptop runs the Flower server (FedAvg) and the dashboard. Each teammate's laptop runs one bank client (A–D). All laptops share a Wi-Fi network or phone hotspot. Each laptop holds only its own shard (never in Git, never sent). Only weights and scalar metrics travel, and every message's size and content type is logged. Clients print per-round logs, and the server logs which bank's update arrived. Fallback: five terminal processes on one laptop, using the same code path. One private GitHub repo for code only.
- **Scope:** live demo only. The six-arm experiments stay in single-machine simulation (speed, reproducibility). The single-machine constraint was amended to allow this, at zero cost.
- **Achieves the guide's goal:** the model visibly moves between separate machines, while the data stays put.
- **Follow-up:** update the project report and synopsis to describe this demo setup. Confirm the design with the guide.

## D-021: sentence-transformers kept as a diagnostic; numeric DCR makes the privacy/diversity decisions
- **Date:** 2026-10-08 (Step 6). **Approved by:** Shubham.
- **Decision:** embed each row as text with `all-MiniLM-L6-v2` (CPU), report cosine similarities, and reject only essentially identical text (cosine ≥ 0.999). Add a numeric distance-to-closest-record (DCR) check that decides privacy and diversity rejections.
- **Alternatives:** gate on embeddings only (as originally specified); drop embeddings entirely.
- **Reason:** measured cosine similarity was 0.96–0.995 for synthetic↔real, real↔real and synthetic↔synthetic alike, so text embeddings of numeric rows cannot separate near-copies. The specified tool stays in the pipeline (no silent substitution), and the added check is documented.

## D-022: Schema/range rules (Pandera)
- **Date:** 2026-10-08 (Step 6). **Approved by:** Shubham.
- **Decision:** exact column set (strict), numeric, no missing values, Class = 1; each value within the bank's real training-fraud [min, max] widened by 10% of the range (Time also within 0–172,792; Amount ≥ 0); reject rows with 3+ values exactly on a real min/max (edge clamping).
- **Alternatives:** reject any clamped value (keeps only 36% / 20% of A / B); ignore clamping.
- **Reason:** removes CTGAN's artificial boundary spikes while tolerating occasional extremes. Bounds are computed locally and never shared. Observed side effect: CTGAN validated-set fidelity dropped (0.833 → 0.804 for A, 0.843 → 0.800 for B).

## D-023: Patterned decimals are report-only
- **Date:** 2026-10-08 (Step 6). **Approved by:** Shubham.
- **Decision:** measure and report the share of patterned decimals; never reject on it.
- **Alternatives:** reject rows with > 25% or > 50% patterned decimals (would keep only ~18–32% of LLM rows).
- **Reason:** the pattern is in the 3rd decimal, which barely affects model inputs. The harm it signals (clumped rows) is addressed by the diversity check.

## D-024: Data-driven privacy/diversity thresholds and a fidelity minimum
- **Date:** 2026-10-08 (Step 6). **Approved by:** Shubham.
- **Decision:** DCR threshold = 5th percentile of real-to-real nearest-neighbour distances in the bank (standardised with the bank's real fraud mean/std). Rows below it are rejected as too close to real; rows within it of an already-kept synthetic row are dropped as near-duplicates (greedy, in file order). SDMetrics overall quality of the admitted set must be ≥ 0.70, otherwise the whole batch is refused.
- **Alternatives:** privacy only, without the diversity filter; stricter 25th percentile.
- **Reason:** thresholds adapt to each bank's own data rather than being invented constants. Result: thresholds 0.495 / 1.582 / 1.067 / 0.922; 0 rows too close to real; 37 / 9 LLM near-duplicates removed; all banks ≥ 0.746 fidelity.

## D-025: Augmentation ratio 1:1 with real training fraud
- **Date:** 2026-10-08 (Step 7). **Approved by:** Shubham.
- **Decision:** each bank adds validated synthetic fraud rows equal in number to its real training fraud rows (A 105, B 82, C 28, D 18), sampled from `synthetic_validated.csv` with the run seed.
- **Alternatives:** all validated rows (synthetic would outnumber real 8–15×); top up to a fixed fraud count (mixes ratio and mode effects).
- **Reason:** the same rule for every bank (fair across modes); real fraud keeps half of the fraud weight. Other ratios can be a Step 9 sensitivity check.

## D-026: Thresholds tuned on the model owner's own validation data
- **Date:** 2026-10-08 (Step 7). **Approved by:** Shubham.
- **Decision:** isolated bank → its own local val; centralized → union of the banks' local vals (49 fraud); federated (Step 8) → the same data, combined only through aggregated counts.
- **Alternatives:** the global val set for every arm (identical protocol, but unrealistic for isolated banks).
- **Reason:** realistic, and it exposes the true cost of isolation for small banks. PR-AUC (threshold-free) is reported for all arms.

## D-027: Centralized arms pool the banks' local train parts
- **Date:** 2026-10-08 (Step 7).
- **Decision:** Arms 5/6 train on the union of `bank_x/train.csv` (139,022 rows), not on `global_train.csv`.
- **Reason:** `global_train.csv` also contains every bank's local val/test rows; training on it would leak local test rows into training.

## D-028: Scores are raw logits (float64), not float32 probabilities
- **Date:** 2026-10-08 (Step 7). Bug fix.
- **Decision:** `predict_scores()` returns float64 logits, used for ranking, thresholds, PR-AUC and ROC-AUC. `predict_proba()` (sigmoid) is for display only. The default cut-off "probability 0.5" is logit 0.
- **Reason:** with pos_weight ≈ 600, confident logits reach 17–46, and float32 sigmoid rounds them to exactly 1.0. In one model, 49 test rows (41 frauds) tied, losing ranking information. The sigmoid preserves order, so logits give the same ranking without ties. Effect: P/R/F1 unchanged, PR-AUC was under-reported (Step 3 sanity: 0.6978 → 0.7272). Step 3 results were re-run and its doc corrected.

## D-029: 30 FL rounds
- **Date:** 2026-10-08 (Step 8). **Chosen by:** Shubham (option B).
- **Decision:** 30 FedAvg rounds, all four banks every round.
- **Alternatives:** 15 rounds (matches the 15 epochs of the non-federated arms exactly, but risks stopping before convergence); 50 rounds (more time, multiplied by seeds in Step 9).
- **Reason:** room for FedAvg to converge under non-IID data. Observed: global PR-AUC plateaued around rounds 13–15. Training effort (30 passes) is 2× the non-federated arms, which is disclosed.

## D-030: 1 local epoch per round
- **Date:** 2026-10-08 (Step 8). **Chosen by:** Shubham (option A).
- **Decision:** each bank trains 1 epoch per round (same Adam settings as D-012; a fresh optimiser each round, so clients keep no state).
- **Alternatives:** 2 epochs; 5 epochs (faster per round, more client drift).
- **Reason:** the least client drift for very unequal banks (105 vs 18 training frauds).

## D-031: Federated threshold from summed per-bank counts on a fixed grid
- **Date:** 2026-10-08 (Step 8). **Chosen by:** Shubham (option A).
- **Decision:** a fixed grid of 241 logit cut-offs (−10 to 50, step 0.25) set before training. Each bank returns integer TP/FP counts on its own local val set at each cut-off; the server sums them and takes the max-F1 cut-off.
- **Alternatives:** each bank sends its own best threshold, averaged (sends a score value; Bank D's comes from 4 frauds); a fixed cut-off of logit 0 (precision ≈ 0.10).
- **Reason:** counts only, never scores or rows. Mathematically equal to pooled-validation tuning on the grid (tested), so comparable with the centralized arms (D-026).

## D-032: Fixed aggregation order (reproducibility fix)
- **Date:** 2026-10-08 (Step 8).
- **Decision:** `OrderedFedAvg` sorts client replies by bank before calling FedAvg's own aggregation.
- **Reason:** parallel clients reply in varying order. Floating-point sums in a different order differ by ~1e-8, which training amplified: two same-seed runs drifted apart from the first round where the order changed (max per-round PR-AUC difference 8.7e-5). After the fix, two full runs were bit-identical. The FedAvg maths is unchanged.

## D-033: Ready-made files for every bank; no zip passwords
- **Date:** 2026-10-10 (Step 8B). **Chosen by:** Shubham.
- **Decision:** every bank, including Bank B (CTGAN), receives pre-generated, gate-validated files from Steps 4–6. Zips carry no password.
- **Alternatives:** Prachi re-runs CTGAN live (~1.5–2 min); a password per zip (extra package such as `py7zr`).
- **Reason:** the same files as the experiments, no Groq/CTGAN dependency on demo day, and in-person handover with deletion. Disclosed: synthetic data was generated on Shubham's laptop.

## D-034: Deployment set-up for the demo (flwr 1.39)
- **Date:** 2026-10-10 (Step 8B).
- **Decision:** SuperLink on the server laptop (Fleet API 0.0.0.0:9092 for banks; HTTP Control API bound to 127.0.0.1:9093); one SuperNode per bank started with `--node-config "bank='bank_x'"`; runs started with `flwr run`; insecure mode (no TLS) on a private phone hotspot. The app bundle is restricted by `fab-include` to `ml/**/*.py` and `configs/*.yaml` (built size 154 KB, no CSV or JSON).
- **Alternatives:** TLS certificates and SuperNode authentication (more set-up, out of demo scope); Flower's default bundle patterns (would include every `*.json`).
- **Reason:** the matching deployment API for the installed version (not the simulation API), the minimum ports, and a bundle that provably carries no data.

## D-035: Data isolation for the demo
- **Date:** 2026-10-10 (Step 8B).
- **Decision:** `FRAUDNET_DATA_ROOT` separates the data folder from the code. The client runs `check_my_data` before its first round and refuses on failure. The export refuses to build a zip unless the Step 2 checksums match, every real row comes from global train, no row is shared with another bank or global val/test, and no synthetic row equals a real row (fingerprints). The one-laptop fallback gives every process its own sandbox folder.
- **Reason:** turns "each laptop holds only its own bank" from a promise into a check, which is visible evidence for the review.

## D-036: Demo run defaults
- **Date:** 2026-10-10 (Step 8B).
- **Decision:** Arm 4 (headline), 10 rounds, `expected-banks` configurable (2 for the two-laptop test), per-message logging on; demo outputs written to `results/demo_8b/` and never over experiment results.
- **Reason:** 10 rounds ≈ 5–6 minutes in deployment (≈ 25–30 s per round), which fits the 5-minute demo. Demo numbers are never reported as results.

## D-037: 5 seeds for the six-arm results
- **Date:** 2026-10-11 (Step 9). **Approved by:** Shubham.
- **Decision:** seeds 42–46 for every arm.
- **Alternatives:** 10 seeds (tighter CIs, ~2× time); 3 seeds (std from 3 points is unreliable).
- **Reason:** mean ± std and paired CIs at ~22 min of CPU. Seed 42 reproduces Steps 7–8 exactly.

## D-038: Seeds vary training randomness only
- **Date:** 2026-10-11 (Step 9). **Approved by:** Shubham.
- **Decision:** a seed changes initial weights, batch order, FL client seeds and the 1:1 synthetic sample. The data split, bank shards, CTGAN/LLM pools and validation gate stay fixed.
- **Alternatives:** re-running CTGAN per seed (LLM banks couldn't follow, which would be unequal); re-partitioning per seed (would require Steps 2–6 per seed).
- **Reason:** isolates training noise and is feasible on the free tier. Disclosed as a limitation.

## D-039: Paired differences with t-based 95% confidence intervals
- **Date:** 2026-10-11 (Step 9). **Approved by:** Shubham.
- **Decision:** for each comparison, per-seed paired differences: mean, std, CI = mean ± t(0.975, n−1)·std/√n, and "improved in k of N seeds". A difference is "clear" only if the CI excludes 0. The number of comparisons (56) and the expected chance hits (~3) are reported.
- **Alternatives:** mean ± std only (easy to over-read); Wilcoxon signed-rank (with n = 5 the smallest two-sided p is 0.0625, so it can never reach p < 0.05).
- **Reason:** simple, honest and explainable in a viva.

## D-040: /predict uses the federated Arm 3 model (seed 42)
- **Date:** 2026-10-11 (Step 10). **Chosen by:** Shubham.
- **Decision:** the final FedAvg global model of Arm 3, seed 42, 30 rounds, with its count-based threshold (logit 3.75), saved by the `prepare_model` job (re-run of the bit-reproducible simulation; verified identical to Step 9).
- **Alternatives:** Arm 4 (statistically the same performance); both models side by side.
- **Reason:** the privacy-preserving model, and Step 9 found no augmentation benefit.

## D-041: Generation through the API = replay by default, versioned regenerate on request
- **Date:** 2026-10-11 (Step 10). **Chosen by:** Shubham.
- **Decision:** *replay* retrains CTGAN with its original seed (a reloaded model does **not** resample the same rows) and rebuilds LLM rows from the cache (0 API calls), runs the same gate into `generations/replay_<job>/` inside the bank folder, and compares SHA-256 with the experiment files. *Regenerate* writes new data to `generations/vN/`. Experiment files are never overwritten.
- **Alternatives:** replay only; full regenerate in place (would invalidate Steps 6–9).
- **Reason:** reproducibility proof plus safe experimentation. Replay of all four banks was byte-identical.

## D-042: Background jobs as separate processes, one at a time
- **Date:** 2026-10-11 (Step 10).
- **Decision:** a queue and worker thread; each job is `python -m services.orchestrator.tasks …`; state persisted to `results/api_jobs/<id>/job.json`; unfinished jobs are marked "interrupted" after an API restart.
- **Alternatives:** threads inside the API process (training would freeze the server); Celery/Redis (extra infrastructure, not justified on one laptop).
- **Reason:** isolation, simplicity, zero cost.

## D-043: Runs = any subset of arms 1–6 or all six, over chosen seeds
- **Date:** 2026-10-11 (Step 10). **Chosen by:** Shubham.
- **Decision:** `POST /runs {arms, seeds}`; outputs go to the job folder, never over Step 9's; a full report is built when all six arms run over ≥ 2 seeds.
- **Reason:** quick single-arm demos and full reproductions from the dashboard (Step 13).

## D-044: Orchestrator listens on localhost with an internal key
- **Date:** 2026-10-11 (Step 10).
- **Decision:** bind to 127.0.0.1:8100; every endpoint except `/health` requires `X-Internal-Key` (≥ 16 characters, in `.env`, compared in constant time); the WebSocket accepts it as a header or `?key=`. User authentication lives in the gateway (Step 11).
- **Reason:** Tier 3 is never exposed directly; one clear trust boundary.

## D-045: Synthetic export only to the bank's own operator, audited (requirement conflict resolved)
- **Date:** 2026-10-11 (Step 10). **Chosen by:** Shubham.
- **Conflict:** CLAUDE.md lists "dataset export", while the privacy invariant says no synthetic row crosses a client boundary.
- **Decision:** export returns only that bank's `synthetic_validated.csv` (never real rows), only when `X-Bank-Scope` (set by the gateway from the user's role) equals the bank; every attempt is logged in `results/api_audit/exports.log`.
- **Alternatives:** aggregates-only export; any logged-in user (contradicts the invariant).
- **Reason:** data goes only to its owner. Disclosed in the report.

## D-046: JWT bearer tokens with bcrypt-hashed passwords
- **Date:** 2026-10-11 (Step 11).
- **Decision:** login checks the bcrypt hash (cost 12) and issues an HS256 JWT {username, role, bank}, valid 8 h, sent as `Authorization: Bearer`. Login is rate-limited (20 / 15 min) and audited; errors don't reveal whether a username exists. The secret and demo passwords live in `.env`.
- **Alternatives:** session cookies (needs session storage and CSRF handling); httpOnly-cookie JWT (better against XSS, but more set-up for the WebSocket).
- **Reason:** simple, stateless and easy to explain. Local deployment.

## D-047: Roles: admin, four bank operators, viewer
- **Date:** 2026-10-11 (Step 11). **Chosen by:** Shubham.
- **Decision:** `admin` (start runs, generation, model preparation; cancel; audit log); `bank_operator` for bank_a–bank_d (read, plus export of their own bank's synthetic data, with scope taken from the token); `viewer` (guide, read-only).
- **Alternatives:** admin + viewer only; a single demo user.
- **Reason:** demonstrates role-based access and enforces D-045 end to end.

## D-048: Mongoose schemas, rounds in their own collection
- **Date:** 2026-10-11 (Step 11). **Chosen by:** Shubham (Mongoose).
- **Decision:** collections users, runs, rounds, modelresults, banks, validationreports, auditevents; unique (jobId, arm, seed, round) for rounds; only metrics and metadata stored, never transaction rows.
- **Alternatives:** the official driver without schemas; rounds embedded in run documents (would grow to hundreds per run).
- **Reason:** explainable schema, data integrity, idempotent writes.

## D-049: Server-side job watcher + plain WebSocket relay
- **Date:** 2026-10-11 (Step 11). **Chosen by:** Shubham (plain WebSocket).
- **Decision:** the gateway watches each job's FastAPI WebSocket itself, persists events in order, reconnects and resumes after restarts, and relays events to browsers at `/ws/runs/:jobId?token=…` with catch-up from MongoDB.
- **Alternatives:** Socket.IO; Server-Sent Events; browsers connecting to FastAPI directly (would expose Tier 3 and the key).
- **Reason:** complete history without an open browser; the same protocol end to end.

## D-050: Gateway uses its own MongoDB database `fraudnet_gateway`
- **Date:** 2026-10-11 (Step 11). **Chosen by:** Shubham.
- **Context:** database `fraudnet_synth` already held an earlier prototype's data (7 `runs` from 2026-08-03 with different fields; collections `round_metrics`, `client_metrics`, `validation_reports`).
- **Decision:** `GATEWAY_MONGO_URI` (default `mongodb://localhost:27017/fraudnet_gateway`); the old database is left untouched. Tests use `fraudnet_gateway_test`.
- **Alternatives:** share `fraudnet_synth` and filter (mixes two systems); delete the old data (not ours to delete).

## D-051: Recharts for dashboard charts
- **Date:** 2026-10-11 (Step 12). **Chosen by:** Shubham.
- **Decision:** Recharts 3.10.1 (charts as React components; live data updates state).
- **Alternatives:** Chart.js via react-chartjs-2 (canvas, configured with option objects).
- **Reason:** simplest to read and explain; both are named in the architecture.

## D-052: Vite + React, built and served by the gateway
- **Date:** 2026-10-11 (Step 12). **Chosen by:** Shubham.
- **Decision:** Vite 8 dev server (proxying `/api` and `/ws`) for development; `npm run build` → `frontend/dist`, served by the gateway at http://127.0.0.1:8200 with a fallback to the app for page routes.
- **Alternatives:** Vite dev server during demos; Create React App (deprecated).
- **Reason:** one address, no CORS, works offline, fewer moving parts in the viva.

## D-053: JavaScript + plain CSS with design tokens
- **Date:** 2026-10-11 (Step 12). **Chosen by:** Shubham.
- **Decision:** JavaScript (JSX) and one CSS file with light/dark tokens; chart colours from the validated dataviz palette in `theme.js`, with fixed roles (real = blue, augmented = aqua, banks A–D = slots 1–4); status = icon + label.
- **Alternatives:** TypeScript; Tailwind CSS.
- **Reason:** same language as the gateway; nothing extra to learn or explain.

## D-054: Login token in memory + sessionStorage
- **Date:** 2026-10-11 (Step 12). **Chosen by:** Shubham.
- **Decision:** survives a refresh in the same tab, cleared when the tab closes; automatic logout on any 401.
- **Alternatives:** memory only (a refresh logs you out); localStorage (persists across browser restarts).
- **Reason:** demo-friendly without long-lived tokens on a shared laptop.

## D-055: Results annexure generated by code, with provenance
- **Date:** 2026-10-12 (Step 13).
- **Decision:** `ml/evaluation/annexure.py` builds `docs/RESULTS_ANNEXURE.md` and `results/final_annexure/` from one succeeded dashboard job (all six arms): tables rendered from CSV/JSON, figures copied byte-for-byte, caption numbers computed, SHA-256 of every source in `provenance.json`, and a row-by-row dashboard-vs-CLI check in `consistency.json`.
- **Alternatives:** hand-written annexure (violates "never invent results"); tables copied manually (error-prone, untraceable).
- **Reason:** every reported number is traceable to a saved run.

## D-056: Integration evidence from committed, guarded scripts; retakes labelled
- **Date:** 2026-10-12 (Step 13). **Chosen by:** Shubham (rewrite + re-run).
- **Decision:** dashboard runs are driven by `services/gateway/scripts/full_run_ui.js` (committed). It refuses unknown arguments, judges "finished" by the run's own card only, resizes before capturing, and `--reopen` retakes are logged in `retake_output.txt`. Main evidence: job `20261012-142636-run-eed13e`; two earlier dashboard runs are kept as additional confirmation (all three identical to Step 9).
- **Context:** an earlier session's run and annexure existed without their generating code; one run was started by mistake during the fix (see `results/integration/README.md`).

## D-057: Offline demo = pre-flight check + one-command launcher
- **Date:** 2026-10-12 (Step 14). **Chosen by:** Shubham.
- **Decision:** `python -m demo.preflight` (19 checks with fix hints) and `python -m demo.start` (pre-flight, FastAPI + gateway with `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, health wait, browser; `--smoke` tour). The demo uses existing results and cached synthetic data.
- **Alternatives:** pre-flight only (more to remember on viva day); a frozen bundle of results and model (~10 MB duplicated).
- **Reason:** one command, failures explained before the examiners see them. Verified with Python's internet blocked.

## D-058: Viva sheet = curated top 20 + generated step questions
- **Date:** 2026-10-12 (Step 14). **Chosen by:** Shubham.
- **Decision:** `demo/viva_sheet.py` combines `docs/viva_top20_source.md` (hand-written, citing verified results) with every step doc's "Likely viva questions" into `docs/VIVA_QA.md` (100 Q&As).
- **Alternatives:** generated only (no big-picture questions); hand-written only (drifts from step docs).
