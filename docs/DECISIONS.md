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
