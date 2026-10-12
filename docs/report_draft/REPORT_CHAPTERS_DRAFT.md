# FraudNet-Synth: draft report chapters (results, discussion, extensions, limitations)

> **How to use this draft.** Chapter numbers are placeholders; fit them into the VTU report template. Every number has its source in square brackets, so it can be checked and cited as a table or figure. Rewrite the prose in your own words before submission. Literature values marked **[verify at source]** must be checked against the original papers before you cite them.

---

## Chapter A. Results and evaluation

### A.1 Experimental set-up
The ULB credit-card dataset has 284,807 transactions, of which 492 (0.17%) are fraudulent [results/eda/eda_summary.json]. After removing 1,081 exact duplicates, a stratified global split held out 56,746 test transactions (95 frauds) and 28,373 validation transactions (47 frauds) [results/partition/shard_stats.csv]. The remaining training data was divided into four non-IID banks using explicit fraud quotas: Banks A and B are fraud-rich (149 and 116 frauds) and Banks C and D fraud-poor (40 and 26) [same file]. Each bank split its shard 70/15/15 into local train, validation and test sets.

All six arms use the same classifier: a multi-layer perceptron (31 → 64 → 32 → 1, 4,161 weights) with a class-balanced weighted loss. Each model's decision threshold maximises fraud-class F1 on the validation data its owner would have. The federated arms use FedAvg for 30 rounds with 1 local epoch, and their threshold is chosen from integer counts summed across banks [docs/DECISIONS.md D-009–D-012, D-026, D-029–D-031]. Every arm was repeated over five seeds (42–46), and arms are compared with paired differences and t-based 95% confidence intervals [D-037–D-039].

**Headline metrics** are fraud-class precision, recall, F1 and PR-AUC. Accuracy is secondary: a model that labels everything "genuine" already reaches 99.83% [eda_summary.json].

### A.2 Synthetic data generation and validation
| Bank | Mode | Real training frauds | Candidates | Mean KS distance | Mean correlation error | Validated (pass rate) | Fidelity (validated) |
|---|---|---|---|---|---|---|---|
| A | CTGAN | 105 | 1,000 | 0.219 | 0.148 | 868 (86.8%) | 0.804 |
| B | CTGAN | 82 | 1,000 | 0.223 | 0.153 | 814 (81.4%) | 0.800 |
| C | LLM | 28 | 300 | 0.260 | 0.493 | 234 (78.0%) | 0.746 |
| D | LLM | 18 | 300 | 0.252 | 0.493 | 264 (88.0%) | 0.756 |

[results/final_annexure/tables/T03_synthetic_and_gate.csv; results/ctgan/*_summary.json; results/llm/*_summary.json; results/validation/validation_summary.csv]

- **Both modes passed the shared gate at similar rates:** 84.1% for Augment mode and 83.0% for Schema mode [results/validation/console_output.txt]. No synthetic row was rejected as a near-copy of a real transaction.
- **CTGAN** matched real fraud distributions more closely, but SDV's min/max clamping left 63.7% (A) and 79.7% (B) of candidate rows with at least one value stuck on a range boundary [results/ctgan/*_summary.json]. The gate removed rows with three or more such values.
- **The LLM** received only aggregate statistics (privacy option (b)). It reproduced column centres but not the relationships between columns (correlation error 0.493), and 68.9% (C) and 63.6% (D) of its values showed artificial digit patterns, against about 1.6% expected by chance [results/llm/*_summary.json].

### A.3 Six-arm comparison
**Table A.1** Global test set (95 frauds), mean ± std over 5 seeds [results/final_annexure/tables/T04_six_arm_global.csv]

| Arm | Model | Precision | Recall | F1 | PR-AUC | ROC-AUC* | Accuracy* |
|---|---|---|---|---|---|---|---|
| 1 Isolated, real | Bank A | 0.855 ± 0.013 | 0.768 ± 0.013 | 0.809 ± 0.005 | 0.705 ± 0.006 | 0.959 | 0.9994 |
| 1 Isolated, real | Bank D | 0.799 ± 0.010 | 0.320 ± 0.035 | 0.456 ± 0.037 | 0.662 ± 0.016 | 0.937 | 0.9987 |
| 2 Isolated, augmented | Bank D | 0.811 ± 0.012 | 0.362 ± 0.027 | 0.500 ± 0.029 | 0.651 ± 0.020 | 0.932 | 0.9988 |
| 3 Federated, real | — | 0.838 ± 0.036 | 0.760 ± 0.022 | **0.796 ± 0.006** | **0.716 ± 0.006** | 0.963 | 0.9993 |
| 4 Federated, augmented | — | 0.845 ± 0.046 | 0.756 ± 0.012 | 0.797 ± 0.017 | 0.717 ± 0.007 | 0.962 | 0.9994 |
| 5 Centralized, real | — | 0.891 ± 0.006 | 0.686 ± 0.039 | 0.775 ± 0.028 | 0.727 ± 0.004 | 0.967 | 0.9993 |
| 6 Centralized, augmented | — | 0.854 ± 0.045 | 0.707 ± 0.074 | 0.770 ± 0.031 | 0.725 ± 0.006 | 0.964 | 0.9993 |

\*Secondary metrics. The full table (all four banks in Arms 1 and 2) is in the annexure.

**Key paired comparisons (95% CI)** [results/final_annexure/tables/T06_paired_global.csv]:
- **Federation vs isolation, data-poor banks:** Bank D F1 +0.340 [0.299, 0.381], higher in 5/5 seeds; Bank C F1 +0.098 [0.077, 0.119], 5/5 seeds. On its own local test, Bank D caught 0 of 4 frauds alone and 3 of 4 with the federated model, in every seed [T05_local_tests.csv].
- **Federation vs isolation, data-rich bank A:** F1 −0.013 [−0.020, −0.006] but PR-AUC +0.011 [0.004, 0.018].
- **Federated vs centralized (cost of privacy):** F1 +0.021 [−0.017, 0.060] (no clear difference); PR-AUC −0.011 [−0.021, −0.001].
- **Augmentation:** 33 of 36 augmentation comparisons (global and local test, F1 and PR-AUC) showed no clear difference. Of the other three, one was higher and two lower [results/six_arms/paired_comparisons.csv].

**Figures:** F06 (F1 and PR-AUC by arm), F07 (paired differences), F08 (federated convergence: mean global PR-AUC 0.688 at round 1, 0.715 at round 10, 0.716 at round 30) [results/final_annexure/figures/].

### A.4 System integration
A full experiment (six arms × five seeds) was started from the React dashboard as the admin user. It passed through the Node.js gateway (JWT authentication, role check, MongoDB persistence) to the FastAPI orchestrator and the four simulated banks, streamed 300 federated rounds live, and stored 60 model results. All 180 metric rows were **identical** to the command-line run (maximum absolute difference 0.0) [results/final_annexure/consistency.json; results/integration/README.md]. A live deployment across two laptops over a phone hotspot also completed 10 rounds (Section C.3).

---

## Chapter B. Discussion

### B.1 Federation helps most where data is scarce
The clearest result is that federated training lifts the banks that have too little fraud data to learn alone. Bank D, with 18 training frauds, reached F1 0.456 alone and 0.796 inside the federation, while sharing only model weights (17,478 bytes per round) [Table A.1; results/federated/]. Isolated small banks did not fail because their models were useless: their PR-AUC (0.66–0.72 across the four isolated banks) was close to the others'. They failed because a threshold tuned on 4–6 validation frauds is unreliable. Federation fixes both problems: the shared model learns from all banks' frauds, and the threshold is tuned on summed counts from 49 validation frauds.

### B.2 The cost of privacy is small
Federated learning matched centralized training on F1 and trailed it by 0.011 PR-AUC, without any transaction leaving a bank. Literature reference points report about 91% (federated) against 95% (centralized) accuracy, and FedFraud reports F1 0.90 / AUC 0.96 under non-IID data **[verify at source]**. Our accuracy (99.93%) can't be compared meaningfully because of the 0.17% fraud rate. Our federated ROC-AUC (0.963) is similar to the reported AUC, while our F1 (0.796) is lower; the datasets, splits and models differ.

### B.3 Synthetic augmentation did not reliably help
Neither CTGAN (Augment mode) nor LLM generation (Schema mode) improved fraud detection reliably, in any arm. We report this as a finding, consistent with literature that reports CTGAN augmentation degrading fraud models **[verify at source]**. Likely reasons, all supported by our measurements:
1. **Little new information:** the generators learnt from only 18–105 real frauds, so synthetic rows mostly repeat what the real rows already show.
2. **The loss already balances classes:** the weighted loss gives real fraud equal total weight, so adding rows adds examples but no extra emphasis [D-010, D-025].
3. **Schema-mode rows lack structure:** the LLM saw no correlations (correlation error 0.493 vs 0.148 for CTGAN) and produced patterned digits.
4. **The gate removed CTGAN's extremes:** filtering lowered CTGAN fidelity (A 0.833 → 0.804), because real fraud also has extreme values.

The contribution therefore lies in the **pipeline**: client-adaptive dual-mode generation under a shared, privacy-checked validation gate, evaluated rigorously enough to show when augmentation does *not* help. It doesn't lie in an accuracy gain.

### B.4 Reproducibility as a design goal
Fixed seeds, a fixed aggregation order (D-032), versioned decisions and generated provenance made every reported number reproducible bit for bit, including runs started from the dashboard. Two bugs found this way were fixed and disclosed: float32 score saturation that under-reported PR-AUC (D-028), and non-deterministic aggregation order (D-032).

---

## Chapter C. Extensions beyond the synopsis

> These three items were **not** in the approved synopsis and are reported as extensions.

### C.1 E1: Fairness-aware aggregation (q-FedAvg)
**Question:** does q-FedAvg, which gives more weight to banks where the global model has a higher loss, improve the worst bank's fraud F1?
**Method:** Flower's QFedAvg with each bank reporting the received global model's loss. Flower's default step constant (L = 1000 for our Adam clients) froze training in a probe (PR-AUC stayed 0.005), so L was tuned on validation data (L = 1). We compared q ∈ {0, 0.5, 1, 2} with FedAvg over 5 seeds, using each bank's own local test [docs/steps/STEP_E1_qfedavg.md].

| Config | Global F1 | Worst-bank F1 | Bank D F1 |
|---|---|---|---|
| FedAvg | 0.796 ± 0.006 | 0.733 ± 0.037 | 0.750 ± 0.000 |
| q-FedAvg q = 1 | 0.805 ± 0.006 | 0.717 ± 0.046 | 0.717 ± 0.046 |
| q-FedAvg q = 2 | 0.795 ± 0.009 | 0.671 ± 0.009 | 0.683 ± 0.037 |

[results/extensions/e1_qfedavg/summary_table.csv]

**Result:** no q improved fairness; q = 2 clearly lowered worst-bank F1 (−0.063 [−0.108, −0.018]) [paired_vs_fedavg.csv]. **Reason:** the global model's loss was nearly equal across banks (max/min ratio ≤ 1.17), because the balanced loss already equalises each bank's classes, so q-FedAvg had no signal to act on.

### C.2 E2: Conformal risk control of the miss rate
**Question:** can the threshold *guarantee* a maximum expected miss rate α, and should calibration be global or per bank?
**Method:** conformal risk control (Angelopoulos et al., 2022 **[verify at source]**): the strictest cut-off with (misses + 1)/(n + 1) ≤ α, computed from per-bank counts [docs/steps/STEP_E2_conformal.md].

| Threshold (global calibration, 49 frauds) | Miss rate, global test | Transactions flagged | Precision |
|---|---|---|---|
| F1-tuned | 0.240 | 0.15% | 0.838 |
| α = 0.20 | 0.128 | 2.27% | 0.076 |
| α = 0.10 | 0.061 | 18.62% | 0.009 |
| α = 0.05 | 0.055 | 29.37% | 0.006 |

[results/extensions/e2_conformal/summary_table.csv]

**Results:**
1. The guarantee held on average over 2,000 random re-splits × 5 seeds for every pool and α [guarantee_check.csv].
2. Strong guarantees cost many false alarms.
3. Banks C and D cannot certify α below 0.143 and 0.200 alone and must flag every transaction at α = 0.10, whereas global calibration through counts gives them the guarantee at about 19% of transactions flagged. Federation therefore also helps small banks *calibrate*.

### C.3 Multi-laptop federated demonstration
To show the model moving between machines while data stays put (a guide request, D-020), the system was deployed with Flower's deployment API. A server laptop ran the SuperLink and Bank A, and a second laptop ran Bank B, over a phone hotspot.
- **Before training:** each laptop passed a data check proving it held only its own bank.
- **During training:** each round, both banks received the global model, trained locally and returned 17,478 bytes of weights; the threshold step exchanged only integer counts.
- **Run:** 10 rounds completed in 274 s; global test F1 0.781, PR-AUC 0.726 (two banks only, demonstration rather than an experimental arm) [docs/steps/STEP_08B_multi_device_demo.md §5.4].

The four-laptop run is **pending** at the time of this draft.

---

## Chapter D. Limitations and future scope

### D.1 Limitations
- **One dataset, one split:** all results come from the public, PCA-anonymised ULB dataset with a single global split and partition; seeds varied only training randomness [D-038].
- **Very few frauds per bank:** Bank D's local test has 4 frauds and C's has 6, so per-bank F1 moves in large steps.
- **Simulation:** reported results come from single-machine simulation; the multi-laptop deployment is a demonstration.
- **Synthetic-data scope:** the LLM saw aggregates only (no rows, no min/max, no correlations), a privacy choice that limits fidelity; generation was not re-run per seed (free-tier quota).
- **Privacy scope:** no formal privacy guarantee. Model updates can still leak information, and the Groq API is an external service (only aggregates were sent).
- **Validation layer:** the sentence-transformers embedding check could not separate numeric rows (cosine 0.96–0.995 for all pairs) and was kept as a diagnostic beside a numeric distance check [D-021].
- **Conformal guarantees** assume exchangeability, which concept drift would break.

### D.2 Future scope
Differential privacy and secure aggregation, graph-based models (FinGraphFL), more than four clients, and replication on IEEE-CIS were outside this project's scope by design. Further directions suggested by our results: generation that preserves correlations under privacy (for example, privately computed correlation summaries), augmentation ratios other than 1:1, fairness signals not tied to the balanced loss (E1), and conformal calibration under drift (E2).
