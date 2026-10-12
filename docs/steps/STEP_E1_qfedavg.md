# Extension E1: q-FedAvg vs FedAvg, per-bank fairness (optional, beyond the synopsis)

> **Report note:** this is an extension beyond the synopsis and must be written into the project report as such.

## 1. Goal
Test whether **fairness-aware aggregation (q-FedAvg)** gives the weakest bank a better fraud model than plain FedAvg, and at what cost to the other banks and to overall performance. Setting: real data only (the Arm 3 setting), 30 rounds × 1 local epoch, 5 seeds, the same model and data as Steps 8–9.

## 2. Concepts explained simply
- **Fairness in FL:** how evenly the shared model performs across banks. We measure it as the **worst bank's** fraud F1 on its own local test, and the **spread** (standard deviation) of F1 across the four banks.
- **q-FedAvg:** instead of averaging bank updates by data size, it gives more say to banks where the current global model has a **higher loss**. *Analogy:* a teacher who lets the struggling students steer the next lesson.
- **q:** the strength of that push. q = 0 means every bank gets an *equal* say (which is **not** our FedAvg, since FedAvg weights by row count); larger q leans harder towards high-loss banks.
- **L (step constant):** controls the size of each q-FedAvg update. Flower derives it as 1 / learning rate (1000 for our Adam clients), which froze training in a probe, so we tuned it on validation data.
- **Global-model loss:** each bank scores the *received* global model on its own training data before training and reports that one number. This is what q-FedAvg reacts to.

## 3. What we built
| File | Purpose |
|---|---|
| `ml/federated/client_app.py` | Each bank also reports `global-model-loss` (one scalar, before local training). FedAvg ignores it. |
| `ml/federated/server_app.py` | `OrderedQFedAvg` (Flower's QFedAvg with the fixed reply order of D-032) and `make_strategy()`; FedAvg stays the default. |
| `configs/e1_qfedavg.yaml` | q grid {0.5, 1, 2}, L grid {1, 3, 10}, tuning seed 42, final seeds 42–46. |
| `ml/experiments/e1_qfedavg.py` | Tune L per q on validation → 5-seed runs → per-bank tables, paired CIs vs FedAvg, chart. Resumable. |
| `tests/test_qfedavg.py` | 4 tests: default is FedAvg; settings and loss key; q = 0 is an unweighted average; q > 0 favours the higher-loss bank. |

## 4. How to run it
```powershell
python -m ml.experiments.e1_qfedavg                 # ~35 min (34 simulations)
python -m ml.experiments.e1_qfedavg --report-only   # rebuild tables/chart from saved runs
```

## 5. Evidence (real outputs, 2026-10-12, `results/extensions/e1_qfedavg/`)
**Probe (before the design, seed 42, 10 rounds):** with Flower's default L = 1000, q = 1 never learned (global PR-AUC 0.005 in every round), while q = 0 learned like FedAvg (0.705 → 0.721).

**Tuning on validation only** (`tuning_validation.csv`; pooled val F1 from summed bank counts): L = 1 was best for every q (val F1 0.7500 for q = 0.5, 1, 2; FedAvg's was 0.7556 in Step 8).

**Final, 5 seeds** (`summary_table.csv`; per-bank F1 on each bank's own local test):

| Config | Global F1 | Global PR-AUC | Worst-bank F1 | F1 spread (std) | Bank D F1 |
|---|---|---|---|---|---|
| FedAvg | 0.7961 ± 0.0063 | 0.7158 ± 0.0061 | 0.7333 ± 0.0373 | 0.0569 ± 0.0124 | 0.7500 ± 0.0000 |
| q-FedAvg q = 0 | 0.7949 ± 0.0069 | 0.7150 ± 0.0084 | 0.7000 ± 0.0456 | 0.0647 ± 0.0201 | 0.7167 ± 0.0456 |
| q-FedAvg q = 0.5 | 0.8076 ± 0.0088 | 0.7144 ± 0.0095 | 0.7000 ± 0.0456 | 0.0796 ± 0.0162 | 0.7000 ± 0.0456 |
| q-FedAvg q = 1 | 0.8045 ± 0.0059 | 0.7151 ± 0.0069 | 0.7167 ± 0.0456 | 0.0673 ± 0.0127 | 0.7167 ± 0.0456 |
| q-FedAvg q = 2 | 0.7954 ± 0.0094 | 0.7108 ± 0.0084 | 0.6705 ± 0.0085 | 0.0787 ± 0.0155 | 0.6833 ± 0.0373 |

**Paired vs FedAvg (95% CI, `paired_vs_fedavg.csv`):** for q = 0, 0.5 and 1, no metric differs clearly from FedAvg. For **q = 2**, worst-bank F1 is clearly **lower** (−0.0629, CI −0.1077 to −0.0180, lower in 5 of 5 seeds) and Bank D's F1 is clearly lower (−0.0667, CI −0.1129 to −0.0204).

**Bank D caught 3 of 4 local-test frauds in every configuration and seed**; its F1 changes come only from false alarms (seed 42: FedAvg TP 3 / FP 1 / FN 1; q = 2: TP 3 / FP 2 / FN 1).

**Why q-FedAvg had nothing to act on:** the mean global-model loss per bank (q = 1 runs) was nearly identical: max/min ratio 1.00 in round 1, 1.17 in round 10, 1.09 in round 30, when Bank D had the *lowest* loss (0.274).

**Reproducibility:** after adding the loss scalar, FedAvg still reproduces Step 9 bit-for-bit (all 5 seeds).

**Chart:** `results/extensions/e1_qfedavg/e1_per_bank_f1.png` (per-bank F1 bars, and worst-bank vs global F1).

## 6. Explain it to the guide (script)
"Sir, as an extension beyond our synopsis, I tested q-FedAvg, a fairness-aware version of federated averaging that gives more weight to banks where the shared model performs badly. I first found that Flower's default setting froze training with our optimiser, so I tuned its step constant on validation data only. Then I compared it with FedAvg over five seeds and measured each bank's fraud F1 on its own test customers. q-FedAvg did not make the system fairer: no setting improved the worst bank, and the strongest setting, q equals 2, made the worst bank clearly worse because of extra false alarms. The reason is that our balanced loss already makes every bank's loss almost the same, so q-FedAvg had no struggling bank to help; by the last round Bank D actually had the lowest loss. So for this system, plain FedAvg is the better choice, and we report this negative result honestly."

## 7. Likely viva questions
1. **What is q-FedAvg?** An FL aggregation rule that gives more weight to clients where the global model's loss is higher; q controls how strongly. q = 0 gives every client an equal say.
2. **Did it improve fairness?** No. No q improved the worst bank's F1; q = 2 made it clearly worse (−0.063, 95% CI −0.108 to −0.018).
3. **Why not?** The per-bank losses of the global model were nearly equal (max/min ratio ≤ 1.17) because our balanced weighted loss equalises the classes in every bank, so q-FedAvg had no signal; Bank D even had the lowest loss by round 30.
4. **Why did you tune L, and on what data?** Flower sets L = 1 / learning rate = 1000 for our Adam clients, which froze training (PR-AUC stayed 0.005). We chose L from {1, 3, 10} by pooled validation F1, never test data.
5. **Is q = 0 the same as FedAvg?** No: q = 0 averages banks equally, while our FedAvg weights them by row count. That's why both were run.

## 8. Limitations and honest notes
- **L = 1 was the edge of the tuning grid**; smaller L values were not tried and might behave differently.
- q-FedAvg's theory assumes plain gradient descent; our clients use Adam, so L is only an approximate step constant.
- **Bank D's local test has 4 frauds**, so its F1 moves in large steps (one false alarm ≈ 0.08); worst-bank F1 is therefore noisy.
- Only the real-data setting (Arm 3) was tested; augmented data was not repeated, because Step 9 found augmentation had no reliable effect.
- A possible follow-up (not run): a fairness signal less tied to the balanced loss, e.g. per-bank validation recall.

## 9. Next step
**Extension E2: conformal risk control** for a false-negative-rate bound, global vs per-bank, started only on your `continue`. Remember to add both E1 and E2 to the report as extensions beyond the synopsis.
