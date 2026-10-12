# Extension E2: conformal risk control of the miss rate, global vs per-bank (optional, beyond the synopsis)

> **Report note:** this is an extension beyond the synopsis and must be written into the project report as such.

## 1. Goal
Replace "the threshold that maximises F1" with a threshold that **guarantees a maximum expected miss rate α** (the share of frauds not flagged), using **conformal risk control (CRC)**. Compare **global** calibration (all banks' validation frauds, combined through counts only) with **per-bank** calibration (each bank on its own validation frauds), and measure the price in false alarms. Model: federated Arm 3 (real data), seeds 42–46.

## 2. Concepts explained simply
- **Miss rate (false-negative rate, FNR):** missed frauds ÷ all frauds. Our F1-tuned threshold misses about 24% of frauds on the global test.
- **Conformal risk control:** choose the strictest cut-off t whose calibration frauds satisfy **(misses + 1) / (n + 1) ≤ α**. If new frauds resemble the calibration frauds (*exchangeability*), the **expected** miss rate is ≤ α. It's a guarantee **on average**, not for every single test set. (Method: Angelopoulos et al., 2022; verify at the source before citing.)
- **Price of small samples:** the "+1" terms mean n calibration frauds can never certify α below **1/(n+1)**. Bank D has 4 validation frauds, so its minimum is α = 0.20; for anything stricter it must flag *every* transaction.
- **Global calibration through counts:** each bank sends only how many of its validation frauds fall below each fixed cut-off (the same mechanism as the federated threshold, D-031); the server adds them up. No scores or rows leave a bank.
- *Analogy:* airport security sets the scanner from bags with known contraband so that at most 1 in 10 gets through, accepting more false alarms on innocent bags.

## 3. What we built
| File | Purpose |
|---|---|
| `ml/evaluation/conformal.py` | The CRC rule on the fixed grid (`crc_threshold`), miss counts, minimum feasible α. |
| `ml/experiments/e2_conformal.py` | Re-runs Arm 3 with the final model saved (must equal Step 9, otherwise it stops); global and per-bank CRC thresholds; fixed-test evaluation; guarantee check; tables and chart. |
| `configs/e2_conformal.yaml` | α ∈ {0.05, 0.10, 0.20}, seeds 42–46, 2,000 re-splits. |
| `tests/test_conformal.py` | 6 tests: hand-checked thresholds, counts agree with the federated counts, infeasible small banks, a 3,000-trial simulation showing the average miss rate ≤ α. |

## 4. How to run it
```powershell
python -m ml.experiments.e2_conformal     # ~6 min the first time (5 simulations), then seconds
```

## 5. Evidence (real outputs, 2026-10-12, `results/extensions/e2_conformal/`)
**Global test set (95 frauds), global calibration (49 validation frauds), mean ± std over 5 seeds** (`summary_table.csv`):

| Threshold | Miss rate | Frauds caught per seed | Precision | Transactions flagged |
|---|---|---|---|---|
| F1-tuned (Step 8/9) | 0.240 ± 0.022 | 70 74 74 70 73 /95 | 0.838 ± 0.036 | 0.15% |
| CRC α = 0.20 | 0.128 ± 0.009 | 83 83 82 84 82 /95 | 0.076 ± 0.032 | 2.27% |
| CRC α = 0.10 | 0.061 ± 0.014 | 90 90 89 90 87 /95 | 0.009 ± 0.004 | 18.62% |
| CRC α = 0.05 | 0.055 ± 0.005 | 90 90 90 90 89 /95 | 0.006 ± 0.002 | 29.37% |

**Per bank at α = 0.10** (each bank's own local test): with **global** calibration, every bank flags about 18–19% of its transactions (miss rates A 0.064, B 0.118, C 0.000, D 0.000). With **per-bank** calibration, A and B flag about 24%, while **Banks C and D must flag 100%**: with 6 and 4 validation frauds they can't certify α below 0.143 and 0.200. At α = 0.20, per-bank calibration costs Bank D 28.9% flagged versus 2.3% with global calibration.

**Guarantee check** (`guarantee_check.csv`; 2,000 random re-splits of each pool's val+test frauds × 5 seeds): the average miss rate stayed within α for every pool and α, e.g. all banks pooled: 0.035 / 0.093 / 0.198 for α = 0.05 / 0.10 / 0.20. Bank D at α = 0.20 averaged 0.2030 with a Monte Carlo standard error of 0.0024 (within noise; at its minimum feasible α the bound is tight by construction). Individual splits often exceeded α (e.g. 46% of pooled splits at α = 0.10), which is expected for an *average* guarantee.

**Model check:** all 5 re-run models reproduced Step 9's global-test results exactly before being analysed.

**Chart:** `results/extensions/e2_conformal/e2_conformal.png`.

## 6. Explain it to the guide (script)
"Sir, as a second extension beyond our synopsis, I used conformal risk control to choose the decision threshold with a guarantee on how many frauds we miss. Our F1-tuned threshold misses about 24% of frauds. With conformal risk control at a 10% target, the federated model missed 6.1% of frauds on the global test, but it had to flag about 19% of all transactions, so most alerts are false alarms. The interesting federated result is calibration: the small banks C and D have only 6 and 4 validation frauds, so on their own they can't certify a 10% miss rate and would have to flag every transaction. When the four banks pool only their counts, they get the guarantee with about 19% alerts each, so federation helps here too. I also checked the guarantee with 2,000 random re-splits, and the average miss rate stayed within the target."

## 7. Likely viva questions
1. **What does conformal risk control guarantee?** That the expected miss rate on new frauds is at most α, if new frauds are exchangeable with the calibration frauds. It's an average guarantee, not a promise for every test set.
2. **Why did α = 0.05 give 5.5% misses on the test set, above 5%?** The guarantee holds on average over calibration sets; one fixed split can land above α. Our resampling check averaged 3.5% at α = 0.05.
3. **Why must Banks C and D flag everything at α = 0.10?** With n calibration frauds the smallest certifiable α is 1/(n+1): 1/7 ≈ 0.143 for C and 1/5 = 0.20 for D.
4. **How is global calibration private?** Each bank sends only counts of its validation frauds below fixed cut-offs; the server sums them, as for the federated threshold.
5. **What's the cost of a strict guarantee?** Many false alarms: at α = 0.10, 18.6% of transactions are flagged and precision drops to about 0.01, versus 0.15% flagged and precision 0.84 with the F1-tuned threshold.

## 8. Limitations and honest notes
- The guarantee assumes **exchangeability**; real fraud changes over time (concept drift), which would break it.
- It controls the miss rate **on average**; single test sets can exceed α.
- Bank C's and D's local tests have only 6 and 4 frauds, so their realised miss rates (often 0.000) are very coarse.
- The resampling check re-uses each pool's val+test frauds, so its standard error is approximate. It is a check of the method only; no reported threshold is chosen on test data.
- Thresholds are restricted to the fixed 241-point grid, which can only make them more cautious.

## 9. Next step
The build plan and both optional extensions are complete. Remaining work is outside the code: the Step 8B four-laptop test, two rehearsals, and writing the report (including E1 and E2 as extensions beyond the synopsis).
