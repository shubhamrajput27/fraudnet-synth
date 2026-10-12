# Step 13 integration evidence: full six-arm runs started from the dashboard

Three full runs (all six arms, seeds 42–46) were started from the React dashboard as `admin`. Each went
Tier 1 (browser) → Tier 2 (gateway, stored in MongoDB `fraudnet_gateway`) → Tier 3 (FastAPI job) →
Tier 4 (four simulated banks). **All three are identical to the command-line Step 9 run** (180 metric rows each,
max |difference| 0.0 for precision, recall, F1, PR-AUC, ROC-AUC, accuracy, TP, FP, FN).

| Job | Started → finished (UTC) | Duration | Evidence here | Status of evidence |
|---|---|---|---|---|
| `20261012-142636-run-eed13e` | 2026-10-12 08:56:36 → 09:11:25 | 14 min 49 s | top-level files (01–04, `full_run_ui_output.txt`, `retake_output.txt`) | **Main evidence; used by the annexure.** |
| `20261012-140559-run-9d99cb` | 2026-10-12 08:35:59 → 08:55:28 | 19 min 29 s | `second_run_20261012-140559/` (03, 04, log) | 01 and 02 were overwritten (see below). |
| `20261012-013254-run-b5ac84` | 2026-10-11 20:02:54 → 20:31:47 | 28 min 53 s | `first_run_20261012-013254/` (01, 04, logs) | Driven by a script that was never committed; 02 and 03 missing. |

Duration differences come from other load on the laptop; the results are identical.

## Honest notes
- **First run:** its driver script and the annexure generator were not in the repository, so the evidence could not
  be reproduced. Both were rewritten in Step 13 (`services/gateway/scripts/full_run_ui.js`,
  `ml/evaluation/annexure.py`). Regenerating the annexure for this job reproduced all 7 tables and 11 of 12 figures
  exactly. The 12th differs only because its screenshot was retaken two minutes after the original annexure was made.
- **Third run was started by mistake.** A failed edit left the script without its new `--reopen` option, and the
  next command ran it anyway, so it started a new full run instead of re-opening the second one. It overwrote the
  second run's 01 and 02 screenshots. The run itself is valid, so it became the main evidence. The script now
  refuses unknown arguments.
- **`04_full_run_finished.png` is a retake.** The original capture was taken ~2 s before the run's end event
  arrived (the finish check matched "Succeeded" on an older history row), so it showed "Running". It was retaken
  with `--reopen`, which re-opens the finished run from history (rounds replayed from MongoDB); see
  `retake_output.txt`. The finish check now looks only at this run's own card.
- **Squashed charts in some screenshots** (`second_run_*/03`, an earlier retake of 04) were a capture artefact: the
  full-page capture resized the page while charts redrew. The capture now resizes first and waits 1.5 s.
