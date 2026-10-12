# Step 14: Testing, documentation and demo hardening (Phase 8)

## 1. Goal
Make the project safe to present and easy to understand. Every test suite now runs from one command; the viva demo works with no internet and is started by one command after an automatic pre-flight check; and the final README, architecture document, 5-minute demo script and a 100-question viva sheet are in place.

## 2. Concepts explained simply
- **Regression test:** a test that catches something that used to work breaking after a change. *Analogy:* a checklist the pilot runs before every flight, not just the first.
- **Pre-flight check:** a script that verifies everything the demo needs before you're in front of examiners, and tells you how to fix anything missing.
- **Offline mode:** forcing libraries never to reach the internet (e.g. `HF_HUB_OFFLINE=1`), so a hidden download can't hang the demo.
- **Smoke test:** a quick end-to-end check that the main paths work ("does it switch on without smoke?").

## 3. What we built
| File | Purpose |
|---|---|
| `tests/test_metrics.py` | 7 tests of precision, recall, F1, PR-AUC, ROC-AUC, the confusion matrix and threshold tuning, against answers worked out by hand. |
| `tests/run_all.py` | Runs pytest, the gateway tests and the dashboard build; prints a summary. |
| `demo/preflight.py` | 19 PASS/WARN/FAIL checks: secrets, bank data, LLM caches, CTGAN models, predict model, offline embedding model, results, gateway packages, dashboard build, MongoDB + users, ports. |
| `demo/start.py` | Pre-flight → starts FastAPI + gateway with offline flags → waits for health → opens the dashboard; Ctrl+C stops both. `--smoke` does an automatic tour and stops. |
| `demo/viva_sheet.py` + `docs/viva_top20_source.md` | Builds `docs/VIVA_QA.md` from 20 curated questions + every step doc's viva questions. |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/DEMO_SCRIPT.md`, `docs/VIVA_QA.md` | Final documentation. |

## 4. How to run it
```powershell
python -m tests.run_all            # all tests
python -m demo.preflight           # is the laptop ready?
python -m demo.start               # the demo (Ctrl+C to stop)
python -m demo.viva_sheet          # rebuild the viva sheet after editing step docs
```

## 5. Evidence (real outputs, 2026-10-12)
**All tests, one command:**
```
Suite                      Result        Time
Python (pytest)            77 passed        23s
Gateway (node --test)      9 passed          6s
Dashboard build (vite)     built             1s
ALL PASSED
```
**Offline demo** (Python's internet access blocked by pointing it at a dead proxy, `127.0.0.1:9`, with localhost exempt; a direct request to huggingface.co then failed with `URLError`):
```
19 passed, 0 warnings, 0 failed -> READY for the demo
Health: {'gateway': 'ok', 'mongo': 'ok', 'orchestrator': 'ok'}
  login as guide: OK | banks: bank_a, bank_b, bank_c, bank_d
  six-arm results: Arm 3 (federated, real) F1 0.7961 ± 0.0063, PR-AUC 0.7158 ± 0.0061
  predict fraud example (true label 1): is_fraud=True, logit 8.5424 vs threshold 3.75
Smoke tour passed.
```
**Generation replay offline** (Bank C, LLM mode): 300 candidates rebuilt from the cache with **0 Groq calls**, 234 validated, byte-identical to the experiment files (job folder `results/api_jobs/offline_replay_check_20261012/`).

**Viva sheet:** `docs/VIVA_QA.md`, 20 curated + 80 step questions from 16 step documents (including this one).

## 6. Explain it to the guide (script)
"Sir, Step 14 makes the project ready for the viva. All tests, 77 Python tests, 9 gateway tests and a build check of the dashboard, now run with one command and pass. I added tests for our metric code with answers I worked out by hand. For the demo I wrote a pre-flight check with 19 checks and a launcher that starts everything with one command. To prove it works without internet, I blocked Python's internet access and ran it: every check passed, and login, results and a fraud prediction all worked. I also replayed the LLM bank's synthetic data generation offline, and it made zero calls to Groq and reproduced the same files. Finally I wrote the README, an architecture document, a five-minute demo script with each teammate's part, and a viva sheet with 100 questions."

## 7. Likely viva questions
1. **How do you know your metrics are computed correctly?** `tests/test_metrics.py` checks them on a 10-transaction example whose precision, recall, F1, PR-AUC and ROC-AUC were worked out by hand.
2. **What happens if the internet fails during the viva?** Nothing: synthetic data comes from saved files and the LLM cache, the embedding model loads from the local cache with offline flags, and the services only use localhost. We tested this with Python's internet blocked.
3. **How do you check the laptop is ready?** `python -m demo.preflight` runs 19 checks and prints a fix for anything that fails.
4. **How many tests are there and what do they cover?** 77 Python tests (data split, partition, CTGAN/LLM engines, validation gate, model, metrics, federated threshold, privacy invariant, demo data checks, statistics, FastAPI API) and 9 gateway tests (login, roles, proxy, MongoDB, WebSocket relay, export scope), plus a dashboard build check.
5. **How is the viva sheet kept consistent with the step documents?** A script extracts every step's viva questions automatically, so editing a step doc and re-running it updates the sheet.

## 8. Limitations and honest notes
- The offline test **blocked Python's internet via a dead proxy**; it did not switch off Wi-Fi. It covers every Python library that downloads (Hugging Face, Groq). Node only talks to localhost and MongoDB.
- The predict model file, the LLM cache and the CTGAN synthesizers are **not in Git** (size / privacy). A fresh clone must restore them or rebuild them; the pre-flight check says how.
- The dashboard-level UI tests (`ui_check.js`, `full_run_ui.js`) need Edge and the running services, so they're not part of `tests.run_all`; they were run in Steps 12–13.
- The Step 8B four-laptop test is still pending (planned with the team).
- The demo uses existing results; a fresh full run takes 15–30 minutes, too long for a 5-minute demo.

## 9. Next step
The core build plan is complete. Remaining: the Step 8B four-laptop test and two full rehearsals, and updating the project report and synopsis (including the multi-laptop demo). The optional extensions **E1 (q-FedAvg)** and **E2 (conformal risk control)** start only if you confirm; if built, they must also be written into the report as extensions beyond the synopsis.
