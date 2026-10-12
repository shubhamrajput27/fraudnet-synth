# FraudNet-Synth: 5-minute demo script (dashboard)

Everything below works **offline** (`python -m demo.start`). The multi-laptop federation demo has its own script: `docs/DEMO_SCRIPT_8B.md`.

## Before the examiners arrive (10 minutes)
1. `python -m demo.preflight` must end with **READY** (fix any FAIL line it prints).
2. `python -m demo.start` and leave that window open; the browser opens http://127.0.0.1:8200.
3. Log in once as `admin`, open **Runs (live)**, tick **only Arm 3**, seeds `42`, and press **Start run** so a short live run (about 1 minute) is ready to show; leave it on screen. Then sign out.
4. Have `docs/RESULTS_ANNEXURE.md` and `docs/VIVA_QA.md` open in another window.

## The five minutes
| Time | Who | Screen | What to say (in your own words) |
|---|---|---|---|
| 0:00–0:45 | **Shubham** | Login as `guide`, **Overview** | "Four banks hold private slices of a real credit-card dataset; only 0.17% of transactions are fraud. Banks A and B have more fraud examples and use CTGAN; C and D have very few and use an LLM. No bank's data ever leaves it." |
| 0:45–1:30 | **Prachi** | Overview: Bank A/B cards, pass-rate chart | "CTGAN is a forger-versus-detective network trained inside each rich bank on its own fraud rows. Bank B learnt from 82 fraud rows; 814 of its 1,000 synthetic rows passed the validation gate." |
| 1:30–2:15 | **Pradeepa** | Overview: Bank C/D cards, rejection table | "Banks C and D have only 28 and 18 training frauds. The LLM never saw a real row, only averages and percentiles. It matched the column centres but not the relationships between columns, and some rows were rejected as near-duplicates." |
| 2:15–3:15 | **Shubham** (sign in as `admin`) | **Runs (live)**: the prepared Arm 3 run | "This is federated averaging: each round the server sends the model, every bank trains on its own data and sends back only 17,478 bytes of weights, and the server averages them. You can see the 30 rounds and each bank's loss." |
| 3:15–4:15 | **Sathvik** | **Six-arm results** | "Headline metrics are fraud precision, recall, F1 and PR-AUC; accuracy is about 99.9% for every model, so it's greyed out. Over 5 seeds, federated F1 is 0.796, as good as pooling all data, and Bank D jumps from 0.456 alone. Synthetic data showed no reliable gain, and we report that honestly: 33 of 36 comparisons were unclear." |
| 4:15–5:00 | **Shubham** | **Test a transaction** → Load a fraud example → Score | "The federated model flags this real fraud from the test set: score 8.54 against a threshold of 3.75. A full six-arm run started from this dashboard gave results identical to our command-line run." |

## If something goes wrong
| Problem | Do this |
|---|---|
| Dashboard won't load | Look at the `demo.start` window; run `python -m demo.preflight` and follow its fix lines. |
| A live run is slow | Talk over the prepared finished run in **Run history** (it replays from MongoDB). |
| Login fails | Passwords are in `.env` (`GATEWAY_PW_...`); after 20 login attempts in 15 minutes the gateway refuses more until the window passes. |
| No internet | Nothing changes: the demo needs no internet. |
| Laptop fails completely | Show the screenshots in `results/integration/` and `results/dashboard_demo/` and the tables in `docs/RESULTS_ANNEXURE.md`. |
