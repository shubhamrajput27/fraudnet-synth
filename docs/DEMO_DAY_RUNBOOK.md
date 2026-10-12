# Demo-Day Runbook: COE Final-Year Demonstration, Tue 13/10/2026, 1:30 PM

One page that ties the existing demo documents together. The detailed scripts are in:
- [DEMO_SCRIPT_8B.md](DEMO_SCRIPT_8B.md): live federated run on four laptops (who says what)
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md): dashboard walkthrough (who says what)
- [MULTI_DEVICE_SETUP.md](MULTI_DEVICE_SETUP.md): teammate setup and troubleshooting
- [VIVA_QA.md](VIVA_QA.md) and [RESULTS_ANNEXURE.md](RESULTS_ANNEXURE.md): answers and numbers. Quote numbers **only** from these files.

**Rule for today: no new features, no code changes.** Only rehearse and fix what is actually broken.

## 1. Status at 00:45 on demo day (checked, not assumed)

| Item | Status |
|---|---|
| `python -m demo.preflight` | ✅ 17 PASS, 0 FAIL → READY (2 warnings: services were already running) |
| Dashboard smoke tour (login, banks, six-arm results, predict) | ✅ passed; the fraud example was flagged as fraud |
| One-laptop fallback (`ml.federated.demo.fallback`) | ✅ completed 2026-10-12 20:04 in 177 s, logs in `results/demo_8b/fallback_20261012_195939/` |
| Two-laptop test (Shubham + Prachi) | ✅ 2026-10-11 |
| **Four-laptop test** | ❌ **never run**: this morning's top priority |
| Clean demo folder `C:\Projects\fraudnet-demo` | ✅ exists |

## 2. Timeline for today

| Time | What | Who |
|---|---|---|
| Night | Sleep. Charge all 4 laptops, the phone and a power bank. | All |
| 09:00 | Pradeepa and Sathvik: receive their zip by pen drive, unzip, `check_my_data` must pass ([MULTI_DEVICE_SETUP.md](MULTI_DEVICE_SETUP.md) Parts 1–2). Delete the zips, wipe the pen drive. | Shubham + each teammate |
| 09:45 | **Rehearsal 1:** full four-laptop run on Shubham's phone hotspot, speaking the [DEMO_SCRIPT_8B.md](DEMO_SCRIPT_8B.md) lines. Note every problem. | All |
| 10:30 | Fix only what broke. Then **Rehearsal 2:** 8B run + dashboard script ([DEMO_SCRIPT.md](DEMO_SCRIPT.md)), timed end to end. | All |
| 11:30 | **Record a backup video** of one good run (Win + G → record), and copy it to a pen drive. | Shubham |
| 11:45 | Freeze. Lunch. Nobody changes code or data after this. | All |
| 12:45 | At the venue: connect to the hotspot, run the pre-demo checks (§3). | All |
| 13:30 | Demo (§4). | All |

## 3. Pre-demo checks at the venue (15 minutes before)

**Every laptop:** charger plugged in · other apps closed · Windows Update paused · sleep set to "Never" · connected to **Shubham's hotspot** (not college Wi-Fi).

**Teammates (Prachi = bank_b, Pradeepa = bank_c, Sathvik = bank_d):**
```powershell
python -m ml.federated.demo.check_my_data --bank bank_x      # must print checksums OK, others empty
```

**Shubham (server + Bank A), in `C:\Projects\fraudnet-demo`, main working folder closed:**
```powershell
$PY = "C:\Users\shubh\Desktop\Shubham Major Project\fraudnet-synth\.venv\Scripts\python.exe"
ipconfig                                    # tell the team the hotspot IPv4 address
& $PY -m ml.federated.demo.check_my_data --bank bank_a --server-laptop
```
Then open the three windows from [MULTI_DEVICE_SETUP.md](MULTI_DEVICE_SETUP.md) Part 4 (server, Bank A client, `run_demo`). Teammates start `start_client --bank bank_x --server <IP>`.

## 4. Demo order (about 10 minutes)

1. **Part A: live federated run on 4 laptops** (~5 min), following [DEMO_SCRIPT_8B.md](DEMO_SCRIPT_8B.md). This proves the model travels and the data stays.
2. Stop the 8B windows. Shubham switches to the main working folder:
   ```powershell
   .venv\Scripts\python -m demo.preflight     # must say READY
   .venv\Scripts\python -m demo.start         # opens http://127.0.0.1:8200
   ```
   *(Better: start `demo.start` before the examiners arrive and keep the browser minimised. It uses ports 8100/8200, Flower uses 9092/9093, so they don't clash.)*
3. **Part B: dashboard** (~5 min), following [DEMO_SCRIPT.md](DEMO_SCRIPT.md): Overview → Runs (live) → Six-arm results → Test a transaction.
4. Close with limitations (honest): demo-only multi-laptop setup, reported numbers come from single-machine runs, synthetic data showed no reliable gain.

## 5. Fallback ladder: if a level fails, go down one step calmly

| Level | What | Command / where |
|---|---|---|
| 1 | Four laptops live on the hotspot | §3–§4 |
| 2 | Fewer laptops (e.g. 2–3 that connect) | `run_demo --expected-banks N` |
| 3 | **One-laptop fallback**: 5 processes, same code | `.venv\Scripts\python -m ml.federated.demo.fallback` (~3 min) |
| 4 | Dashboard with saved results and run history | `python -m demo.start` (needs no internet) |
| 5 | Saved screenshots and tables | `results/integration/`, `results/final_annexure/figures/`, [RESULTS_ANNEXURE.md](RESULTS_ANNEXURE.md) |
| 6 | Backup video | pen drive |

**What to say if something fails:** "The network is not cooperating; this is the same code running on one laptop as five separate processes." Then continue. Examiners care more about how you handle it than about the failure.

## 6. Quick fixes

| Problem | Fix |
|---|---|
| Teammate's client can't connect | Same hotspot? Correct IP? Firewall rule on the server laptop with `-Profile Any`? Some phones block device-to-device traffic: try another phone's hotspot. |
| Client "REFUSING TO START" | Read the message; usually the zip wasn't unzipped into `data\clients\`. |
| Dashboard won't load | Look at the `demo.start` window; run `demo.preflight` and follow its fix lines. |
| Port 8100/8200 busy | An old demo is still running: close that window (or use it). |
| Login fails | Passwords are in `.env` (`GATEWAY_PW_...`). |
