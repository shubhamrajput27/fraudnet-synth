# FraudNet-Synth — Live Demo Script

Click-by-click checklist for the live demo portion of the presentation (Slide 13,
`docs/presentation_deck.md`). Rehearse this at least once end-to-end before presenting —
`federated_augmented` with the reduced settings below takes roughly 30–60 seconds to complete on
a typical laptop, which is demo-friendly; the full default settings (10 rounds) take longer.

## Before the room fills up

1. Confirm MongoDB is running: `Get-Service MongoDB` (Windows) should show `Running`.
2. Start all three services — either run `.\start_demo.ps1` from the repo root (opens three
   labeled terminal windows), or manually:
   ```
   # Terminal 1
   ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000
   # Terminal 2
   cd gateway && npm start
   # Terminal 3
   cd dashboard && npm run dev
   ```
3. Open the dashboard URL (default `http://localhost:5173`) in a browser window, sized so the
   audience can read it.
4. Log in once *before* presenting so the demo itself starts from a warm, working session (log
   back out right before if you want to show the login step deliberately).

## The demo path

1. **Login** — show the login form, sign in with the `ADMIN_USERNAME`/`ADMIN_PASSWORD` from
   `.env`. *Say:* "Minimal single-login auth — this is a research demo, not a multi-tenant
   product."
2. **Trigger Run tab** — select arm `federated_augmented`. Reduce `FedAvg rounds` to `4` and
   `Local epochs/round` to `1` for a fast, demo-friendly run (the committed results in
   `docs/phase7_results.md` use the full 10×2 settings — mention this explicitly so the audience
   knows the live numbers won't exactly match the report). Click **Trigger run**.
3. **Watch it train live** — point out:
   - The per-bank mode indicator badges: Banks A/B show "Augment (CTGAN)", Banks C/D show
     "Schema (LLM)".
   - The line chart updating round-by-round *as it happens* — this is real WebSocket push, not a
     page refresh.
   - *Say:* "Every one of these updates is a real training round completing on real data —
     nothing here is pre-recorded."
4. **Run History tab** — show the just-completed run in the table, plus any earlier runs.
5. **Comparison Grid tab** — show the six-arm bar chart. *Say:* "This is populated from whatever
   arms have actually been run — for the full, final numbers see the committed report."
6. **Synthetic Quality tab** — show the validation verdicts, specifically Bank B
   (`REJECT_TOO_FEW_ROWS`) and Bank C (`REJECT_MODE_COLLAPSE`). *Say:* "This is the shared
   validation layer catching real generation failures — not every synthetic batch makes it
   through, and that's the point." Click a "Candidates CSV" export button to show the download
   working.
7. **Test a Transaction tab** — select the run just completed, pick a bank, click "Load a random
   holdout transaction", then **Predict**. *Say:* "This is a real transaction from the held-out
   test set the model never saw during training — the label shown is for reference only, not fed
   to the model." Optionally load a second sample to show a different prediction.

## If something goes wrong live

- **A service didn't start / crashed**: check the terminal window for that service — most likely
  culprits are MongoDB not running (gateway won't start) or the orchestrator not running (gateway
  requests will 502). Restart the affected service; the dashboard itself doesn't need a reload.
- **A run gets stuck at "running"**: refresh the Run History tab — the underlying training is a
  real background thread and will finish on its own; there's no way to speed it up mid-run, so
  have a fallback: switch to discussing the committed `docs/phase7_results.md` numbers while it
  finishes.
- **Fallback if live demo genuinely fails**: the screenshots taken during Phase 6's Playwright
  browser testing (see `PLAN.md`'s Phase 6 progress log) prove the exact same flow worked
  end-to-end — mention this if you need to pivot to "here's what it looks like when it works."
