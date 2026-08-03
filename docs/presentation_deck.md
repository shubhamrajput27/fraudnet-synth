# FraudNet-Synth — Presentation Deck Outline

Structured Markdown, one `##` per slide: a short **Slide** line for what goes on the slide itself
(keep slides sparse — bullets, not paragraphs) and **Speaker notes** for what to actually say.
Paste into PowerPoint/Google Slides/Keynote, or render as-is with a tool like
[Marp](https://marp.app/) if you want literal slides from this file. Figures referenced below are
already committed under `docs/phase7_figures/`.

Estimated: ~16 slides, 12–15 min talk + live demo + Q&A.

---

## Slide 1 — Title

**Slide:**
- FraudNet-Synth: Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation
- Team: Palleti Pradeepa, Prachi Yadav, Sathvik D, Shubham Kumar Singh
- Guide: Dr. Chethan L S — Dept. of CSE, PESITM Shivamogga, VTU

**Speaker notes:** Introduce the team and one-sentence framing: "Four simulated banks train a
shared fraud model without ever sharing a single transaction row."

---

## Slide 2 — The Problem

**Slide:**
- Fraud data is extremely imbalanced (~0.17% positive on the benchmark used here) and often too
  scarce at any single institution to train a strong model alone
- Transaction data can't be legally/practically pooled across institutions
- Two separate literatures address these separately: federated learning (privacy) and synthetic
  augmentation (scarcity) — rarely combined *and* evaluated honestly together

**Speaker notes:** Set up why neither federated learning alone nor synthetic augmentation alone
is the full answer — motivate why this project combines both.

---

## Slide 3 — Research Gap

**Slide:**
- Published work shows CTGAN augmentation can *degrade* downstream fraud models — no method is
  universally best
- **Our contribution:** dual-mode, *client-adaptive* generation — CTGAN for data-rich clients,
  LLM schema-and-few-shot generation for data-poor clients — selected per client by how much real
  fraud data it holds
- Gated by a shared validation layer with thresholds frozen *before* results are seen
- Evaluated across a complete isolated / federated / centralized grid, with and without
  augmentation (six arms)

**Speaker notes:** This is the thesis of the whole project — say it once, clearly, and refer back
to it during the results slide.

---

## Slide 4 — System Architecture

**Slide:** (reproduce the four-tier diagram from `CLAUDE.md` / `docs/final_report.md` §3)
- Tier 1: React Dashboard
- Tier 2: Node/Express Gateway (auth, run management, MongoDB)
- Tier 3: FastAPI Orchestrator (augmentation + federated training)
- Tier 4: 4 simulated bank clients, private non-IID shards

**Speaker notes:** Emphasize the privacy invariant: only serialized model weights ever cross a
client boundary — never a raw or synthetic row. This held throughout every phase.

---

## Slide 5 — Data & Non-IID Partition

**Slide:**
- ULB Credit Card Fraud dataset: 284,807 rows, 492 fraud → 283,726 clean after dedup
- Dirichlet fraud-skew + deliberate quantity-skew partition, seed 42
- Bank A: 79,621 rows / 310 fraud · Bank B: 79,349 / 38 · Bank C: 34,005 / 15 · Bank D: 34,005 / 15
- Banks C/D's scarcity (15 fraud rows each) is *by design* — it's what makes Schema Mode necessary

**Speaker notes:** The shard sizes weren't arbitrary — call out that they were computed from the
real dataset and explicitly approved before use, and that the scarcity is the whole reason two
different generation modes exist.

---

## Slide 6 — Client-Adaptive Augmentation

**Slide:**
- **Augment Mode** (Banks A, B — data-rich): CTGAN, trained per-client on real fraud rows only
- **Schema Mode** (Banks C, D — data-poor): Groq LLM, prompted with aggregate statistics +
  correlations + synthetic prototype rows — **never a real transaction row**
- V1–V28 are anonymized PCA components with no semantic meaning an LLM could otherwise ground on

**Speaker notes:** Stress the "never a real row" point — it resolves a real tension in the
original design between wanting few-shot examples and never leaking raw data.

---

## Slide 7 — Shared Validation Layer

**Slide:**
- Pandera → structural/range checks
- SDMetrics `QualityReport` → fidelity vs. real fraud rows
- SDMetrics `NewRowSynthesis` → novelty + PII/leakage guard
- sentence-transformers → mode-collapse detection within a batch
- Thresholds **frozen before any arm was evaluated**

**Speaker notes:** This is the gate that makes "client-adaptive" mean something — augmentation
isn't assumed to help, it's tested and can fail the gate.

---

## Slide 8 — A Real Finding: Two Batches Rejected

**Slide:**
- **Bank B rejected**: CTGAN, trained on only 38 real fraud rows, generated `Time` values ~100x
  the real range
- **Bank C rejected**: mode collapse — one near-duplicate synthetic pair
- Banks A and D passed cleanly

**Speaker notes:** This is a good slide to linger on — it demonstrates the validation layer
*actually works* rather than rubber-stamping everything, and it's an honest, reportable result
per the project's own evaluation philosophy (never hide an inconvenient finding).

---

## Slide 9 — Federated Training Setup

**Slide:**
- Classifier: small PyTorch MLP, input(30)→64→32→1
- Real Flower (`flwr`) `NumPyClient` + `FedAvg.aggregate_fit` for aggregation
- Per-client `StandardScaler` — never pooled, even at evaluation time
- 10 FedAvg rounds × 2 local epochs, seed 42

**Speaker notes:** Optionally mention the ray/Windows/Python 3.13 blocker as a "real engineering
problem we hit and solved" anecdote if time allows — shows genuine implementation depth.

---

## Slide 10 — Six-Arm Results

**Slide:** (embed `docs/phase7_figures/six_arm_comparison.png`)
- Table: F1 / Precision / Recall / AUC for all six arms (see `docs/phase7_results.md`)
- **federated_augmented wins on F1: 0.6146**

**Speaker notes:** Walk the table left to right: isolated (privacy-preserving lower bound) →
federated → centralized (non-privacy-preserving upper bound). Point out federated beats isolated
on every metric, and federated_augmented beats everything including centralized.

---

## Slide 11 — Convergence Over Rounds

**Slide:** (embed `docs/phase7_figures/federated_augmented_convergence.png`)
- F1/precision/recall per FedAvg round
- Trend still improving at round 10

**Speaker notes:** Note this suggests more rounds would likely help further — a natural future-
work point, not a weakness to gloss over.

---

## Slide 12 — Comparison to Literature

**Slide:**
- Accuracy exceeds literature reference points (~91%/~95%) — but accuracy is misleading here
  (>94% trivially on this imbalanced dataset)
- F1 vs. a "FedFraud" non-IID reference (0.90): ours reaches 0.61 — a real, unresolved gap
- Named, unconfirmed contributors: smaller epoch/round budget, more extreme non-IID split, only
  half the banks' augmentation surviving validation

**Speaker notes:** This is the slide where the project's honesty matters most — don't spin the
gap, name it and explain why it's plausible, per the evaluation philosophy repeated throughout
this deck.

---

## Slide 13 — Live Demo

**Slide:**
- (transition slide — mostly blank, just "Live Demo" and the dashboard URL)

**Speaker notes:** Switch to the actual running dashboard. Suggested demo path (see
`docs/demo_script.md` for the detailed click-by-click script): log in → trigger
`federated_augmented` with a small round count → show the live WebSocket chart updating → show
the quality panel → run "test a transaction" against a completed run.

---

## Slide 14 — Limitations

**Slide:**
- CPU-only, laptop-scale epoch/round budget
- CTGAN failed on Bank B's tiny real fraud count
- Fixed 0.5 decision threshold interacts with extreme class imbalance (centralized_real artifact)
- Groq free-tier daily token budget caps Schema Mode throughput

**Speaker notes:** Keep this brief and factual — each point traces to a specific, already-
investigated finding in the report, not a vague hedge.

---

## Slide 15 — Future Scope

**Slide:**
- Explicitly out of scope this semester: GNNs/FinGraphFL, differential privacy, secure
  aggregation, robust/trust-aware aggregation, IEEE-CIS replication, >4 clients
- Natural next steps: larger epoch/round budget, alternative generator for Bank B's failure mode,
  a sharper mode-collapse metric

**Speaker notes:** Frame the "out of scope" list as deliberate scoping decisions, not omissions —
they were named and set aside on purpose from the start.

---

## Slide 16 — Conclusion / Thank You

**Slide:**
- Complete, working, privacy-preserving pipeline — end-to-end from data to a live browser demo
- Headline finding: federated + validated client-adaptive augmentation beats every other
  configuration tested, honestly arrived at
- Thank you — Q&A

**Speaker notes:** Close on the headline finding one more time. Open the floor for questions.

---

## Appendix: anticipated Q&A

- **"Why not just use accuracy?"** → point to the ~0.17%-positive dataset; a trivial
  always-legitimate classifier scores ~99.8%.
- **"Why did you use CPU only?"** → hard project constraint (zero infrastructure cost,
  single-machine simulation) — also why the ray/Windows blocker mattered.
- **"Why does Bank B's augmentation fail but Bank A's doesn't?"** → real fraud row count (38 vs.
  310) — CTGAN needs more data than Bank B had.
- **"Isn't a 0.61 vs 0.90 F1 gap a problem?"** → named honestly in §12/Slide 12; not hidden, with
  plausible (not confirmed) explanations offered.
