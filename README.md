# FraudNet-Synth

**Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation**

Final-year B.E. CSE project, PES Institute of Technology & Management, Shivamogga (VTU).
Team: Shubham Kumar Singh, Palleti Pradeepa, Prachi Yadav, Sathvik D. Guide: Dr. Chethan L S.

Four simulated banks hold private, non-IID slices of the ULB Credit Card Fraud dataset and jointly train one fraud classifier with **Flower / FedAvg**, sharing only model weights. Each bank can add synthetic fraud rows, and the method depends on how much fraud it holds: **CTGAN** for data-rich banks (Augment mode) and an **LLM via Groq**, seeing only aggregate statistics, for data-poor banks (Schema mode). Every synthetic row passes a **shared validation gate**. The system is evaluated across six arms: {isolated, federated, centralized} × {real only, real + synthetic}.

## Results (5 seeds, global test set with 95 frauds)
| Arm | Fraud F1 | PR-AUC |
|---|---|---|
| 1 · Isolated, real (Bank A / Bank D) | 0.809 / 0.456 | 0.705 / 0.662 |
| 3 · **Federated, real** | **0.796 ± 0.006** | **0.716 ± 0.006** |
| 4 · Federated, real + synthetic | 0.797 ± 0.017 | 0.717 ± 0.007 |
| 5 · Centralized, real (illegal upper bound) | 0.775 ± 0.028 | 0.727 ± 0.004 |

- Federation matches pooled training on F1 without sharing data, and lifts the smallest bank most (Bank D +0.34 F1 in every seed).
- Synthetic augmentation showed **no reliable benefit** (33 of 36 paired comparisons unclear); this is reported as a finding.
- Accuracy is ≈ 99.9% for every model and is not a headline metric.

Full tables, figures and provenance: [docs/RESULTS_ANNEXURE.md](docs/RESULTS_ANNEXURE.md).

## Architecture
React dashboard → Node/Express gateway (JWT, roles, MongoDB) → FastAPI orchestrator (CTGAN, Groq, validation, Flower) → four banks. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (Windows, CPU only)
Prerequisites: Python 3.12, Node.js 22, MongoDB Community Server running as a service, Microsoft Edge (for the UI checks).
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            # includes the CPU-only torch index
copy .env.example .env                     # fill in the keys and passwords (see comments)
cd services\gateway; npm install; npm run seed; cd ..\..
cd frontend; npm install; npm run build; cd ..
```
Place `creditcard.csv` (Kaggle, "Credit Card Fraud Detection", ULB) in `data/raw/`, then rebuild the pipeline:
```powershell
python -m ml.data.prepare                  # split + four bank shards
python -m ml.augmentation.ctgan_engine     # Banks A, B (~2 min each)
python -m ml.augmentation.llm_engine       # Banks C, D (needs GROQ_API_KEY; cached afterwards)
python -m ml.validation.gate               # shared validation layer
python -m ml.experiments.run_all_arms      # six arms x 5 seeds (~25 min)
python -m ml.evaluation.annexure --job <dashboard job id>
```

## Demo (no internet needed)
```powershell
python -m demo.preflight                   # must say READY
python -m demo.start                       # starts everything and opens http://127.0.0.1:8200
```
5-minute script: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md). Multi-laptop federation: [docs/MULTI_DEVICE_SETUP.md](docs/MULTI_DEVICE_SETUP.md), [docs/DEMO_SCRIPT_8B.md](docs/DEMO_SCRIPT_8B.md).

## Tests
```powershell
python -m tests.run_all                    # Python (pytest), gateway (node --test), dashboard build
```

## Documentation
| Document | For |
|---|---|
| [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) | beginner introduction to the whole project |
| [docs/steps/](docs/steps/) | one "show your guide" document per build step (0–14, 8B) |
| [docs/VIVA_QA.md](docs/VIVA_QA.md) | 100 viva questions with answers |
| [docs/DECISIONS.md](docs/DECISIONS.md) | every design decision, alternatives and reasons |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | every technical term |
| [docs/PROGRESS.md](docs/PROGRESS.md) | step status |

## Honest limitations
One public, anonymised dataset and one data split; small fraud counts make per-bank results noisy (Bank D's local test has 4 frauds); the multi-laptop setup is a live demo, while reported results come from single-machine simulation; differential privacy and secure aggregation are future scope.
