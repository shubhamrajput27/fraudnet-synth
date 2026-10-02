# FraudNet-Synth

**Privacy-Preserving Fraud Detection using Federated Learning and Synthetic Data Augmentation**

Final-year B.E. CSE project, PES Institute of Technology & Management, Shivamogga (VTU).
Team: Shubham Kumar Singh, Palleti Pradeepa, Prachi Yadav, Sathvik D. Guide: Dr. Chethan L S.

Four simulated banks hold private, non-IID shards of the ULB Credit Card Fraud dataset and jointly train a
fraud classifier with Flower/FedAvg without sharing data. Data-rich banks (A, B) augment their fraud examples
with CTGAN (Augment Mode). Data-poor banks (C, D) use an LLM via Groq (Schema Mode). All synthetic rows pass a
shared validation layer before use.

> Status: under construction. See [docs/PROGRESS.md](docs/PROGRESS.md).

## Setup (Windows, CPU-only)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env   # then fill in GROQ_API_KEY and GROQ_MODEL
```

1. Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection", ULB Machine Learning Group) and place it at `data/raw/creditcard.csv`.
2. MongoDB Community Server must be running locally (default `mongodb://localhost:27017`).

## Documentation
- [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md): beginner introduction to the whole project
- [docs/PROGRESS.md](docs/PROGRESS.md): step status
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and the reasons for them
- [docs/GLOSSARY.md](docs/GLOSSARY.md): terms
- [docs/steps/](docs/steps/): one "show your guide" file per step
