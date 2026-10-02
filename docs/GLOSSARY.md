# Glossary

Terms are added as each step introduces them. The step number is in brackets.

| Term | Plain meaning |
|---|---|
| **Virtual environment (venv)** [0] | A private, isolated copy of Python for this project, so its libraries don't clash with other projects. Lives in `.venv/`. |
| **Pinned versions / requirements.txt** [0] | A file listing the exact version of every library, so anyone can rebuild an identical setup. |
| **.env file** [0] | A local, git-ignored file holding secrets (API keys). `.env.example` is the safe template. |
| **Fraud detection model** [0] | A program that scores a transaction by how likely it is to be fraud. |
| **Class imbalance** [0] | One class (fraud) is far rarer than the other (genuine), which makes accuracy misleading. |
| **Precision / Recall / F1 / PR-AUC** [0] | Fraud-class metrics: correctness of alarms / share of frauds caught / balance of the two / quality across all thresholds. Detailed in Step 1. |
| **Federated Learning (FL)** [0] | Training one shared model across many data owners without moving their data; only model weights travel. |
| **Client / Server / Round** [0] | A data owner (bank) / the coordinator that aggregates / one send-train-return-aggregate cycle. |
| **Model weights** [0] | The numbers inside a model that determine its predictions; what FL clients share instead of data. |
| **FedAvg** [0] | Federated Averaging: the server averages client weights, weighted by each client's training-set size. |
| **Flower (flwr)** [0] | The Python federated-learning framework we use. |
| **IID / Non-IID** [0] | Clients' data are / are not similar random samples of one population. Real banks are non-IID. |
| **Synthetic data** [0] | Artificially generated rows that imitate real data's statistics. |
| **GAN** [0] | Generative Adversarial Network: a generator (forger) and a discriminator (detective) trained against each other. |
| **CTGAN** [0] | Conditional Tabular GAN, a GAN designed for tabular data; used in Augment Mode via the SDV library. |
| **SDV** [0] | Synthetic Data Vault, the Python library providing CTGAN. |
| **LLM** [0] | Large Language Model; used in Schema Mode via the Groq API. |
| **Schema** [0] | The description of a table: column names, types, allowed ranges. |
| **Augment Mode / Schema Mode** [0] | Our two generation modes: CTGAN for data-rich banks / LLM for data-poor banks. |
| **Validation layer** [0] | The shared quality gate (fidelity, diversity, schema checks) every synthetic row must pass. |
| **Fidelity** [0] | How statistically similar synthetic data is to real data. |
| **Mode collapse** [0] | When a generator keeps producing the same few outputs instead of varied ones. |
| **Six arms** [0] | Our experiment grid: {isolated, federated, centralized} × {real only, real + synthetic}. |
| **Data leakage** [0] | Test information sneaking into training, which makes results look falsely good. |
