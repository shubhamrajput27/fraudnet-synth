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
| **EDA (Exploratory Data Analysis)** [1] | Studying a dataset's size, quality and patterns with statistics and charts before modelling. |
| **PCA (Principal Component Analysis)** [1] | A transformation that mixes original columns into new uncorrelated "components", ordered by how much variation they capture. In ULB it anonymizes the features (V1–V28). |
| **Accuracy paradox** [1] | On imbalanced data, a useless model (always "genuine") gets very high accuracy. Here it is 99.83%. |
| **Confusion matrix (TP/FP/FN/TN)** [1] | The 2×2 count of correct and wrong predictions: frauds caught, false alarms, frauds missed, genuine passed. |
| **Decision threshold** [1] | The fraud-score cut-off above which a transaction is flagged. Moving it trades precision against recall. |
| **ROC-AUC** [1] | Area under the true-positive vs false-positive rate curve. Secondary here, because it can look flattering on very imbalanced data. |
| **Cohen's d** [1] | Difference between two group means divided by a pooled standard deviation. Used in EDA to rank which features separate fraud from genuine. |
| **Duplicate rows** [1] | Identical rows. If copies fall in both train and test sets, test scores are inflated (a form of leakage). |
| **Stratified split** [1→2] | A split that keeps the same class proportions (fraud %) in every part. |
| **Hold-out test set** [2] | Data locked away until final evaluation. Ours is the global test set shared by all six arms. |
| **Validation set** [2] | Data used for tuning (e.g. thresholds) so the test set stays untouched. |
| **Label skew** [2] | Non-IID type where clients differ in class proportions (fraud %). |
| **Quantity skew** [2] | Non-IID type where clients differ in the number of rows. |
| **Explicit quota partition** [2] | Assigning each client a fixed share of each class. Our method for Banks A–D. |
| **Dirichlet partition** [2] | Drawing random unequal client shares from a Dirichlet distribution. α controls how unequal they are. Considered, not used. |
| **Largest-remainder rounding** [2] | Rounding shares down, then giving leftover units to the largest fractions, so whole counts sum exactly. |
| **Shard** [2] | One client's private slice of the data. |
| **SHA-256 hash / manifest** [2] | A fingerprint of a file. Identical hashes prove two runs produced byte-identical data. |
