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
| **MLP (multi-layer perceptron)** [3] | A small neural network of stacked layers. Ours: 31 → 64 → 32 → 1, with 4,161 weights. |
| **ReLU** [3] | An activation function, max(0, x), that lets a network learn non-linear boundaries. |
| **Dropout** [3] | Randomly switching off some neurons during training to reduce memorising. |
| **Loss function / BCE** [3] | The penalty a model minimises. Binary Cross-Entropy is the standard loss for yes/no labels. |
| **pos_weight (weighted loss)** [3] | A multiplier on the fraud-class loss. Ours = #genuine / #fraud of the model's own training data. |
| **Epoch / batch / learning rate** [3] | One full data pass / rows processed per update / size of each weight update. |
| **Adam** [3] | An optimiser that adapts the update size for each weight. |
| **Logit / sigmoid** [3] | The raw model output / the function that squashes it into a 0–1 score. |
| **Feature transform** [3] | Converting raw columns into model inputs (log1p for Amount, sin/cos hour for Time). |
| **Average Precision** [3] | How we compute PR-AUC: a step-wise summary of the precision–recall curve. |
| **Calibration** [3] | Whether a score of 0.9 really means a 90% chance. Our weighted model's scores are not calibrated; they are used for ranking and thresholding. |
| **get_weights / set_weights** [3] | Export and import model weights as NumPy arrays. This is what FL clients exchange. |
| **Generator / discriminator** [4] | The two networks in a GAN: the forger that makes rows / the detective that judges real vs fake. |
| **Mode-specific normalisation** [4] | CTGAN's way of modelling each numeric column as a mix of bell curves, so skewed columns are handled. |
| **PacGAN (pac)** [4] | The discriminator judges several rows at once (10 here), which discourages mode collapse. |
| **Synthesizer** [4] | SDV's name for a trained generative model (here `CTGANSynthesizer`). |
| **Candidate rows** [4] | Synthetic rows before validation. Only gate-approved rows may be used for training. |
| **KS statistic** [4] | Kolmogorov–Smirnov distance between two distributions of one column (0 = identical, 1 = completely different). |
| **Edge clamping** [4] | Forcing out-of-range synthetic values onto the real min/max, which creates spikes at the edges. |
| **Memorisation / near-copy** [4] | A synthetic row almost identical to a real one, which is a privacy risk. Checked in Step 6. |
| **Prompt engineering** [5] | Designing an LLM's instructions so it returns exactly what you need, in the format you need. |
| **Zero-shot / few-shot** [5] | Prompting without / with example answers. Ours is zero-shot plus aggregate statistics: no real row is shown. |
| **Aggregate statistics** [5] | Summaries over many rows (mean, std, percentiles) rather than any single row's values. |
| **Percentile (p10/p50/p90)** [5] | The value below which 10% / 50% (median) / 90% of the data falls. |
| **Token** [5] | The unit LLMs read and write (roughly a short word or number fragment). Free-tier limits are counted in tokens. |
| **Rate limit / exponential backoff** [5] | An API's cap on usage per minute or day / waiting 2, 4, 8 … seconds before retrying after "too fast". |
| **Structured output (JSON schema, strict)** [5] | Making the API check the reply against an exact JSON shape and reject replies that don't fit. |
| **Reasoning model** [5] | An LLM that "thinks" (generates hidden reasoning tokens) before answering. Slower, and uses more tokens. |
| **Response cache** [5] | Saved API replies, so reruns and demos need no live API call. |
| **Patterned decimals** [5] | Values like −3.210 or 5.432 whose digits form runs. A sign that an LLM is writing number-shaped text rather than irregular values. |
| **Validation gate** [6] | The shared sequence of checks every synthetic row must pass before it may be used for training. |
| **Pandera** [6] | A Python library for declaring what a valid table looks like (columns, types, ranges) and checking data against it. |
| **Standardisation (z-score)** [6] | (value − mean) / std per column, so every column counts equally in a distance. |
| **DCR (distance to closest record)** [6] | Distance from a synthetic row to the nearest real row. Too small suggests a near-copy (privacy risk). |
| **Nearest-neighbour distance** [6] | Distance from a row to the most similar other row. Real-to-real spacing sets our thresholds. |
| **Greedy de-duplication** [6] | Go through rows in order and keep one only if it isn't too close to any row already kept. |
| **SDMetrics QualityReport** [6] | Set-level fidelity score (0–1): Column Shapes (per-column distributions) and Column Pair Trends (relationships between columns). |
| **Sentence embedding / cosine similarity** [6] | A vector representing text meaning / a 0–1 measure of how similar two such vectors are. |
| **Pass rate** [6] | Share of candidate rows admitted by the gate. |
| **Isolated arm** [7] | Each bank trains alone on its own data (Arms 1, 2). The lower bound. |
| **Centralized arm** [7] | One model on all banks' pooled data (Arms 5, 6). The upper bound, illegal in practice. |
| **Augmentation ratio** [7] | Synthetic rows added per real fraud row. Ours is 1:1. |
| **Paired comparison** [7] | Two models identical except for one factor (here the training data), so differences can be attributed to that factor. |
| **Upper / lower bound** [7] | Best and worst reference results that federated learning is compared against. |
| **Global vs local test** [7] | The shared test set every arm is graded on vs one bank's own held-out customers. |
| **Saturation (float32)** [7] | Very confident sigmoid outputs rounding to exactly 1.0, creating ties. Fixed by ranking on logits. |
