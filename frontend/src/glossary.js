// Plain-English meanings for the jargon on screen, shown on hover/focus by <Term>.
// Every threshold quoted here comes from configs/validation.yaml or a DECISIONS.md entry.

export const TERMS = {
  precision: { label: "Precision", text: "Of the transactions the model flagged as fraud, the share that really were fraud. Low precision = many false alarms." },
  recall: { label: "Recall", text: "Of all real frauds, the share the model caught. Low recall = frauds slip through." },
  f1: { label: "F1", text: "One number balancing precision and recall (their harmonic mean), for the fraud class. Our headline metric." },
  pr_auc: { label: "PR-AUC", text: "Area under the precision–recall curve: how well the model ranks frauds above genuine transactions at every threshold. 1.0 is perfect; random is about 0.0017 here." },
  roc_auc: { label: "ROC-AUC", text: "Area under the ROC curve. Looks high on rare-fraud data even for weak models, so it is secondary." },
  accuracy: { label: "Accuracy", text: "Share of all transactions labelled correctly. Useless as a headline: calling everything genuine already scores about 99.8%." },
  sd: { label: "± std", text: "Standard deviation over the training seeds: how much the result moves when only the random seed changes." },
  seeds: { label: "Seeds", text: "Each experiment is repeated with different random seeds; we report the mean and spread so one lucky run can't mislead." },
  delta: { label: "Δ mean", text: "Average difference between the two arms, paired seed by seed (same seed, same data split)." },
  ci: { label: "95% CI", text: "95% confidence interval of the difference. If it includes 0, we can't claim either arm is better." },
  seeds_up: { label: "Seeds up", text: "In how many of the seeds the first arm beat the second." },
  logit: { label: "Fraud score (logit)", text: "The model's raw output before it is turned into a probability. Higher = more fraud-like." },
  threshold: { label: "Alert threshold", text: "Scores at or above this are flagged. Chosen on validation data from the banks' summed counts, never on the test set." },
  probability: { label: "Probability", text: "The score squashed to 0–100%. Not calibrated (fraud was up-weighted in training), so the decision uses the threshold instead." },
  fedavg: { label: "FedAvg", text: "Federated Averaging: the server averages the banks' model weights, weighted by how many training rows each bank has." },
  centralized: { label: "Centralized", text: "All banks' data pooled in one place. An upper bound only: real banks may not share data like this." },
  fidelity: { label: "Fidelity", text: "SDMetrics quality score (0–1): how closely synthetic rows match the bank's real fraud rows, column by column and in column pairs." },
  pass_rate: { label: "Gate pass rate", text: "Share of candidate synthetic rows that passed every validation check and joined the training pool." },
  augment: { label: "Augment mode", text: "Data-rich banks (A, B) train CTGAN, a tabular GAN, on their own fraud rows to make more." },
  schema: { label: "Schema mode", text: "Data-poor banks (C, D) ask an LLM for rows, sending only the column list and summary statistics, never real rows." },
  r_schema: { label: "Out of range / schema", text: "A value is missing, the wrong type, or outside the bank's real [min, max] range widened by 10% of its width." },
  r_clamped: { label: "Edge-clamped", text: "3 or more values sit exactly on a real minimum or maximum: a sign the generator was pushed to the edge." },
  r_close: { label: "Too close to a real row", text: "Closer to a real fraud row than 95% of real fraud rows are to their own nearest neighbour: possibly a memorised copy, a privacy risk." },
  r_embed: { label: "Identical text embedding", text: "The row written as text is essentially identical (cosine ≥ 0.999) to a real row's text." },
  r_dup: { label: "Near-duplicate", text: "Almost the same as a synthetic row already kept: rejected to avoid mode collapse (many copies of one pattern)." },
  r_batch: { label: "Batch fidelity", text: "If the bank's admitted rows score below 0.70 fidelity overall, the whole batch is refused." },
};

// The rejection-reason labels used by the API, mapped to their glossary entries.
export const REASON_TERM = {
  "Out of range / schema": "r_schema", "Edge-clamped (3+ values)": "r_clamped", "Too close to a real row": "r_close",
  "Identical text embedding": "r_embed", "Near-duplicate of a kept synthetic row": "r_dup",
  "Batch fidelity below minimum": "r_batch",
};
