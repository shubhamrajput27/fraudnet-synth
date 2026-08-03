import { useEffect, useState } from "react";
import { getPredictManifest, getSampleTransaction, listRuns, predict } from "../api.js";

const FEATURE_COLUMNS = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

export default function PredictDemo() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState("");
  const [manifest, setManifest] = useState(null);
  const [bank, setBank] = useState("");
  const [features, setFeatures] = useState(null);
  const [actualClass, setActualClass] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    listRuns().then((all) => setRuns(all.filter((r) => r.status === "complete"))).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!runId) return setManifest(null);
    getPredictManifest(runId)
      .then((m) => { setManifest(m); setBank(m.banks?.[0] ?? ""); })
      .catch((err) => setError(err.message));
  }, [runId]);

  async function handleSample() {
    setError(null);
    try {
      const { features: f, actual_class } = await getSampleTransaction();
      setFeatures(f);
      setActualClass(actual_class);
      setResult(null);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handlePredict(e) {
    e.preventDefault();
    if (!features) return;
    setBusy(true);
    setError(null);
    try {
      const res = await predict({ run_id: runId, bank: manifest?.needs_bank ? bank : undefined, features });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2>Test a transaction</h2>
      <div className="predict-controls">
        <label>
          Trained run
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="">Select a completed run...</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>{r.run_id} ({r.arm})</option>
            ))}
          </select>
        </label>
        {manifest?.needs_bank && (
          <label>
            Bank (this arm has no shared model/scaler across banks)
            <select value={bank} onChange={(e) => setBank(e.target.value)}>
              {manifest.banks.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </label>
        )}
        <button type="button" onClick={handleSample}>Load a random holdout transaction</button>
      </div>

      {error && <p className="error">{error}</p>}

      {features && (
        <form onSubmit={handlePredict}>
          <p className="muted">
            Sample's actual label (for reference only, not sent to the model): <strong>{actualClass}</strong>
          </p>
          <div className="feature-grid">
            {FEATURE_COLUMNS.map((col) => (
              <label key={col} className="feature-input">
                {col}
                <input
                  type="number"
                  step="any"
                  value={features[col]}
                  onChange={(e) => setFeatures({ ...features, [col]: Number(e.target.value) })}
                />
              </label>
            ))}
          </div>
          <button type="submit" disabled={busy || !runId}>{busy ? "Predicting..." : "Predict"}</button>
        </form>
      )}

      {result && (
        <div className={`predict-result predict-${result.prediction}`}>
          <strong>{result.prediction === "fraud" ? "Predicted: FRAUD" : "Predicted: legitimate"}</strong>
          <span> — probability {result.probability.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}
