// "Test a transaction": score one transaction with the federated model (Arm 3, seed 42).
import { useState } from "react";
import { api } from "../api.js";
import { fmt } from "../components.jsx";

const V = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const FIELDS = ["Time", ...V, "Amount"];
const blank = Object.fromEntries(FIELDS.map((f) => [f, ""]));

export default function Predict() {
  const [tx, setTx] = useState(blank);
  const [truth, setTruth] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function loadExample(kind) {
    setError(null);
    setResult(null);
    try {
      const ex = await api("/predict/examples");
      setTx(Object.fromEntries(FIELDS.map((f) => [f, String(ex[kind].transaction[f])])));
      setTruth(ex[kind].true_label);
    } catch (e) { setError(e.message); }
  }

  async function predict(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const body = Object.fromEntries(FIELDS.map((f) => [f, Number(tx[f])]));
      setResult(await api("/predict", { method: "POST", body }));
    } catch (err) { setError(err.message); setResult(null); } finally { setBusy(false); }
  }

  const set = (f) => (e) => { setTx({ ...tx, [f]: e.target.value }); setTruth(null); setResult(null); };

  return (
    <div className="grid cols-2">
      <form className="card" onSubmit={predict}>
        <header><h2>Test a transaction</h2>
          <div className="row">
            <button type="button" className="btn" onClick={() => loadExample("fraud")}>Load a fraud example</button>
            <button type="button" className="btn" onClick={() => loadExample("genuine")}>Load a genuine example</button>
            <button type="button" className="btn" onClick={() => { setTx(blank); setTruth(null); setResult(null); }}>Clear</button>
          </div>
        </header>
        <div className="row">
          <div className="field"><label htmlFor="f-Time">Time (seconds, 0–172,792)</label>
            <input id="f-Time" type="number" step="any" min="0" max="172792" required value={tx.Time} onChange={set("Time")} /></div>
          <div className="field"><label htmlFor="f-Amount">Amount</label>
            <input id="f-Amount" type="number" step="any" min="0" required value={tx.Amount} onChange={set("Amount")} /></div>
        </div>
        <div className="field"><label>V1–V28 (anonymised PCA components)</label>
          <div className="vgrid">
            {V.map((v) => (
              <div key={v} className="field"><label htmlFor={`f-${v}`}>{v}</label>
                <input id={`f-${v}`} type="number" step="any" required value={tx[v]} onChange={set(v)} /></div>
            ))}
          </div>
        </div>
        <button className="btn primary" disabled={busy}>{busy ? "Scoring…" : "Score transaction"}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      <article className="card" aria-live="polite">
        <header><h2>Result</h2></header>
        {!result && <p className="muted">Load an example (from the global test set) or type values, then score.</p>}
        {result && (
          <>
            <div className={`verdict ${result.is_fraud ? "fraud" : "genuine"}`}>
              <span aria-hidden="true">{result.is_fraud ? "⚠" : "✓"}</span>{result.is_fraud ? "Flagged as fraud" : "Looks genuine"}
            </div>
            <div className="stats">
              <div className="stat"><div className="v num">{fmt(result.score_logit, 2)}</div><div className="k">fraud score (logit)</div></div>
              <div className="stat"><div className="v num">{fmt(result.threshold_logit, 2)}</div><div className="k">alert threshold</div></div>
              <div className="stat"><div className="v num">{(result.fraud_probability * 100).toFixed(2)}%</div><div className="k">probability (display only)</div></div>
              {truth !== null && <div className="stat"><div className="v">{truth === 1 ? "Fraud" : "Genuine"}</div>
                <div className="k">true label · model {(truth === 1) === result.is_fraud ? "correct" : "wrong"}</div></div>}
            </div>
            <p className="note">Flagged when the score is at or above the threshold, which was chosen from the banks'
              summed validation counts (no scores or rows shared). The probability isn't calibrated (weighted training), so the
              decision uses the threshold.</p>
            <p className="note">Model: federated FedAvg, {result.model.num_rounds} rounds, seed {result.model.seed}
              {" "}(global test F1 {fmt(result.model.global_test_f1, 3)}, PR-AUC {fmt(result.model.global_test_pr_auc, 3)}).</p>
          </>
        )}
      </article>
    </div>
  );
}
