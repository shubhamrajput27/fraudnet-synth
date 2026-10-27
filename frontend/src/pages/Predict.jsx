// "Test a transaction": score one transaction with the federated model (Arm 3, seed 42).
import { useState } from "react";
import { api } from "../api.js";
import { fmt } from "../components.jsx";

const V = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const FIELDS = ["Time", ...V, "Amount"];
const blank = Object.fromEntries(FIELDS.map((f) => [f, ""]));
const KIND_TEXT = { fraud: "fraud", genuine: "genuine", missed_fraud: "missed fraud" };
const POOL_TEXT = { fraud: "correctly flags", genuine: "correctly passes", missed_fraud: "misses (false negatives)" };

export default function Predict() {
  const [tx, setTx] = useState(blank);
  const [truth, setTruth] = useState(null);
  const [source, setSource] = useState(null);   // which global-test row the example came from
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showV, setShowV] = useState(false);     // the 28 PCA boxes mean nothing to most viewers: folded by default

  async function score(values) {
    setBusy(true);
    setError(null);
    try {
      const body = Object.fromEntries(FIELDS.map((f) => [f, Number(values[f])]));
      setResult(await api("/predict", { method: "POST", body }));
    } catch (err) { setError(err.message); setResult(null); } finally { setBusy(false); }
  }

  // Loading an example also scores it at once: one click shows the whole story.
  async function loadExample(kind) {
    setError(null);
    setResult(null);
    try {
      const ex = await api("/predict/examples");     // a NEW random row on every click
      if (!ex[kind]) { setError(`no ${KIND_TEXT[kind]} example available for this model`); return; }
      const values = Object.fromEntries(FIELDS.map((f) => [f, String(ex[kind].transaction[f])]));
      setTx(values);
      setTruth(ex[kind].true_label);
      setSource({ kind, row: ex[kind].row, total: ex[kind].rows_in_test_set, candidates: ex[kind].candidates,
                  classTotal: ex[kind].class_total, filtered: ex[kind].filtered_by_model });
      await score(values);
    } catch (e) { setError(e.message); }
  }

  function predict(e) {
    e.preventDefault();
    // The V boxes may be folded away, where the browser can't point at a blank one: check them here.
    const blankV = V.filter((v) => tx[v].trim() === "");
    if (blankV.length) {
      setShowV(true);
      setError(`Fill in ${blankV.length === 28 ? "V1–V28" : blankV.join(", ")}, or load an example.`);
      return;
    }
    score(tx);
  }

  const set = (f) => (e) => { setTx({ ...tx, [f]: e.target.value }); setTruth(null); setSource(null); setResult(null); };

  return (
    <div className="grid cols-2">
      <form className="card" onSubmit={predict}>
        <header><h2>Test a transaction</h2>
          <div className="row">
            <button type="button" className="btn" onClick={() => loadExample("fraud")}>Load a fraud example</button>
            <button type="button" className="btn" onClick={() => loadExample("genuine")}>Load a genuine example</button>
            <button type="button" className="btn" onClick={() => loadExample("missed_fraud")}>Load a missed fraud</button>
            <button type="button" className="btn" onClick={() => { setTx(blank); setTruth(null); setSource(null); setResult(null); setError(null); }}>Clear</button>
          </div>
        </header>
        <p className="note">
          {source
            ? <>Loaded {KIND_TEXT[source.kind]} transaction <strong className="num">#{source.row.toLocaleString("en-IN")}</strong> of
                the {source.total.toLocaleString("en-IN")}-row global test set
                {source.filtered
                  ? <> (one of the {source.candidates.toLocaleString("en-IN")} of {source.classTotal.toLocaleString("en-IN")}{" "}
                      {source.kind === "genuine" ? "genuine" : "fraud"} rows the model {POOL_TEXT[source.kind]})</>
                  : <> ({source.candidates.toLocaleString("en-IN")} rows to pick from)</>}.
                Click again for a different one.</>
            : <>Each click loads a different random transaction from the global test set (never used for training)
                and scores it. The model doesn't catch every fraud, so "Load a missed fraud" shows one it gets wrong.</>}
        </p>
        <div className="row">
          <div className="field"><label htmlFor="f-Time">Time (seconds, 0–172,792)</label>
            <input id="f-Time" type="number" step="any" min="0" max="172792" required value={tx.Time} onChange={set("Time")} /></div>
          <div className="field"><label htmlFor="f-Amount">Amount</label>
            <input id="f-Amount" type="number" step="any" min="0" required value={tx.Amount} onChange={set("Amount")} /></div>
        </div>
        <details className="vfold" open={showV} onToggle={(e) => setShowV(e.currentTarget.open)}>
          <summary>V1–V28: 28 anonymised PCA components {tx.V1 !== "" ? "(filled in)" : "(empty)"} · {showV ? "hide" : "show"}</summary>
          <div className="vgrid">
            {V.map((v) => (
              <div key={v} className="field"><label htmlFor={`f-${v}`}>{v}</label>
                <input id={`f-${v}`} type="number" step="any" value={tx[v]} onChange={set(v)} /></div>
            ))}
          </div>
        </details>
        <button className="btn primary" disabled={busy}>{busy ? "Scoring…" : "Score transaction"}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      <article className="card result-card" aria-live="polite">
        <header><h2>Result</h2></header>
        {!result && <p className="muted">{busy ? "Scoring…" : "Load an example (from the global test set) or type values, then score."}</p>}
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
              {" "}(global test F1 {fmt(result.model.global_test_f1, 3)}, PR-AUC {fmt(result.model.global_test_pr_auc, 3)}
              {result.model.global_test_recall != null && <>; it catches {fmt(result.model.global_test_recall * 100, 1)}% of test frauds</>}).</p>
          </>
        )}
      </article>
    </div>
  );
}
