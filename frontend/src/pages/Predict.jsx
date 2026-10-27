// "Test a transaction": score one transaction with the federated model (Arm 3, seed 42; D-040).
// Picking a real test transaction is the main path; typing values is optional (V1–V28 mean nothing to a person).
import { useState } from "react";
import { api } from "../api.js";
import { More, PageHead, Term, fmt } from "../components.jsx";
import { parseRow } from "../format.js";

const V = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const FIELDS = ["Time", ...V, "Amount"];
const blank = Object.fromEntries(FIELDS.map((f) => [f, ""]));
const KIND_TEXT = { fraud: "fraud", genuine: "genuine", missed_fraud: "missed fraud" };
const POOL_TEXT = { fraud: "correctly flags", genuine: "correctly passes", missed_fraud: "misses, i.e. false negatives" };
const CHOICES = [
  { kind: "fraud", title: "A fraud", sub: "a real fraud from the test set" },
  { kind: "genuine", title: "A genuine purchase", sub: "a normal transaction" },
  { kind: "missed_fraud", title: "A missed fraud", sub: "one the model gets wrong" },
];

export default function Predict() {
  const [tx, setTx] = useState(blank);
  const [truth, setTruth] = useState(null);
  const [source, setSource] = useState(null);   // which global-test row the example came from
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showV, setShowV] = useState(false);     // the 28 PCA boxes mean nothing to most viewers: folded by default
  const [paste, setPaste] = useState("");

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
      setError(`Fill in ${blankV.length === 28 ? "V1–V28" : blankV.join(", ")}, or pick a transaction above.`);
      return;
    }
    score(tx);
  }

  // Edited values are no longer the test-set row, so its true label no longer applies.
  const forget = () => { setTruth(null); setSource(null); setResult(null); };
  const set = (f) => (e) => { setTx({ ...tx, [f]: e.target.value }); forget(); };
  function fillFromPaste() {
    try { setTx(parseRow(paste, FIELDS)); forget(); setError(null); } catch (e) { setError(`Pasted row: ${e.message}`); }
  }

  return (
    <>
      <PageHead title="Test a transaction"
                lead="Pick a real transaction the model has never seen and watch it decide: fraud or genuine?" />
      <div className="grid cols-2">
        <form className="card" onSubmit={predict}>
          <header><h2>1 · Pick a transaction</h2></header>
          <div className="choices">
            {CHOICES.map((c) => (
              <button key={c.kind} type="button" className={`choice ${c.kind}`} disabled={busy}
                      aria-pressed={source?.kind === c.kind} onClick={() => loadExample(c.kind)}>
                <strong>{c.title}</strong><small>{c.sub}</small>
              </button>
            ))}
          </div>
          <p className="note">
            {source
              ? <>Loaded {KIND_TEXT[source.kind]} transaction <strong className="num">#{source.row.toLocaleString("en-IN")}</strong> of
                  the {source.total.toLocaleString("en-IN")}-row global test set
                  {source.filtered
                    ? <> (one of the {source.candidates.toLocaleString("en-IN")} of {source.classTotal.toLocaleString("en-IN")}{" "}
                        {source.kind === "genuine" ? "genuine" : "fraud"} rows the model {POOL_TEXT[source.kind]})</>
                    : <> ({source.candidates.toLocaleString("en-IN")} rows to pick from)</>}.
                  Click again for a different one.</>
              : <>Each click picks a different random row from the global test set, which was never used for training.</>}
          </p>

          <header><h2>2 · Change it <span className="muted">(optional)</span></h2></header>
          <div className="row">
            <div className="field"><label htmlFor="f-Amount">Amount</label>
              <input id="f-Amount" type="number" step="any" min="0" required value={tx.Amount} onChange={set("Amount")} /></div>
            <div className="field"><label htmlFor="f-Time">Time (seconds, 0–172,792)</label>
              <input id="f-Time" type="number" step="any" min="0" max="172792" required value={tx.Time} onChange={set("Time")} /></div>
          </div>
          <p className="note">Try another amount, then score again to see whether the verdict changes.</p>
          <details className="vfold" open={showV} onToggle={(e) => setShowV(e.currentTarget.open)}>
            <summary>Advanced: V1–V28 (anonymised PCA features) {tx.V1 !== "" ? "· filled in" : "· empty"}</summary>
            <div className="field">
              <label htmlFor="paste">Paste a whole row (Time, V1–V28, Amount; a trailing Class column is ignored)</label>
              <div className="row">
                <input id="paste" type="text" value={paste} onChange={(e) => setPaste(e.target.value)} style={{ flex: 1 }}
                       placeholder="e.g. a line copied from global_test.csv" />
                <button type="button" className="btn" disabled={!paste.trim()} onClick={fillFromPaste}>Fill</button>
              </div>
            </div>
            <div className="vgrid">
              {V.map((v) => (
                <div key={v} className="field"><label htmlFor={`f-${v}`}>{v}</label>
                  <input id={`f-${v}`} type="number" step="any" value={tx[v]} onChange={set(v)} /></div>
              ))}
            </div>
          </details>
          <div className="row">
            <button className="btn primary" disabled={busy}>{busy ? "Scoring…" : "Score transaction"}</button>
            <button type="button" className="btn" onClick={() => { setTx(blank); forget(); setError(null); setPaste(""); }}>Clear</button>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
        </form>

        <article className="card result-card" aria-live="polite">
          <header><h2>Result</h2></header>
          {!result && <p className="muted">{busy ? "Scoring…" : "Pick a transaction on the left to see the model's verdict here."}</p>}
          {result && (
            <>
              <div className={`verdict ${result.is_fraud ? "fraud" : "genuine"}`}>
                <span aria-hidden="true">{result.is_fraud ? "⚠" : "✓"}</span>{result.is_fraud ? "Flagged as fraud" : "Looks genuine"}
              </div>
              {truth !== null && (
                <p className={`truth ${(truth === 1) === result.is_fraud ? "right" : "wrong"}`}>
                  Actually <strong>{truth === 1 ? "fraud" : "genuine"}</strong>, so the model is{" "}
                  <strong>{(truth === 1) === result.is_fraud ? "right" : "wrong"}</strong>.
                </p>
              )}
              <div className="stats">
                <div className="stat"><div className="v num">{fmt(result.score_logit, 2)}</div><div className="k"><Term k="logit">fraud score</Term></div></div>
                <div className="stat"><div className="v num">{fmt(result.threshold_logit, 2)}</div><div className="k"><Term k="threshold">alert threshold</Term></div></div>
                <div className="stat"><div className="v num">{(result.fraud_probability * 100).toFixed(2)}%</div><div className="k"><Term k="probability">probability</Term> (display only)</div></div>
              </div>
              <More label="How is the decision made?">Flagged when the score is at or above the threshold. The threshold was chosen
                from the banks' summed validation counts (no scores or rows shared). The probability isn't calibrated
                (fraud was up-weighted in training), so the decision uses the threshold.</More>
              <p className="note">Model: federated <Term k="fedavg">FedAvg</Term>, Arm 3, {result.model.num_rounds} rounds, seed {result.model.seed}
                {" "}· test <Term k="f1">F1</Term> {fmt(result.model.global_test_f1, 3)}
                {result.model.global_test_recall != null && <> · catches {fmt(result.model.global_test_recall * 100, 1)}% of test frauds</>}</p>
              <More label="Why Arm 3, not Arm 4 (FraudNet-Synth)?">On the global test set, Step 9 found no clear difference
                between Arm 4 (with synthetic data) and Arm 3 (real data only) on any headline metric over 5 seeds; on Bank A's
                own local test, Arm 4 was even slightly lower. So the demo uses the simpler privacy-preserving model trained
                without synthetic rows (decision D-040).</More>
            </>
          )}
        </article>
      </div>
    </>
  );
}
