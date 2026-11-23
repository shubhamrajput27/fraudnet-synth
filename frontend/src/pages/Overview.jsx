// Overview: the federated round, the four banks (mode indicators) and the synthetic data quality panel.
import { Loading, ModeBadge, More, PageHead, Term, bankName, fmt, int, pct, useApi } from "../components.jsx";
import { REASON_TERM } from "../glossary.js";

const REASONS = ["Out of range / schema", "Edge-clamped (3+ values)", "Too close to a real row",
  "Identical text embedding", "Near-duplicate of a kept synthetic row", "Batch fidelity below minimum"];

// The core idea in one picture: model weights travel, transactions never do.
function FlowDiagram({ banks }) {
  const list = banks ?? ["bank_a", "bank_b", "bank_c", "bank_d"].map((bank) => ({ bank }));
  return (
    <figure className="flow" aria-label="One federated training round">
      <div className="flow-server">
        <strong>Server · <Term k="fedavg">FedAvg</Term></strong>
        <small>averages the banks' weights, weighted by data size · never sees a transaction</small>
      </div>
      <div className="flow-links" aria-hidden="true">
        {list.map((b) => (
          <div key={b.bank} className="flow-link">
            <span className="down">▼ global model</span><span className="up">▲ weights only</span>
          </div>
        ))}
      </div>
      <div className="flow-banks">
        {list.map((b) => {
          const augment = String(b.mode ?? "").toLowerCase().startsWith("augment");
          return (
            <div key={b.bank} className={`flow-bank ${b.mode ? (augment ? "augment" : "schema") : ""}`}>
              <strong>{bankName(b.bank)}</strong>
              {b.mode && <small>{augment ? "CTGAN" : "LLM"} synthetic frauds</small>}
              <span className="lock">🔒 data stays here</span>
            </div>
          );
        })}
      </div>
      <figcaption>
        <ol>
          <li>The server sends the current model to every bank.</li>
          <li>Each bank trains it on its own private rows (real + validated synthetic).</li>
          <li>Each bank sends back only the updated weights, a list of numbers, not data.</li>
          <li>The server averages the weights into a new global model. Repeat.</li>
        </ol>
      </figcaption>
    </figure>
  );
}

export default function Overview() {
  const banks = useApi("/banks");
  const validation = useApi("/validation");

  return (
    <>
      <PageHead title="Four banks, one shared model"
                lead="The banks train one fraud detector together. Only model weights travel; no transaction ever leaves its bank." />
      <section className="card hero">
        <FlowDiagram banks={banks.data} />
        <More label="Where do the synthetic frauds come from?">
          Fraud is rare, so each bank makes extra fraud examples. Data-rich banks (A, B) train <strong>CTGAN</strong> on their
          own fraud rows; data-poor banks (C, D) ask an <strong>LLM</strong> that sees only summary statistics, never a real row.
          Every synthetic row must pass the same validation gate before it is used for training.
        </More>
      </section>

      <Loading state={banks} height={150}>
        <div className="grid cols-4">
          {(banks.data ?? []).map((b) => (
            <article key={b.bank} className="card">
              <header><h2>{bankName(b.bank)}</h2><ModeBadge mode={b.mode} /></header>
              <div className="stats">
                <div className="stat"><div className="v num">{int(b.rows)}</div><div className="k">transactions</div></div>
                <div className="stat"><div className="v num">{int(b.fraud)}</div><div className="k">real frauds ({pct(b.fraudPct, 3)})</div></div>
                <div className="stat"><div className="v num">{int(b.trainFraud)}</div><div className="k">training frauds</div></div>
                <div className="stat"><div className="v num">{int(b.syntheticValidated)}</div>
                  <div className="k">synthetic rows passed (of {int(b.syntheticCandidates)})</div></div>
              </div>
              <p className="note"><Term k="pass_rate">Pass rate</Term> <strong>{pct(b.passRatePct)}</strong> ·{" "}
                <Term k="fidelity">fidelity</Term> <strong className="num">{fmt(b.fidelityValidated, 3)}</strong> <span className="muted">(0–1)</span></p>
            </article>
          ))}
        </div>
      </Loading>

      <article className="card">
        <header><h2>Synthetic data quality: why candidate rows were rejected</h2><span className="muted">primary reason, row counts</span></header>
        <Loading state={validation} height={240}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th className="wrap">Reason</th>{(validation.data ?? []).map((v) => <th key={v.bank} className="num">{bankName(v.bank)}</th>)}</tr>
              </thead>
              <tbody>
                {REASONS.map((r) => (
                  <tr key={r}><td className="wrap"><Term k={REASON_TERM[r]}>{r}</Term></td>{(validation.data ?? []).map((v) => <td key={v.bank} className="num">{v[r]}</td>)}</tr>
                ))}
                <tr><td className="wrap"><strong><Term k="fidelity">Fidelity</Term>, kept rows</strong> <span className="muted">(all candidates)</span></td>
                  {(validation.data ?? []).map((v) => (
                    <td key={v.bank} className="num"><strong>{fmt(v.fidelity_validated, 3)}</strong> <span className="muted">({fmt(v.fidelity_all, 3)})</span></td>))}
                </tr>
              </tbody>
            </table>
          </div>
        </Loading>
        <p className="note">No kept synthetic row is a near-copy of a real one (privacy check).</p>
        <More label="Known weakness">The LLM rows that pass are still clumped and use patterned digits (Steps 5–6).
          This is reported, not hidden.</More>
      </article>
    </>
  );
}
