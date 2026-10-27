// Overview: the four banks (mode indicators) and the synthetic data quality panel.
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, Legend, Loading, ModeBadge, bankName, fmt, int, pct, useApi } from "../components.jsx";
import { axisProps, usePalette } from "../theme.js";

const REASONS = ["Out of range / schema", "Edge-clamped (3+ values)", "Too close to a real row",
  "Identical text embedding", "Near-duplicate of a kept synthetic row", "Batch fidelity below minimum"];

// The core idea in one picture: model weights travel, transactions never do.
function FlowDiagram({ banks }) {
  const list = banks ?? ["bank_a", "bank_b", "bank_c", "bank_d"].map((bank) => ({ bank }));
  return (
    <figure className="flow" aria-label="One federated training round">
      <div className="flow-server">
        <strong>Server · FedAvg</strong>
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
              {b.rows !== undefined && <small className="num">{int(b.rows)} rows · {int(b.fraud)} frauds</small>}
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
  const p = usePalette();
  const modeColor = (mode) => (String(mode).startsWith("Augment") ? p.series[0] : p.series[2]);

  return (
    <>
      <section className="card">
        <header><h2>Four banks, one shared model</h2><span className="muted">no bank's data ever leaves it</span></header>
        <p className="ink2">
          Each bank holds a private slice of the ULB credit-card dataset. Data-rich banks (A, B) make extra fraud
          examples with <strong>CTGAN</strong>; data-poor banks (C, D) with an <strong>LLM</strong> that sees only summary statistics.
          Every synthetic row passes the same validation gate before training.
        </p>
        <FlowDiagram banks={banks.data} />
      </section>

      <Loading state={banks}>
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
              <p className="note">Gate pass rate <strong>{pct(b.passRatePct)}</strong> · fidelity <strong className="num">{fmt(b.fidelityValidated, 3)}</strong> (SDMetrics, 0–1)</p>
            </article>
          ))}
        </div>
      </Loading>

      <section className="grid cols-2">
        <article className="card">
          <header><h2>Synthetic quality: gate pass rate</h2><span className="muted">% of candidate rows admitted</span></header>
          <Legend items={[{ label: "Augment mode (CTGAN)", color: p.series[0], box: true },
                          { label: "Schema mode (LLM)", color: p.series[2], box: true }]} />
          <Loading state={banks}>
            <div style={{ height: 240 }}>
              <ResponsiveContainer>
                <BarChart data={banks.data ?? []} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                  <CartesianGrid vertical={false} stroke={p.grid} />
                  <XAxis dataKey="bank" tickFormatter={bankName} {...axisProps(p)} />
                  <YAxis domain={[0, 100]} unit="%" {...axisProps(p)} />
                  <Tooltip cursor={{ fill: p.grid, opacity: 0.4 }}
                           content={<ChartTooltip labelFormatter={bankName} valueDigits={1} />} />
                  <Bar dataKey="passRatePct" name="pass rate (%)" radius={[4, 4, 0, 0]} maxBarSize={56}>
                    {(banks.data ?? []).map((b) => <Cell key={b.bank} fill={modeColor(b.mode)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Loading>
        </article>

        <article className="card">
          <header><h2>Why rows were rejected</h2><span className="muted">primary reason, counts</span></header>
          <Loading state={validation}>
            <div className="table-wrap">
              <table className="compact">
                <thead>
                  <tr><th className="wrap">Reason</th>{(validation.data ?? []).map((v) => <th key={v.bank} className="num">{bankName(v.bank)}</th>)}</tr>
                </thead>
                <tbody>
                  {REASONS.map((r) => (
                    <tr key={r}><td className="wrap">{r}</td>{(validation.data ?? []).map((v) => <td key={v.bank} className="num">{v[r]}</td>)}</tr>
                  ))}
                  <tr><td className="wrap"><strong>Fidelity, validated</strong> (all candidates)</td>
                    {(validation.data ?? []).map((v) => (
                      <td key={v.bank} className="num"><strong>{fmt(v.fidelity_validated, 3)}</strong> <span className="muted">({fmt(v.fidelity_all, 3)})</span></td>))}
                  </tr>
                </tbody>
              </table>
            </div>
          </Loading>
          <p className="note">No synthetic row was a near-copy of a real one (privacy check). The LLM rows that pass are
            still clumped and use patterned digits (Step 5–6), which is reported, not hidden.</p>
        </article>
      </section>
    </>
  );
}
