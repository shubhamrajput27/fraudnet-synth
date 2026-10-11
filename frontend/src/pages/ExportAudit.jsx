// Dataset export (bank operators, own bank only, D-045) and the audit log (admin).
import { useState } from "react";
import { download, getUser } from "../api.js";
import { Loading, bankName, int, useApi } from "../components.jsx";

export function ExportPage() {
  const user = getUser();
  const banks = useApi("/banks");
  const [msg, setMsg] = useState(null);
  const mine = banks.data?.find((b) => b.bank === user.bank);

  async function go() {
    setMsg(null);
    try {
      await download(`/banks/${user.bank}/synthetic/export`, `${user.bank}_synthetic_validated.csv`);
      setMsg("Downloaded.");
    } catch (e) { setMsg(`Error: ${e.message}`); }
  }

  return (
    <article className="card">
      <header><h2>Export synthetic data</h2></header>
      <div className="callout">Only a bank's <strong>own operator</strong> may download that bank's <strong>validated synthetic</strong>{" "}
        rows. Real transactions are never exported, and every attempt is written to the audit log.</div>
      {user.role !== "bank_operator" ? (
        <p className="ink2">Your account ({user.role}) isn't a bank operator, so it can't export any bank's data.</p>
      ) : (
        <Loading state={banks}>
          <p className="ink2">You are the operator of <strong>{bankName(user.bank)}</strong>. Its validated synthetic pool has
            <strong> {int(mine?.syntheticValidated)}</strong> fraud rows.</p>
          <div className="row"><button className="btn primary" onClick={go}>Download {bankName(user.bank)} synthetic CSV</button></div>
        </Loading>
      )}
      {msg && <p className={msg.startsWith("Error") ? "error" : "note"}>{msg}</p>}
    </article>
  );
}

export function AuditPage() {
  const audit = useApi("/audit?limit=200");
  return (
    <article className="card">
      <header><h2>Audit log</h2><button className="btn" onClick={audit.reload}>Refresh</button></header>
      <Loading state={audit}>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Time</th><th>User</th><th>Role</th><th>Action</th><th>Target</th><th>Allowed</th></tr></thead>
            <tbody>
              {(audit.data ?? []).map((e) => (
                <tr key={e._id}>
                  <td className="num">{new Date(e.time).toLocaleString()}</td><td>{e.user}</td><td>{e.role}</td>
                  <td>{e.action}</td><td>{e.target}</td>
                  <td>{e.allowed ? <span className="badge"><span style={{ color: "var(--good)" }}>✓</span>yes</span>
                                  : <span className="badge"><span style={{ color: "var(--critical)" }}>✕</span>no</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Loading>
    </article>
  );
}
