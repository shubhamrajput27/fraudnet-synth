import { useEffect, useState } from "react";
import { getUser, logout, onAuthChange } from "./api.js";
import Login from "./pages/Login.jsx";
import Overview from "./pages/Overview.jsx";
import Results from "./pages/Results.jsx";
import Runs from "./pages/Runs.jsx";
import Predict from "./pages/Predict.jsx";
import { AuditPage, ExportPage } from "./pages/ExportAudit.jsx";

// Which pages each role sees (the gateway enforces the same rules on the server side).
const PAGES = [
  { id: "overview", label: "Overview", el: Overview, roles: ["admin", "bank_operator", "viewer"] },
  { id: "results", label: "Six-arm results", el: Results, roles: ["admin", "bank_operator", "viewer"] },
  { id: "runs", label: "Runs (live)", el: Runs, roles: ["admin", "bank_operator", "viewer"] },
  { id: "predict", label: "Test a transaction", el: Predict, roles: ["admin", "bank_operator", "viewer"] },
  { id: "export", label: "Export", el: ExportPage, roles: ["bank_operator"] },
  { id: "audit", label: "Audit log", el: AuditPage, roles: ["admin"] },
];

export default function App() {
  const [user, setUser] = useState(getUser());
  const [page, setPage] = useState(() => location.hash.slice(1) || "overview");
  useEffect(() => onAuthChange(setUser), []);
  useEffect(() => { location.hash = page; }, [page]);

  if (!user) return <Login />;
  const pages = PAGES.filter((p) => p.roles.includes(user.role));
  const Current = (pages.find((p) => p.id === page) ?? pages[0]).el;

  return (
    <>
      <nav className="topbar" aria-label="main">
        <div className="brand"><strong>FraudNet-Synth</strong><small>federated fraud detection · synthetic augmentation</small></div>
        <div className="tabs">
          {pages.map((p) => (
            <button key={p.id} className="tab" aria-current={p.id === page ? "page" : undefined} onClick={() => setPage(p.id)}>{p.label}</button>
          ))}
        </div>
        <div className="who">
          <span>{user.displayName} · <em>{user.role.replace("_", " ")}</em></span>
          <button className="btn" onClick={logout}>Sign out</button>
        </div>
      </nav>
      <main><Current /></main>
    </>
  );
}
