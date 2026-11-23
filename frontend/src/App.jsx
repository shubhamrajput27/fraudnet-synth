import { Suspense, lazy, useEffect, useState } from "react";
import { getUser, logout, onAuthChange } from "./api.js";
import { PageBoundary } from "./components.jsx";
import Logo from "./Logo.jsx";
import Login from "./pages/Login.jsx";

// Each page is its own download, fetched when first opened: the login screen doesn't pull in the
// charting library, and the first page appears sooner.
const Overview = lazy(() => import("./pages/Overview.jsx"));
const Results = lazy(() => import("./pages/Results.jsx"));
const Runs = lazy(() => import("./pages/Runs.jsx"));
const Predict = lazy(() => import("./pages/Predict.jsx"));
const ExportPage = lazy(() => import("./pages/ExportAudit.jsx").then((m) => ({ default: m.ExportPage })));
const AuditPage = lazy(() => import("./pages/ExportAudit.jsx").then((m) => ({ default: m.AuditPage })));

// Which pages each role sees (the gateway enforces the same rules on the server side).
const PAGES = [
  { id: "overview", label: "Overview", el: Overview, roles: ["admin", "bank_operator", "viewer"] },
  { id: "results", label: "Six-arm results", el: Results, roles: ["admin", "bank_operator", "viewer"] },
  { id: "runs", label: "Runs", el: Runs, roles: ["admin", "bank_operator", "viewer"] },
  { id: "predict", label: "Test a transaction", el: Predict, roles: ["admin", "bank_operator", "viewer"] },
  { id: "export", label: "Export", el: ExportPage, roles: ["bank_operator"] },
  { id: "audit", label: "Audit log", el: AuditPage, roles: ["admin"] },
];

export default function App() {
  const [user, setUser] = useState(getUser());
  const [page, setPage] = useState(() => location.hash.slice(1) || "overview");
  useEffect(() => onAuthChange(setUser), []);
  // A new tab starts at the top, like a new page would (not halfway down a long table).
  useEffect(() => { location.hash = page; window.scrollTo(0, 0); }, [page]);
  // Browser Back/Forward change the hash: follow it, so the tabs behave like normal pages.
  useEffect(() => {
    const on = () => setPage(location.hash.slice(1) || "overview");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const pages = user ? PAGES.filter((p) => p.roles.includes(user.role)) : [];
  const current = pages.find((p) => p.id === page) ?? pages[0];
  // A page this role can't see (e.g. #audit as a viewer) falls back to the first one; keep the URL honest.
  useEffect(() => { if (current && current.id !== page) setPage(current.id); }, [current, page]);

  if (!user) return <Login />;
  const Current = current.el;

  return (
    <>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main").focus(); }}>Skip to content</a>
      <nav className="topbar" aria-label="main">
        <div className="brand"><Logo />
          <div><strong>FraudNet-Synth</strong><small>federated fraud detection · synthetic augmentation</small></div></div>
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
      <main id="main" tabIndex={-1}>
        {/* key: a page that failed gets a fresh chance when you switch tabs */}
        <PageBoundary key={current.id}>
          <Suspense fallback={<div className="skeleton" style={{ height: 320 }} role="status" aria-label="Loading" />}>
            <Current />
          </Suspense>
        </PageBoundary>
      </main>
    </>
  );
}
