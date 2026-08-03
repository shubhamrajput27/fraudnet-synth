import { useState } from "react";
import "./App.css";
import { isLoggedIn, logout } from "./api.js";
import ComparisonGrid from "./components/ComparisonGrid.jsx";
import Login from "./components/Login.jsx";
import PredictDemo from "./components/PredictDemo.jsx";
import QualityPanel from "./components/QualityPanel.jsx";
import RunHistory from "./components/RunHistory.jsx";
import TriggerRun from "./components/TriggerRun.jsx";

const TABS = {
  trigger: { label: "Trigger Run", Component: TriggerRun },
  history: { label: "Run History", Component: RunHistory },
  comparison: { label: "Comparison Grid", Component: ComparisonGrid },
  quality: { label: "Synthetic Quality", Component: QualityPanel },
  predict: { label: "Test a Transaction", Component: PredictDemo },
};

export default function App() {
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());
  const [tab, setTab] = useState("trigger");

  if (!loggedIn) {
    return <Login onLoggedIn={() => setLoggedIn(true)} />;
  }

  const { Component } = TABS[tab];

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>FraudNet-Synth</h1>
        <button className="logout-btn" onClick={() => { logout(); setLoggedIn(false); }}>Log out</button>
      </header>
      <nav className="tabs">
        {Object.entries(TABS).map(([key, { label }]) => (
          <button key={key} className={tab === key ? "tab active" : "tab"} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </nav>
      <main className="app-main">
        <Component />
      </main>
    </div>
  );
}
