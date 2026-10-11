// Runs: start (admin), history (everyone), and a live view of the selected run.
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, getUser } from "../api.js";
import { ChartTooltip, Legend, StatusBadge, bankName, fmt, int } from "../components.jsx";
import { useRunStream } from "../live.js";
import { ROLE, axisProps, usePalette } from "../theme.js";

const ARMS = { 1: "Isolated, real", 2: "Isolated, augmented", 3: "Federated, real",
               4: "Federated, augmented", 5: "Centralized, real", 6: "Centralized, augmented" };
const ARM_KEY = { arm3_federated_real: "Arm 3 · federated, real", arm4_federated_aug: "Arm 4 · federated, augmented" };

function StartPanel({ onStarted }) {
  const [arms, setArms] = useState(new Set([3]));
  const [seeds, setSeeds] = useState("42");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const toggle = (a) => setArms((s) => { const n = new Set(s); n.has(a) ? n.delete(a) : n.add(a); return n; });

  async function start(path, body) {
    setBusy(true);
    setMsg(null);
    try {
      const run = await api(path, { method: "POST", body });
      setMsg(`Started ${run.jobId}`);
      onStarted(run.jobId);
    } catch (e) { setMsg(`Error: ${e.message}`); } finally { setBusy(false); }
  }
  const seedList = seeds.split(/[\s,]+/).filter(Boolean).map(Number);
  const allSix = arms.size === 6;

  return (
    <article className="card">
      <header><h2>Start a run</h2><span className="muted">admin only</span></header>
      <div className="row">
        {Object.entries(ARMS).map(([a, label]) => (
          <label key={a} className="check"><input type="checkbox" checked={arms.has(+a)} onChange={() => toggle(+a)} />Arm {a} · {label}</label>
        ))}
      </div>
      <div className="row">
        <div className="field"><label htmlFor="seeds">Seeds (comma-separated)</label>
          <input id="seeds" type="text" value={seeds} onChange={(e) => setSeeds(e.target.value)} /></div>
        <button className="btn primary" disabled={busy || !arms.size || !seedList.length || seedList.some(Number.isNaN)}
                onClick={() => start("/runs", { arms: allSix ? "all" : [...arms].sort(), seeds: seedList })}>
          Start run
        </button>
        <button className="btn" disabled={busy} onClick={() => start("/generation", { mode: "replay" })}
                title="Rebuild all synthetic data deterministically into a separate folder and compare checksums">
          Replay generation
        </button>
      </div>
      <p className="note">Rough CPU times: one federated arm ≈ 1–1.5 min per seed; all six arms ≈ 4.5 min per seed.
        Results go to a new job folder, never over the reported Step 9 results.</p>
      {msg && <p className={msg.startsWith("Error") ? "error" : "note"}>{msg}</p>}
    </article>
  );
}

function RoundsChart({ rounds, p }) {
  const keys = [...new Set(rounds.map((r) => `${r.arm}|${r.seed}`))];
  const data = useMemo(() => {
    const byRound = new Map();
    for (const r of rounds) {
      const row = byRound.get(r.round) ?? { round: r.round };
      row[`${r.arm}|${r.seed}`] = r.globalTestPrAuc;
      byRound.set(r.round, row);
    }
    return [...byRound.values()].sort((a, b) => a.round - b.round);
  }, [rounds]);
  const color = (k) => p.series[k.startsWith("arm4") ? ROLE.augmented : ROLE.real];
  const label = (k) => `${ARM_KEY[k.split("|")[0]] ?? k.split("|")[0]} · seed ${k.split("|")[1]}`;
  return (
    <>
      <Legend items={keys.map((k) => ({ label: label(k), color: color(k) }))} />
      <div style={{ height: 260 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -6 }}>
            <CartesianGrid vertical={false} stroke={p.grid} />
            <XAxis dataKey="round" {...axisProps(p)} />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} {...axisProps(p)} />
            <Tooltip content={<ChartTooltip labelFormatter={(r) => `Round ${r}`} />} />
            {keys.map((k, i) => (
              <Line key={k} dataKey={k} name={label(k)} stroke={color(k)} strokeWidth={2} isAnimationActive={false}
                    strokeDasharray={i >= 2 ? "5 3" : undefined} dot={{ r: 3 }} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

function BankLossChart({ rounds, p }) {
  if (!rounds.length) return null;
  const last = rounds[rounds.length - 1];
  const series = rounds.filter((r) => r.arm === last.arm && r.seed === last.seed);
  const data = series.map((r) => ({ round: r.round, ...Object.fromEntries(r.banks.map((b) => [b.bank, b.trainLoss])) }));
  const banks = last.banks.map((b) => b.bank);
  return (
    <article className="card">
      <header><h3>Each bank's local training loss</h3><span className="muted">{ARM_KEY[last.arm] ?? last.arm} · seed {last.seed}</span></header>
      <Legend items={banks.map((b) => ({ label: bankName(b), color: p.series[ROLE.banks[b]] }))} />
      <div style={{ height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -6 }}>
            <CartesianGrid vertical={false} stroke={p.grid} />
            <XAxis dataKey="round" {...axisProps(p)} />
            <YAxis tickFormatter={(v) => v.toFixed(1)} {...axisProps(p)} />
            <Tooltip content={<ChartTooltip labelFormatter={(r) => `Round ${r}`} />} />
            {banks.map((b) => <Line key={b} dataKey={b} name={bankName(b)} stroke={p.series[ROLE.banks[b]]}
                                    strokeWidth={2} dot={false} isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="note">What travelled each round: every bank sent back only its model weights
        ({int(last.banks[0]?.uploadBytes)} bytes), never a transaction.</p>
    </article>
  );
}

function RunDetail({ jobId, onEnded }) {
  const s = useRunStream(jobId);
  const p = usePalette();
  useEffect(() => { if (s.ended) onEnded?.(); }, [s.ended]);   // refresh the history list at once
  const lastRound = s.rounds[s.rounds.length - 1];
  return (
    <>
      <article className="card">
        <header>
          <h2>Run {jobId}</h2>
          <div className="row">{s.status && <StatusBadge status={s.status} />}{!s.ended && s.status === "running" && <span className="muted">live</span>}</div>
        </header>
        {s.error && <p className="error">{s.error}</p>}
        {lastRound && <p className="ink2">Latest: round <strong>{lastRound.round}</strong> of {ARM_KEY[lastRound.arm] ?? lastRound.arm},
          seed {lastRound.seed} → global PR-AUC <strong className="num">{fmt(lastRound.globalTestPrAuc)}</strong></p>}
        {s.rounds.length > 0 ? <RoundsChart rounds={s.rounds} p={p} />
          : <p className="muted">No federated rounds yet (non-federated arms report when each model finishes).</p>}
        {s.messages.length > 0 && <div className="log" aria-label="progress messages">{s.messages.map((m, i) => <div key={i}>{m}</div>)}</div>}
      </article>
      <div className="grid cols-2">
        <BankLossChart rounds={s.rounds} p={p} />
        <article className="card">
          <header><h3>Finished models (global test)</h3><span className="muted">{s.models.length} so far</span></header>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Seed</th><th>Arm</th><th>Model</th><th className="num headline">Precision</th>
                <th className="num headline">Recall</th><th className="num headline">F1</th><th className="num headline">PR-AUC</th></tr></thead>
              <tbody>
                {[...s.models].sort((a, b) => a.seed - b.seed || a.arm.localeCompare(b.arm) || a.model.localeCompare(b.model)).map((m) => (
                  <tr key={`${m.seed}-${m.arm}-${m.model}`}>
                    <td className="num">{m.seed}</td><td>{m.arm.replace(/_/g, " ")}</td>
                    <td>{m.model.startsWith("bank_") ? bankName(m.model) : m.model}</td>
                    <td className="num">{fmt(m.precision)}</td><td className="num">{fmt(m.recall)}</td>
                    <td className="num headline">{fmt(m.f1)}</td><td className="num headline">{fmt(m.prAuc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {s.banksDone.length > 0 && (
            <div className="note">{s.banksDone.map((b) => (
              <div key={b.bank}>{bankName(b.bank)}: {b.validated}/{b.candidates} passed ({b.pass_rate_pct}%)
                {b.identical_to_experiment_files && ` · identical to experiment files: ${Object.values(b.identical_to_experiment_files).every(Boolean) ? "yes" : "NO"}`}</div>))}
            </div>
          )}
        </article>
      </div>
    </>
  );
}

export default function Runs() {
  const isAdmin = getUser()?.role === "admin";
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let alive = true;
    const load = () => api("/runs?limit=30").then((r) => { if (alive) { setRuns(r); setError(null); } })
      .catch((e) => alive && setError(e.message));
    load();
    const t = setInterval(load, 4000);          // refresh the history list while the page is open
    return () => { alive = false; clearInterval(t); };
  }, [refresh]);
  useEffect(() => { if (!selected && runs.length) setSelected(runs[0].jobId); }, [runs, selected]);

  return (
    <>
      {isAdmin && <StartPanel onStarted={setSelected} />}
      <article className="card">
        <header><h2>Run history</h2><span className="muted">stored in MongoDB by the gateway</span></header>
        {error && <p className="error">{error}</p>}
        <div className="table-wrap">
          <table>
            <thead><tr><th className="num">Started</th><th>Type</th><th>Arms · seeds</th><th>By</th><th>Status</th><th className="num">Rounds</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.jobId} className={`clickable ${r.jobId === selected ? "selected" : ""}`} onClick={() => setSelected(r.jobId)}>
                  <td className="num">{new Date(r.createdAt).toLocaleString()}</td>
                  <td>{r.type}</td>
                  <td>{r.type === "run" ? `${Array.isArray(r.params.arms) ? r.params.arms.join(", ") : r.params.arms} · ${r.params.seeds?.join(", ")}`
                       : r.params.mode ?? "—"}</td>
                  <td>{r.requestedBy}</td><td><StatusBadge status={r.status} /></td>
                  <td className="num">{r.roundsReceived}</td>
                </tr>
              ))}
              {!runs.length && <tr><td colSpan={6} className="muted">No runs yet{isAdmin ? ": start one above." : "."}</td></tr>}
            </tbody>
          </table>
        </div>
      </article>
      {selected && <RunDetail jobId={selected} key={selected} onEnded={() => setRefresh((r) => r + 1)} />}
    </>
  );
}
