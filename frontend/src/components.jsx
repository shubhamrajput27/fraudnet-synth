// Small shared building blocks.
import { useEffect, useState } from "react";
import { api } from "./api.js";

// Job status: the status palette is reserved for states, and always pairs colour with an icon + label.
const STATUS = {
  queued: { icon: "◷", color: "var(--muted)", label: "Queued" },
  running: { icon: "▶", color: "var(--accent)", label: "Running" },
  succeeded: { icon: "✓", color: "var(--good)", label: "Succeeded" },
  failed: { icon: "✕", color: "var(--critical)", label: "Failed" },
  cancelled: { icon: "■", color: "var(--serious)", label: "Cancelled" },
  interrupted: { icon: "!", color: "var(--warning)", label: "Interrupted" },
};
export function StatusBadge({ status }) {
  const s = STATUS[status] ?? { icon: "?", color: "var(--muted)", label: status ?? "unknown" };
  return (
    <span className="badge" title={s.label}>
      <span aria-hidden="true" style={{ color: s.color }}>{s.icon}</span>{s.label}
    </span>
  );
}

export function ModeBadge({ mode }) {
  const augment = String(mode).toLowerCase().startsWith("augment");
  return (
    <span className={`badge ${augment ? "augment" : "schema"}`}>
      <span className="dot" aria-hidden="true" />{augment ? "Augment mode · CTGAN" : "Schema mode · LLM"}
    </span>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}><i className={it.box ? "box" : ""} style={{ background: it.color }} />{it.label}</span>
      ))}
    </div>
  );
}

// Fetch helper: { data, error, loading, reload }.
export function useApi(path, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    api(path)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: e.message, loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, tick, ...deps]);
  return { ...state, reload: () => setTick((t) => t + 1) };
}

export function Loading({ state, children }) {
  if (state.loading && !state.data) return <p className="muted">Loading…</p>;
  if (state.error) return <p className="error">Couldn't load: {state.error}</p>;
  return children;
}

export const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}%`);
export const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-IN"));
export const bankName = (b) => `Bank ${String(b).slice(-1).toUpperCase()}`;

// "0.8093 ± 0.0048" -> { mean: 0.8093, std: 0.0048 }
export function parsePm(s) {
  const [m, sd] = String(s).split("±").map((x) => parseFloat(x));
  return { mean: m, std: Number.isFinite(sd) ? sd : 0 };
}

// Chart tooltip: values first, then labels; colour keys as short strokes (dataviz interaction rules).
export function ChartTooltip({ active, payload, label, labelFormatter, valueDigits = 4 }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card" style={{ padding: "8px 10px", gap: 4, fontSize: 12 }}>
      <div className="muted">{labelFormatter ? labelFormatter(label) : label}</div>
      {payload.filter((p) => p.value !== undefined && p.value !== null).map((p) => (
        <div key={p.dataKey} className="row" style={{ gap: 6 }}>
          <i style={{ width: 12, height: 3, background: p.color, display: "inline-block", borderRadius: 2 }} />
          <strong className="num">{typeof p.value === "number" ? p.value.toFixed(valueDigits) : p.value}</strong>
          <span className="ink2">{p.name}</span>
        </div>
      ))}
    </div>
  );
}
