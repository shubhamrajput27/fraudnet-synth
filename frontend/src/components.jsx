// Small shared building blocks.
import { useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "./api.js";
import { TERMS } from "./glossary.js";

export { bankName, fmt, int, parsePm, pct } from "./format.js";

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
      <span className="dot" aria-hidden="true" />
      <Term k={augment ? "augment" : "schema"}>{augment ? "Augment mode · CTGAN" : "Schema mode · LLM"}</Term>
    </span>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}><i className={it.box ? "box" : ""} style={{ background: it.dash
          ? `repeating-linear-gradient(90deg, ${it.color} 0 5px, transparent 5px 8px)` : it.color }} />{it.label}</span>
      ))}
    </div>
  );
}

// A jargon word with its plain meaning on hover, focus or tap. The tip is portalled to <body> with
// fixed position, so table scroll containers (overflow: auto) can't clip it.
export function Term({ k, children }) {
  const t = TERMS[k];
  const id = useId();
  const [pos, setPos] = useState(null);
  useEffect(() => {
    if (!pos) return undefined;
    const hide = () => setPos(null);
    window.addEventListener("scroll", hide, true);   // a fixed tip would drift away from its word
    return () => window.removeEventListener("scroll", hide, true);
  }, [pos]);
  if (!t) return children;
  const show = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const below = r.bottom + 140 < window.innerHeight;
    setPos({ left: Math.max(8, Math.min(r.left + r.width / 2 - 140, window.innerWidth - 288)),
             ...(below ? { top: r.bottom + 6 } : { bottom: window.innerHeight - r.top + 6 }) });
  };
  return (
    <>
      <span className="term" tabIndex={0} aria-describedby={pos ? id : undefined}
            onMouseEnter={show} onMouseLeave={() => setPos(null)} onFocus={show} onBlur={() => setPos(null)}
            onKeyDown={(e) => e.key === "Escape" && setPos(null)}>{children ?? t.label}</span>
      {pos && createPortal(<span id={id} role="tooltip" className="tip" style={pos}><strong>{t.label}.</strong> {t.text}</span>,
        document.body)}
    </>
  );
}

// The detail behind a claim, folded away: the page stays scannable, and nothing honest is removed.
export function More({ label = "Why?", children }) {
  return (
    <details className="more">
      <summary>{label}</summary>
      <div className="note">{children}</div>
    </details>
  );
}

// Title + one-line lead at the top of every page.
export function PageHead({ title, lead, children }) {
  return (
    <header className="pagehead">
      <div><h1>{title}</h1>{lead && <p className="ink2">{lead}</p>}</div>
      {children}
    </header>
  );
}

// Recharts axis tick that wraps on "\n", so six group names fit under a half-width chart.
export function WrapTick({ x, y, payload, fill, fontSize = 11 }) {
  const lines = String(payload.value).split("\n");
  return (
    <text x={x} y={y + 4} textAnchor="middle" fill={fill} fontSize={fontSize}>
      {lines.map((l, i) => <tspan key={i} x={x} dy={i ? fontSize + 2 : fontSize}>{l}</tspan>)}
    </text>
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

// While loading: grey placeholder blocks shaped like the content (no layout jump when it arrives).
export function Loading({ state, height = 120, children }) {
  if (state.loading && !state.data) {
    return <div className="skeleton" style={{ height }} role="status" aria-label="Loading" />;
  }
  if (state.error) {
    return (
      <p className="error" role="alert">Couldn't load: {state.error}{" "}
        {state.reload && <button className="btn" onClick={state.reload}>Try again</button>}</p>
    );
  }
  return children;
}

// Chart tooltip: values first, then labels; colour keys as short strokes (dataviz interaction rules).
export function ChartTooltip({ active, payload, label, labelFormatter, valueDigits = 4 }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card" style={{ padding: "8px 10px", gap: 4, fontSize: 12 }}>
      <div className="muted">{labelFormatter ? labelFormatter(label) : String(label).replace("\n", " ")}</div>
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
