// Six-arm comparison grid (Step 9 multi-seed results). Headline = fraud-class precision, recall,
// F1 and PR-AUC; ROC-AUC and accuracy are shown greyed out as secondary.
import { useMemo, useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer,
         Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, Legend, Loading, bankName, parsePm, useApi } from "../components.jsx";
import { ROLE, axisProps, usePalette } from "../theme.js";

const GROUPS = [
  { label: "Bank A alone", real: [1, "bank_a"], aug: [2, "bank_a"] },
  { label: "Bank B alone", real: [1, "bank_b"], aug: [2, "bank_b"] },
  { label: "Bank C alone", real: [1, "bank_c"], aug: [2, "bank_c"] },
  { label: "Bank D alone", real: [1, "bank_d"], aug: [2, "bank_d"] },
  { label: "Federated", real: [3, "federated"], aug: [4, "federated"] },
  { label: "Centralized*", real: [5, "centralized"], aug: [6, "centralized"] },
];
const VERDICT = { higher: "▲ higher", lower: "▼ lower", "no clear difference": "● unclear" };

function ArmBars({ rows, metric, title, p }) {
  const find = ([arm, model]) => rows.find((r) => r.arm === arm && r.model === model);
  const data = GROUPS.map((g) => {
    const r = parsePm(find(g.real)?.[metric]);
    const a = parsePm(find(g.aug)?.[metric]);
    return { group: g.label, real: r.mean, realSd: r.std, aug: a.mean, augSd: a.std };
  });
  return (
    <article className="card">
      <header><h3>{title}</h3><span className="muted">mean ± std, 5 seeds, global test</span></header>
      <Legend items={[{ label: "Real data only", color: p.series[ROLE.real], box: true },
                      { label: "Real + validated synthetic", color: p.series[ROLE.augmented], box: true }]} />
      <div style={{ height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2}>
            <CartesianGrid vertical={false} stroke={p.grid} />
            <XAxis dataKey="group" interval={0} {...axisProps(p)} tick={{ fill: p.muted, fontSize: 11 }} />
            <YAxis domain={[0, 1]} {...axisProps(p)} />
            <Tooltip cursor={{ fill: p.grid, opacity: 0.4 }} content={<ChartTooltip />} />
            <Bar dataKey="real" name="real only" fill={p.series[ROLE.real]} radius={[4, 4, 0, 0]} maxBarSize={30}>
              <ErrorBar dataKey="realSd" width={4} stroke={p.ink2} />
            </Bar>
            <Bar dataKey="aug" name="real + synthetic" fill={p.series[ROLE.augmented]} radius={[4, 4, 0, 0]} maxBarSize={30}>
              <ErrorBar dataKey="augSd" width={4} stroke={p.ink2} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function Convergence({ curves, centralPrAuc, p }) {
  const data = useMemo(() => {
    if (!curves) return [];
    const stat = (arm, i) => {
      const v = Object.values(curves[arm]).map((s) => s[i]);
      const m = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
      return [m, sd];
    };
    const n = Object.values(curves.arm3_federated_real)[0].length;
    return Array.from({ length: n }, (_, i) => {
      const [m3, s3] = stat("arm3_federated_real", i);
      const [m4, s4] = stat("arm4_federated_aug", i);
      return { round: i + 1, arm3: m3, arm3band: [m3 - s3, m3 + s3], arm4: m4, arm4band: [m4 - s4, m4 + s4] };
    });
  }, [curves]);
  return (
    <article className="card">
      <header><h3>Federated training converges</h3><span className="muted">global PR-AUC per round, mean ± std over 5 seeds</span></header>
      <Legend items={[{ label: "Arm 3 · federated, real", color: p.series[ROLE.real] },
                      { label: "Arm 4 · federated, augmented", color: p.series[ROLE.augmented] },
                      { label: "centralized real (mean)", color: p.muted }]} />
      <div style={{ height: 260 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -6 }}>
            <CartesianGrid vertical={false} stroke={p.grid} />
            <XAxis dataKey="round" {...axisProps(p)} />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} {...axisProps(p)} />
            <Tooltip content={<ChartTooltip labelFormatter={(r) => `Round ${r}`} />} />
            <Area dataKey="arm3band" stroke="none" fill={p.series[ROLE.real]} fillOpacity={0.15} isAnimationActive={false} name="Arm 3 ± std" />
            <Area dataKey="arm4band" stroke="none" fill={p.series[ROLE.augmented]} fillOpacity={0.15} isAnimationActive={false} name="Arm 4 ± std" />
            <Line dataKey="arm3" name="Arm 3 mean" stroke={p.series[ROLE.real]} strokeWidth={2} dot={false} />
            <Line dataKey="arm4" name="Arm 4 mean" stroke={p.series[ROLE.augmented]} strokeWidth={2} dot={false} />
            {centralPrAuc && <ReferenceLine y={centralPrAuc} stroke={p.muted} strokeDasharray="4 4" />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="note">The y-axis is zoomed (≈0.67–0.73), so small differences look bigger than they are.</p>
    </article>
  );
}

export default function Results() {
  const res = useApi("/results/six-arms");
  const conv = useApi("/results/six-arms/convergence");
  const p = usePalette();
  const [family, setFamily] = useState("federation");
  const rows = res.data?.global_table ?? [];
  const central = rows.find((r) => r.arm === 5);

  return (
    <Loading state={res}>
      <section className="card">
        <header><h2>Six-arm comparison</h2><span className="muted">seeds {res.data?.seeds?.join(", ")} · 95 frauds in the global test set</span></header>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Arm</th><th>Model</th>
                <th className="num headline">Precision</th><th className="num headline">Recall</th>
                <th className="num headline">F1 (fraud)</th><th className="num headline">PR-AUC</th>
                <th className="num secondary">ROC-AUC</th><th className="num secondary">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.arm}-${r.model}`}>
                  <td>{r.arm} · {r.setting}</td>
                  <td>{r.model.startsWith("bank_") ? bankName(r.model) : r.model}</td>
                  {["precision", "recall", "f1", "pr_auc"].map((k) => <td key={k} className="num headline">{r[k]}</td>)}
                  {["roc_auc", "accuracy"].map((k) => <td key={k} className="num secondary">{r[k]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">Accuracy is ≈99.9% for every model because 99.83% of transactions are genuine, so it is shown
          only as a secondary column. *Centralized pools all banks' data: an upper bound that would be illegal in practice.</p>
      </section>

      <section className="grid cols-2">
        <ArmBars rows={rows} metric="f1" title="Fraud F1 by arm" p={p} />
        <ArmBars rows={rows} metric="pr_auc" title="PR-AUC by arm" p={p} />
      </section>

      <section className="grid cols-2">
        <Loading state={conv}>
          <Convergence curves={conv.data} centralPrAuc={central ? parsePm(central.pr_auc).mean : null} p={p} />
        </Loading>
        <article className="card">
          <header>
            <h3>Is the difference real? (paired, 95% CI)</h3>
            <select value={family} onChange={(e) => setFamily(e.target.value)} aria-label="comparison family">
              <option value="federation">Federation vs alone</option>
              <option value="augmentation">Augmentation effect</option>
              <option value="fed_vs_central">Federated vs centralized</option>
            </select>
          </header>
          <div className="table-wrap">
            <table className="compact">
              <thead><tr><th>Comparison</th><th>Bank</th><th>Test set</th><th>Metric</th><th className="num">Δ mean</th>
                <th className="num">95% CI</th><th className="num">Seeds up</th><th>Verdict</th></tr></thead>
              <tbody>
                {(res.data?.paired_comparisons ?? []).filter((c) => c.family === family && ["f1", "pr_auc"].includes(c.metric))
                  .map((c, i) => (
                    <tr key={i}>
                      <td>{c.comparison}</td><td>{c.bank.startsWith("bank_") ? bankName(c.bank) : c.bank}</td>
                      <td>{c.eval_set === "global test" ? "global" : "own local"}</td><td>{c.metric === "f1" ? "F1" : "PR-AUC"}</td>
                      <td className="num">{c.mean_diff >= 0 ? "+" : ""}{c.mean_diff.toFixed(4)}</td>
                      <td className="num">[{c.ci_low.toFixed(3)}, {c.ci_high.toFixed(3)}]</td>
                      <td className="num">{c.improved_in}/{c.n}</td>
                      <td>{VERDICT[c.verdict] ?? c.verdict}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="note">"Unclear" = the 95% CI includes 0. A difference counts only if its 95% CI excludes 0. With 56 comparisons, about 3 "clear" results
            are expected by chance.</p>
        </article>
      </section>
    </Loading>
  );
}
