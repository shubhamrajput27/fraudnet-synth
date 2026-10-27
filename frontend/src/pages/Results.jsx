// Six-arm comparison grid (Step 9 multi-seed results). Headline = fraud-class precision, recall,
// F1 and PR-AUC; ROC-AUC and accuracy are shown greyed out as secondary.
import { useMemo, useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer,
         Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, Legend, Loading, More, PageHead, Term, WrapTick, bankName, fmt, int, parsePm, useApi } from "../components.jsx";
import { ROLE, axisProps, usePalette } from "../theme.js";

const GROUPS = [
  // "\n" = line break under the bar group (WrapTick), so six names fit in a half-width card.
  { label: "Bank A\nalone", real: [1, "bank_a"], aug: [2, "bank_a"] },
  { label: "Bank B\nalone", real: [1, "bank_b"], aug: [2, "bank_b"] },
  { label: "Bank C\nalone", real: [1, "bank_c"], aug: [2, "bank_c"] },
  { label: "Bank D\nalone", real: [1, "bank_d"], aug: [2, "bank_d"] },
  { label: "Federated", real: [3, "federated"], aug: [4, "federated"] },
  { label: "Central-\nized*", real: [5, "centralized"], aug: [6, "centralized"] },
];
const VERDICT = { higher: "▲ higher", lower: "▼ lower", "no clear difference": "● unclear" };

function ArmBars({ rows, metric, title, nSeeds, p }) {
  const find = ([arm, model]) => rows.find((r) => r.arm === arm && r.model === model);
  const data = GROUPS.map((g) => {
    const r = parsePm(find(g.real)?.[metric]);
    const a = parsePm(find(g.aug)?.[metric]);
    return { group: g.label, real: r.mean, realSd: r.std, aug: a.mean, augSd: a.std };
  });
  return (
    <article className="card">
      <header><h3>{title}</h3><span className="muted">mean <Term k="sd">± std</Term>, {nSeeds} <Term k="seeds">seeds</Term>, global test</span></header>
      <Legend items={[{ label: "Real data only", color: p.series[ROLE.real], box: true },
                      { label: "Real + validated synthetic", color: p.series[ROLE.augmented], box: true }]} />
      <div style={{ height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2}>
            <CartesianGrid vertical={false} stroke={p.grid} />
            <XAxis dataKey="group" interval={0} height={36} {...axisProps(p)} tick={<WrapTick fill={p.muted} />} />
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
  const nSeeds = curves ? Object.keys(curves.arm3_federated_real).length : 0;
  const span = data.flatMap((d) => [...d.arm3band, ...d.arm4band]);
  return (
    <article className="card">
      <header><h3>Federated training converges</h3><span className="muted">global <Term k="pr_auc">PR-AUC</Term> per round, mean ± std over {nSeeds} seeds</span></header>
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
      {span.length > 0 && <p className="note">Zoomed y-axis ({fmt(Math.min(...span), 2)}–{fmt(Math.max(...span), 2)}):
        small gaps look bigger than they are.</p>}
    </article>
  );
}

// The headline figures, read from the results table: fraud F1 (mean ± std) for the main settings.
function Headline({ rows, paired }) {
  const f1 = (arm, model) => rows.find((r) => r.arm === arm && r.model === model)?.f1;
  const alone = rows.filter((r) => r.arm === 1).map((r) => parsePm(r.f1).mean);
  const tiles = [
    { k: "Federated + synthetic", sub: "Arm 4 · FraudNet-Synth", v: f1(4, "federated"), main: true },
    { k: "Federated, real only", sub: "Arm 3", v: f1(3, "federated") },
    { k: "Each bank alone", sub: "Arm 1 · range over Banks A–D",
      v: alone.length ? `${fmt(Math.min(...alone), 3)} – ${fmt(Math.max(...alone), 3)}` : null },
    { k: "Centralized (upper bound)", term: "centralized", sub: "Arm 5 · pools all data, not private", v: f1(5, "centralized") },
  ];
  // The tiles alone invite "Arm 4 > Arm 3"; show the paired test's verdict right beside them.
  const aug = paired.find((c) => c.comparison === "Arm 4 - Arm 3" && c.eval_set === "global test" && c.metric === "f1");
  return (
    <>
    <section className="grid cols-4">
      {tiles.map((t) => (
        <article key={t.k} className={`card kpi${t.main ? " main" : ""}`}>
          <div className="k">{t.term ? <Term k={t.term}>{t.k}</Term> : t.k}</div>
          <div className="v num">{t.v ? String(t.v).split("±")[0].trim() : "—"}
            {String(t.v ?? "").includes("±") && <small> ± {String(t.v).split("±")[1].trim()}</small>}</div>
          <div className="muted">fraud <Term k="f1">F1</Term> · {t.sub}</div>
        </article>
      ))}
    </section>
    {aug && <p className="callout">Does synthetic data help inside federation? <strong>{aug.verdict === "no clear difference"
      ? "No clear difference" : `Arm 4 is clearly ${aug.verdict}`}</strong>: Arm 4 − Arm 3 F1 = {aug.mean_diff >= 0 ? "+" : ""}
      {aug.mean_diff.toFixed(4)}, <Term k="ci">95% CI</Term> [{aug.ci_low.toFixed(3)}, {aug.ci_high.toFixed(3)}] over {aug.n} seeds.</p>}
    </>
  );
}

export default function Results() {
  const res = useApi("/results/six-arms");
  const conv = useApi("/results/six-arms/convergence");
  const p = usePalette();
  const [family, setFamily] = useState("federation");
  const rows = res.data?.global_table ?? [];
  const central = rows.find((r) => r.arm === 5);
  const nSeeds = res.data?.seeds?.length ?? 0;
  const minAcc = rows.length ? Math.min(...rows.map((r) => parsePm(r.accuracy).mean)) : null;
  const genuinePct = res.data?.test_rows ? 100 * (1 - res.data.test_frauds / res.data.test_rows) : null;
  // How many F1/PR-AUC comparisons the table can show in total, and how many "clear" ones chance alone would give.
  const nShown = (res.data?.paired_comparisons ?? []).filter((c) => ["f1", "pr_auc"].includes(c.metric)).length;

  return (
    <>
    <PageHead title="Six-arm results"
              lead="Fraud F1 on the same global test set for every arm. Read top to bottom: headline, charts, full table, then whether the gaps are real." />
    <Loading state={res} height={320}>
      <Headline rows={rows} paired={res.data?.paired_comparisons ?? []} />
      <section className="grid cols-2">
        <ArmBars rows={rows} metric="f1" title="Fraud F1 by arm" nSeeds={nSeeds} p={p} />
        <ArmBars rows={rows} metric="pr_auc" title="PR-AUC by arm" nSeeds={nSeeds} p={p} />
      </section>

      <section className="card">
        <header><h2>Six-arm comparison</h2><span className="muted">seeds {res.data?.seeds?.join(", ")} ·{" "}
          {int(res.data?.test_frauds)} frauds among {int(res.data?.test_rows)} global test transactions</span></header>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Arm</th><th>Model</th>
                <th className="num headline"><Term k="precision">Precision</Term></th><th className="num headline"><Term k="recall">Recall</Term></th>
                <th className="num headline"><Term k="f1">F1 (fraud)</Term></th><th className="num headline"><Term k="pr_auc">PR-AUC</Term></th>
                <th className="num secondary"><Term k="roc_auc">ROC-AUC</Term></th><th className="num secondary"><Term k="accuracy">Accuracy</Term></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.arm}-${r.model}`} className={r.arm === 4 ? "highlight" : undefined}>
                  <td>{r.arm} · {r.setting}{r.arm === 4 && <span className="badge tag">FraudNet-Synth</span>}</td>
                  <td>{r.model.startsWith("bank_") ? bankName(r.model) : r.model}</td>
                  {["precision", "recall", "f1", "pr_auc"].map((k) => <td key={k} className="num headline">{r[k]}</td>)}
                  {["roc_auc", "accuracy"].map((k) => <td key={k} className="num secondary">{r[k]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">Accuracy is greyed out: it is at least {fmt(minAcc === null ? null : minAcc * 100, 2)}% for every model,
          because {fmt(genuinePct, 2)}% of test transactions are genuine. *Centralized pools all banks' data: an upper bound, not private.</p>
      </section>

      <section className="grid cols-2">
        <Loading state={conv} height={300}>
          <Convergence curves={conv.data} centralPrAuc={central ? parsePm(central.pr_auc).mean : null} p={p} />
        </Loading>
        <article className="card">
          <header>
            <h3>Is the difference real? <span className="muted">(paired, <Term k="ci">95% CI</Term>)</span></h3>
            <select value={family} onChange={(e) => setFamily(e.target.value)} aria-label="comparison family">
              <option value="federation">Federation vs alone</option>
              <option value="augmentation">Augmentation effect</option>
              <option value="fed_vs_central">Federated vs centralized</option>
            </select>
          </header>
          <div className="table-wrap">
            <table className="compact">
              <thead><tr><th>Comparison</th><th>Bank</th><th>Test set</th><th>Metric</th><th className="num"><Term k="delta">Δ mean</Term></th>
                <th className="num"><Term k="ci">95% CI</Term></th><th className="num"><Term k="seeds_up">Seeds up</Term></th><th>Verdict</th></tr></thead>
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
          <p className="note">"Unclear" = the 95% CI includes 0, so neither side is shown to be better.</p>
          <More label="How many “clear” results could be luck?">With {nShown} comparisons across all three families, about{" "}
            {Math.round(nShown * 0.05)} would look "clear" by chance alone at 95% confidence, so one isolated ▲ or ▼ is weak evidence.</More>
        </article>
      </section>
    </Loading>
    </>
  );
}
