import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { listRuns } from "../api.js";

const ARMS = [
  "isolated_real", "isolated_augmented",
  "federated_real", "federated_augmented",
  "centralized_real", "centralized_augmented",
];

// CLAUDE.md's six-arm grid, populated from the most recently *completed* run of each arm in
// history — the dashboard doesn't force a fresh sweep, it just shows whatever's been run so far.
export default function ComparisonGrid() {
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    listRuns().then(setRuns).catch((err) => setError(err.message));
  }, []);

  if (error) return <p className="error">{error}</p>;

  const latestPerArm = {};
  for (const r of runs) {
    if (r.status !== "complete") continue;
    if (!latestPerArm[r.arm] || new Date(r.created_at) > new Date(latestPerArm[r.arm].created_at)) {
      latestPerArm[r.arm] = r;
    }
  }

  const data = ARMS.map((arm) => {
    const run = latestPerArm[arm];
    return {
      arm,
      f1: run ? run.final_metrics.f1 : 0,
      precision: run ? run.final_metrics.precision : 0,
      recall: run ? run.final_metrics.recall : 0,
      hasData: Boolean(run),
    };
  });

  const missing = data.filter((d) => !d.hasData).map((d) => d.arm);

  return (
    <div className="panel">
      <h2>Six-arm comparison</h2>
      {missing.length > 0 && (
        <p className="muted">No completed run yet for: {missing.join(", ")}. Trigger them from the "Trigger Run" tab.</p>
      )}
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="arm" angle={-30} textAnchor="end" interval={0} height={80} />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="f1" fill="#2563eb" name="F1" />
          <Bar dataKey="precision" fill="#16a34a" name="Precision" />
          <Bar dataKey="recall" fill="#d97706" name="Recall" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
