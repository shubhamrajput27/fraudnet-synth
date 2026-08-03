import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// Live FL round chart (CLAUDE.md Tier 1). `roundMetrics` is the macro-averaged metric per round,
// as pushed over WebSocket — one point appears per completed FedAvg round.
export default function LiveRoundChart({ roundMetrics }) {
  const data = roundMetrics
    .filter((r) => r.round != null)
    .sort((a, b) => a.round - b.round)
    .map((r) => ({ round: r.round, f1: r.metrics.f1, precision: r.metrics.precision, recall: r.metrics.recall }));

  if (data.length === 0) {
    return <p className="muted">Waiting for the first round to complete...</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -4 }} />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="f1" stroke="#2563eb" strokeWidth={2} dot />
        <Line type="monotone" dataKey="precision" stroke="#16a34a" strokeWidth={2} dot />
        <Line type="monotone" dataKey="recall" stroke="#d97706" strokeWidth={2} dot />
      </LineChart>
    </ResponsiveContainer>
  );
}
