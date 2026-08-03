import { useEffect, useState } from "react";
import { listRuns } from "../api.js";

export default function RunHistory() {
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="muted">Loading run history...</p>;
  if (error) return <p className="error">{error}</p>;
  if (runs.length === 0) return <p className="muted">No runs yet — trigger one from the "Trigger Run" tab.</p>;

  return (
    <div className="panel">
      <h2>Run history</h2>
      <table>
        <thead>
          <tr>
            <th>Run ID</th><th>Arm</th><th>Status</th><th>F1</th><th>Precision</th><th>Recall</th><th>AUC</th><th>Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id}>
              <td>{r.run_id}</td>
              <td>{r.arm}</td>
              <td><span className={`status status-${r.status}`}>{r.status}</span></td>
              <td>{r.final_metrics?.f1?.toFixed(4) ?? "-"}</td>
              <td>{r.final_metrics?.precision?.toFixed(4) ?? "-"}</td>
              <td>{r.final_metrics?.recall?.toFixed(4) ?? "-"}</td>
              <td>{r.final_metrics?.auc?.toFixed(4) ?? "-"}</td>
              <td>{new Date(r.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
