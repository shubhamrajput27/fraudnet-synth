import { useEffect, useState } from "react";
import { downloadExport, getQualityPanel } from "../api.js";

const VERDICT_LABEL = {
  PASS: "Passed",
  REJECT_TOO_FEW_ROWS: "Rejected (too few valid rows)",
  REJECT_FIDELITY: "Rejected (fidelity below floor)",
  REJECT_NOVELTY: "Rejected (novelty below floor)",
  REJECT_MODE_COLLAPSE: "Rejected (mode collapse)",
};

export default function QualityPanel() {
  const [panel, setPanel] = useState([]);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(null);

  useEffect(() => {
    getQualityPanel().then(setPanel).catch((err) => setError(err.message));
  }, []);

  async function handleExport(bank, type) {
    setDownloading(`${bank}-${type}`);
    try {
      await downloadExport(bank, type);
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloading(null);
    }
  }

  if (error) return <p className="error">{error}</p>;

  return (
    <div className="panel">
      <h2>Synthetic quality panel</h2>
      <p className="muted">Phase 3 validation results (frozen thresholds — CLAUDE.md D6), per bank.</p>
      <table>
        <thead>
          <tr>
            <th>Bank</th><th>Mode</th><th>Candidates</th><th>Fidelity</th><th>Novelty</th>
            <th>Mode collapse (max)</th><th>Verdict</th><th>Validated rows</th><th>Export</th>
          </tr>
        </thead>
        <tbody>
          {panel.map((row) => (
            <tr key={row.bank}>
              <td>{row.bank}</td>
              <td>{row.mode === "augment_ctgan" ? "Augment (CTGAN)" : "Schema (LLM)"}</td>
              <td>{row.n_candidates}</td>
              <td>{row.validation?.fidelity_score?.toFixed(4) ?? "-"}</td>
              <td>{row.validation?.novelty_score?.toFixed(4) ?? "-"}</td>
              <td>{row.validation?.mode_collapse_max?.toFixed(4) ?? "-"}</td>
              <td>
                <span className={`badge ${row.validation?.batch_verdict === "PASS" ? "badge-augment" : "badge-rejected"}`}>
                  {VERDICT_LABEL[row.validation?.batch_verdict] ?? "-"}
                </span>
              </td>
              <td>{row.n_validated}</td>
              <td>
                <button disabled={row.n_candidates === 0} onClick={() => handleExport(row.bank, "candidates")}>
                  {downloading === `${row.bank}-candidates` ? "..." : "Candidates CSV"}
                </button>{" "}
                <button disabled={row.n_validated === 0} onClick={() => handleExport(row.bank, "validated")}>
                  {downloading === `${row.bank}-validated` ? "..." : "Validated CSV"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
