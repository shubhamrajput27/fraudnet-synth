import { useState } from "react";
import { triggerRun } from "../api.js";
import { useLiveRun } from "../useLiveRun.js";
import LiveRoundChart from "./LiveRoundChart.jsx";
import ModeIndicator from "./ModeIndicator.jsx";

const ARMS = [
  "isolated_real", "isolated_augmented",
  "federated_real", "federated_augmented",
  "centralized_real", "centralized_augmented",
];
const BANKS = ["A", "B", "C", "D"];

export default function TriggerRun() {
  const [arm, setArm] = useState("federated_augmented");
  const [seed, setSeed] = useState(42);
  const [epochs, setEpochs] = useState(20);
  const [numRounds, setNumRounds] = useState(10);
  const [localEpochs, setLocalEpochs] = useState(2);
  const [runId, setRunId] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const live = useLiveRun(runId);
  const augmented = arm.endsWith("_augmented");
  const isFederatedOrIsolated = arm.startsWith("federated") || arm.startsWith("isolated");

  async function handleTrigger(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await triggerRun({
        arm, seed: Number(seed), epochs: Number(epochs),
        num_rounds: Number(numRounds), local_epochs: Number(localEpochs),
      });
      setRunId(res.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const latestByBank = {};
  for (const cm of live.clientMetrics) {
    if (!latestByBank[cm.bank] || cm.round > latestByBank[cm.bank].round) latestByBank[cm.bank] = cm;
  }

  return (
    <div className="panel">
      <h2>Trigger a run</h2>
      <form className="run-form" onSubmit={handleTrigger}>
        <label>
          Arm
          <select value={arm} onChange={(e) => setArm(e.target.value)}>
            {ARMS.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
        </label>
        {isFederatedOrIsolated && arm.startsWith("federated") ? (
          <>
            <label>
              FedAvg rounds
              <input type="number" value={numRounds} onChange={(e) => setNumRounds(e.target.value)} />
            </label>
            <label>
              Local epochs/round
              <input type="number" value={localEpochs} onChange={(e) => setLocalEpochs(e.target.value)} />
            </label>
          </>
        ) : (
          <label>
            Local epochs
            <input type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
          </label>
        )}
        <button type="submit" disabled={busy}>{busy ? "Starting..." : "Trigger run"}</button>
      </form>
      {error && <p className="error">{error}</p>}

      {runId && (
        <div className="live-run">
          <h3>
            {runId} — <span className={`status status-${live.status}`}>{live.status}</span>
          </h3>

          {isFederatedOrIsolated && (
            <div className="mode-row">
              {BANKS.map((bank) => (
                <div key={bank} className="mode-cell">
                  <strong>Bank {bank}</strong>
                  <ModeIndicator bank={bank} augmented={augmented} verdict={null} />
                  {latestByBank[bank] && <span className="mini-metric">f1={latestByBank[bank].metrics.f1.toFixed(3)}</span>}
                </div>
              ))}
            </div>
          )}

          {arm.startsWith("federated") && <LiveRoundChart roundMetrics={live.roundMetrics} />}

          {live.finalMetrics && (
            <div className="final-metrics">
              <strong>Final:</strong> F1={live.finalMetrics.f1.toFixed(4)} Precision=
              {live.finalMetrics.precision.toFixed(4)} Recall={live.finalMetrics.recall.toFixed(4)} AUC=
              {live.finalMetrics.auc.toFixed(4)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
