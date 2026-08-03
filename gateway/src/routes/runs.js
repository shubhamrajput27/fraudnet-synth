// Run management (CLAUDE.md Tier 2): triggers runs via the FastAPI orchestrator (Tier 3) and
// owns MongoDB persistence (D8 collections: runs, round_metrics, client_metrics). The gateway
// never runs ML code itself — it only calls the orchestrator over internal HTTP and persists
// whatever comes back, matching the architecture's tier split.
import { Router } from "express";
import { getDb } from "../db.js";
import { finalizeRun, upsertClientMetric, upsertRoundMetric } from "../persistence.js";

const router = Router();

function fastApiUrl(path) {
  const base = process.env.FASTAPI_URL ?? "http://127.0.0.1:8000";
  return `${base}${path}`;
}

router.post("/", async (req, res) => {
  const { arm, seed = 42, epochs = 20, num_rounds = 10, local_epochs = 2 } = req.body ?? {};
  if (!arm) return res.status(400).json({ error: "arm is required" });

  const orchestratorRes = await fetch(fastApiUrl("/runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arm, seed, epochs, num_rounds, local_epochs }),
  });
  if (!orchestratorRes.ok) {
    const detail = await orchestratorRes.text();
    return res.status(502).json({ error: "Orchestrator rejected the run request", detail });
  }
  const { run_id } = await orchestratorRes.json();

  const db = getDb();
  await db.collection("runs").insertOne({
    run_id,
    arm,
    seed,
    config: { epochs, num_rounds, local_epochs },
    status: "running",
    final_metrics: null,
    created_at: new Date(),
    updated_at: new Date(),
  });

  res.status(202).json({ run_id, arm, status: "running" });
});

router.get("/", async (req, res) => {
  const db = getDb();
  const runs = await db.collection("runs").find({}).sort({ created_at: -1 }).toArray();
  res.json(runs);
});

router.get("/:runId", async (req, res) => {
  const { runId } = req.params;
  const db = getDb();
  const runDoc = await db.collection("runs").findOne({ run_id: runId });
  if (!runDoc) return res.status(404).json({ error: `Unknown run_id '${runId}'` });

  if (runDoc.status === "complete" || runDoc.status === "failed") {
    const roundMetrics = await db.collection("round_metrics").find({ run_id: runId }).toArray();
    const clientMetrics = await db.collection("client_metrics").find({ run_id: runId }).toArray();
    return res.json({ ...runDoc, round_metrics: roundMetrics, client_metrics: clientMetrics });
  }

  // Still running as of our last write — poll the orchestrator for fresh state.
  const orchestratorRes = await fetch(fastApiUrl(`/runs/${runId}`));
  if (!orchestratorRes.ok) {
    return res.status(502).json({ error: "Orchestrator lost track of this run" });
  }
  const live = await orchestratorRes.json();

  if (live.status === "complete" || live.status === "failed") {
    await persistCompletedRun(db, live);
  } else {
    await db.collection("runs").updateOne({ run_id: runId }, { $set: { updated_at: new Date() } });
  }

  res.json({ ...runDoc, ...live });
});

async function persistCompletedRun(db, live) {
  const { round_metrics, client_metrics } = live;
  for (const doc of round_metrics) await upsertRoundMetric(db, doc);
  for (const doc of client_metrics) await upsertClientMetric(db, doc);
  await finalizeRun(db, live);
}

export default router;
