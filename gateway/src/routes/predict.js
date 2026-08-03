// Proxies "test a transaction" requests to the orchestrator's /predict (CLAUDE.md Tier 1 demo
// feature). No persistence here — a prediction isn't a run, it's a one-off inference query.
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Router } from "express";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HOLDOUT_PATH = path.resolve(__dirname, "..", "..", "..", "data", "processed", "global_test_holdout.csv");

const router = Router();

function fastApiUrl(path) {
  const base = process.env.FASTAPI_URL ?? "http://127.0.0.1:8000";
  return `${base}${path}`;
}

// Convenience for the demo form — a real transaction's features (never used for training/eval
// anywhere else), so a user doesn't have to hand-type 30 PCA values to try the model.
router.get("/sample", (req, res) => {
  if (!existsSync(HOLDOUT_PATH)) return res.status(404).json({ error: "Holdout file not found" });
  const lines = readFileSync(HOLDOUT_PATH, "utf8").split("\n").filter((l) => l.trim().length > 0);
  const header = lines[0].split(",");
  const pick = lines[1 + Math.floor(Math.random() * (lines.length - 1))].split(",").map(Number);

  const row = Object.fromEntries(header.map((col, i) => [col, pick[i]]));
  const actualClass = row.Class;
  delete row.Class;
  res.json({ features: row, actual_class: actualClass === 1 ? "fraud" : "legitimate" });
});

router.get("/manifest/:runId", async (req, res) => {
  const orchestratorRes = await fetch(fastApiUrl(`/predict/manifest/${req.params.runId}`));
  const body = await orchestratorRes.json();
  res.status(orchestratorRes.status).json(body);
});

router.post("/", async (req, res) => {
  const orchestratorRes = await fetch(fastApiUrl("/predict"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
  const body = await orchestratorRes.json();
  res.status(orchestratorRes.status).json(body);
});

export default router;
