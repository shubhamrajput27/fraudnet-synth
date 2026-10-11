// Job watcher + live hub (decision D-049).
//
// For every job the gateway starts, a SERVER-SIDE watcher connects to FastAPI's job WebSocket and
// (1) saves each event to MongoDB, so a run's history is complete even if no browser is open, and
// (2) relays it to any browsers subscribed via the gateway's own WebSocket (/ws/runs/:jobId).
import WebSocket from "ws";
import { config } from "./config.js";
import { orch, orchestratorWsUrl } from "./orchestrator.js";
import { ModelResult, Round, Run } from "./models.js";

const subscribers = new Map();   // jobId -> Set of browser sockets
const watching = new Set();      // jobIds with an active watcher
const FINAL = ["succeeded", "failed", "cancelled", "interrupted"];

export function subscribe(jobId, socket) {
  if (!subscribers.has(jobId)) subscribers.set(jobId, new Set());
  subscribers.get(jobId).add(socket);
  socket.on("close", () => subscribers.get(jobId)?.delete(socket));
}

function broadcast(jobId, event) {
  for (const s of subscribers.get(jobId) ?? []) {
    if (s.readyState === WebSocket.OPEN) s.send(JSON.stringify(event));
  }
}

// FastAPI's snake_case round event -> the camelCase shape stored in MongoDB and sent to browsers.
export function roundToClient(e) {
  return { type: "round", arm: e.arm, seed: e.seed, round: e.round,
           globalTestPrAuc: e.global_test_pr_auc, globalTestRocAuc: e.global_test_roc_auc,
           banks: (e.banks ?? []).map((b) => ({ bank: b.bank, trainLoss: b.train_loss,
                                                 numExamples: b.num_examples, uploadBytes: b.upload_bytes })) };
}

async function handle(jobId, e) {
  if (e.type === "job_status") {
    const set = { status: e.status };
    if (e.status === "running") set.startedAt = new Date();
    await Run.updateOne({ jobId }, { $set: set });
    broadcast(jobId, { type: "status", status: e.status });
  } else if (e.type === "round") {
    const r = roundToClient(e);
    const { type, ...doc } = r;
    const res = await Round.updateOne({ jobId, arm: r.arm, seed: r.seed, round: r.round },
                                      { $setOnInsert: { jobId, ...doc } }, { upsert: true });
    if (res.upsertedCount) await Run.updateOne({ jobId }, { $inc: { roundsReceived: 1 } });
    broadcast(jobId, r);
  } else if (e.type === "model_done") {
    const doc = { jobId, seed: e.seed, arm: e.arm, model: e.model, precision: e.precision,
                  recall: e.recall, f1: e.f1, prAuc: e.pr_auc };
    await ModelResult.updateOne({ jobId, seed: e.seed, arm: e.arm, model: e.model }, { $set: doc }, { upsert: true });
    broadcast(jobId, { type: "model_done", ...doc });
  } else if (e.type === "end") {
    await Run.updateOne({ jobId }, { $set: { status: e.status, finishedAt: new Date(), result: e.result ?? null } });
    broadcast(jobId, { type: "end", status: e.status, result: e.result ?? null });
  } else if (e.type === "status") {
    // FastAPI's progress text ("seed 42: training ..."). Renamed so it can't be confused with the
    // job-status changes above, which also travel as type "status".
    broadcast(jobId, { type: "message", message: e.message });
  } else {
    broadcast(jobId, e);                                   // bank_done (generation), ...
  }
}

export function watchJob(jobId, attempt = 0) {
  if (watching.has(jobId)) return;
  watching.add(jobId);
  const ws = new WebSocket(orchestratorWsUrl(jobId), { headers: { "X-Internal-Key": config.orchestratorKey } });
  let ended = false;
  // Events are handled strictly in order (a promise chain), so rounds are never saved out of order.
  let chain = Promise.resolve();
  ws.on("message", (data) => {
    const e = JSON.parse(data.toString());
    if (e.type === "end") ended = true;
    chain = chain.then(() => handle(jobId, e)).catch((err) => console.error(`[watcher ${jobId}]`, err.message));
  });
  const retry = async () => {
    watching.delete(jobId);
    await chain;
    if (ended) return;
    // Lost the connection (e.g. FastAPI restarted): check the job; resume watching if still active.
    try {
      const job = await orch(`/jobs/${encodeURIComponent(jobId)}`);
      if (FINAL.includes(job.status)) {
        await handle(jobId, { type: "end", status: job.status, result: job.result });
      } else if (attempt < 20) {
        setTimeout(() => watchJob(jobId, attempt + 1), 3000);
      }
    } catch {
      if (attempt < 20) setTimeout(() => watchJob(jobId, attempt + 1), 3000);
    }
  };
  ws.on("close", retry);
  ws.on("error", () => {});                                // 'close' follows and handles it
}

// On gateway start-up: resume watching every run that wasn't finished when we last stopped.
export async function resumeWatchers() {
  const open = await Run.find({ status: { $in: ["queued", "running"] } }, { jobId: 1 });
  open.forEach((r) => watchJob(r.jobId));
  return open.length;
}

export function isFinal(status) {
  return FINAL.includes(status);
}
