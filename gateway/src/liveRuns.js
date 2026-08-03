// Live round-by-round push updates (CLAUDE.md Tier 1: "live FL round charts (WebSocket)"). One
// watcher per actively-viewed run_id: polls the orchestrator internally (the orchestrator itself
// stays plain HTTP, per Phase 5's design — see D14), persists any newly-seen round/client metric
// exactly once (idempotent upserts, see persistence.js), and broadcasts it to every subscribed
// WebSocket client for that run. Stops polling once the run reaches complete/failed.
import { finalizeRun, upsertClientMetric, upsertRoundMetric } from "./persistence.js";

const POLL_INTERVAL_MS = 1500;
const watchers = new Map(); // run_id -> watcher state

function fastApiUrl(path) {
  const base = process.env.FASTAPI_URL ?? "http://127.0.0.1:8000";
  return `${base}${path}`;
}

function broadcast(watcher, message) {
  const payload = JSON.stringify(message);
  for (const ws of watcher.subscribers) {
    if (ws.readyState === ws.OPEN) ws.send(payload);
  }
}

async function poll(runId, watcher, db) {
  let live;
  try {
    const res = await fetch(fastApiUrl(`/runs/${runId}`));
    if (!res.ok) return;
    live = await res.json();
  } catch {
    return; // orchestrator hiccup — try again on the next tick
  }

  for (const doc of live.round_metrics) {
    if (watcher.seenRounds.has(doc.round)) continue;
    watcher.seenRounds.add(doc.round);
    await upsertRoundMetric(db, doc);
    broadcast(watcher, { type: "round_metric", data: doc });
  }

  for (const doc of live.client_metrics) {
    const key = `${doc.round}:${doc.bank}`;
    if (watcher.seenClientKeys.has(key)) continue;
    watcher.seenClientKeys.add(key);
    await upsertClientMetric(db, doc);
    broadcast(watcher, { type: "client_metric", data: doc });
  }

  if (live.status === "complete" || live.status === "failed") {
    await finalizeRun(db, live);
    broadcast(watcher, { type: "done", status: live.status, final_metrics: live.final_metrics });
    clearInterval(watcher.intervalId);
    watchers.delete(runId);
  }
}

export function subscribe(runId, ws, db) {
  let watcher = watchers.get(runId);
  if (!watcher) {
    watcher = { subscribers: new Set(), seenRounds: new Set(), seenClientKeys: new Set(), intervalId: null };
    watcher.intervalId = setInterval(() => poll(runId, watcher, db), POLL_INTERVAL_MS);
    watchers.set(runId, watcher);
    poll(runId, watcher, db); // fire immediately, don't wait for the first tick
  }
  watcher.subscribers.add(ws);

  ws.on("close", () => {
    watcher.subscribers.delete(ws);
    // Keep polling even with zero subscribers until completion, so a late-joining client still
    // gets a fast-forwarded view via GET /api/runs/:id — the watcher only stops itself in poll().
  });
}
