// Idempotent Mongo writes for round_metrics/client_metrics/runs (D8). Upserts, not blind
// inserts, because both the REST poll path (routes/runs.js) and the WebSocket live-watcher
// (liveRuns.js) can observe and persist the same round concurrently while a run is in progress —
// upserting on a natural key makes that race harmless instead of producing duplicate documents.
export async function upsertRoundMetric(db, doc) {
  await db.collection("round_metrics").updateOne(
    { run_id: doc.run_id, round: doc.round },
    { $set: doc },
    { upsert: true }
  );
}

export async function upsertClientMetric(db, doc) {
  await db.collection("client_metrics").updateOne(
    { run_id: doc.run_id, round: doc.round, bank: doc.bank },
    { $set: doc },
    { upsert: true }
  );
}

export async function finalizeRun(db, { run_id, arm, status, final_metrics }) {
  await db.collection("runs").updateOne(
    { run_id },
    { $set: { status, final_metrics, arm, updated_at: new Date() } }
  );
}
