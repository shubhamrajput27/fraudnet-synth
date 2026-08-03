// Verifies the idempotent-upsert contract itself (D16): both the REST poll path and the
// WebSocket watcher can call these for the same round concurrently, so every write MUST be an
// upsert keyed on the natural (run_id, round[, bank]) key, never a blind insert.
import assert from "node:assert/strict";
import { test } from "node:test";

import { finalizeRun, upsertClientMetric, upsertRoundMetric } from "../src/persistence.js";

function fakeDb() {
  const calls = [];
  return {
    calls,
    collection(name) {
      return {
        async updateOne(filter, update, options) {
          calls.push({ collection: name, filter, update, options });
        },
      };
    },
  };
}

test("upsertRoundMetric upserts keyed on run_id + round", async () => {
  const db = fakeDb();
  const doc = { run_id: "run1", round: 3, arm: "federated_real", metrics: { f1: 0.5 } };
  await upsertRoundMetric(db, doc);

  assert.equal(db.calls.length, 1);
  const call = db.calls[0];
  assert.equal(call.collection, "round_metrics");
  assert.deepEqual(call.filter, { run_id: "run1", round: 3 });
  assert.deepEqual(call.update, { $set: doc });
  assert.equal(call.options.upsert, true);
});

test("upsertClientMetric upserts keyed on run_id + round + bank", async () => {
  const db = fakeDb();
  const doc = { run_id: "run1", round: 2, bank: "A", metrics: { f1: 0.7 } };
  await upsertClientMetric(db, doc);

  const call = db.calls[0];
  assert.equal(call.collection, "client_metrics");
  assert.deepEqual(call.filter, { run_id: "run1", round: 2, bank: "A" });
  assert.equal(call.options.upsert, true);
});

test("calling upsertRoundMetric twice with the same key both go through updateOne (never insertMany)", async () => {
  const db = fakeDb();
  const doc = { run_id: "run1", round: 1, metrics: {} };
  await upsertRoundMetric(db, doc);
  await upsertRoundMetric(db, doc);

  assert.equal(db.calls.length, 2);
  assert.deepEqual(db.calls[0].filter, db.calls[1].filter); // same key both times — safe to race
});

test("finalizeRun sets status and final_metrics without upsert (the run doc always exists by then)", async () => {
  const db = fakeDb();
  await finalizeRun(db, { run_id: "run1", arm: "isolated_real", status: "complete", final_metrics: { f1: 0.4 } });

  const call = db.calls[0];
  assert.equal(call.collection, "runs");
  assert.deepEqual(call.filter, { run_id: "run1" });
  assert.equal(call.update.$set.status, "complete");
  assert.equal(call.update.$set.final_metrics.f1, 0.4);
  assert.equal(call.options, undefined); // no upsert: true here, deliberately
});
