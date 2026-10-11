// Gateway tests: run with `npm test` (needs the local MongoDB service; uses database fraudnet_synth_test).
// A fake orchestrator stands in for FastAPI so these tests are fast and independent of Tier 3.
import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import express from "express";
import { WebSocketServer } from "ws";
import WebSocket from "ws";

const KEY = "test-internal-key-123456";
let fake, fakePort, gw, gwPort, request, mongoose, models;
const pw = { admin: "admin-pass-1234", bank_a: "bank-a-pass-1234", bank_b: "bank-b-pass-1234",
             bank_c: "bank-c-pass-1234", bank_d: "bank-d-pass-1234", guide: "guide-pass-1234" };
const seen = { exportScopes: [], keys: new Set() };

function startFakeOrchestrator() {
  const app = express();
  app.use(express.json());
  app.use((req, res, next) => { seen.keys.add(req.get("x-internal-key")); next(); });
  app.get("/health", (req, res) => res.json({ status: "ok" }));
  app.get("/banks", (req, res) => res.json(["bank_a", "bank_b", "bank_c", "bank_d"].map((b) => ({
    bank: b, mode: "Augment (CTGAN)", rows: 100, fraud: 5, fraud_pct: 5, train_fraud: 3,
    synthetic_candidates: 10, synthetic_validated: 8, pass_rate_pct: 80, fidelity_validated: 0.8 }))));
  app.post("/runs", (req, res) => res.status(202).json({ id: "job-1", type: "run", params: req.body, status: "queued",
                                                          created: new Date().toISOString() }));
  app.get("/jobs/:id", (req, res) => res.json({ id: req.params.id, status: "succeeded", result: { ok: true } }));
  app.post("/predict", (req, res) => res.json({ score_logit: 5, is_fraud: true }));
  app.get("/banks/:bank/synthetic/export", (req, res) => {
    seen.exportScopes.push(req.get("x-bank-scope"));
    res.type("text/csv").send("Time,V1,Class\n1,2,1\n");
  });
  const server = http.createServer(app);
  const wss = new WebSocketServer({ server, path: "/ws/jobs/job-1" });
  wss.on("connection", async (ws, req) => {
    if (req.headers["x-internal-key"] !== KEY) return ws.close(4401);
    const send = (o) => ws.send(JSON.stringify(o));
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    send({ type: "job_status", status: "running" });
    for (const round of [1, 2]) {
      await wait(50);
      send({ type: "round", arm: "arm3_federated_real", seed: 42, round, global_test_pr_auc: 0.70 + round / 100,
             global_test_roc_auc: 0.95, banks: [{ bank: "bank_a", train_loss: 0.3, num_examples: 100, upload_bytes: 17478 }] });
    }
    send({ type: "round", arm: "arm3_federated_real", seed: 42, round: 2, global_test_pr_auc: 0.72,  // duplicate
           global_test_roc_auc: 0.95, banks: [] });
    send({ type: "model_done", seed: 42, arm: "arm3_federated_real", model: "federated",
           precision: 0.9, recall: 0.7, f1: 0.79, pr_auc: 0.72 });
    send({ type: "job_status", status: "succeeded" });
    send({ type: "end", status: "succeeded", result: { arms: [3] } });
    ws.close();
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

before(async () => {
  fake = await startFakeOrchestrator();
  fakePort = fake.address().port;
  Object.assign(process.env, { ORCHESTRATOR_URL: `http://127.0.0.1:${fakePort}`, ORCHESTRATOR_INTERNAL_KEY: KEY,
    GATEWAY_JWT_SECRET: "test-jwt-secret-0123456789-abcdefghij", GATEWAY_MONGO_URI: "mongodb://localhost:27017/fraudnet_gateway_test" });
  mongoose = (await import("mongoose")).default;
  await mongoose.connect(process.env.GATEWAY_MONGO_URI);
  await mongoose.connection.dropDatabase();
  models = await import("../src/models.js");
  await (await import("../src/seed.js")).seedUsers(pw);
  const { createApp, attachLiveSocket } = await import("../src/app.js");
  gw = http.createServer(createApp());
  attachLiveSocket(gw);
  await new Promise((r) => gw.listen(0, "127.0.0.1", r));
  gwPort = gw.address().port;
  request = (await import("supertest")).default(gw);
});

after(async () => {
  await mongoose.connection.dropDatabase();
  await mongoose.disconnect();
  gw.close();
  fake.close();
});

async function tokenFor(user) {
  const r = await request.post("/api/auth/login").send({ username: user, password: pw[user] });
  assert.equal(r.status, 200);
  return r.body.token;
}

test("health is open; everything else needs a login", async () => {
  assert.equal((await request.get("/api/health")).body.gateway, "ok");
  assert.equal((await request.get("/api/banks")).status, 401);
  assert.equal((await request.get("/api/banks").set("Authorization", "Bearer forged.token.value")).status, 401);
});

test("wrong password is refused with a generic message and audited", async () => {
  const r = await request.post("/api/auth/login").send({ username: "admin", password: "nope-nope-nope" });
  assert.equal(r.status, 401);
  const r2 = await request.post("/api/auth/login").send({ username: "nobody", password: "nope-nope-nope" });
  assert.deepEqual(r.body, r2.body);                    // doesn't reveal whether the user exists
  assert.ok(await models.AuditEvent.findOne({ action: "login", allowed: false }));
});

test("passwords are stored only as bcrypt hashes", async () => {
  const u = await models.User.findOne({ username: "bank_b" }).lean();
  assert.match(u.passwordHash, /^\$2[aby]\$12\$/);
  assert.ok(!JSON.stringify(u).includes(pw.bank_b));
});

test("banks are proxied with the internal key and synced to MongoDB", async () => {
  const t = await tokenFor("guide");
  const r = await request.get("/api/banks").set("Authorization", `Bearer ${t}`);
  assert.equal(r.status, 200);
  assert.equal(r.body.length, 4);
  assert.equal(r.body[0].passRatePct, 80);
  assert.ok(seen.keys.has(KEY));
  assert.equal(await models.Bank.countDocuments(), 4);
});

test("only admin may start runs (viewer and bank operator get 403)", async () => {
  for (const user of ["guide", "bank_b"]) {
    const r = await request.post("/api/runs").set("Authorization", `Bearer ${await tokenFor(user)}`).send({ arms: [3] });
    assert.equal(r.status, 403);
  }
});

test("admin run: persisted, watched, rounds de-duplicated, model result and final status stored", async () => {
  const r = await request.post("/api/runs").set("Authorization", `Bearer ${await tokenFor("admin")}`)
    .send({ arms: [3], seeds: [42] });
  assert.equal(r.status, 202);
  assert.equal(r.body.jobId, "job-1");
  assert.equal(r.body.requestedBy, "admin");
  for (let i = 0; i < 50; i++) {                       // the watcher works in the background
    const run = await models.Run.findOne({ jobId: "job-1" }).lean();
    if (run.status === "succeeded") break;
    await new Promise((res) => setTimeout(res, 100));
  }
  const run = await models.Run.findOne({ jobId: "job-1" }).lean();
  assert.equal(run.status, "succeeded");
  assert.deepEqual(run.result, { arms: [3] });
  assert.equal(run.roundsReceived, 2);
  const rounds = await models.Round.find({ jobId: "job-1" }).sort({ round: 1 }).lean();
  assert.deepEqual(rounds.map((x) => x.round), [1, 2]);  // the duplicate round 2 was not stored twice
  assert.equal(rounds[0].banks[0].uploadBytes, 17478);
  assert.equal((await models.ModelResult.findOne({ jobId: "job-1" })).f1, 0.79);
  assert.ok(await models.AuditEvent.findOne({ action: "start:run", user: "admin" }));
});

test("browser WebSocket replays stored rounds for a finished run; rejects missing token", async () => {
  const t = await tokenFor("guide");
  const msgs = await new Promise((resolve, reject) => {
    const got = [];
    const ws = new WebSocket(`ws://127.0.0.1:${gwPort}/ws/runs/job-1?token=${t}`);
    ws.on("message", (d) => got.push(JSON.parse(d.toString())));
    ws.on("close", () => resolve(got));
    ws.on("error", reject);
  });
  assert.deepEqual(msgs.filter((m) => m.type === "round").map((m) => m.round), [1, 2]);
  assert.equal(msgs.at(-1).type, "end");
  const status = await new Promise((resolve) => {
    const ws = new WebSocket(`ws://127.0.0.1:${gwPort}/ws/runs/job-1`);
    ws.on("unexpected-response", (req, res) => resolve(res.statusCode));
    ws.on("error", () => {});
  });
  assert.equal(status, 401);
});

test("export: only the bank's own operator; scope comes from the token, not the request", async () => {
  const bankB = await tokenFor("bank_b");
  const own = await request.get("/api/banks/bank_b/synthetic/export").set("Authorization", `Bearer ${bankB}`);
  assert.equal(own.status, 200);
  assert.match(own.text, /Class/);
  const other = await request.get("/api/banks/bank_c/synthetic/export").set("Authorization", `Bearer ${bankB}`);
  assert.equal(other.status, 403);
  // A user can't smuggle a different scope header: the gateway sets it from the token.
  const spoof = await request.get("/api/banks/bank_b/synthetic/export")
    .set("Authorization", `Bearer ${bankB}`).set("X-Bank-Scope", "bank_a");
  assert.equal(spoof.status, 200);
  assert.deepEqual(seen.exportScopes, ["bank_b", "bank_b"]);   // never bank_c, never the spoofed bank_a
  for (const user of ["admin", "guide"]) {
    const r = await request.get("/api/banks/bank_b/synthetic/export").set("Authorization", `Bearer ${await tokenFor(user)}`);
    assert.equal(r.status, 403);
  }
  assert.equal(await models.AuditEvent.countDocuments({ action: "export", allowed: false }), 3);
});

test("audit log is admin-only", async () => {
  assert.equal((await request.get("/api/audit").set("Authorization", `Bearer ${await tokenFor("bank_a")}`)).status, 403);
  const r = await request.get("/api/audit").set("Authorization", `Bearer ${await tokenFor("admin")}`);
  assert.equal(r.status, 200);
  assert.ok(r.body.length > 5);
});
