// Step 11 milestone check: trigger a FULL run through the gateway API alone, watch it live, and read
// its persisted history back from MongoDB.   node scripts/e2e_milestone.js [seed]
// Needs: FastAPI (python -m services.orchestrator) and the gateway (npm start) running, users seeded.
// Passwords are read from .env and never printed.
import { config } from "../src/config.js";   // also loads .env
import mongoose from "mongoose";
import WebSocket from "ws";

const BASE = "http://127.0.0.1:8200";
const seed = Number(process.argv[2] ?? 42);
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0).padStart(4)}s]`, ...a);

async function call(path, { token, method = "GET", body } = {}) {
  const res = await fetch(BASE + path, { method, headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(body ? { "Content-Type": "application/json" } : {}) }, body: body ? JSON.stringify(body) : undefined });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = text; }
  return { status: res.status, data };
}
const login = async (u) => (await call("/api/auth/login", { method: "POST",
  body: { username: u, password: process.env[`GATEWAY_PW_${u.toUpperCase()}`] } })).data.token;

const admin = await login("admin");
await call("/api/banks", { token: admin });                         // syncs client metadata into MongoDB
for (const b of ["bank_a", "bank_b", "bank_c", "bank_d"]) await call(`/api/validation/${b}`, { token: admin });
const guide = await login("guide");
const bankB = await login("bank_b");
log("logged in as admin, guide (viewer), bank_b (operator)");

log("viewer tries to start a run ->", (await call("/api/runs", { token: guide, method: "POST", body: { arms: "all", seeds: [seed] } })).status);
const start = await call("/api/runs", { token: admin, method: "POST", body: { arms: "all", seeds: [seed] } });
log(`admin starts ALL SIX arms, seed ${seed} -> ${start.status}, job ${start.data.jobId}, requestedBy ${start.data.requestedBy}`);
const jobId = start.data.jobId;

// Watch live as the guide through the GATEWAY's WebSocket (never FastAPI directly).
const live = { rounds: 0, models: 0 };
await new Promise((resolve, reject) => {
  const ws = new WebSocket(`ws://127.0.0.1:8200/ws/runs/${encodeURIComponent(jobId)}?token=${guide}`);
  ws.on("message", (d) => {
    const e = JSON.parse(d.toString());
    if (e.type === "round") {
      live.rounds += 1;
      if (e.round === 1 || e.round % 10 === 0) log(`live round ${String(e.round).padStart(2)} ${e.arm}: global PR-AUC ${e.globalTestPrAuc.toFixed(4)} (banks ${e.banks.map((b) => b.bank.at(-1).toUpperCase()).join("")})`);
    } else if (e.type === "model_done") {
      live.models += 1;
      log(`model done: ${e.arm.padEnd(20)} ${String(e.model).padEnd(11)} F1 ${e.f1.toFixed(4)}  PR-AUC ${e.prAuc.toFixed(4)}`);
    } else if (e.type === "status") {
      log(`status: ${e.status}`);
    } else if (e.type === "message") {
      log(`message: ${e.message}`);
    } else if (e.type === "end") {
      log(`end: ${e.status}`);
      ws.close();
    }
  });
  ws.on("close", resolve);
  ws.on("error", reject);
});

// Read everything back from MongoDB (the persisted history).
await mongoose.connect(config.mongoUri);
const db = mongoose.connection.db;
const run = await db.collection("runs").findOne({ jobId });
const nRounds = await db.collection("rounds").countDocuments({ jobId });
const perArm = await db.collection("rounds").aggregate([{ $match: { jobId } }, { $group: { _id: "$arm", n: { $sum: 1 } } }, { $sort: { _id: 1 } }]).toArray();
const models = await db.collection("modelresults").find({ jobId }).sort({ arm: 1, model: 1 }).toArray();
log(`MongoDB runs: status ${run.status}, requestedBy ${run.requestedBy}, roundsReceived ${run.roundsReceived}, `
    + `started ${run.startedAt?.toISOString()}, finished ${run.finishedAt?.toISOString()}`);
log(`MongoDB rounds: ${nRounds} documents (${perArm.map((a) => `${a._id}: ${a.n}`).join(", ")}); live stream delivered ${live.rounds}`);
log(`MongoDB modelresults: ${models.length} models; live stream delivered ${live.models}`);
for (const m of models) console.log(`        ${m.arm.padEnd(20)} ${String(m.model).padEnd(11)} P ${m.precision.toFixed(4)} R ${m.recall.toFixed(4)} F1 ${m.f1.toFixed(4)} PR-AUC ${m.prAuc.toFixed(4)}`);
const counts = Object.fromEntries(await Promise.all(["users", "runs", "rounds", "modelresults", "banks", "validationreports", "auditevents"]
  .map(async (c) => [c, await db.collection(c).countDocuments()])));
log("collection sizes:", JSON.stringify(counts));

// Role checks against the REAL orchestrator.
log("bank_b exports bank_b ->", (await call("/api/banks/bank_b/synthetic/export", { token: bankB })).status);
log("bank_b exports bank_c ->", (await call("/api/banks/bank_c/synthetic/export", { token: bankB })).status);
log("guide reads audit log ->", (await call("/api/audit", { token: guide })).status);
const ex = (await call("/api/predict/examples", { token: guide })).data;
const p = (await call("/api/predict", { token: guide, method: "POST", body: ex.fraud.transaction })).data;
log(`guide predicts the fraud example -> is_fraud ${p.is_fraud}, logit ${p.score_logit.toFixed(4)} (threshold ${p.threshold_logit})`);
await mongoose.disconnect();
