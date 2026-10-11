// Express app: routes, roles, proxying to Tier 3, persistence to MongoDB.
//
// Role table (D-047):
//   everyone logged in : read banks / validation / results / runs / rounds, predict
//   admin              : start runs, generation, model preparation; cancel jobs; read the audit log
//   bank_operator      : additionally export ONLY their own bank's validated synthetic data
//   viewer (guide)     : read-only
import express from "express";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import mongoose from "mongoose";
import { Readable } from "node:stream";
import { WebSocketServer } from "ws";
import fs from "node:fs";
import path from "node:path";
import { PROJECT_ROOT, config } from "./config.js";
import { login, requireAuth, requireRole, verifyToken } from "./auth.js";
import { OrchestratorError, orch } from "./orchestrator.js";
import { AuditEvent, Bank, ModelResult, Round, Run, ValidationReport, audit } from "./models.js";
import { isFinal, subscribe, watchJob } from "./watcher.js";

const camelBank = (b) => ({ bank: b.bank, mode: b.mode, rows: b.rows, fraud: b.fraud, fraudPct: b.fraud_pct,
  trainFraud: b.train_fraud, syntheticCandidates: b.synthetic_candidates, syntheticValidated: b.synthetic_validated,
  passRatePct: b.pass_rate_pct, fidelityValidated: b.fidelity_validated });

export function createApp() {
  const app = express();
  app.disable("x-powered-by");
  app.use(helmet());
  app.use(express.json({ limit: "100kb" }));

  // ---------------------------------------------------------------- open ---
  app.get("/api/health", async (req, res) => {
    let orchestrator = "unreachable";
    try { orchestrator = (await orch("/health")).status; } catch { /* reported below */ }
    res.json({ gateway: "ok", mongo: mongoose.connection.readyState === 1 ? "ok" : "down", orchestrator });
  });
  // Slow down password guessing: 20 login attempts per 15 minutes per IP.
  app.post("/api/auth/login", rateLimit({ windowMs: 15 * 60 * 1000, limit: 20, standardHeaders: true,
                                          legacyHeaders: false }), login);

  // ------------------------------------------------- everything else: JWT ---
  const api = express.Router();
  api.use(requireAuth);
  const admin = requireRole("admin");

  api.get("/auth/me", (req, res) => res.json(req.user));

  api.get("/banks", async (req, res) => {
    const banks = (await orch("/banks")).map(camelBank);
    // Keep the "client metadata" collection in sync with Tier 3.
    await Promise.all(banks.map((b) => Bank.updateOne({ bank: b.bank }, { $set: b }, { upsert: true })));
    res.json(banks);
  });
  api.get("/validation", async (req, res) => res.json(await orch("/validation")));
  api.get("/validation/:bank", async (req, res) => {
    const report = await orch(`/validation/${encodeURIComponent(req.params.bank)}`);
    await ValidationReport.updateOne({ bank: req.params.bank }, { $set: { report } }, { upsert: true });
    res.json(report);
  });
  api.get("/results/six-arms", async (req, res) => res.json(await orch("/results/six-arms")));
  api.get("/results/six-arms/convergence", async (req, res) => res.json(await orch("/results/six-arms/convergence")));

  // ---- jobs: started through Tier 3, recorded and watched here ----
  async function startJob(req, res, path, body) {
    const job = await orch(path, { method: "POST", body });
    const run = await Run.create({ jobId: job.id, type: job.type, params: job.params, status: job.status,
                                   requestedBy: req.user.username });
    await audit(req.user, `start:${job.type}`, job.id, true, JSON.stringify(job.params));
    watchJob(job.id);
    res.status(202).json(run);
  }
  api.post("/runs", admin, (req, res) => startJob(req, res, "/runs", req.body));
  api.post("/generation", admin, (req, res) => startJob(req, res, "/generation", req.body));
  api.post("/admin/prepare-model", admin, (req, res) => startJob(req, res, "/admin/prepare-model"));

  api.get("/runs", async (req, res) => {
    const limit = Math.min(Number(req.query.limit ?? 50), 200);
    res.json(await Run.find().sort({ createdAt: -1 }).limit(limit).lean());
  });
  api.get("/runs/:jobId", async (req, res) => {
    const run = await Run.findOne({ jobId: req.params.jobId }).lean();
    if (!run) return res.status(404).json({ error: "unknown run" });
    res.json(run);
  });
  api.get("/runs/:jobId/rounds", async (req, res) => {
    res.json(await Round.find({ jobId: req.params.jobId }).sort({ arm: 1, seed: 1, round: 1 }).lean());
  });
  api.get("/runs/:jobId/models", async (req, res) => {
    res.json(await ModelResult.find({ jobId: req.params.jobId }).sort({ seed: 1, arm: 1, model: 1 }).lean());
  });
  api.delete("/runs/:jobId", admin, async (req, res) => {
    const job = await orch(`/jobs/${encodeURIComponent(req.params.jobId)}`, { method: "DELETE" });
    await Run.updateOne({ jobId: job.id }, { $set: { status: job.status } });
    await audit(req.user, "cancel", job.id, true);
    res.json(job);
  });

  // ---- prediction ----
  api.post("/predict", async (req, res) => res.json(await orch("/predict", { method: "POST", body: req.body })));
  api.get("/predict/examples", async (req, res) => res.json(await orch("/predict/examples")));

  // ---- export (D-045): own bank's validated synthetic data, bank operators only ----
  api.get("/banks/:bank/synthetic/export", async (req, res) => {
    const { bank } = req.params;
    const allowed = req.user.role === "bank_operator" && req.user.bank === bank;
    if (!allowed) {
      await audit(req.user, "export", bank, false, "not this bank's operator");
      return res.status(403).json({ error: `only ${bank}'s own operator may export ${bank}'s synthetic data` });
    }
    const upstream = await orch(`/banks/${encodeURIComponent(bank)}/synthetic/export`,
                                { raw: true, headers: { "X-Bank-Scope": req.user.bank, "X-User": req.user.username } });
    await audit(req.user, "export", bank, upstream.ok);
    if (!upstream.ok) return res.status(upstream.status).json(await upstream.json());
    res.set("Content-Type", "text/csv");
    res.set("Content-Disposition", upstream.headers.get("content-disposition") ?? `attachment; filename="${bank}.csv"`);
    Readable.fromWeb(upstream.body).pipe(res);
  });

  api.get("/audit", admin, async (req, res) => {
    res.json(await AuditEvent.find().sort({ time: -1 }).limit(Math.min(Number(req.query.limit ?? 100), 500)).lean());
  });

  app.use("/api", api);
  app.use("/api", (req, res) => res.status(404).json({ error: "unknown API route" }));

  // Tier 1: the built React dashboard (frontend/dist), served from the same address (D-052).
  const dist = path.join(PROJECT_ROOT, "frontend", "dist");
  if (fs.existsSync(dist)) {
    app.use(express.static(dist, { index: "index.html", maxAge: "1h" }));
    app.get(/^(?!\/api\/|\/ws\/).*/, (req, res) => res.sendFile(path.join(dist, "index.html")));
  }

  // Errors: orchestrator errors keep their status; Mongoose validation -> 400; anything else -> 500.
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err instanceof OrchestratorError) return res.status(err.status).json(err.body ?? { error: err.message });
    if (err?.name === "ValidationError") return res.status(400).json({ error: err.message });
    console.error(err);
    return res.status(500).json({ error: "internal error" });
  });
  return app;
}

// Browser WebSocket: /ws/runs/<jobId>?token=<JWT>. Browsers can't add headers to WebSockets,
// so the login token comes as a query parameter (local deployment only, D-049).
export function attachLiveSocket(server) {
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", async (req, socket, head) => {
    const url = new URL(req.url, "http://localhost");
    const m = url.pathname.match(/^\/ws\/runs\/([^/]+)$/);
    let user = null;
    try { user = verifyToken(url.searchParams.get("token") ?? ""); } catch { user = null; }
    if (!m || !user) {
      socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n");
      return socket.destroy();
    }
    const jobId = decodeURIComponent(m[1]);
    const run = await Run.findOne({ jobId }).lean();
    if (!run) {
      socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
      return socket.destroy();
    }
    wss.handleUpgrade(req, socket, head, async (ws) => {
      // Catch-up first: everything already stored for this run, then live events.
      ws.send(JSON.stringify({ type: "status", status: run.status }));
      const rounds = await Round.find({ jobId }).sort({ arm: 1, seed: 1, round: 1 }).lean();
      rounds.forEach(({ _id, __v, jobId: _j, receivedAt, ...r }) => ws.send(JSON.stringify({ type: "round", ...r })));
      if (isFinal(run.status)) {
        ws.send(JSON.stringify({ type: "end", status: run.status, result: run.result }));
        return ws.close();
      }
      subscribe(jobId, ws);
    });
  });
  return wss;
}

export { config };
