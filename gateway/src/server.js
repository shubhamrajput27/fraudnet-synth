import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import path from "node:path";

import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { WebSocketServer } from "ws";

// The project's single .env lives at the repo root (see .env.example), not gateway/ — dotenv's
// default only checks process.cwd(), so point it there explicitly regardless of where this
// process is launched from.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "..", "..", ".env"), quiet: true });

import { login, requireAuth, verifyToken } from "./auth.js";
import { connectDb, getDb } from "./db.js";
import { subscribe } from "./liveRuns.js";
import predictRouter from "./routes/predict.js";
import qualityRouter from "./routes/quality.js";
import runsRouter from "./routes/runs.js";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (req, res) => res.json({ status: "ok" }));

app.post("/auth/login", (req, res) => {
  const { username, password } = req.body ?? {};
  const token = login(username, password);
  if (!token) return res.status(401).json({ error: "Invalid credentials" });
  res.json({ token });
});

app.use("/api/runs", requireAuth, runsRouter);
app.use("/api/predict", requireAuth, predictRouter);
app.use("/api/quality", requireAuth, qualityRouter);

// Central error handler — Express 5 auto-forwards rejected async route handlers here.
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

const PORT = process.env.GATEWAY_PORT ?? 4000;
const httpServer = createServer(app);

// Live FL round charts (CLAUDE.md Tier 1 <-> Tier 2: "REST / WebSocket"). Path: /ws/runs/:runId
// ?token=<jwt> — the WebSocket handshake can't carry an Authorization header from a browser, so
// the JWT travels as a query param instead, verified the same way as requireAuth.
const wss = new WebSocketServer({ server: httpServer, path: "/ws" });
wss.on("connection", (ws, req) => {
  const url = new URL(req.url, "http://localhost");
  const runId = url.searchParams.get("run_id");
  const token = url.searchParams.get("token");

  if (!runId || !token) return ws.close(4000, "run_id and token query params are required");
  try {
    verifyToken(token);
  } catch {
    return ws.close(4001, "Invalid or expired token");
  }

  subscribe(runId, ws, getDb());
});

connectDb()
  .then(() => {
    httpServer.listen(PORT, () => console.log(`Gateway listening on port ${PORT}`));
  })
  .catch((err) => {
    console.error("Failed to connect to MongoDB:", err.message);
    process.exit(1);
  });
