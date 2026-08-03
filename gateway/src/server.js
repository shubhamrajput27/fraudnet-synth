import { fileURLToPath } from "node:url";
import path from "node:path";

import cors from "cors";
import dotenv from "dotenv";
import express from "express";

// The project's single .env lives at the repo root (see .env.example), not gateway/ — dotenv's
// default only checks process.cwd(), so point it there explicitly regardless of where this
// process is launched from.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "..", "..", ".env"), quiet: true });

import { login, requireAuth } from "./auth.js";
import { connectDb } from "./db.js";
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

// Central error handler — Express 5 auto-forwards rejected async route handlers here.
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

const PORT = process.env.GATEWAY_PORT ?? 4000;

connectDb()
  .then(() => {
    app.listen(PORT, () => console.log(`Gateway listening on port ${PORT}`));
  })
  .catch((err) => {
    console.error("Failed to connect to MongoDB:", err.message);
    process.exit(1);
  });
