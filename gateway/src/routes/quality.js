// Synthetic quality panel (CLAUDE.md Tier 1) + D8's `validation_reports` collection, which
// nothing before Phase 6 populated (Phase 3 only wrote data/validated/validation_report.json to
// disk). Reads straight from the repo's data/ files — gateway and the ML pipeline share the same
// filesystem on this single-machine deployment — and upserts into Mongo on each request so the
// collection stays in sync with whatever the pipeline last produced, per bank.
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Router } from "express";
import { getDb } from "../db.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const DATA_DIR = path.join(REPO_ROOT, "data");

const BANK_MODE = { A: "augment_ctgan", B: "augment_ctgan", C: "schema_llm", D: "schema_llm" };

function countCsvRows(filePath) {
  if (!existsSync(filePath)) return 0;
  const lines = readFileSync(filePath, "utf8").split("\n").filter((l) => l.trim().length > 0);
  return Math.max(0, lines.length - 1); // minus header
}

const router = Router();

router.get("/", async (req, res) => {
  let validationReport = [];
  const reportPath = path.join(DATA_DIR, "validated", "validation_report.json");
  if (existsSync(reportPath)) {
    validationReport = JSON.parse(readFileSync(reportPath, "utf8"));
  }

  const db = getDb();
  const panel = [];
  for (const bank of Object.keys(BANK_MODE)) {
    const validation = validationReport.find((r) => r.bank === bank) ?? null;
    const shardPath = path.join(DATA_DIR, "shards", `bank_${bank}.csv`);
    const candidatesPath = path.join(DATA_DIR, "synthetic", `bank_${bank}_candidates.csv`);
    const validatedPath = path.join(DATA_DIR, "validated", `bank_${bank}_validated.csv`);

    const entry = {
      bank,
      mode: BANK_MODE[bank],
      n_candidates: countCsvRows(candidatesPath),
      n_validated: countCsvRows(validatedPath),
      validation,
    };
    panel.push(entry);

    if (validation) {
      await db.collection("validation_reports").updateOne(
        { bank, mode: BANK_MODE[bank] },
        { $set: { ...entry, updated_at: new Date() } },
        { upsert: true }
      );
    }
  }

  res.json(panel);
});

// Synthetic dataset export (CLAUDE.md Tier 1). type=candidates -> Phase 2 raw generation output;
// type=validated -> Phase 3 output that actually passed the shared validation layer.
router.get("/:bank/export", (req, res) => {
  const { bank } = req.params;
  const type = req.query.type === "validated" ? "validated" : "candidates";
  if (!(bank in BANK_MODE)) return res.status(404).json({ error: `Unknown bank '${bank}'` });

  const filePath =
    type === "validated"
      ? path.join(DATA_DIR, "validated", `bank_${bank}_validated.csv`)
      : path.join(DATA_DIR, "synthetic", `bank_${bank}_candidates.csv`);
  if (!existsSync(filePath)) return res.status(404).json({ error: `No ${type} file for bank '${bank}'` });

  res.download(filePath, `bank_${bank}_${type}.csv`);
});

export default router;
