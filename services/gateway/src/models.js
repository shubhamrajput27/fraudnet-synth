// MongoDB collections (Mongoose schemas), decision D-048.
//
// users              who may log in, and their role (admin / bank_operator / viewer)
// runs               one document per job started through the gateway (run, generation, model prep)
// rounds             one document per federated round of a run (live metrics, per-bank sizes)
// modelresults       final global-test metrics of every model trained in a run
// banks              client metadata snapshot (mode, shard size, synthetic pool) synced from Tier 3
// validationreports  latest Step 6 validation report per bank, synced from Tier 3
// auditevents        logins, run starts, exports: who did what, and whether it was allowed
//
// Only aggregate metrics and metadata are stored: no transaction rows, real or synthetic.
import mongoose from "mongoose";

const { Schema } = mongoose;
const BANKS = ["bank_a", "bank_b", "bank_c", "bank_d"];

const UserSchema = new Schema({
  username: { type: String, required: true, unique: true, lowercase: true, trim: true },
  displayName: { type: String, required: true },
  role: { type: String, required: true, enum: ["admin", "bank_operator", "viewer"] },
  // A bank operator belongs to exactly one bank; other roles have none.
  bank: { type: String, enum: [...BANKS, null], default: null,
          validate: { validator(v) { return this.role === "bank_operator" ? BANKS.includes(v) : v === null; },
                      message: "bank operators need a bank; other roles must not have one" } },
  passwordHash: { type: String, required: true },   // bcrypt; the password itself is never stored
  lastLoginAt: Date,
}, { timestamps: true });

const RunSchema = new Schema({
  jobId: { type: String, required: true, unique: true },   // the orchestrator's job id
  type: { type: String, required: true, enum: ["run", "generate", "prepare_model"] },
  params: { type: Schema.Types.Mixed, default: {} },
  status: { type: String, required: true,
            enum: ["queued", "running", "succeeded", "failed", "cancelled", "interrupted"] },
  requestedBy: { type: String, required: true },
  startedAt: Date,
  finishedAt: Date,
  result: { type: Schema.Types.Mixed, default: null },     // aggregate metrics / checksums from Tier 3
  roundsReceived: { type: Number, default: 0 },
}, { timestamps: true });
RunSchema.index({ createdAt: -1 });
RunSchema.index({ status: 1 });

const RoundSchema = new Schema({
  jobId: { type: String, required: true, index: true },
  arm: { type: String, required: true },
  seed: { type: Number, required: true },
  round: { type: Number, required: true },
  globalTestPrAuc: Number,
  globalTestRocAuc: Number,
  banks: [{ _id: false, bank: String, trainLoss: Number, numExamples: Number, uploadBytes: Number }],
}, { timestamps: { createdAt: "receivedAt", updatedAt: false } });
// One document per (run, arm, seed, round): a reconnecting watcher can't create duplicates.
RoundSchema.index({ jobId: 1, arm: 1, seed: 1, round: 1 }, { unique: true });

const ModelResultSchema = new Schema({
  jobId: { type: String, required: true, index: true },
  seed: Number, arm: String, model: String,
  precision: Number, recall: Number, f1: Number, prAuc: Number,
}, { timestamps: true });
ModelResultSchema.index({ jobId: 1, seed: 1, arm: 1, model: 1 }, { unique: true });

const BankSchema = new Schema({
  bank: { type: String, required: true, unique: true, enum: BANKS },
  mode: String, rows: Number, fraud: Number, fraudPct: Number, trainFraud: Number,
  syntheticCandidates: Number, syntheticValidated: Number, passRatePct: Number, fidelityValidated: Number,
}, { timestamps: { createdAt: false, updatedAt: "syncedAt" } });

const ValidationReportSchema = new Schema({
  bank: { type: String, required: true, unique: true, enum: BANKS },
  report: { type: Schema.Types.Mixed, required: true },
}, { timestamps: { createdAt: false, updatedAt: "syncedAt" } });

const AuditEventSchema = new Schema({
  user: String, role: String, action: { type: String, required: true }, target: String,
  allowed: { type: Boolean, required: true }, detail: String,
}, { timestamps: { createdAt: "time", updatedAt: false } });
AuditEventSchema.index({ time: -1 });

export const User = mongoose.model("User", UserSchema);
export const Run = mongoose.model("Run", RunSchema);
export const Round = mongoose.model("Round", RoundSchema);
export const ModelResult = mongoose.model("ModelResult", ModelResultSchema);
export const Bank = mongoose.model("Bank", BankSchema);
export const ValidationReport = mongoose.model("ValidationReport", ValidationReportSchema);
export const AuditEvent = mongoose.model("AuditEvent", AuditEventSchema);

export async function audit(user, action, target, allowed, detail = "") {
  await AuditEvent.create({ user: user?.username ?? "anonymous", role: user?.role ?? "none",
                            action, target, allowed, detail });
}
