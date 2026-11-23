// Pure formatting helpers (no React, no browser APIs), so they can be unit-tested in Node.

export const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}%`);
export const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-IN"));
// "12 Oct 2026, 2:26 pm": month spelled out, so 10/12 can't be read as either 10 Dec or 12 Oct.
export const when = (v) => (v ? new Date(v).toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric",
                                                                    hour: "numeric", minute: "2-digit" }) : "—");
export const bankName = (b) => `Bank ${String(b).slice(-1).toUpperCase()}`;

// Arm keys from the experiment runner -> the names used everywhere else on screen.
const ARM_NAMES = { 1: "Isolated, real", 2: "Isolated, augmented", 3: "Federated, real",
                    4: "Federated, augmented", 5: "Centralized, real", 6: "Centralized, augmented" };
// "arm3_federated_real" (or 3) -> "Arm 3 · Federated, real"; anything unknown is shown as is.
export function armName(key) {
  const n = Number(String(key).match(/^(?:arm)?(\d)/)?.[1]);
  return ARM_NAMES[n] ? `Arm ${n} · ${ARM_NAMES[n]}` : String(key);
}
export const ARMS = ARM_NAMES;

// "0.8093 ± 0.0048" -> { mean: 0.8093, std: 0.0048 }
export function parsePm(s) {
  const [m, sd] = String(s).split("±").map((x) => parseFloat(x));
  return { mean: m, std: Number.isFinite(sd) ? sd : 0 };
}

// One pasted row from a CSV (e.g. a line of test.csv) -> { Time, V1..V28, Amount }.
// Accepts 30 values, or 31 with a trailing Class column (ignored: the model must not see the answer).
export function parseRow(text, fields) {
  const parts = String(text).trim().split(/[\s,;]+/).filter(Boolean);
  if (parts.length === fields.length + 1) parts.pop();
  if (parts.length !== fields.length) {
    throw new Error(`expected ${fields.length} numbers (Time, V1–V28, Amount), got ${parts.length}`);
  }
  const nums = parts.map(Number);
  const bad = parts.find((p, i) => !Number.isFinite(nums[i]));
  if (bad !== undefined) throw new Error(`"${bad}" is not a number`);
  return Object.fromEntries(fields.map((f, i) => [f, String(nums[i])]));
}

// FastAPI's 422 body is a list like [{ loc: ["body", "seeds"], msg: "List should have at least 1 item" }]:
// turn it into "seeds: List should have at least 1 item" instead of showing raw JSON.
export function errorText(data) {
  const d = data?.detail;
  if (Array.isArray(d)) return d.map((e) => `${(e.loc ?? []).filter((x) => x !== "body").join(".")}: ${e.msg}`).join("; ");
  if (typeof d === "string") return d;
  return data?.error ?? null;
}
