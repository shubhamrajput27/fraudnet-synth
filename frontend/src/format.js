// Pure formatting helpers (no React, no browser APIs), so they can be unit-tested in Node.

export const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}%`);
export const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-IN"));
export const bankName = (b) => `Bank ${String(b).slice(-1).toUpperCase()}`;

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
