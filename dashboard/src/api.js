// Thin fetch wrapper for gateway/ (CLAUDE.md Tier 1 <-> Tier 2). Token lives in memory + sessionStorage
// only — no cookies, matching the gateway's stateless Bearer-token auth (D9).
const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL ?? "http://localhost:4000";
const GATEWAY_WS_URL = GATEWAY_URL.replace(/^http/, "ws");

export { GATEWAY_URL, GATEWAY_WS_URL };

function authHeaders() {
  const token = sessionStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(path, options = {}) {
  const res = await fetch(`${GATEWAY_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...options.headers },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? body.detail ?? `Request failed (${res.status})`);
  return body;
}

export async function login(username, password) {
  const { token } = await request("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
  sessionStorage.setItem("token", token);
  return token;
}

export function logout() {
  sessionStorage.removeItem("token");
}

export function isLoggedIn() {
  return Boolean(sessionStorage.getItem("token"));
}

export function triggerRun(params) {
  return request("/api/runs", { method: "POST", body: JSON.stringify(params) });
}

export function getRun(runId) {
  return request(`/api/runs/${runId}`);
}

export function listRuns() {
  return request("/api/runs");
}

export function getQualityPanel() {
  return request("/api/quality");
}

// A plain <a href> can't carry an Authorization header, and putting the JWT in the URL would
// leak it into browser history/server logs — so download via fetch+blob instead.
export async function downloadExport(bank, type) {
  const res = await fetch(`${GATEWAY_URL}/api/quality/${bank}/export?type=${type}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Export failed (${res.status})`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bank_${bank}_${type}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function getSampleTransaction() {
  return request("/api/predict/sample");
}

export function getPredictManifest(runId) {
  return request(`/api/predict/manifest/${runId}`);
}

export function predict(params) {
  return request("/api/predict", { method: "POST", body: JSON.stringify(params) });
}
