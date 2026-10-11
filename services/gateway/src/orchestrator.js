// Calls to Tier 3 (FastAPI). Every request carries the internal key; browsers never see it.
import { config } from "./config.js";

export class OrchestratorError extends Error {
  constructor(status, body) {
    super(`orchestrator responded ${status}`);
    this.status = status;
    this.body = body;
  }
}

export async function orch(path, { method = "GET", body, headers = {}, raw = false } = {}) {
  let res;
  try {
    res = await fetch(`${config.orchestratorUrl}${path}`, {
      method,
      headers: { "X-Internal-Key": config.orchestratorKey,
                 ...(body !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new OrchestratorError(502, { error: "orchestrator (FastAPI) is not reachable; is it running?" });
  }
  if (raw) return res;                                    // e.g. CSV export: caller streams it
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new OrchestratorError(res.status, data);
  return data;
}

export function orchestratorWsUrl(jobId) {
  return `${config.orchestratorUrl.replace(/^http/, "ws")}/ws/jobs/${encodeURIComponent(jobId)}`;
}
