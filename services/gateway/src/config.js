// Gateway configuration. Secrets come from the project's .env file (never committed).
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

const here = path.dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = path.resolve(here, "../../..");
dotenv.config({ path: path.join(PROJECT_ROOT, ".env"), quiet: true });

function required(name, minLength = 1) {
  const v = process.env[name] ?? "";
  if (v.length < minLength) {
    throw new Error(`Set ${name} in .env (at least ${minLength} characters).`);
  }
  return v;
}

export const config = {
  host: process.env.GATEWAY_HOST ?? "127.0.0.1",
  port: Number(process.env.GATEWAY_PORT ?? 8200),
  // Own database (D-050): 'fraudnet_synth' holds an earlier prototype's data, which we leave untouched.
  mongoUri: process.env.GATEWAY_MONGO_URI ?? "mongodb://localhost:27017/fraudnet_gateway",
  // Tier 3 (FastAPI). Only the gateway knows the internal key; browsers never see it.
  orchestratorUrl: process.env.ORCHESTRATOR_URL ?? "http://127.0.0.1:8100",
  orchestratorKey: required("ORCHESTRATOR_INTERNAL_KEY", 16),
  jwtSecret: required("GATEWAY_JWT_SECRET", 32),
  jwtTtl: process.env.GATEWAY_JWT_TTL ?? "8h",
  banks: ["bank_a", "bank_b", "bank_c", "bank_d"],
};
