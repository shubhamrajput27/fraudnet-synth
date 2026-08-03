// MongoDB connection (CLAUDE.md D8): native driver, not an ODM, matching the locked stack's own
// rationale ("MongoDB | Flexible schema for run/round/client/arm metadata") — every document
// already carries run_id/arm and shapes vary slightly across the six arms, so a rigid schema
// layer would fight the data rather than help it.
import { MongoClient } from "mongodb";

let client;
let db;

export async function connectDb() {
  const uri = process.env.MONGO_URI;
  if (!uri) throw new Error("MONGO_URI not set — copy .env.example to .env and fill it in.");

  client = new MongoClient(uri);
  await client.connect();
  db = client.db(); // uses the database named in the URI path (fraudnet_synth)
  await db.command({ ping: 1 });
  return db;
}

export function getDb() {
  if (!db) throw new Error("connectDb() must be called before getDb()");
  return db;
}

export async function closeDb() {
  if (client) await client.close();
}
