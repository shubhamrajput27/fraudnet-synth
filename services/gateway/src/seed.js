// Create (or update) the demo accounts:  npm run seed
// Passwords are read from .env (GATEWAY_PW_<USERNAME>) and stored only as bcrypt hashes.
import bcrypt from "bcryptjs";
import mongoose from "mongoose";
import { pathToFileURL } from "node:url";
import { config } from "./config.js";
import { User } from "./models.js";

export const DEMO_USERS = [
  { username: "admin", displayName: "Shubham (admin)", role: "admin", bank: null },
  { username: "bank_a", displayName: "Shubham (Bank A operator)", role: "bank_operator", bank: "bank_a" },
  { username: "bank_b", displayName: "Prachi (Bank B operator)", role: "bank_operator", bank: "bank_b" },
  { username: "bank_c", displayName: "Pradeepa (Bank C operator)", role: "bank_operator", bank: "bank_c" },
  { username: "bank_d", displayName: "Sathvik (Bank D operator)", role: "bank_operator", bank: "bank_d" },
  { username: "guide", displayName: "Dr. Chethan L S (viewer)", role: "viewer", bank: null },
];

export async function seedUsers(passwords) {
  for (const u of DEMO_USERS) {
    const pw = passwords[u.username];
    if (!pw || pw.length < 10) throw new Error(`password for ${u.username} missing or shorter than 10 characters`);
    const passwordHash = await bcrypt.hash(pw, 12);   // cost 12: deliberately slow to guess
    // Load-and-save (not updateOne) so the role/bank validator sees the whole document.
    const doc = (await User.findOne({ username: u.username })) ?? new User();
    doc.set({ ...u, passwordHash });
    await doc.save();
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {   // run directly, not imported
  await mongoose.connect(config.mongoUri);
  const pw = Object.fromEntries(DEMO_USERS.map((u) => [u.username, process.env[`GATEWAY_PW_${u.username.toUpperCase()}`]]));
  await seedUsers(pw);
  console.log(`Seeded ${DEMO_USERS.length} users into ${mongoose.connection.name}: ${DEMO_USERS.map((u) => `${u.username} (${u.role}${u.bank ? `, ${u.bank}` : ""})`).join(", ")}`);
  await mongoose.disconnect();
}
