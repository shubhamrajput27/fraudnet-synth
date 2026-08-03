// Minimal single-login JWT auth (CLAUDE.md D9): one admin credential pair from env, not a user
// database. Proportionate for a single-machine demo — the credential's only storage is the
// gitignored .env file, so bcrypt-hashing it here wouldn't add real protection (there's no
// separate at-rest store to protect against), but comparisons are still constant-time to avoid
// trivial timing attacks.
import { timingSafeEqual } from "node:crypto";
import jwt from "jsonwebtoken";

const TOKEN_TTL = "12h";

function safeEqual(a, b) {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

export function login(username, password) {
  const expectedUsername = process.env.ADMIN_USERNAME;
  const expectedPassword = process.env.ADMIN_PASSWORD;
  if (!expectedUsername || !expectedPassword) {
    throw new Error("ADMIN_USERNAME/ADMIN_PASSWORD not set in .env");
  }

  if (!safeEqual(username ?? "", expectedUsername) || !safeEqual(password ?? "", expectedPassword)) {
    return null;
  }

  return jwt.sign({ sub: username }, process.env.JWT_SECRET, { expiresIn: TOKEN_TTL });
}

export function requireAuth(req, res, next) {
  const header = req.headers.authorization ?? "";
  const [scheme, token] = header.split(" ");
  if (scheme !== "Bearer" || !token) {
    return res.status(401).json({ error: "Missing or malformed Authorization header" });
  }

  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}
