// Authentication (JWT) and role-based access control (RBAC), decisions D-046 / D-047.
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { config } from "./config.js";
import { User, audit } from "./models.js";

export function signToken(user) {
  // The token is a signed "visitor badge": who you are, your role, your bank (if any).
  return jwt.sign({ sub: user.username, role: user.role, bank: user.bank ?? null },
                  config.jwtSecret, { expiresIn: config.jwtTtl, algorithm: "HS256" });
}

export function verifyToken(token) {
  const p = jwt.verify(token, config.jwtSecret, { algorithms: ["HS256"] });
  return { username: p.sub, role: p.role, bank: p.bank };
}

export async function login(req, res) {
  const { username, password } = req.body ?? {};
  if (typeof username !== "string" || typeof password !== "string") {
    return res.status(400).json({ error: "username and password are required" });
  }
  const user = await User.findOne({ username: username.toLowerCase().trim() });
  // Same message whether the user or the password is wrong: don't reveal which usernames exist.
  const ok = user ? await bcrypt.compare(password, user.passwordHash) : false;
  await audit(user ? { username: user.username, role: user.role } : { username }, "login", "", ok);
  if (!ok) return res.status(401).json({ error: "wrong username or password" });
  user.lastLoginAt = new Date();
  await user.save();
  return res.json({ token: signToken(user),
                    user: { username: user.username, displayName: user.displayName, role: user.role, bank: user.bank } });
}

// Express middleware: require a valid "Authorization: Bearer <token>" header.
export function requireAuth(req, res, next) {
  const h = req.get("authorization") ?? "";
  const token = h.startsWith("Bearer ") ? h.slice(7) : "";
  try {
    req.user = verifyToken(token);
    return next();
  } catch {
    return res.status(401).json({ error: "please log in (missing, invalid or expired token)" });
  }
}

// Express middleware: only these roles may continue.
export function requireRole(...roles) {
  return async (req, res, next) => {
    if (roles.includes(req.user.role)) return next();
    await audit(req.user, `forbidden:${req.method} ${req.baseUrl}${req.path}`, "", false);
    return res.status(403).json({ error: `requires role: ${roles.join(" or ")}` });
  };
}
