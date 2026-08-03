// D9's single-login JWT auth. Sets its own env vars rather than relying on a real .env, so this
// test is self-contained and doesn't depend on the developer's local credentials.
import assert from "node:assert/strict";
import { test, before } from "node:test";

import jwt from "jsonwebtoken";
import { login, requireAuth, verifyToken } from "../src/auth.js";

before(() => {
  process.env.ADMIN_USERNAME = "testadmin";
  process.env.ADMIN_PASSWORD = "testpassword123";
  process.env.JWT_SECRET = "test-secret-do-not-use-in-prod";
});

test("login succeeds with correct credentials and returns a valid JWT", () => {
  const token = login("testadmin", "testpassword123");
  assert.ok(token);
  const decoded = verifyToken(token);
  assert.equal(decoded.sub, "testadmin");
});

test("login fails with wrong password", () => {
  assert.equal(login("testadmin", "wrongpassword"), null);
});

test("login fails with wrong username", () => {
  assert.equal(login("someoneelse", "testpassword123"), null);
});

test("login fails when username/password are missing", () => {
  assert.equal(login(undefined, undefined), null);
});

test("verifyToken rejects a token signed with a different secret", () => {
  const forged = jwt.sign({ sub: "testadmin" }, "wrong-secret");
  assert.throws(() => verifyToken(forged));
});

test("verifyToken rejects garbage input", () => {
  assert.throws(() => verifyToken("not-a-real-jwt"));
});

test("requireAuth rejects a request with no Authorization header", () => {
  const req = { headers: {} };
  let statusCode, body;
  const res = { status(code) { statusCode = code; return this; }, json(b) { body = b; } };
  requireAuth(req, res, () => assert.fail("next() should not be called"));
  assert.equal(statusCode, 401);
  assert.match(body.error, /Missing or malformed/);
});

test("requireAuth rejects a malformed (non-Bearer) Authorization header", () => {
  const req = { headers: { authorization: "Basic abc123" } };
  let statusCode;
  const res = { status(code) { statusCode = code; return this; }, json() {} };
  requireAuth(req, res, () => assert.fail("next() should not be called"));
  assert.equal(statusCode, 401);
});

test("requireAuth calls next() and sets req.user for a valid token", () => {
  const token = login("testadmin", "testpassword123");
  const req = { headers: { authorization: `Bearer ${token}` } };
  const res = {};
  let nextCalled = false;
  requireAuth(req, res, () => { nextCalled = true; });
  assert.ok(nextCalled);
  assert.equal(req.user.sub, "testadmin");
});
