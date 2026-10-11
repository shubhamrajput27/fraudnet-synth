// Step 12 browser check: drives a real (headless) Microsoft Edge through the dashboard via the
// Chrome DevTools Protocol, logs in as different roles, starts a live run from the UI, and saves
// screenshots to results/dashboard_demo/. Passwords come from .env and are never printed.
//   node scripts/ui_check.js            (needs FastAPI + gateway running, dashboard built)
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import WebSocket from "ws";
import { PROJECT_ROOT } from "../src/config.js";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const BASE = "http://127.0.0.1:8200";
const OUT = path.join(PROJECT_ROOT, "results", "dashboard_demo");
const PORT = 9333;
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0).padStart(3)}s]`, ...a);

const profile = fs.mkdtempSync(path.join(os.tmpdir(), "fns-edge-"));
const edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--window-size=1400,1000", "--no-first-run", "--disable-extensions", "about:blank"], { stdio: "ignore" });

let ws, id = 0;
const pending = new Map();
const problems = [];
function send(method, params = {}) {
  const msgId = ++id;
  ws.send(JSON.stringify({ id: msgId, method, params }));
  return new Promise((resolve, reject) => pending.set(msgId, { resolve, reject }));
}
async function evaluate(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? "evaluate failed");
  return r.result.value;
}
async function waitFor(expr, timeoutMs = 20000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    if (await evaluate(`Boolean(${expr})`)) return true;
    await sleep(250);
  }
  throw new Error(`timed out waiting for: ${expr}`);
}
async function shot(name) {
  const { contentSize } = await send("Page.getLayoutMetrics");
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: 1400, height: Math.min(Math.ceil(contentSize.height), 4000), scale: 1 } });
  fs.writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(data, "base64"));
  log(`screenshot ${name}.png`);
}
const text = () => evaluate("document.body.innerText");
const clickText = (sel, t) => evaluate(`(() => { const el = [...document.querySelectorAll(${JSON.stringify(sel)})]
  .find(e => e.textContent.trim().startsWith(${JSON.stringify(t)})); if (!el) throw new Error('no ' + ${JSON.stringify(t)}); el.click(); return true; })()`);
// React-controlled inputs need the native value setter + an input event.
const setInput = (selector, value) => evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)});
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);

async function loginAs(user) {
  if (await evaluate("Boolean(document.querySelector('nav.topbar'))")) {
    await clickText("button", "Sign out");          // switch user the way a person would
  } else {
    await send("Page.navigate", { url: `${BASE}/#overview` });
  }
  await waitFor("document.querySelector('#u')");
  await setInput("#u", user);
  await setInput("#p", process.env[`GATEWAY_PW_${user.toUpperCase()}`]);
  await clickText("button", "Sign in");
  await waitFor("document.querySelector('nav.topbar')");
  log(`logged in as ${user}; tabs: ${await evaluate("[...document.querySelectorAll('.tab')].map(t => t.textContent).join(' | ')")}`);
}
async function openTab(label, readyExpr) {
  await clickText(".tab", label);
  await waitFor(readyExpr, 30000);
  await sleep(1200);                                  // let charts finish drawing
}

try {
  let version;
  for (let i = 0; i < 40 && !version; i++) {
    try { version = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); } catch { await sleep(250); }
  }
  const target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json();
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.on("open", r));
  ws.on("message", (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.id && pending.has(m.id)) { pending.get(m.id).resolve(m.result ?? {}); pending.delete(m.id); }
    if (m.method === "Runtime.exceptionThrown") problems.push(`exception: ${m.params.exceptionDetails.text}`);
    if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
      problems.push(`console.error: ${m.params.args.map((a) => a.value ?? a.description).join(" ")}`);
    if (m.method === "Log.entryAdded" && m.params.entry.level === "error") problems.push(`log: ${m.params.entry.text}`);
  });
  await send("Page.enable"); await send("Runtime.enable"); await send("Log.enable");
  log(`browser: ${version.Browser}`);

  // --- viewer (guide): read-only ---
  await loginAs("guide");
  await openTab("Overview", "document.querySelectorAll('article.card .stats').length === 4");
  await shot("01_overview_guide");
  await openTab("Six-arm results", "document.querySelectorAll('.recharts-bar-rectangle').length > 10");
  await shot("02_six_arm_results");
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
  await sleep(800);
  await shot("02b_six_arm_results_light");
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
  await openTab("Runs (live)", "document.body.innerText.includes('Run history')");
  log(`guide sees "Start a run"? ${(await text()).includes("Start a run")}`);

  // --- admin: start a run from the UI and watch it live ---
  await loginAs("admin");
  await openTab("Runs (live)", "document.body.innerText.includes('Start a run')");
  await setInput("#seeds", "44");
  await clickText("button", "Start run");
  await waitFor("document.body.innerText.includes('Started ')", 20000);
  const job = (await text()).match(/Started (\S+)/)[1];
  log(`run started from the UI: ${job}`);
  // "Finished" = THIS run's detail card shows Succeeded and its model table has a row.
  const runCard = `[...document.querySelectorAll('article.card')].find(c => c.querySelector('h2')?.textContent === 'Run ${job}')`;
  await waitFor("document.querySelectorAll('.recharts-line-dot').length >= 8", 240000);
  await sleep(800);
  await shot("03_live_run_in_progress");
  log(`mid-run rounds visible: ${await evaluate("document.querySelectorAll('.recharts-line-dot').length")}`);
  await waitFor(`${runCard}?.innerText.includes('Succeeded') && [...document.querySelectorAll('article.card')].some(c => c.innerText.startsWith('Finished models') && c.querySelectorAll('tbody tr').length >= 1)`, 300000);
  await sleep(1500);
  await sleep(500);
  log(`history row of this run: ${await evaluate(`[...document.querySelectorAll('tbody tr')].find(r => r.classList.contains('selected'))?.innerText.replace(/\s+/g, ' ')`)}`);
  await shot("04_live_run_finished");
  log(`finished run shows: ${(await text()).match(/Latest: [^\n]+/)?.[0]}`);

  await openTab("Test a transaction", "document.body.innerText.includes('Load a fraud example')");
  await clickText("button", "Load a fraud example");
  await waitFor("document.querySelector('#f-V1').value !== ''");
  await clickText("button", "Score transaction");
  await waitFor("document.querySelector('.verdict')");
  await sleep(300);
  log(`prediction: ${await evaluate("document.querySelector('.verdict').textContent")}`);
  await shot("05_predict_fraud_example");
  await openTab("Audit log", "document.querySelectorAll('tbody tr').length > 3");
  await shot("06_audit_log_admin");

  // --- bank operator: export only own bank ---
  await loginAs("bank_b");
  await openTab("Export", "document.body.innerText.includes('Download Bank B')");
  await shot("07_export_bank_b");
  log(`bank_b export page: ${(await text()).match(/You are the operator[^\n]+/)?.[0]}`);

  log(problems.length ? `BROWSER PROBLEMS:\n  ${problems.join("\n  ")}` : "browser console: no errors or exceptions");
} catch (e) {
  log(`FAILED: ${e.message}`);
  log(problems.length ? `browser problems: ${problems.join(" | ")}` : "");
  try { await shot("zz_failure"); } catch { /* ignore */ }
  process.exitCode = 1;
} finally {
  ws?.close();
  edge.kill();
  await sleep(500);
  fs.rmSync(profile, { recursive: true, force: true });
}
