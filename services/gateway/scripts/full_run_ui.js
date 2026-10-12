// Step 13 integration run: start the FULL six-arm x five-seed experiment FROM THE DASHBOARD in a real
// (headless) Microsoft Edge, watch it live to the end, and save evidence to results/integration/.
// The path exercised is Tier 1 (React) -> Tier 2 (gateway, MongoDB) -> Tier 3 (FastAPI) -> Tier 4 (banks).
//   node scripts/full_run_ui.js                    start a full run (needs FastAPI + gateway, built dashboard; ~20-30 min)
//   node scripts/full_run_ui.js --reopen <jobId>   NEVER starts a run: re-opens a finished run from the history
//                                                  (replayed from MongoDB) and retakes 04_full_run_finished.png
// The admin password comes from .env and is never printed.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import WebSocket from "ws";
import { PROJECT_ROOT } from "../src/config.js";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const BASE = "http://127.0.0.1:8200";
const OUT = path.join(PROJECT_ROOT, "results", "integration");
const PORT = 9334;
const args = process.argv.slice(2);
// Refuse anything unexpected rather than fall through to starting a 20-minute run by accident.
if (args.length && !(args.length === 2 && args[0] === "--reopen")) {
  console.error("usage: node scripts/full_run_ui.js [--reopen <jobId>]");
  process.exit(2);
}
const REOPEN = args.length ? args[1] : null;
const SEEDS = "42, 43, 44, 45, 46";
const EXPECTED_ROUNDS = 2 * 5 * 30;           // 2 federated arms x 5 seeds x 30 rounds
const EXPECTED_MODELS = 12 * 5;               // 12 models per seed (Step 7/8) x 5 seeds
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const lines = [];
const log = (...a) => {
  const line = `[${((Date.now() - t0) / 1000).toFixed(0).padStart(5)}s] ${a.join(" ")}`;
  console.log(line);
  lines.push(line);
};

const profile = fs.mkdtempSync(path.join(os.tmpdir(), "fns-edge-"));
const edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--window-size=1400,1000", "--no-first-run", "--disable-extensions", "about:blank"], { stdio: "ignore" });

let ws, id = 0;
const pending = new Map();
const problems = [];
function send(method, params = {}) {
  const msgId = ++id;
  ws.send(JSON.stringify({ id: msgId, method, params }));
  return new Promise((resolve) => pending.set(msgId, { resolve }));
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
    await sleep(500);
  }
  throw new Error(`timed out waiting for: ${expr}`);
}
async function shot(name) {
  // Grow the window to the full page height FIRST and let the charts re-measure, then capture.
  // (captureBeyondViewport resizes during capture, and charts sometimes redrew squashed mid-resize.)
  const { contentSize } = await send("Page.getLayoutMetrics");
  const height = Math.min(Math.ceil(contentSize.height), 4000);
  await send("Emulation.setDeviceMetricsOverride", { width: 1400, height, deviceScaleFactor: 1, mobile: false });
  await sleep(1500);
  const { data } = await send("Page.captureScreenshot", { format: "png",
    clip: { x: 0, y: 0, width: 1400, height, scale: 1 } });
  await send("Emulation.clearDeviceMetricsOverride");
  fs.writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(data, "base64"));
  log(`screenshot ${name}.png`);
}
const clickText = (sel, t) => evaluate(`(() => { const el = [...document.querySelectorAll(${JSON.stringify(sel)})]
  .find(e => e.textContent.trim().startsWith(${JSON.stringify(t)})); if (!el) throw new Error('no ' + ${JSON.stringify(t)}); el.click(); return true; })()`);
const setInput = (selector, value) => evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)});
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
// This run's own detail card (its <h2> is "Run <jobId>"); other runs in the history must not count.
const card = (job) => `[...document.querySelectorAll('article.card')].find(c => c.querySelector('h2')?.textContent === 'Run ${job}')`;
// What the live run view shows right now: round points on the chart, rows in "Finished models".
const progress = () => evaluate(`({ rounds: document.querySelectorAll('.recharts-line-dot').length,
  models: [...document.querySelectorAll('article.card')].find(c => c.innerText.startsWith('Finished models'))
          ?.querySelectorAll('tbody tr').length ?? 0 })`);

try {
  for (let i = 0; i < 40; i++) {
    try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(250); }
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
  });
  await send("Page.enable"); await send("Runtime.enable");

  // Log in as admin, open the Runs page.
  await send("Page.navigate", { url: `${BASE}/#runs` });
  await waitFor("document.querySelector('#u')");
  await setInput("#u", "admin");
  await setInput("#p", process.env.GATEWAY_PW_ADMIN);
  await clickText("button", "Sign in");
  await waitFor("document.querySelector('nav.topbar')");
  await clickText(".tab", "Runs (live)");
  await waitFor("document.body.innerText.includes('Start a run')");

  if (REOPEN) {
    // Click history rows until this job's card opens (rows don't show the job id).
    const n = await evaluate("document.querySelectorAll('tr.clickable').length");
    for (let i = 0; i < n && !(await evaluate(`Boolean(${card(REOPEN)})`)); i++) {
      await evaluate(`document.querySelectorAll('tr.clickable')[${i}].click()`);
      await sleep(800);
    }
    await waitFor(`${card(REOPEN)}?.innerText.includes('Succeeded')`, 60000);
    await waitFor(`document.querySelectorAll('.recharts-line-dot').length >= ${EXPECTED_ROUNDS}`, 60000);
    await sleep(2000);
    const p = await progress();
    log(`RETAKE: re-opened finished run ${REOPEN} from the history: ${p.rounds} round points, ${p.models} models (replayed from MongoDB)`);
    await shot("04_full_run_finished");
  } else {

  // Tick all six arms (only the unticked ones) and enter the five seeds.
  await evaluate(`[...document.querySelectorAll('label.check input[type=checkbox]')].forEach(cb => { if (!cb.checked) cb.click(); })`);
  const ticked = await evaluate("[...document.querySelectorAll('label.check input[type=checkbox]')].filter(c => c.checked).length");
  await setInput("#seeds", SEEDS);
  log(`arms ticked: ${ticked} of 6; seeds: ${SEEDS}`);
  if (ticked !== 6) throw new Error("could not tick all six arms");

  await clickText("button", "Start run");
  await waitFor("document.body.innerText.includes('Started ')");
  const job = (await evaluate("document.body.innerText")).match(/Started (\S+)/)[1];
  fs.writeFileSync(path.join(OUT, "dashboard_job_id.txt"), `${job}\n`);
  log(`full run started FROM THE DASHBOARD: ${job}`);
  await sleep(1500);
  await shot("01_full_run_started");

  // Watch live. Screenshot when the first federated arm is drawn, again around half-way, and at the end.
  await waitFor("document.querySelectorAll('.recharts-line-dot').length >= 30", 30 * 60 * 1000);
  log("first federated arm on screen (30 round points)");
  await sleep(1000);
  await shot("02_full_run_first_federated_rounds");
  let midShot = false, lastRounds = -1;
  const deadline = Date.now() + 60 * 60 * 1000;
  for (;;) {
    const p = await progress();
    if (p.rounds !== lastRounds && p.rounds % 60 === 0 && p.rounds > 0) log(`progress: ${p.rounds} round points, ${p.models} models finished`);
    lastRounds = p.rounds;
    if (!midShot && p.rounds >= EXPECTED_ROUNDS / 2) { await shot("03_full_run_midway"); midShot = true; }
    if (p.rounds >= EXPECTED_ROUNDS && p.models >= EXPECTED_MODELS) break;
    if (Date.now() > deadline) throw new Error(`run did not finish within 60 min (${p.rounds} rounds, ${p.models} models)`);
    await sleep(5000);
  }
  await waitFor(`${card(job)}?.innerText.includes('Succeeded')`, 120000);   // THIS run, not older history rows
  await sleep(2000);
  const p = await progress();
  log(`finished: ${p.rounds} round points, ${p.models} models`);
  await shot("04_full_run_finished");
  }
  log(problems.length ? `BROWSER PROBLEMS:\n  ${problems.join("\n  ")}` : "browser console: no errors or exceptions");
} catch (e) {
  log(`FAILED: ${e.message}`);
  try { await shot("zz_failure"); } catch { /* ignore */ }
  process.exitCode = 1;
} finally {
  fs.writeFileSync(path.join(OUT, REOPEN ? "retake_output.txt" : "full_run_ui_output.txt"), lines.join("\n") + "\n");
  ws?.close();
  edge.kill();
  await sleep(500);
  fs.rmSync(profile, { recursive: true, force: true });
}
