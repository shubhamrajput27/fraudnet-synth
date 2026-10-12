"""Run every test suite in the project with one command (Step 14):

    python -m tests.run_all

1. Python tests (pytest): data split/partition, CTGAN/LLM engines, validation gate, model, metrics,
   federated threshold, privacy invariant, demo checks, six-arm statistics, FastAPI API.
2. Gateway tests (Node test runner): login, roles, proxy, MongoDB persistence, WebSocket relay, export scope.
3. Dashboard build (Vite): the React app must compile, or the demo can't be served.
Prints a summary and exits non-zero if anything failed.
"""
import re
import subprocess
import sys
import time

from ml.data.load import PROJECT_ROOT

NPM = "npm.cmd" if sys.platform == "win32" else "npm"
SUITES = [
    ("Python (pytest)", [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:warnings"], PROJECT_ROOT,
     r"(\d+) passed"),
    ("Gateway (node --test)", [NPM, "test"], PROJECT_ROOT / "services" / "gateway", r"# pass (\d+)"),
    ("Dashboard build (vite)", [NPM, "run", "build"], PROJECT_ROOT / "frontend", r"(built) in"),
]


def main():
    rows, ok_all = [], True
    for name, cmd, cwd, pattern in SUITES:
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = p.stdout + p.stderr
        ok = p.returncode == 0
        ok_all &= ok
        m = re.findall(pattern, out)
        detail = (f"{m[-1]} passed" if m and m[-1].isdigit() else "built") if ok else "FAILED"
        rows.append((name, detail, time.perf_counter() - t0))
        if not ok:
            print(f"\n--- {name} output (last 40 lines) ---\n" + "\n".join(out.splitlines()[-40:]))
    print("\nSuite                      Result        Time")
    for name, detail, secs in rows:
        print(f"{name:<26} {detail:<13} {secs:5.0f}s")
    print("ALL PASSED" if ok_all else "SOME SUITES FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
