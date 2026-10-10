"""Step 8B ONE-LAPTOP FALLBACK: server + four banks as five processes, one command.

    python -m ml.federated.demo.export_bank --all        # once, from the main working folder
    python -m ml.federated.demo.fallback                 # builds the sandbox and runs the demo

Uses the SAME code path as the multi-laptop demo (SuperLink + SuperNodes + `flwr run`). To make
one laptop behave like five, it builds demo_sandbox/ with one isolated folder per process:
    server/  -> only data/processed/global_test.csv
    bank_a/  -> only data/clients/bank_a/ (unzipped from exports/bank_a.zip)   ... and so on.
Each process sees only its own folder (FRAUDNET_DATA_ROOT), so every bank's start-up check
genuinely passes. All logs are copied to results/demo_8b/fallback_<time>/ at the end.
"""
import argparse
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from ml.data.load import PROJECT_ROOT, load_config
from ml.federated.demo.bankfiles import BANKS
from ml.federated.demo.check_my_data import check
from ml.federated.demo.launch import (DEMO, popen, superlink_cmd, supernode_cmd, wait_for_port,
                                      write_cli_config)
from ml.federated.demo.run_demo import run


def build_sandbox(sandbox: Path) -> dict[str, Path]:
    """Fresh isolated folders from the export zips (exactly what teammates receive)."""
    if sandbox.exists():
        shutil.rmtree(sandbox)
    folders = {"server": sandbox / "server", **{b: sandbox / b for b in BANKS}}
    processed = load_config("partition")["processed_dir"]
    (folders["server"] / processed).mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / processed / "global_test.csv", folders["server"] / processed / "global_test.csv")
    exports = PROJECT_ROOT / DEMO["export"]["out_dir"]
    for b in BANKS:
        z = exports / f"{b}.zip"
        if not z.exists():
            sys.exit(f"{z} not found: run `python -m ml.federated.demo.export_bank --all` first")
        dest = folders[b] / load_config("partition")["clients_dir"]
        dest.mkdir(parents=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)          # zip root is <bank>/ -> data/clients/<bank>/...
    return folders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=DEMO["run"]["arm"], choices=["arm3_federated_real", "arm4_federated_aug"])
    p.add_argument("--rounds", type=int, default=DEMO["run"]["num_rounds"])
    args = p.parse_args()

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    sandbox = PROJECT_ROOT / DEMO["fallback"]["sandbox_dir"]
    out = PROJECT_ROOT / DEMO["fallback"]["results_dir"] / f"fallback_{stamp}"
    out.mkdir(parents=True)
    folders = build_sandbox(sandbox)

    print("== Data check for every 'laptop' (each process sees only its own folder) ==")
    report = []
    for b in BANKS:
        ok, lines = check(b, root=folders[b])
        report += [f"[{b}] {line}" for line in lines] + [f"[{b}] RESULT: {'OK' if ok else 'FAILED'}"]
        if not ok:
            print("\n".join(report))
            sys.exit(f"{b} failed its data check; aborting")
    print("\n".join(report))
    (out / "check_my_data.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    procs = []
    try:
        srv_home = folders["server"] / ".flwr_server"
        write_cli_config(srv_home, f"127.0.0.1:{DEMO['server']['control_port']}")
        procs.append(popen(superlink_cmd(), folders["server"], srv_home, out / "superlink.log"))
        for port in (DEMO["server"]["fleet_port"], DEMO["server"]["control_port"]):
            if not wait_for_port("127.0.0.1", port, 60):
                sys.exit(f"SuperLink did not open port {port}; see {out / 'superlink.log'}")
        print(f"== SuperLink up (fleet :{DEMO['server']['fleet_port']}, control :{DEMO['server']['control_port']}) ==")

        for i, b in enumerate(BANKS):
            port = DEMO["client"]["first_runtime_port"] + i
            procs.append(popen(supernode_cmd(b, "127.0.0.1", port, server_laptop=False),
                               folders[b], folders[b] / f".flwr_{b}", out / f"supernode_{b}.log"))
        print(f"== Started 4 SuperNodes ({', '.join(BANKS)}); starting the run ==")
        time.sleep(3)

        t0 = time.perf_counter()
        code = run(args.arm, args.rounds, 4, srv_home, out / "flwr_run.log")
        print(f"== flwr run finished with exit code {code} in {time.perf_counter() - t0:.0f}s ==")
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=15)
            except Exception:
                pr.kill()
        # Collect the demo logs each "laptop" wrote into its own folder.
        for name, folder in folders.items():
            src = folder / "results" / "demo_8b"
            if src.exists():
                for f in src.iterdir():
                    shutil.copy2(f, out / f"{name}__{f.name}")
        print(f"== Logs saved to {out.relative_to(PROJECT_ROOT)} ==")


if __name__ == "__main__":
    main()
