"""Step 8B, SERVER laptop: start a federated run once every bank is connected.

    python -m ml.federated.demo.run_demo                      # 4 banks, demo defaults
    python -m ml.federated.demo.run_demo --expected-banks 2   # two-laptop test

`flwr run` packages the app (code + configs only, ~150 KB, no data) and hands it to the SuperLink,
which ships the code to every bank and starts the ServerApp. Output is streamed here and saved
to results/demo_8b/run_<time>.log in the server's data folder.
"""
import argparse
import subprocess
from datetime import datetime

from ml.data.load import DATA_ROOT
from ml.federated.demo.launch import DEMO, env_for, flwr_run_cmd


def run(arm: str, rounds: int, expected_banks: int, flwr_home, log_path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(flwr_run_cmd(arm, rounds, expected_banks), env=env_for(DATA_ROOT, flwr_home),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=DEMO["run"]["arm"], choices=["arm3_federated_real", "arm4_federated_aug"])
    p.add_argument("--rounds", type=int, default=DEMO["run"]["num_rounds"])
    p.add_argument("--expected-banks", type=int, default=4)
    args = p.parse_args()
    log = DATA_ROOT / "results" / "demo_8b" / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    raise SystemExit(run(args.arm, args.rounds, args.expected_banks, DATA_ROOT / ".flwr_server", log))


if __name__ == "__main__":
    main()
