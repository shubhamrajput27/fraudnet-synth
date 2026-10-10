"""Step 8B: build Shubham's CLEAN demo folder for the multi-laptop demo (server + Bank A).

    python -m ml.federated.demo.make_demo_folder --dest C:\\Projects\\fraudnet-demo

The main working folder holds every bank (it made the split), so it can never pass the data check.
This creates a fresh clone of the code (`git clone` of this repo: code only, no data), then adds
exactly two things: Bank A's handover files (from exports/bank_a.zip) and the global test set.
Run the server and Bank A from that folder, with the main folder closed:

    cd C:\\Projects\\fraudnet-demo
    <main .venv>\\Scripts\\python -m ml.federated.demo.check_my_data --bank bank_a --server-laptop
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from ml.data.load import PROJECT_ROOT, load_config
from ml.federated.demo.check_my_data import check
from ml.federated.demo.launch import DEMO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", required=True, type=Path)
    args = p.parse_args()
    dest = args.dest.resolve()
    if dest.exists():
        sys.exit(f"{dest} already exists; choose a new folder (nothing was changed)")
    subprocess.run(["git", "clone", "--quiet", str(PROJECT_ROOT), str(dest)], check=True)

    zpath = PROJECT_ROOT / DEMO["export"]["out_dir"] / "bank_a.zip"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest / load_config("partition")["clients_dir"])
    processed = load_config("partition")["processed_dir"]
    (dest / processed).mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / processed / "global_test.csv", dest / processed / "global_test.csv")

    ok, lines = check("bank_a", root=dest, server_laptop=True)
    print(f"Clean demo folder: {dest}")
    for line in lines:
        print("  " + line)
    print("RESULT: OK" if ok else "RESULT: FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
