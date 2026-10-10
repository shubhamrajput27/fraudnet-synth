"""Step 8B, BANK laptop: check this laptop's data, then connect this bank to the server.

    python -m ml.federated.demo.start_client --bank bank_b --server 192.168.43.10

The client REFUSES to start unless check_my_data passes (own bank only, checksums OK, no other
bank's files, no raw/global split). Each round it prints, and logs to results/demo_8b/:
    Round N: received global model -> trained on R rows (F fraud) -> sent weights back (X KB)
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from ml.data.load import DATA_ROOT, PROJECT_ROOT
from ml.federated.demo.bankfiles import BANKS
from ml.federated.demo.check_my_data import check
from ml.federated.demo.launch import DEMO, popen, supernode_cmd


def main():
    load_dotenv(PROJECT_ROOT / ".env")        # optional FRAUDNET_SERVER_IP
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, choices=BANKS)
    p.add_argument("--server", default=os.environ.get("FRAUDNET_SERVER_IP", DEMO["server"]["default_ip"]),
                   help="server laptop IP (or FRAUDNET_SERVER_IP in .env)")
    p.add_argument("--server-laptop", action="store_true", help="this bank runs on the server laptop")
    p.add_argument("--runtime-port", type=int, default=DEMO["client"]["first_runtime_port"])
    args = p.parse_args()

    ok, lines = check(args.bank, server_laptop=args.server_laptop)
    print(f"Data folder: {DATA_ROOT}")
    for line in lines:
        print("  " + line)
    if not ok:
        print(f"REFUSING TO START {args.bank}: fix the problems above (see docs/MULTI_DEVICE_SETUP.md).")
        sys.exit(1)
    print(f"Check passed. Connecting {args.bank} to {args.server}:{DEMO['server']['fleet_port']} ... (Ctrl+C to stop)")
    proc = popen(supernode_cmd(args.bank, args.server, args.runtime_port, args.server_laptop),
                 DATA_ROOT, DATA_ROOT / f".flwr_{args.bank}")
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
