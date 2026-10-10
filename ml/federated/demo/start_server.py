"""Step 8B, SERVER laptop (Shubham): start the Flower SuperLink and leave this window open.

    python -m ml.federated.demo.start_server

Run it from the clean demo folder (it should hold only data/clients/bank_a and the global test
set). Banks connect to this laptop's IP on the fleet port (9092). Then, in another window:
    python -m ml.federated.demo.start_client --bank bank_a --server 127.0.0.1 --server-laptop
and, once every bank is connected:
    python -m ml.federated.demo.run_demo
"""
import socket

from ml.data.load import DATA_ROOT
from ml.federated.demo.launch import DEMO, popen, superlink_cmd, write_cli_config


def lan_ips() -> list[str]:
    try:
        return sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    except OSError:
        return []


def main():
    flwr_home = DATA_ROOT / ".flwr_server"
    write_cli_config(flwr_home, f"127.0.0.1:{DEMO['server']['control_port']}")
    print(f"Server data folder: {DATA_ROOT}")
    print(f"Teammates connect to one of these IPs on port {DEMO['server']['fleet_port']}: {', '.join(lan_ips())}")
    print("(On a phone hotspot it is usually the 192.168.x.x address. Ctrl+C stops the server.)")
    popen(superlink_cmd(), DATA_ROOT, flwr_home).wait()


if __name__ == "__main__":
    main()
