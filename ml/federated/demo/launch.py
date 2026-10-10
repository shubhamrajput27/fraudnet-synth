"""Process launchers for the Step 8B deployment (flwr 1.39: SuperLink, SuperNodes, `flwr run`).

Every process gets its OWN data folder (FRAUDNET_DATA_ROOT) and its own Flower home (FLWR_HOME),
which is how one laptop can stand in for five separate machines in the fallback.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from ml.data.load import PROJECT_ROOT, load_config

DEMO = load_config("demo")
SCRIPTS = Path(sys.executable).parent          # the venv's Scripts folder (flower-superlink.exe etc.)
CONNECTION = "fraudnet"                        # name of the SuperLink connection used by `flwr run`


def env_for(data_root: Path, flwr_home: Path) -> dict:
    env = dict(os.environ)
    env["FRAUDNET_DATA_ROOT"] = str(data_root)
    env["FLWR_HOME"] = str(flwr_home)
    # Child processes (flwr-serverapp / flwr-clientapp) must come from this venv.
    env["PATH"] = str(SCRIPTS) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"          # Flower's CLI prints emoji; Windows consoles need UTF-8
    env["PYTHONUNBUFFERED"] = "1"
    return env


def write_cli_config(flwr_home: Path, control_address: str) -> None:
    """Tell `flwr run` where the SuperLink's control API is (insecure = no TLS, local network demo)."""
    flwr_home.mkdir(parents=True, exist_ok=True)
    (flwr_home / "config.toml").write_text(
        f'[superlink]\ndefault = "{CONNECTION}"\n\n[superlink.{CONNECTION}]\n'
        f'address = "{control_address}"\ninsecure = true\n', encoding="utf-8")


def wait_for_port(host: str, port: int, timeout: float = 60) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def superlink_cmd() -> list[str]:
    s = DEMO["server"]
    # flwr 1.39: the Control API (used by `flwr run`) is HTTP, set with --host/--port; we bind it to
    # localhost so only the server laptop itself can start runs. Banks use the Fleet API on the LAN.
    return [str(SCRIPTS / "flower-superlink"), "--insecure",
            "--fleet-api-address", f"0.0.0.0:{s['fleet_port']}",
            "--host", "127.0.0.1", "--port", str(s["control_port"])]


def supernode_cmd(bank: str, server_ip: str, runtime_port: int, server_laptop: bool) -> list[str]:
    node_cfg = f"bank='{bank}'" + (" server-laptop=true" if server_laptop else "")
    return [str(SCRIPTS / "flower-supernode"), "--insecure",
            "--superlink", f"{server_ip}:{DEMO['server']['fleet_port']}",
            "--node-config", node_cfg, "--port", str(runtime_port)]


def flwr_run_cmd(arm: str, rounds: int, expected_banks: int) -> list[str]:
    run_cfg = f"arm='{arm}' num-rounds={rounds} expected-banks={expected_banks}"
    return [str(SCRIPTS / "flwr"), "run", str(PROJECT_ROOT), CONNECTION, "--stream", "--run-config", run_cfg]


def popen(cmd: list[str], data_root: Path, flwr_home: Path, log_file: Path | None = None, cwd: Path | None = None):
    """Start a process; with log_file, its output goes to that file (background process)."""
    out = open(log_file, "w", encoding="utf-8") if log_file else None
    return subprocess.Popen(cmd, env=env_for(data_root, flwr_home), cwd=str(cwd or data_root),
                            stdout=out, stderr=subprocess.STDOUT if out else None)
