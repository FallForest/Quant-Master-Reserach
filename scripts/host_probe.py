#!/usr/bin/env python
"""Read-only host probe for local xiadan-related interfaces.

The probe intentionally does not log in, submit orders, mutate files, or enable
any live-trading path. It only reports local process, port, config-path, and
optional localhost HTTP reachability metadata as structured JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


XIADAN_KEYWORDS = (
    "xiadan",
    "tdx",
    "tongdaxin",
    "silkriver",
    "addinflatjy",
    "tc.dll",
)

DEFAULT_ENDPOINTS = ("/TOUCH", "/ALIVE")
DEFAULT_CONFIG_ROOT = Path(r"C:\silkriver")
KEY_CONFIG_PATHS = (
    "xiadan.exe",
    "TdxW.exe",
    "Tc.dll",
    "MfcHlpr520.dll",
    "connect.cfg",
    "config.ini",
    "xiadan.ini",
    "TdxW.ini",
    "userdata",
    "T0002",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: Sequence[str], timeout: float = 5.0) -> dict:
    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout}s",
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def is_related_process(name: str, command_line: str = "") -> bool:
    haystack = f"{name} {command_line}".lower()
    return any(keyword in haystack for keyword in XIADAN_KEYWORDS)


def powershell_process_args() -> list[str]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,ExecutablePath,CommandLine | "
        "ConvertTo-Csv -NoTypeInformation"
    )
    return ["powershell", "-NoProfile", "-Command", script]


def parse_process_csv(text: str) -> list[dict]:
    rows = []
    for row in csv.DictReader(text.splitlines()):
        try:
            pid = int(row.get("ProcessId") or 0)
        except ValueError:
            continue
        name = row.get("Name") or ""
        command_line = row.get("CommandLine") or ""
        if not is_related_process(name, command_line):
            continue
        rows.append(
            {
                "pid": pid,
                "name": name,
                "executable_path": row.get("ExecutablePath") or "",
                "command_line": command_line,
            }
        )
    return sorted(rows, key=lambda item: (item["name"].lower(), item["pid"]))


def collect_processes() -> dict:
    result = run_command(powershell_process_args())
    if not result["ok"]:
        return {"ok": False, "error": result["stderr"], "items": []}
    return {"ok": True, "error": "", "items": parse_process_csv(result["stdout"])}


NETSTAT_RE = re.compile(
    r"^\s*(TCP|UDP)\s+(\S+)\s+(\S+)\s+(?:(LISTENING|ESTABLISHED|TIME_WAIT|CLOSE_WAIT)\s+)?(\d+)\s*$",
    re.IGNORECASE,
)


def parse_endpoint(endpoint: str) -> tuple[str, int | None]:
    if endpoint.startswith("["):
        host, _, rest = endpoint[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port_text = endpoint.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        port = None
    return host, port


def parse_netstat(text: str, related_pids: Iterable[int] = ()) -> list[dict]:
    related_pid_set = set(related_pids)
    ports = []
    for line in text.splitlines():
        match = NETSTAT_RE.match(line)
        if not match:
            continue
        proto, local_addr, foreign_addr, state, pid_text = match.groups()
        host, port = parse_endpoint(local_addr)
        if port is None:
            continue
        pid = int(pid_text)
        is_listening = proto.upper() == "UDP" or (state or "").upper() == "LISTENING"
        if not is_listening:
            continue
        ports.append(
            {
                "protocol": proto.upper(),
                "local_address": host,
                "port": port,
                "pid": pid,
                "state": (state or "LISTENING").upper(),
                "related_process": pid in related_pid_set,
            }
        )
    return sorted(ports, key=lambda item: (item["port"], item["protocol"], item["pid"]))


def collect_listening_ports(related_pids: Iterable[int] = ()) -> dict:
    result = run_command(["netstat", "-ano"])
    if not result["ok"]:
        return {"ok": False, "error": result["stderr"], "items": []}
    return {"ok": True, "error": "", "items": parse_netstat(result["stdout"], related_pids)}


def inspect_config_root(root: Path = DEFAULT_CONFIG_ROOT) -> dict:
    root_exists = root.exists()
    entries = []
    for rel_path in KEY_CONFIG_PATHS:
        path = root / rel_path
        entries.append(
            {
                "relative_path": rel_path,
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
            }
        )
    return {
        "root": str(root),
        "root_exists": root_exists,
        "key_paths": entries,
    }


def normalize_endpoint(endpoint: str) -> str:
    if not endpoint.startswith("/"):
        return f"/{endpoint}"
    return endpoint


def probe_http_port(port: int, endpoints: Sequence[str], timeout: float = 1.0) -> list[dict]:
    results = []
    for endpoint in endpoints:
        safe_endpoint = normalize_endpoint(endpoint)
        url = f"http://127.0.0.1:{port}{safe_endpoint}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(256)
                results.append(
                    {
                        "url": url,
                        "ok": True,
                        "status": response.status,
                        "reason": response.reason,
                        "bytes_read_capped": len(body),
                        "error": "",
                    }
                )
        except (urllib.error.URLError, OSError, socket.timeout) as exc:
            results.append(
                {
                    "url": url,
                    "ok": False,
                    "status": None,
                    "reason": "",
                    "bytes_read_capped": 0,
                    "error": str(exc),
                }
            )
    return results


def build_report(args: argparse.Namespace) -> dict:
    processes = collect_processes()
    related_pids = [item["pid"] for item in processes["items"]]
    listening_ports = collect_listening_ports(related_pids)
    config_root = inspect_config_root(Path(args.config_root))

    http_probes = []
    if args.probe_http:
        endpoints = tuple(normalize_endpoint(endpoint) for endpoint in args.endpoints)
        for port in args.ports:
            http_probes.extend(probe_http_port(port, endpoints, timeout=args.timeout))

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "safety": {
            "mode": "scan-only",
            "live_enabled": False,
            "login_attempted": False,
            "order_attempted": False,
            "http_probe_enabled": bool(args.probe_http),
            "notes": "Read-only probe. No login, no order placement, no --enable-live support.",
        },
        "processes": processes,
        "listening_ports": listening_ports,
        "silkriver_config": config_root,
        "http_probes": http_probes,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv and "--enable-live" in argv:
        raise SystemExit("--enable-live is forbidden for host_probe.py")
    parser = argparse.ArgumentParser(description="Read-only xiadan host/interface probe")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--probe-http", action="store_true", help="GET localhost endpoints only")
    parser.add_argument("--ports", nargs="*", type=int, default=[], help="localhost ports for optional GET probes")
    parser.add_argument("--endpoints", nargs="*", default=list(DEFAULT_ENDPOINTS), help="GET paths for optional probes")
    parser.add_argument("--timeout", type=float, default=1.0, help="per-request timeout in seconds")
    args, unknown = parser.parse_known_args(argv)
    if "--enable-live" in unknown:
        raise SystemExit("--enable-live is forbidden for host_probe.py")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
