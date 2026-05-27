#!/usr/bin/env python
"""Check whether the bundled Tc.dll helper can start and answer commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_master.contrib.broker import TcDllBroker


def _probe_name(value: str) -> str:
    return value.replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tc.dll connectivity check")
    parser.add_argument("--caller-path", default=None)
    parser.add_argument("--xiadan-work-dir", default=None, help="Working directory containing the xiadan/Tc.dll runtime")
    parser.add_argument("--dll-dir", default=None, help="Directory used to resolve Tc.dll and its side-by-side DLLs")
    parser.add_argument(
        "--probe",
        choices=["ready", "ping", "functions", "createall", "version", "status", "status-raw", "gridjy-dryrun"],
        default="createall",
        help="Probe to run after the helper prints READY",
    )
    parser.add_argument("--gridjy-entry", default="DRYRUN", help="Entry name for --probe gridjy-dryrun")
    parser.add_argument("--gridjy-params", default="", help="Parameter string for --probe gridjy-dryrun")
    args = parser.parse_args()

    try:
        with TcDllBroker(
            caller_path=args.caller_path,
            xiadan_work_dir=args.xiadan_work_dir,
            dll_dir=args.dll_dir,
        ) as broker:
            payload = {
                "connected": broker.is_connected(),
                "probe": _probe_name(args.probe),
                "diagnostics": broker.diagnostics(),
            }
            if args.probe == "ready":
                payload["result"] = {"ok": True, "status": "READY", "raw": "READY", "probe": "ready"}
            else:
                payload["result"] = broker.probe(
                    args.probe,
                    gridjy_entry=args.gridjy_entry,
                    gridjy_params=args.gridjy_params,
                )
            payload["diagnostics"] = broker.diagnostics()
    except Exception as exc:
        diagnostics = None
        try:
            diagnostics = broker.diagnostics()  # type: ignore[name-defined]
        except Exception:
            pass
        payload = {
            "connected": False,
            "probe": _probe_name(args.probe),
            "error": f"{type(exc).__name__}: {exc}",
            "diagnostics": diagnostics,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
