#!/usr/bin/env python
"""Safe AddinFlatJy.dll probe CLI.

The default probe parses PE metadata only.  ``--probe load`` asks Windows to
load the DLL but still never calls exported trading functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_master.contrib.broker.addinflatjy_broker import DEFAULT_DLL_PATH, format_probe_result, inspect_addinflatjy_dll


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe AddinFlatJy.dll probe")
    parser.add_argument("--dll-path", default=str(DEFAULT_DLL_PATH), help="Path to AddinFlatJy.dll")
    parser.add_argument(
        "--probe",
        choices=["functions", "dependencies", "load"],
        default="functions",
        help="Probe mode. load only loads the DLL; no export is invoked.",
    )
    args = parser.parse_args()

    payload = inspect_addinflatjy_dll(args.dll_path, load=args.probe == "load")
    payload["probe"] = args.probe
    print(format_probe_result(payload))
    if not payload.get("exists"):
        return 2
    if args.probe == "load" and not payload.get("loaded"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
