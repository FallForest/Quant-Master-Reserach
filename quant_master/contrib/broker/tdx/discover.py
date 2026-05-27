# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""TQL Entry name discovery tool.

Probes a TDX trading server to find valid Entry names by trying
candidate combinations from CANDIDATE_PREFIXES and CANDIDATE_OPERATIONS.

Usage::

    python -m quant_master.contrib.broker.tdx.discover --host 61.135.173.138 --port 7708
    python -m quant_master.contrib.broker.tdx.discover --host 61.135.173.138 --port 7708 --operation buy
"""

import argparse
import itertools
import sys
import time

from .client import TDXClient
from .consts import CANDIDATE_OPERATIONS, CANDIDATE_PREFIXES
from .exceptions import TDXConnectionError, TDXTradeError


def discover_entries(
    host: str,
    port: int = 7708,
    use_https: bool = False,
    operations: list[str] | None = None,
    delay: float = 0.3,
) -> dict[str, list[str]]:
    """Try candidate Entry names and return those that give non-empty responses.

    Parameters
    ----------
    host : str
        TDX server hostname or IP.
    port : int
        TDX server port (default 7708 for trading).
    use_https : bool
        Whether to use HTTPS.
    operations : list[str], optional
        Subset of operations to probe. If None, probe all.
    delay : float
        Seconds to wait between probes (avoid flooding).

    Returns
    -------
    dict[str, list[str]]
        Mapping of operation -> list of working Entry names.
    """
    client = TDXClient(host=host, port=port, use_https=use_https)

    print(f"Connecting to {host}:{port} ...")
    try:
        client.connect()
    except TDXConnectionError as e:
        print(f"Connection failed: {e}")
        return {}

    print("Connected. Starting discovery ...\n")

    ops_to_probe = operations or list(CANDIDATE_OPERATIONS.keys())
    results: dict[str, list[str]] = {}

    for op in ops_to_probe:
        suffixes = CANDIDATE_OPERATIONS.get(op, [])
        if not suffixes:
            print(f"  [{op}] No candidate suffixes, skipping")
            continue

        results[op] = []
        candidates = list(itertools.product(CANDIDATE_PREFIXES, suffixes))
        print(f"  [{op}] Probing {len(candidates)} candidates ...")

        for prefix, suffix in candidates:
            entry = f"{prefix}.{suffix}"
            try:
                text = client.call_tql(entry)
                if text.strip() and "error" not in text.lower():
                    print(f"    HIT  {entry!r}  →  {text[:120]!r}")
                    results[op].append(entry)
                else:
                    # Quiet skip for empty/error responses
                    pass
            except TDXTradeError:
                # Server rejected the call
                pass
            except Exception:
                pass

            time.sleep(delay)

        if not results[op]:
            print(f"    (no matches for {op})")
        else:
            print(f"    Found {len(results[op])} match(es) for {op}")

    client.quit()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Discover valid TQL Entry names on a TDX trading server."
    )
    parser.add_argument("--host", required=True, help="TDX server hostname or IP")
    parser.add_argument("--port", type=int, default=7708, help="TDX server port (default: 7708)")
    parser.add_argument("--https", action="store_true", help="Use HTTPS")
    parser.add_argument(
        "--operation",
        choices=list(CANDIDATE_OPERATIONS.keys()),
        help="Probe only one operation (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between probes (default: 0.3)",
    )
    args = parser.parse_args()

    ops = [args.operation] if args.operation else None
    results = discover_entries(
        host=args.host,
        port=args.port,
        use_https=args.https,
        operations=ops,
        delay=args.delay,
    )

    print("\n=== Discovery Results ===")
    if not results:
        print("No entries found. Check host/port and try again.")
        sys.exit(1)

    for op, entries in results.items():
        if entries:
            print(f"  {op}: {entries}")
        else:
            print(f"  {op}: (none)")

    # Print a copy-paste ready TDX_ENTRIES dict
    all_found = any(entries for entries in results.values())
    if all_found:
        print("\n=== Paste into consts.py TDX_ENTRIES ===")
        for op, entries in results.items():
            if entries:
                print(f'    "{op}": "{entries[0]}",')


if __name__ == "__main__":
    main()
