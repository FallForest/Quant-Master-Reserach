"""Read-only audit for provider URI consistency in example workflow YAML files."""

import argparse
import os
import re
from collections import Counter
from pathlib import Path


PROVIDER_URI_PATTERN = re.compile(r"^[ \t]*provider_uri:[ \t]*['\"]?([^'\"\s#]+)", re.MULTILINE)


def scan_provider_uris(examples_dir, include_intraday=False):
    records = []
    for path in sorted(Path(examples_dir).rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for match in PROVIDER_URI_PATTERN.finditer(text):
            provider_uri = match.group(1)
            if not include_intraday and provider_uri.rstrip("/").endswith(("_1min", "_15min", "_30min", "_60min")):
                continue
            records.append((path, provider_uri))
    return records


def audit_provider_uris(
    examples_dir="examples",
    expected_provider_uri=None,
    include_intraday=False,
    fail_on_mismatch=False,
):
    records = scan_provider_uris(examples_dir, include_intraday=include_intraday)
    counts = Counter(uri for _, uri in records)
    print("Daily example provider URI distribution:")
    for uri, count in sorted(counts.items()):
        print(f"  {count:4d}  {uri}")

    expected = expected_provider_uri or os.environ.get("QUANT_MASTER_PROVIDER_URI")
    mismatches = [(path, uri) for path, uri in records if expected and uri != expected]
    if expected:
        print(f"Expected provider URI: {expected}")
        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for path, uri in mismatches:
                print(f"  {path}: {uri}")
        else:
            print("All scanned daily providers match the expected URI.")

    if fail_on_mismatch and not expected:
        raise ValueError("Set --expected-provider-uri or QUANT_MASTER_PROVIDER_URI before using --fail-on-mismatch.")
    if fail_on_mismatch and mismatches:
        raise RuntimeError(f"Provider URI audit failed: {len(mismatches)} mismatched workflow files.")
    return {"counts": counts, "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--expected-provider-uri")
    parser.add_argument("--include-intraday", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()
    audit_provider_uris(
        examples_dir=args.examples_dir,
        expected_provider_uri=args.expected_provider_uri,
        include_intraday=args.include_intraday,
        fail_on_mismatch=args.fail_on_mismatch,
    )


if __name__ == "__main__":
    main()
