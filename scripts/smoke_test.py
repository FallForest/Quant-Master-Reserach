"""Smoke test for Yahoo data pipeline.

Tests download + normalize using direct urllib-based Yahoo fetcher.
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# Ensure project root and collector dir are on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data_collector" / "yahoo"))

from data_collector.yahoo.collector import Run


def main():
    smoke_source = ROOT / ".qmData" / "yahoo" / "smoke_source"
    smoke_norm = ROOT / ".qmData" / "yahoo" / "smoke_normalize"

    # Clean previous runs
    for d in [smoke_source, smoke_norm]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"=== Smoke Test: Yahoo Data Pipeline ===")
    print(f"Date range: {start_date} -> {end_date}")
    print()

    # Initialize Run
    run = Run(
        source_dir=str(smoke_source),
        normalize_dir=str(smoke_norm),
        max_workers=1,
        interval="1d",
        region="CN",
    )

    # Phase 1: Download
    print("--- Phase 1: Download ---")
    try:
        run.download_data(
            max_collector_count=1,
            delay=1.0,
            start=start_date,
            end=end_date,
            limit_nums=5,
        )
    except Exception as e:
        print(f"Download error: {type(e).__name__}: {e}")

    csv_files = list(smoke_source.glob("*.csv"))
    print(f"\nDownloaded {len(csv_files)} CSV files:")
    for f in sorted(csv_files)[:10]:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows, cols={list(df.columns)}")

    if not csv_files:
        print("FAIL: No CSV files downloaded!")
        return False

    # Phase 2: Normalize
    print("\n--- Phase 2: Normalize ---")
    try:
        run.normalize_data(date_field_name="date", symbol_field_name="symbol")
    except Exception as e:
        print(f"Normalize error: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    norm_files = list(smoke_norm.glob("*.csv"))
    print(f"\nNormalized {len(norm_files)} CSV files:")
    for f in sorted(norm_files)[:10]:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows, cols={list(df.columns)}")
        expected = {"date", "open", "high", "low", "close", "volume", "adjclose", "symbol", "change", "factor"}
        missing = expected - set(df.columns)
        if missing:
            print(f"    WARNING: missing columns: {missing}")
        if df["close"].isna().all():
            print(f"    WARNING: all close prices are NaN")

    if not norm_files:
        print("FAIL: No normalized CSV files!")
        return False

    print("\n=== PASS: Smoke test completed successfully ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
