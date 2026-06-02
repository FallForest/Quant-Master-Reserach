#!/usr/bin/env python
"""Daily scheduler for live signal pipeline.

Usage:
    # Run once for today
    python scripts/schedule_daily_run.py --once

    # Schedule daily at 15:30 (requires 'schedule' package: pip install schedule)
    python scripts/schedule_daily_run.py --schedule

    # Run for a specific date
    python scripts/schedule_daily_run.py --once --date 2026-05-30
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("schedule_daily_run")


def is_trading_day(date_str: str = None) -> bool:
    """Check if a date is a trading day (weekday)."""
    import pandas as pd
    if date_str:
        dt = pd.Timestamp(date_str)
    else:
        dt = pd.Timestamp.now()
    return dt.dayofweek < 5


def run_pipeline(date: str = None, broker: str = "paper", dry_run: bool = False):
    """Run the live signal pipeline for a given date."""
    from scripts.live_signal_pipeline import run_daily

    if date is None:
        date = str(datetime.now().date())

    if not is_trading_day(date):
        logger.info(f"{date} is not a trading day, skipping.")
        return None

    logger.info(f"Running pipeline for {date}")
    try:
        result = run_daily(trade_date=date, broker_kind=broker, dry_run=dry_run)
        logger.info(f"Pipeline completed successfully for {date}")
        return result
    except Exception as e:
        logger.error(f"Pipeline failed for {date}: {e}", exc_info=True)
        return None


def run_scheduled(broker: str = "paper", dry_run: bool = False):
    """Run the scheduler loop, triggering at 15:30 daily."""
    try:
        import schedule
    except ImportError:
        logger.error("Please install 'schedule' package: pip install schedule")
        sys.exit(1)

    def job():
        date_str = str(datetime.now().date())
        run_pipeline(date_str, broker, dry_run)

    schedule.every().day.at("15:30").do(job)
    logger.info("Scheduler started. Will run daily at 15:30. Press Ctrl+C to stop.")

    now = datetime.now()
    if now.hour >= 15 and now.minute >= 30 and is_trading_day():
        logger.info("Past 15:30 today, running now...")
        job()

    while True:
        schedule.run_pending()
        time.sleep(60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily scheduler for live signal pipeline")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--schedule", action="store_true", help="Run as daily scheduler")
    p.add_argument("--date", default=None, help="Specific date (YYYY-MM-DD), only with --once")
    p.add_argument("--broker", default="paper", help="Broker type: paper, tcdll, tdx")
    p.add_argument("--dry-run", action="store_true", help="Dry run mode")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.once:
        result = run_pipeline(args.date, args.broker, args.dry_run)
        return 0 if result is not None else 1

    if args.schedule:
        run_scheduled(args.broker, args.dry_run)
        return 0

    result = run_pipeline(args.date, args.broker, args.dry_run)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
