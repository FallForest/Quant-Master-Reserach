# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import time
import datetime
import json
import multiprocessing
from pathlib import Path
from typing import List, Tuple

import fire
import numpy as np
import pandas as pd
from loguru import logger

import quant_master
from quant_master.utils import code_to_fname, exists_quant_master_data

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataAll, DumpDataUpdate, verify_dump
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list

# Galaxy Securities market data servers (from connect.cfg)
TDX_SERVERS = [
    ("120.76.1.198", 7709),
    ("123.125.108.101", 7709),
    ("114.141.177.118", 7709),
    ("27.151.2.90", 7709),
    ("202.100.166.12", 7709),
]

# pytdx frequency code: 9=1day, 8=1min, 7=15min, etc.
TDX_FREQ_MAP = {
    "1d": 9,
    "1min": 8,
    "15min": 7,
    "30min": 6,
    "60min": 5,
}

MAX_BARS_PER_REQUEST = 800

# Cache file for stock list (avoids 4s+ paginated scan every run)
_STOCK_LIST_CACHE = Path(__file__).resolve().parent / ".stock_list_cache.json"
_STOCK_LIST_CACHE_TTL = 86400  # 24 hours


def _connect_any():
    """Try connecting to any available TDX server. Returns (api, ip, port) or raises."""
    from pytdx.hq import TdxHq_API
    api = TdxHq_API()
    for ip, port in TDX_SERVERS:
        try:
            api.connect(ip, port)
            return api, ip, port
        except Exception:
            continue
    raise ConnectionError("No TDX server available")


# ---------------------------------------------------------------------------
# Instrument list helpers
# ---------------------------------------------------------------------------

def _tdx_market_to_prefix(market: int) -> str:
    """Convert TDX market code to QuantMaster prefix: 0=SZ, 1=SH."""
    return "SH" if market == 1 else "SZ"


def _get_tdx_stock_list(api, market: int) -> List[Tuple[str, str, int]]:
    """Get stock list from a TDX server.

    Paginates through the full security list and filters to 6-digit A-shares.
    Returns list of (code, name, market) tuples.
    """
    seen = set()
    result = []
    # Paginate in chunks of 1000 (TDX server limit per request)
    for start in range(0, 50000, 1000):
        stocks = api.get_security_list(market, start)
        if not stocks:
            if start > 28000:
                break
            continue
        for s in stocks:
            code = s.get("code", "")
            name = s.get("name", "")
            if len(code) == 6 and code.isdigit() and code not in seen:
                if market == 0 and code.startswith(("0", "3")) and not code.startswith("39"):
                    seen.add(code)
                    result.append((code, name, market))
                elif market == 1 and code.startswith("6"):
                    seen.add(code)
                    result.append((code, name, market))
    return result


def _get_cached_stock_list() -> List[str]:
    """Get stock list, using file cache to avoid slow paginated scans."""
    cache = _load_stock_cache()
    return cache["symbols"]


def _load_stock_cache() -> dict:
    """Load or populate the stock list + names cache."""
    if _STOCK_LIST_CACHE.exists():
        try:
            cache = json.loads(_STOCK_LIST_CACHE.read_text())
            if time.time() - cache.get("ts", 0) < _STOCK_LIST_CACHE_TTL:
                logger.info(f"Using cached stock list ({len(cache['symbols'])} symbols)")
                return cache
        except Exception:
            pass

    logger.info("Fetching stock list from TDX server...")
    api, ip, port = _connect_any()
    try:
        sz_stocks = _get_tdx_stock_list(api, 0)
        sh_stocks = _get_tdx_stock_list(api, 1)
    finally:
        api.disconnect()

    all_stocks = sz_stocks + sh_stocks
    symbols = [f"{_tdx_market_to_prefix(m)}{c}" for c, n, m in all_stocks]
    names = {c: n for c, n, m in all_stocks}
    logger.info(f"Got {len(symbols)} A-share symbols from TDX")

    cache = {"ts": time.time(), "symbols": symbols, "names": names}
    try:
        _STOCK_LIST_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    except Exception:
        pass

    return cache


def _get_cached_stock_names() -> dict:
    """Get code->name mapping from the cached stock list."""
    cache = _load_stock_cache()
    return cache.get("names", {})


# ---------------------------------------------------------------------------
# Fast batch download (multi-threaded, one TDX connection per thread)
# ---------------------------------------------------------------------------

def _download_worker(symbols_chunk: List[str], save_dir: Path, interval: str,
                     start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                     delay: float, counter: dict):
    """Worker: download a chunk of symbols using one persistent TDX connection."""
    from pytdx.hq import TdxHq_API

    freq_code = TDX_FREQ_MAP.get(interval, 9)
    api, ip, port = _connect_any()

    try:
        for symbol in symbols_chunk:
            if symbol.startswith("SZ"):
                market, code = 0, symbol[2:]
            elif symbol.startswith("SH"):
                market, code = 1, symbol[2:]
            else:
                continue

            # Skip if already downloaded and up-to-date
            fname = code_to_fname(symbol)
            fpath = save_dir / f"{fname}.csv"
            if fpath.exists():
                try:
                    old = pd.read_csv(fpath, usecols=["date"])
                    if not old.empty and pd.Timestamp(old["date"].iloc[-1]) >= end_ts - pd.Timedelta(days=1):
                        with counter["lock"]:
                            counter["ok"] += 1
                        continue
                except Exception:
                    pass

            try:
                all_bars = []
                for start_pos in range(0, 50000, MAX_BARS_PER_REQUEST):
                    bars = api.get_security_bars(freq_code, market, code, start_pos, MAX_BARS_PER_REQUEST)
                    if not bars:
                        break
                    all_bars.extend(bars)
                    if len(bars) < MAX_BARS_PER_REQUEST:
                        break

                if not all_bars:
                    with counter["lock"]:
                        counter["empty"] += 1
                    continue

                df = pd.DataFrame(all_bars)
                df["date"] = pd.to_datetime(df["datetime"])
                if interval == "1d":
                    df["date"] = df["date"].dt.normalize()
                df = df[["date", "open", "high", "low", "close", "vol", "amount"]]
                df.rename(columns={"vol": "volume"}, inplace=True)
                df.sort_values("date", inplace=True)
                df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
                df["symbol"] = symbol
                df.reset_index(drop=True, inplace=True)
                df.to_csv(fpath, index=False)
                with counter["lock"]:
                    counter["ok"] += 1
            except Exception as e:
                with counter["lock"]:
                    counter["err"] += 1
                # Reconnect on error
                try:
                    api.disconnect()
                except Exception:
                    pass
                try:
                    api = TdxHq_API()
                    api.connect(ip, port)
                except Exception:
                    api, ip, port = _connect_any()

            if delay > 0:
                time.sleep(delay)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def _download_all(save_dir: Path, symbols: List[str], interval: str,
                  start: str, end: str, delay: float = 0.02, num_workers: int = 5):
    """Download all symbols using multiple threads, one TDX connection per thread.

    ~5x faster than single-threaded download.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    save_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Split symbols into chunks for each worker
    chunks = [symbols[i::num_workers] for i in range(num_workers)]
    counter = {"ok": 0, "empty": 0, "err": 0, "lock": threading.Lock()}
    total = len(symbols)

    logger.info(f"Downloading {total} symbols with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in chunks:
            if chunk:
                fut = executor.submit(_download_worker, chunk, save_dir, interval,
                                      start_ts, end_ts, delay, counter)
                futures.append(fut)

        # Log progress periodically
        import time as _time
        done = False
        while not done:
            _time.sleep(10)
            with counter["lock"]:
                n = counter["ok"] + counter["empty"] + counter["err"]
            pct = n * 100 // total if total else 100
            logger.info(f"  Progress: {n}/{total} ({pct}%) "
                        f"(ok={counter['ok']}, empty={counter['empty']}, err={counter['err']})")
            done = all(f.done() for f in futures)

        # Wait for all to finish
        for f in futures:
            f.result()

    logger.info(f"Download complete: {counter['ok']} ok, {counter['empty']} empty, {counter['err']} errors")
    return counter["ok"]


# ---------------------------------------------------------------------------
# Fast in-process normalize (no ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _normalize_all(source_dir: Path, target_dir: Path, end_date: str = None):
    """Normalize all CSV files in-process. Avoids Windows ProcessPoolExecutor memory issues."""
    target_dir.mkdir(parents=True, exist_ok=True)
    normalizer = TdxNormalize()
    files = sorted(source_dir.glob("*.csv"))
    logger.info(f"Normalizing {len(files)} files...")

    _end_ts = pd.Timestamp(end_date) if end_date else None
    ok = 0

    for f in files:
        try:
            df = pd.read_csv(f, dtype={"symbol": str}, keep_default_na=False)
            df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                    inplace=True, errors="ignore")
            for col in ["open", "high", "low", "close", "volume", "adjclose"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            result = normalizer.normalize(df)
            if result is not None and not result.empty:
                if _end_ts is not None:
                    mask = pd.to_datetime(result["date"]) <= _end_ts
                    result = result[mask]
                result.to_csv(target_dir / f.name, index=False)
                ok += 1
        except Exception as e:
            logger.warning(f"{f.stem} normalize failed: {e}")

    logger.info(f"Normalized {ok}/{len(files)} files")
    return ok


# ---------------------------------------------------------------------------
# Collector (for BaseCollector compatibility)
# ---------------------------------------------------------------------------

class TdxCollector(BaseCollector):
    """Collector that fetches A-share data from TDX (通达信) market data servers."""

    retry = 3

    def __init__(
        self,
        save_dir,
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0.05,
        check_data_length=None,
        limit_nums=None,
    ):
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.date_field_name = "date"
        self.symbol_field_name = "symbol"

    def get_instrument_list(self):
        return _get_cached_stock_list()

    def normalize_symbol(self, symbol):
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"

    @staticmethod
    def _symbol_to_tdx(symbol: str) -> Tuple[int, str]:
        if symbol.startswith("SZ"):
            return 0, symbol[2:]
        elif symbol.startswith("SH"):
            return 1, symbol[2:]
        else:
            raise ValueError(f"Unknown symbol format: {symbol}")

    def get_data(self, symbol, interval, start_datetime, end_datetime):
        market, code = self._symbol_to_tdx(symbol)
        freq_code = TDX_FREQ_MAP.get(interval, 9)

        api, ip, port = _connect_any()
        try:
            all_bars = []
            for start in range(0, 50000, MAX_BARS_PER_REQUEST):
                bars = api.get_security_bars(freq_code, market, code, start, MAX_BARS_PER_REQUEST)
                if not bars:
                    break
                all_bars.extend(bars)
                if len(bars) < MAX_BARS_PER_REQUEST:
                    break
        finally:
            api.disconnect()

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["date"] = pd.to_datetime(df["datetime"])
        if interval == "1d":
            df["date"] = df["date"].dt.normalize()
        df = df[["date", "open", "high", "low", "close", "vol", "amount"]]
        df.rename(columns={"vol": "volume"}, inplace=True)
        df.sort_values("date", inplace=True)
        df = df[(df["date"] >= pd.Timestamp(start_datetime)) & (df["date"] <= pd.Timestamp(end_datetime))]
        df["symbol"] = symbol
        df.reset_index(drop=True, inplace=True)
        time.sleep(self.delay)
        return df

    def download_index_data(self):
        index_map = {
            "csi300": (1, "000300"),
            "csi500": (0, "000905"),
        }
        api, ip, port = _connect_any()
        try:
            for idx_name, (market, code) in index_map.items():
                logger.info(f"Downloading index: {idx_name} ({code})")
                all_bars = []
                for start in range(0, 50000, MAX_BARS_PER_REQUEST):
                    bars = api.get_security_bars(market, code, 9, start, MAX_BARS_PER_REQUEST)
                    if not bars:
                        break
                    all_bars.extend(bars)
                    if len(bars) < MAX_BARS_PER_REQUEST:
                        break
                if not all_bars:
                    continue
                df = pd.DataFrame(all_bars)
                df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
                df = df[["date", "open", "high", "low", "close", "vol", "amount"]]
                df.rename(columns={"vol": "volume"}, inplace=True)
                df = df.astype(float, errors="ignore")
                df["adjclose"] = df["close"]
                df["change"] = df["close"].pct_change(fill_method=None)
                suffix = "SH" if market == 1 else "SZ"
                df["symbol"] = f"{code}.{suffix}"
                self.save_instrument(f"{code}.{suffix}", df)
                logger.info(f"  Saved {len(df)} bars for {idx_name}")
        finally:
            api.disconnect()


class TdxCollectorCN1d(TdxCollector):
    """TDX collector for Chinese A-share daily data."""
    pass


class TdxCollectorCN1min(TdxCollector):
    """TDX collector for Chinese A-share 1-minute data."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("interval", "1min")
        super().__init__(*args, **kwargs)


class TdxCollectorCN15min(TdxCollector):
    """TDX collector for Chinese A-share 15-minute data."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("interval", "15min")
        super().__init__(*args, **kwargs)


class TdxCollectorCN30min(TdxCollector):
    """TDX collector for Chinese A-share 30-minute data."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("interval", "30min")
        super().__init__(*args, **kwargs)


class TdxCollectorCN60min(TdxCollector):
    """TDX collector for Chinese A-share 60-minute data."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("interval", "60min")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TdxNormalize(BaseNormalize):
    """Base normalizer for TDX data."""
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    def __init__(self, date_field_name="date", symbol_field_name="symbol", **kwargs):
        super().__init__(date_field_name, symbol_field_name)
        self._end_date = kwargs.get("end_date", None)

    def _get_calendar_list(self):
        return get_calendar_list("ALL")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        calendar_list = self._get_calendar_list()
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date(): pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ].index
            )
        df.sort_index(inplace=True)
        df = df.reset_index()
        if "index" in df.columns:
            df.rename(columns={"index": self._date_field_name}, inplace=True)
        df = self.adjusted_price(df)
        return df

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["adjclose"] = df["close"]
        df["factor"] = 1.0
        df["change"] = df["close"].pct_change(fill_method=None)
        return df


class TdxNormalizeCN1d(TdxNormalize):
    """Normalizer for Chinese daily TDX data."""
    pass


class TdxNormalizeCN1min(TdxNormalize):
    """Normalizer for Chinese 1-minute TDX data."""

    def _get_calendar_list(self):
        """Skip daily calendar alignment for minute data.

        TDX minute data is already complete within trading hours.
        Reindexing against a daily calendar would collapse minute bars to daily.
        """
        return None

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize minute data without daily calendar reindex."""
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        df = df.reset_index()
        if "index" in df.columns:
            df.rename(columns={"index": self._date_field_name}, inplace=True)
        df = self.adjusted_price(df)
        return df


class TdxNormalizeCN15min(TdxNormalizeCN1min):
    """Normalizer for Chinese 15-minute TDX data."""
    pass


class TdxNormalizeCN30min(TdxNormalizeCN1min):
    """Normalizer for Chinese 30-minute TDX data."""
    pass


class TdxNormalizeCN60min(TdxNormalizeCN1min):
    """Normalizer for Chinese 60-minute TDX data."""
    pass


# ---------------------------------------------------------------------------
# Run (CLI entry point)
# ---------------------------------------------------------------------------

class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d"):
        # Use interval-aware directories to prevent daily and minute data from mixing
        if source_dir is None:
            source_dir = CUR_DIR / f"source_{interval}"
        if normalize_dir is None:
            normalize_dir = CUR_DIR / f"normalize_{interval}"
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"TdxCollectorCN{self.interval}"

    @property
    def normalize_class_name(self):
        return f"TdxNormalizeCN{self.interval}"

    @property
    def default_base_dir(self):
        return CUR_DIR

    def download_data(self, max_collector_count=2, delay=0.02, start=None, end=None,
                      check_data_length=None, limit_nums=None):
        """Download daily data from TDX servers.

        Examples
        -------
            $ python collector.py download_data --source_dir ~/.quant_master/tdx_source --start 2020-01-01 --end 2026-05-25
        """
        if start is None:
            start = "2000-01-01"
        if end is None:
            end = (pd.Timestamp(datetime.datetime.now()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        symbols = _get_cached_stock_list()
        if limit_nums is not None:
            symbols = symbols[:int(limit_nums)]

        _download_all(
            save_dir=self.source_dir,
            symbols=symbols,
            interval=self.interval,
            start=start,
            end=end,
            delay=delay,
        )

    def normalize_data(self, date_field_name="date", symbol_field_name="symbol", end_date=None,
                       quant_master_data_1d_dir=None):
        """Normalize downloaded TDX data.

        Examples
        -------
            $ python collector.py normalize_data --source_dir ~/.quant_master/tdx_source --normalize_dir ~/.quant_master/tdx_normalize
        """
        _normalize_all(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            end_date=end_date,
        )

    def update_data_to_bin(
        self,
        quant_master_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 0.02,
        exists_skip: bool = False,
    ):
        """Download, normalize and dump TDX daily data to QuantMaster binary format.

        Optimized for speed: single connection for download, in-process normalize/dump.

        Parameters
        ----------
        quant_master_data_1d_dir : str
            Path to the QuantMaster data directory.
        end_date : str
            End date (exclusive). Default: tomorrow.
        delay : float
            Delay between requests. Default 0.02.

        Examples
        -------
            $ python collector.py update_data_to_bin --quant_master_data_1d_dir ~/.quant_master/quant_master_data/cn_data --end_date 2026-05-26
        """
        import shutil

        def _cleanup():
            for d in [self.source_dir, self.normalize_dir]:
                if not d.exists():
                    continue
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)

        _cleanup()

        if end_date is None:
            end_date = (pd.Timestamp(datetime.datetime.now()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Determine start date from existing data
        cal_path = Path(quant_master_data_1d_dir) / "calendars" / f"{self.interval}.txt"
        if cal_path.exists():
            lines = cal_path.read_text().strip().split("\n")
            last_date = lines[-1].strip()
            start_date = last_date
            logger.info(f"Incremental update from {start_date} to {end_date}")
        else:
            if self.interval == "1min":
                start_date = (pd.Timestamp(datetime.datetime.now()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start_date = "2000-01-01"

        # Step 1: Download (single connection, fast)
        symbols = _get_cached_stock_list()
        _download_all(
            save_dir=self.source_dir,
            symbols=symbols,
            interval=self.interval,
            start=start_date,
            end=end_date,
            delay=delay,
        )

        # Step 2: Normalize (in-process, no subprocess)
        _normalize_all(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            end_date=end_date,
        )

        # Step 3: Dump to binary
        if cal_path.exists():
            _dump = DumpDataUpdate(
                data_path=self.normalize_dir,
                quant_master_dir=quant_master_data_1d_dir,
                freq=self.interval,
                exclude_fields="symbol,date",
                max_workers=1,
            )
        else:
            _dump = DumpDataAll(
                data_path=self.normalize_dir,
                quant_master_dir=quant_master_data_1d_dir,
                freq=self.interval,
                exclude_fields="symbol,date",
                max_workers=1,
            )
        _dump.dump()

        # Verify
        verify_dump(quant_master_data_1d_dir, expected_end_date=end_date, freq=self.interval)

        # Write stock names for UI export
        try:
            names = _get_cached_stock_names()
            names_path = Path(quant_master_data_1d_dir) / "instruments" / "names.txt"
            with open(names_path, "w", encoding="utf-8") as f:
                for code, name in sorted(names.items()):
                    f.write(f"{code}\t{name}\n")
            logger.info(f"Wrote {len(names)} stock names to {names_path}")
        except Exception as e:
            logger.warning(f"Failed to write stock names (non-fatal): {e}")

        # Cleanup
        _cleanup()
        logger.info("TDX data update complete.")


if __name__ == "__main__":
    fire.Fire(Run)
