"""日线数据增量同步：多线程从 TDX 拉取最新 K 线，批量追加到 bin 文件。"""
import datetime
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .calendar_validation import InvalidCalendarError, load_calendar_file, parse_calendar_date
from .datadir import get_effective_data_dir
from .stock_cache import build_stock_summary
from .tdx_quote import TDXQuote, connect_first_working


PARTIAL_SYNC_COVERAGE_THRESHOLD = 0.8
MAX_SAMPLE_SYMBOLS = 8
FETCH_TIMEOUT_SECONDS = 300
DEFAULT_FETCH_WORKERS = 24
MAX_FETCH_WORKERS = 32

_log = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_sync_status = {
    "running": False, "lastSync": None, "lastError": None, "lastStats": None, "runningSince": None,
    "progressPhase": None, "progressTotal": 0, "progressDone": 0, "progressLabel": None,
}
_cache_lock = threading.Lock()
_cache_status = {
    "running": False,
    "startedAt": None,
    "finishedAt": None,
    "lastError": None,
}
_cache_refresh_pending = False
# Health scans inspect every instrument's close file.  Keep the last result for
# a short interval so the browser and model pages do not repeat the same scan
# on every status poll.  A completed sync always refreshes it with force=True.
_health_snapshot_lock = threading.Lock()
_health_snapshot_cache = {}
_health_refreshing = set()
HEALTH_SNAPSHOT_TTL = 2.0
# 同步运行超过此秒数视为失效，允许清除
STALE_RUNNING_TIMEOUT = 600  # 10 分钟


def get_sync_status():
    """返回同步状态的快照副本（线程安全）。自动检测卡死的 running 标记。"""
    with _sync_lock:
        stale = (
            _sync_status["running"]
            and _sync_status["runningSince"] is not None
            and (time.time() - _sync_status["runningSince"]) > STALE_RUNNING_TIMEOUT
        )
        if stale:
            _log.warning(
                "Sync running flag stuck for > %ds, auto-clearing. "
                "runningSince=%.1f, lastSync=%s, lastError=%s",
                STALE_RUNNING_TIMEOUT,
                _sync_status["runningSince"],
                _sync_status["lastSync"],
                _sync_status["lastError"],
            )
            _sync_status["running"] = False
            _sync_status["runningSince"] = None
            _sync_status["lastError"] = "Data sync stopped responding and was reset"
            _sync_status["progressPhase"] = None
            _sync_status["progressTotal"] = 0
            _sync_status["progressDone"] = 0
            _sync_status["progressLabel"] = None
        return dict(_sync_status)


def _set_sync_status(**kwargs):
    """原子更新同步状态字段。自动记录 running 的开始/结束时间。"""
    with _sync_lock:
        _sync_status.update(kwargs)
        if "running" in kwargs:
            if kwargs["running"] is True:
                _sync_status["runningSince"] = time.time()
            elif kwargs["running"] is False:
                _sync_status["runningSince"] = None


def get_cache_status():
    """Return the independent stock-summary cache refresh status."""
    with _cache_lock:
        return dict(_cache_status)


def start_cache_refresh(data_obj):
    """Refresh the browser summary without extending the market-data lock.

    The cache is a derived artifact.  Its rebuild can continue after the quote
    sync has completed because ``build_stock_summary`` writes one complete JSON
    payload only after all workers finish.
    """
    global _cache_refresh_pending
    if data_obj is None:
        return False
    with _cache_lock:
        if _cache_status["running"]:
            _cache_refresh_pending = True
            return False
        _cache_status.update(
            running=True,
            startedAt=time.time(),
            finishedAt=None,
            lastError=None,
        )

    def _worker():
        global _cache_refresh_pending
        while True:
            try:
                build_stock_summary(data_obj)
            except Exception as exc:  # pragma: no cover - exercised by integration failures
                _log.exception("Stock summary cache refresh failed")
                with _cache_lock:
                    _cache_status["lastError"] = str(exc)
                    _cache_refresh_pending = False
                    _cache_status["running"] = False
                    _cache_status["finishedAt"] = time.time()
                return
            with _cache_lock:
                if _cache_refresh_pending:
                    _cache_refresh_pending = False
                    continue
                _cache_status["running"] = False
                _cache_status["finishedAt"] = time.time()
                return

    threading.Thread(target=_worker, daemon=True, name="stock-summary-refresh").start()
    return True


def _fetch_worker_count():
    """Resolve bounded TDX fetch parallelism from the environment."""
    raw = os.getenv("QUANT_MASTER_SYNC_WORKERS", str(DEFAULT_FETCH_WORKERS))
    try:
        return max(1, min(MAX_FETCH_WORKERS, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_FETCH_WORKERS


def _try_begin_sync():
    """Atomically claim the single sync slot and publish an initial progress state."""
    with _sync_lock:
        if _sync_status["running"]:
            return False
        _sync_status.update(
            running=True,
            runningSince=time.time(),
            lastError=None,
            progressPhase="starting",
            progressTotal=0,
            progressDone=0,
            progressLabel="正在检查本地数据...",
        )
        return True


def start_auto_sync_daily(data_dir, data_obj=None, force=False):
    """Claim the sync slot and start the job, making status polling race-free."""
    # This also clears a stale status before the atomic claim below.
    get_sync_status()
    if not _try_begin_sync():
        return False
    try:
        thread = threading.Thread(
            target=auto_sync_daily,
            args=(data_dir, data_obj),
            kwargs={"force": force, "_already_started": True},
            daemon=True,
            name="daily-data-sync",
        )
        thread.start()
    except Exception:
        _set_sync_status(
            running=False,
            lastError="Failed to start data sync",
            progressPhase=None,
            progressTotal=0,
            progressDone=0,
            progressLabel=None,
        )
        raise
    return True


def _update_instrument_end_dates(inst_path, symbol_end_dates):
    """Update quote coverage dates for an instrument file.

    Use this for instruments/all.txt only. Index files such as csi300.txt store
    membership intervals, not quote coverage, so quote sync must not extend
    their end_date values.
    """
    inst_path = Path(inst_path)
    if not inst_path.exists() or not symbol_end_dates:
        return 0

    lines = inst_path.read_text(encoding="utf-8").strip().split("\n")
    new_lines = []
    updated_count = 0
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[0] in symbol_end_dates:
            new_end = symbol_end_dates[parts[0]]
            if parts[2] != new_end:
                parts[2] = new_end
                updated_count += 1
        new_lines.append("\t".join(parts))
    _atomic_write_text(inst_path, "\n".join(new_lines) + "\n")
    return updated_count


def _update_instrument_start_dates(inst_path, symbol_start_dates):
    """Update the first available calendar date in ``instruments/all.txt``."""
    inst_path = Path(inst_path)
    if not inst_path.exists() or not symbol_start_dates:
        return 0
    lines = inst_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    updated_count = 0
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[0] in symbol_start_dates:
            new_start = symbol_start_dates[parts[0]]
            if parts[1] != new_start:
                parts[1] = new_start
                updated_count += 1
        new_lines.append("\t".join(parts))
    _atomic_write_text(inst_path, "\n".join(new_lines) + "\n")
    return updated_count


def _atomic_write_text(path, content):
    """Publish a small metadata file without exposing a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _discover_instruments_from_tdx(data_dir):
    """Discover the A-share universe and publish its metadata atomically."""
    quote = TDXQuote()
    try:
        discovered = quote.fetch_instruments()
    finally:
        quote._invalidate()

    if not discovered:
        raise RuntimeError("TDX returned no A-share instruments")

    bootstrap_date = "1900-01-01"
    ordered = sorted(discovered.items())
    instrument_root = Path(data_dir) / "instruments"
    _atomic_write_text(
        instrument_root / "all.txt",
        "\n".join(
            f"{symbol}\t{bootstrap_date}\t{bootstrap_date}" for symbol, _ in ordered
        ) + "\n",
    )
    _atomic_write_text(
        instrument_root / "names.txt",
        "\n".join(
            f"{item['code']}\t{item['name']}" for _, item in ordered if item["name"]
        ) + "\n",
    )
    return len(ordered)


def _read_calendar_with_validation(cal_path, strict=False):
    """Read calendars/day.txt with validation diagnostics."""
    try:
        return load_calendar_file(cal_path, strict=strict)
    except InvalidCalendarError:
        raise
    except Exception:
        return [], None


def _calendar_corruption_message(data_dir, result_or_lines):
    lines = getattr(result_or_lines, "invalid_lines", result_or_lines) or []
    samples = lines[:3]
    sample_text = "、".join(f"第 {item.get('line')} 行 {item.get('value')}" for item in samples) or "无非法日期样例"
    return f"数据日历文件损坏，已暂停同步：{data_dir}/calendars/day.txt，非法日期：{sample_text}"


def _get_last_update_date(data_dir):
    """从 calendars/day.txt 读取最后一条有效日期。"""
    cal_path = Path(data_dir) / "calendars" / "day.txt"
    if not cal_path.exists():
        return None
    try:
        calendar, _ = load_calendar_file(cal_path, strict=False)
        return calendar[-1] if calendar else None
    except Exception:
        return None


def _empty_health_snapshot(calendar_last_date=None):
    return {
        "calendarLastDate": calendar_last_date,
        "effectiveLastDate": calendar_last_date,
        "marketEffectiveLastDate": calendar_last_date,
        "equityCoverageAtLastDate": 0.0,
        "equityCount": 0,
        "equityCoveredAtLastDate": 0,
        "calendarCoverage": 0.0,
        "calendarCoveredEquities": 0,
        "overflowedSymbolCount": 0,
        "sampleOverflowSymbols": [],
        "calendarHealthy": True,
        "calendarInvalidLineCount": 0,
        "sampleInvalidCalendarLines": [],
        "calendarDuplicateCount": 0,
        "calendarOrdered": True,
    }






def _scan_data_health_snapshot(data_dir):
    """扫描本地日线落盘情况，返回日历/全量/股票 universe 的最新覆盖信息。"""
    cal_path = Path(data_dir) / "calendars" / "day.txt"
    inst_path = Path(data_dir) / "instruments" / "all.txt"

    if not cal_path.exists():
        return _empty_health_snapshot(None)

    calendar = []
    validation = None
    calendar_last_date = None
    try:
        calendar, validation = load_calendar_file(cal_path, strict=False)
        calendar_last_date = calendar[-1] if calendar else None
        if not calendar:
            snapshot = _empty_health_snapshot(None)
            if validation is not None:
                snapshot.update(validation.diagnostics())
            return snapshot

        instruments = []
        if inst_path.exists():
            with open(inst_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        instruments.append(parts[0])
        equity_symbols = {sym.upper() for sym in instruments if _is_equity_symbol(sym)}

        latest = None
        market_latest = None
        symbol_end_dates = {}
        overflow_symbols = []
        for close_path in (Path(data_dir) / "features").glob("*/close.day.bin"):
            raw = np.fromfile(str(close_path), dtype="<f4")
            if len(raw) < 2:
                continue
            start_idx = int(raw[0])
            arr = raw[1:]
            symbol = close_path.parent.name.upper()
            if start_idx + len(arr) > len(calendar):
                overflow_symbols.append(symbol)
            max_in_range_pos = min(len(arr) - 1, len(calendar) - 1 - start_idx)
            if max_in_range_pos < 0:
                continue
            for pos in range(max_in_range_pos, -1, -1):
                val = float(arr[pos])
                if np.isnan(val):
                    continue
                cal_idx = start_idx + pos
                if 0 <= cal_idx < len(calendar):
                    date_str = calendar[cal_idx]
                    symbol_end_dates[symbol] = date_str
                    if latest is None or date_str > latest:
                        latest = date_str
                    if symbol in equity_symbols and (market_latest is None or date_str > market_latest):
                        market_latest = date_str
                break

        equity_count = len(equity_symbols)
        equity_covered_at_last = 0
        if market_latest:
            equity_covered_at_last = sum(1 for sym in equity_symbols if symbol_end_dates.get(sym) == market_latest)
        equity_coverage = (equity_covered_at_last / equity_count) if equity_count else 0.0

        calendar_covered_equities = 0
        if calendar_last_date:
            calendar_covered_equities = sum(1 for sym in equity_symbols if symbol_end_dates.get(sym) == calendar_last_date)
        calendar_coverage = (calendar_covered_equities / equity_count) if equity_count else 0.0

        snapshot = _empty_health_snapshot(calendar_last_date)
        snapshot.update({
            "effectiveLastDate": latest or calendar_last_date,
            "marketEffectiveLastDate": market_latest or latest or calendar_last_date,
            "equityCoverageAtLastDate": equity_coverage,
            "equityCount": equity_count,
            "equityCoveredAtLastDate": equity_covered_at_last,
            "calendarCoverage": calendar_coverage,
            "calendarCoveredEquities": calendar_covered_equities,
            "overflowedSymbolCount": len(overflow_symbols),
            "sampleOverflowSymbols": overflow_symbols[:MAX_SAMPLE_SYMBOLS],
        })
        if validation is not None:
            snapshot.update(validation.diagnostics())
        return snapshot
    except Exception:
        snapshot = _empty_health_snapshot(calendar_last_date)
        if validation is not None:
            snapshot.update(validation.diagnostics())
        return snapshot


def get_data_health_snapshot(data_dir, force=False):
    """Return a short-lived health snapshot, refreshing after a completed sync."""
    cache_key = str(Path(data_dir).expanduser().resolve())
    now = time.monotonic()
    with _health_snapshot_lock:
        cached = _health_snapshot_cache.get(cache_key)
        # A background refresh owns the expensive scan.  Serve the last
        # complete snapshot while it runs so status requests stay responsive.
        if cached is not None and (not force or cache_key in _health_refreshing):
            if cache_key in _health_refreshing or now - cached[0] < HEALTH_SNAPSHOT_TTL:
                return dict(cached[1])

    snapshot = _scan_data_health_snapshot(data_dir)
    with _health_snapshot_lock:
        _health_snapshot_cache[cache_key] = (time.monotonic(), dict(snapshot))
    return snapshot


def start_health_refresh(data_dir):
    """Refresh the aggregate health snapshot outside the sync critical path."""
    cache_key = str(Path(data_dir).expanduser().resolve())
    with _health_snapshot_lock:
        if cache_key in _health_refreshing:
            return False
        _health_refreshing.add(cache_key)

    def _worker():
        try:
            snapshot = _scan_data_health_snapshot(data_dir)
            with _health_snapshot_lock:
                _health_snapshot_cache[cache_key] = (time.monotonic(), dict(snapshot))
        finally:
            with _health_snapshot_lock:
                _health_refreshing.discard(cache_key)

    threading.Thread(target=_worker, daemon=True, name="health-snapshot-refresh").start()
    return True


def is_health_refreshing(data_dir):
    cache_key = str(Path(data_dir).expanduser().resolve())
    with _health_snapshot_lock:
        return cache_key in _health_refreshing



def get_effective_last_update_date(data_dir):
    """返回实际已写入收盘价数据的最后交易日，而不只是日历最后一天。"""
    return get_data_health_snapshot(data_dir)["effectiveLastDate"]


def _is_trading_day():
    """判断今天是否为交易日（周一至周五，不含节假日）。"""
    return datetime.datetime.now().weekday() < 5


def _bars_to_fetch(last_date, min_bars=3, max_bars=60):
    """根据 last_date 和今天的差距，计算需要拉取的 bar 数。"""
    if not last_date:
        return max_bars
    try:
        last = datetime.date.fromisoformat(last_date)
        today = datetime.date.today()
        gap = (today - last).days
        need = max(min_bars, int(gap * 1.5) + 2)
        return min(need, max_bars)
    except Exception:
        return max_bars


def _is_equity_symbol(symbol: str) -> bool:
    """判断 symbol 是否属于股票 universe，用于覆盖率门控。

    这里显式排除常见指数代码，例如 SH000300 / SH000001 / SZ399001 等，
    只把 A 股/BJ 股票计入“推进交易日”的覆盖率统计。
    """
    s = (symbol or "").upper()
    if len(s) < 4:
        return False
    if s.startswith("SH"):
        return s[2:].startswith("6")
    if s.startswith("SZ"):
        return s[2:].startswith(("0", "3"))
    if s.startswith("BJ"):
        return s[2:].startswith(("4", "8"))
    return False


def _build_sync_stats(**kwargs):
    stats = {
        "dataDir": None,
        "calendarLastDate": None,
        "effectiveLastDate": None,
        "marketEffectiveLastDate": None,
        "targetSyncDate": None,
        "acceptedNewDates": [],
        "rejectedNewDates": [],
        "coverageThreshold": PARTIAL_SYNC_COVERAGE_THRESHOLD,
        "totalSymbols": 0,
        "equitySymbols": 0,
        "symbolsWithBars": 0,
        "equitySymbolsWithBars": 0,
        "symbolsWithTargetDate": 0,
        "equitySymbolsWithTargetDate": 0,
        "updatedSymbols": 0,
        "staleSymbols": 0,
        "connectionErrorCount": 0,
        "workerErrorCount": 0,
        "equityCoverageAtLastDate": 0.0,
        "equityCount": 0,
        "equityCoveredAtLastDate": 0,
        "calendarCoverage": 0.0,
        "calendarCoveredEquities": 0,
        "overflowedSymbolCount": 0,
        "sampleOverflowSymbols": [],
        "calendarHealthy": True,
        "calendarInvalidLineCount": 0,
        "sampleInvalidCalendarLines": [],
        "calendarDuplicateCount": 0,
        "calendarOrdered": True,
        "invalidBarDateCount": 0,
        "sampleInvalidBarDates": [],
        "sampleTargetSymbols": [],
        "sampleStaleSymbols": [],
        "sampleErrors": [],
        "partial": False,
    }
    stats.update(kwargs)
    return stats


def _extend_tail_files(data_dir, calendar, fields, extra_buffer=0, progress_callback=None):
    """确保所有 .day.bin 文件长度至少对齐到 calendar 长度+extra_buffer，不足的用 NaN 补齐。"""
    root = Path(data_dir)
    calendar_len = len(calendar) + extra_buffer
    extended_count = 0
    close_paths = list((root / "features").glob("*/close.day.bin"))
    total = len(close_paths)
    if progress_callback:
        progress_callback(0, total)
    for index, close_path in enumerate(close_paths, start=1):
        # Only the first float and file sizes are needed here. Loading every full
        # feature array made this preparation phase take minutes on large universes.
        if close_path.stat().st_size >= 8:
            header = np.fromfile(str(close_path), dtype="<f4", count=1)
            stock_idx = int(header[0])
            expected_len = calendar_len - stock_idx
            if expected_len > 0:
                extended = False
                for field in fields:
                    bin_path = close_path.parent / f"{field}.day.bin"
                    if not bin_path.exists():
                        continue
                    value_count = bin_path.stat().st_size // np.dtype("<f4").itemsize
                    if value_count < 2:
                        continue
                    n_missing = expected_len - (value_count - 1)
                    if n_missing <= 0:
                        continue
                    with bin_path.open("ab") as fp:
                        np.full(n_missing, np.nan, dtype=np.float32).tofile(fp)
                    extended = True
                extended_count += int(extended)
        if progress_callback and (index == total or index % max(1, total // 100) == 0):
            progress_callback(index, total)
    if extended_count:
        print(f"Auto-sync: extended tail for {extended_count} symbols to match calendar")
    return extended_count


def auto_sync_daily(data_dir, data_obj=None, force=False, _already_started=False):
    """增量同步日线数据。data_obj 用于清除 calendar 缓存。

    Args:
        force: 为 True 时跳过 now_hour<15 和 _is_trading_day 检查（手动触发时使用）。
    """
    if not _already_started and not _try_begin_sync():
        return False
    t0 = time.time()
    try:
        data_dir = get_effective_data_dir(data_obj, data_dir)
        health_snapshot = get_data_health_snapshot(data_dir)
        if health_snapshot.get("calendarInvalidLineCount", 0):
            stats = _build_sync_stats(dataDir=data_dir, **health_snapshot)
            message = _calendar_corruption_message(
                data_dir,
                health_snapshot.get("sampleInvalidCalendarLines", []),
            )
            _set_sync_status(lastError=message, lastStats=stats)
            print(f"Auto-sync: {message}")
            return
        last_date = health_snapshot["effectiveLastDate"]
        calendar_last_date = health_snapshot["calendarLastDate"]
        today = datetime.date.today().strftime("%Y-%m-%d")
        now_hour = datetime.datetime.now().hour
        base_stats = _build_sync_stats(
            dataDir=data_dir,
            **health_snapshot,
        )
        _set_sync_status(
            lastStats=base_stats,
            progressPhase="preparing",
            progressLabel="正在准备本地数据文件...",
        )

        # 提前加载日历并补齐尾部缺口，确保文件长度对齐 calendar
        fields = ["open", "high", "low", "close", "volume", "amount", "adjclose", "change", "factor"]
        cal_path = Path(data_dir) / "calendars" / "day.txt"
        cal = []
        if cal_path.exists():
            cal, cal_v = load_calendar_file(cal_path, strict=False)
            if cal:
                # Alpha labels may reference the next trading day while preparing
                # an inference dataset, so retain one NaN tail slot beyond the
                # calendar itself.  It is never treated as an accepted market bar.
                def _report_prepare_progress(done, total):
                    _set_sync_status(
                        progressPhase="preparing",
                        progressTotal=total,
                        progressDone=done,
                        progressLabel=f"正在准备本地数据文件... {done}/{total}",
                    )

                _extend_tail_files(
                    data_dir,
                    cal,
                    fields,
                    extra_buffer=1,
                    progress_callback=_report_prepare_progress,
                )

        if not force:
            if not _is_trading_day():
                print(f"Auto-sync: today ({today}) is weekend, skipping")
                _set_sync_status(lastSync=last_date, lastStats=base_stats)
                return

            if now_hour < 15:
                print(f"Auto-sync: market not closed yet ({now_hour}:00), skipping EOD sync.")
                _set_sync_status(lastSync=last_date, lastStats=base_stats)
                return

        data_root = Path(data_dir)
        features_dir = data_root / "features"
        cal_path = data_root / "calendars" / "day.txt"
        inst_path = data_root / "instruments" / "all.txt"

        instruments = []
        instrument_end_dates = {}
        if inst_path.exists():
            with open(inst_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        symbol = parts[0]
                        instruments.append(symbol)
                        # instruments/all.txt is updated together with the bin
                        # files.  Use its end date as a cheap per-symbol
                        # freshness index so a daily sync does not issue a
                        # history request for symbols already up to date.
                        instrument_end_dates[symbol] = parts[2]
        if not instruments:
            _set_sync_status(
                progressPhase="discovering",
                progressTotal=0,
                progressDone=0,
                progressLabel="正在从通达信获取股票清单...",
            )
            discovered_count = _discover_instruments_from_tdx(data_dir)
            print(f"Auto-sync: discovered {discovered_count} A-share instruments from TDX")
            with open(inst_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        symbol = parts[0]
                        instruments.append(symbol)
                        instrument_end_dates[symbol] = parts[2]

        equity_symbols = [sym for sym in instruments if _is_equity_symbol(sym)]

        # instruments/all.txt can lag the exchange by one day.  Probe a small
        # representative batch with one TDX request to avoid treating a
        # calendar-only date as a market data gap and launching thousands of
        # unnecessary history requests.
        remote_probe_date = None
        probe_symbols = equity_symbols[:1]
        if probe_symbols:
            try:
                probe_quote = TDXQuote()
                api = probe_quote._get_api()
                if api is not None:
                    for symbol in probe_symbols:
                        market = 1 if symbol.upper().startswith("SH") else 0
                        code = symbol[2:]
                        rows = api.get_security_bars(9, market, code, 0, 3)
                        if rows:
                            for row in rows:
                                row_date = parse_calendar_date(str(row.get("datetime", ""))[:10])
                                if row_date and (remote_probe_date is None or row_date > remote_probe_date):
                                    remote_probe_date = row_date
            except Exception:
                _log.debug("TDX remote date probe failed", exc_info=True)
            finally:
                try:
                    probe_quote._invalidate()
                except Exception:
                    pass

        valid_local_dates = [value for value in instrument_end_dates.values() if parse_calendar_date(value)]
        max_local_date = max(valid_local_dates, default=last_date)
        target_date_hint = max(
            [value for value in (remote_probe_date, last_date) if value],
            default=today,
        )
        # If every symbol shares the same old local date and the probe failed,
        # retain the original full refresh behavior so a newly closed market
        # day is still discovered on the next attempt.
        if (
            remote_probe_date is None
            and len(set(valid_local_dates)) <= 1
            and max_local_date == last_date
            and last_date
            and last_date < today
        ):
            target_date_hint = today

        symbols_to_fetch = []
        for symbol in instruments:
            end_date = instrument_end_dates.get(symbol)
            if not end_date or not target_date_hint or end_date < target_date_hint:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            print(f"Auto-sync: all {len(instruments)} symbols are already current ({today})")
            _set_sync_status(lastSync=last_date, lastStats=base_stats)
            return

        print(
            f"Auto-sync: fetching {len(symbols_to_fetch)}/{len(instruments)} stale symbols "
            f"from TDX ({len(equity_symbols)} equities total) ..."
        )
        _set_sync_status(
            progressPhase="fetching", progressTotal=len(symbols_to_fetch), progressDone=0,
            progressLabel=f"正在从通达信获取 {len(symbols_to_fetch)} 只待更新股票日线数据...",
        )

        num_workers = _fetch_worker_count()
        invalid_bar_dates = []
        _fetch_counter = [0]  # 共享计数器，各线程处理完一只股票后原子递增
        _fetch_lock = threading.Lock()
        _fetch_total = len(symbols_to_fetch)
        _fetch_stop = threading.Event()
        _fetch_report_step = max(1, _fetch_total // 100)
        _fetch_last_reported = [0]

        def _advance_fetch(count=1):
            with _fetch_lock:
                _fetch_counter[0] = min(_fetch_total, _fetch_counter[0] + count)
                done = _fetch_counter[0]
            if done == _fetch_total or done - _fetch_last_reported[0] >= _fetch_report_step:
                _fetch_last_reported[0] = done
                _set_sync_status(
                    progressDone=done,
                    progressLabel=f"正在从通达信获取数据... {done}/{_fetch_total}",
                )

        def _connect(worker_index=0):
            hosts = TDXQuote.TDX_HOSTS
            # Spread workers across the configured TDX endpoints. Connecting
            # every worker to the first host creates a single-server bottleneck.
            offset = worker_index % len(hosts)
            ordered_hosts = hosts[offset:] + hosts[:offset]
            return connect_first_working(ordered_hosts, time_out=3)

        def _worker(worker_index, sym_chunk, results_buf, error_buf):
            api = _connect(worker_index)
            if api is None:
                error_buf.append("TDX connection failed: all hosts unreachable")
                _advance_fetch(len(sym_chunk))
                return
            try:
                for sym in sym_chunk:
                    if _fetch_stop.is_set():
                        break
                    try:
                        s = sym.upper()
                        if s.startswith("SZ") or s.startswith("BJ"):
                            mkt, code = 0, s[2:]
                        elif s.startswith("SH"):
                            mkt, code = 1, s[2:]
                        else:
                            continue
                        # A symbol may lag the market calendar by weeks while
                        # another one only needs today's bar.  Fetch the
                        # smallest history window that can fill that symbol's
                        # own gap instead of using the global market gap.
                        symbol_num_bars = _bars_to_fetch(instrument_end_dates.get(sym) or last_date)
                        bars = api.get_security_bars(9, mkt, code, 0, symbol_num_bars)
                        if not bars:
                            continue
                        for idx, bar in enumerate(bars):
                            raw_bar_date = str(bar.get("datetime", ""))[:10]
                            bar_date = parse_calendar_date(raw_bar_date)
                            if not bar_date:
                                invalid_bar_dates.append({"symbol": sym, "value": raw_bar_date})
                                continue
                            chg = 0.0
                            if idx > 0:
                                pc = float(bars[idx - 1].get("close", 0))
                                if pc != 0:
                                    chg = (float(bar.get("close", 0)) - pc) / pc
                            cl = float(bar.get("close", 0))
                            vals = (
                                float(bar.get("open", 0)),
                                float(bar.get("high", 0)),
                                float(bar.get("low", 0)),
                                cl,
                                float(bar.get("vol", 0)),
                                float(bar.get("amount", 0)),
                                cl,
                                chg,
                                1.0,
                            )
                            results_buf.append((sym, bar_date, vals))
                    except Exception as e:
                        error_buf.append(f"{sym}: {e}")
                    finally:
                        # Empty responses and per-symbol errors are still completed work.
                        _advance_fetch()
            finally:
                try:
                    api.disconnect()
                except Exception:
                    pass

        chunks = [symbols_to_fetch[i::num_workers] for i in range(num_workers)]
        all_results = []
        all_errors = []
        pool = ThreadPoolExecutor(max_workers=num_workers)
        fetch_timed_out = False
        try:
            futs = []
            for index, chunk in enumerate(chunks):
                buf = []
                err_buf = []
                all_results.append(buf)
                all_errors.append(err_buf)
                futs.append(pool.submit(_worker, index, chunk, buf, err_buf))
            fetch_started = time.monotonic()
            while not all(f.done() for f in futs):
                time.sleep(0.5)
                if time.monotonic() - fetch_started > FETCH_TIMEOUT_SECONDS:
                    fetch_timed_out = True
                    _fetch_stop.set()
                    for future in futs:
                        future.cancel()
                    print(f"Auto-sync: fetch timed out after {FETCH_TIMEOUT_SECONDS}s, proceeding with partial data")
                    break
        finally:
            # Workers observe _fetch_stop before starting another symbol. A current
            # pytdx request is bounded by its socket timeout, so shutdown stays bounded.
            pool.shutdown(wait=True, cancel_futures=True)

        with _fetch_lock:
            completed_fetches = _fetch_counter[0]
        _set_sync_status(progressDone=completed_fetches)

        worker_errors = []
        for err_buf in all_errors:
            worker_errors.extend(err_buf)
        conn_failures = [e for e in worker_errors if "connection failed" in e.lower()]
        other_worker_errors = [e for e in worker_errors if "connection failed" not in e.lower()]
        if fetch_timed_out:
            other_worker_errors.append(
                f"fetch timed out after {FETCH_TIMEOUT_SECONDS}s ({completed_fetches}/{_fetch_total} completed)"
            )

        if conn_failures:
            _set_sync_status(
                lastError="TDX connection failed",
                lastStats=_build_sync_stats(
                    **dict(
                        base_stats,
                        totalSymbols=len(instruments),
                        equitySymbols=len(equity_symbols),
                        connectionErrorCount=len(conn_failures),
                        workerErrorCount=len(other_worker_errors),
                        sampleErrors=worker_errors[:MAX_SAMPLE_SYMBOLS],
                    )
                ),
            )
            print("Auto-sync: TDX connection failed, all hosts unreachable")
            return

        if other_worker_errors:
            if len(other_worker_errors) > 5:
                print(f"Auto-sync: {len(other_worker_errors)} worker errors (showing first 5):")
                for e in other_worker_errors[:5]:
                    print(f"  - {e}")
            else:
                for e in other_worker_errors:
                    print(f"Auto-sync: worker error - {e}")

        all_bar_dates = set()
        stock_bars = {}
        symbol_last_bar_date = {}
        symbols_with_bars = set()
        for buf in all_results:
            for sym, bar_date, vals in buf:
                all_bar_dates.add(bar_date)
                symbols_with_bars.add(sym)
                stock_bars.setdefault(sym, {})[bar_date] = vals
                prev_date = symbol_last_bar_date.get(sym)
                if prev_date is None or bar_date > prev_date:
                    symbol_last_bar_date[sym] = bar_date

        if not all_bar_dates:
            elapsed = time.time() - t0
            stats = _build_sync_stats(
                **dict(
                    base_stats,
                    totalSymbols=len(instruments),
                    equitySymbols=len(equity_symbols),
                    symbolsWithBars=0,
                    equitySymbolsWithBars=0,
                    connectionErrorCount=len(conn_failures),
                    workerErrorCount=len(other_worker_errors),
                    sampleErrors=worker_errors[:MAX_SAMPLE_SYMBOLS],
                    invalidBarDateCount=len(invalid_bar_dates),
                    sampleInvalidBarDates=invalid_bar_dates[:MAX_SAMPLE_SYMBOLS],
                )
            )
            _set_sync_status(lastSync=last_date, lastStats=stats)
            print(f"Auto-sync: no new data ({elapsed:.1f}s)")
            return

        cal_set = set(cal)
        new_date_list = sorted(d for d in all_bar_dates if d not in cal_set)
        target_sync_date = max(new_date_list) if new_date_list else None

        equity_with_bars = [sym for sym in equity_symbols if sym in symbols_with_bars]
        equity_with_target_date = [sym for sym in equity_symbols if symbol_last_bar_date.get(sym) == target_sync_date]
        all_with_target_date = [sym for sym in instruments if symbol_last_bar_date.get(sym) == target_sync_date]
        stale_equities = [sym for sym in equity_symbols if symbol_last_bar_date.get(sym) != target_sync_date]

        accepted_new_dates = list(new_date_list)
        rejected_new_dates = []
        partial_message = None
        if target_sync_date:
            coverage_ratio = (len(equity_with_target_date) / len(equity_symbols)) if equity_symbols else 1.0
            if not force and coverage_ratio < PARTIAL_SYNC_COVERAGE_THRESHOLD:
                accepted_new_dates = [d for d in new_date_list if d < target_sync_date]
                rejected_new_dates = [target_sync_date]
                partial_message = (
                    f"partial sync: equity coverage {len(equity_with_target_date)}/{len(equity_symbols)} "
                    f"for {target_sync_date} below threshold {PARTIAL_SYNC_COVERAGE_THRESHOLD:.0%}"
                )
                print(f"Auto-sync: {partial_message}")

        accepted_new_dates = [d for d in accepted_new_dates if parse_calendar_date(d)]
        if accepted_new_dates:
            # 日历只能向后延长：往中间插入日期会让所有已写入 bin 的下标整体错位。
            # 而停牌/退市股能取到的最新 K 线可能停在窗口之前（如整个 4 月），
            # 这些日期落不进窗口，必须丢弃，否则日历会变成乱序的两段拼接。
            latest = max(cal) if cal else None
            extended = [d for d in accepted_new_dates if latest is None or d > latest]
            if len(extended) < len(accepted_new_dates):
                _log.warning(
                    "丢弃 %d 个早于日历窗口的日期（停牌/退市股）",
                    len(accepted_new_dates) - len(extended),
                )
            if extended:
                cal.extend(extended)
                _atomic_write_text(cal_path, "\n".join(cal) + "\n")

        date_to_idx = {date_str: idx for idx, date_str in enumerate(cal)}
        accepted_date_set = set(cal)
        nan_vals = tuple([float("nan")] * len(fields))
        total_updated = 0
        _set_sync_status(
            progressPhase="writing", progressTotal=len(symbols_with_bars), progressDone=0,
            progressLabel="正在写入日线数据到本地存储...",
        )
        symbol_end_dates = {}
        update_symbols = sorted(symbols_with_bars)

        # 写入阶段并行化——每只股票的读写互不依赖
        _write_lock = threading.Lock()
        _write_counter = [0]
        _write_results = []

        def _write_worker(sym_chunk):
            local_end_dates = {}
            local_start_dates = {}
            local_updated = 0
            for sym in sym_chunk:
                bars_dict = {d: vals for d, vals in stock_bars.get(sym, {}).items() if d in accepted_date_set}
                fname = sym.lower()
                sym_dir = features_dir / fname
                sym_dir.mkdir(parents=True, exist_ok=True)

                field_arrays = {}
                stock_idx = None
                last_valid_cal_idx = -1
                for field in fields:
                    bin_path = sym_dir / f"{field}.day.bin"
                    if bin_path.exists() and bin_path.stat().st_size >= 8:
                        raw = np.fromfile(str(bin_path), dtype="<f4")
                        current_idx = int(raw[0])
                        if stock_idx is None:
                            stock_idx = current_idx
                        field_arrays[field] = raw[1:].astype(np.float32)
                        if field == "close":
                            close_arr = field_arrays[field]
                            for pos in range(len(close_arr) - 1, -1, -1):
                                if not np.isnan(float(close_arr[pos])):
                                    last_valid_cal_idx = current_idx + pos
                                    break
                    else:
                        field_arrays[field] = np.array([], dtype=np.float32)

                if stock_idx is None:
                    if bars_dict:
                        stock_idx = min(date_to_idx[d] for d in bars_dict if d in date_to_idx)
                    else:
                        continue

                if 0 <= stock_idx < len(cal):
                    local_start_dates[sym] = cal[stock_idx]

                write_start = max(stock_idx, last_valid_cal_idx + 1)
                write_end = len(cal) - 1
                wrote_any = False

                if write_start <= write_end:
                    for cal_idx in range(write_start, write_end + 1):
                        date_str = cal[cal_idx]
                        vals = bars_dict.get(date_str, nan_vals)
                        pos = cal_idx - stock_idx
                        for i, field in enumerate(fields):
                            arr = field_arrays[field]
                            if pos >= len(arr):
                                pad = np.full(pos + 1 - len(arr), np.nan, dtype=np.float32)
                                arr = np.concatenate([arr, pad])
                            arr[pos] = np.float32(vals[i])
                            field_arrays[field] = arr
                        if date_str in bars_dict or any(not np.isnan(float(v)) for v in vals):
                            wrote_any = True

                if wrote_any or bars_dict or write_start <= write_end:
                    for field in fields:
                        bin_path = sym_dir / f"{field}.day.bin"
                        payload = np.concatenate([
                            np.array([float(stock_idx)], dtype=np.float32),
                            field_arrays[field],
                        ])
                        temporary = bin_path.with_name(
                            f".{bin_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                        )
                        try:
                            payload.tofile(str(temporary))
                            os.replace(temporary, bin_path)
                        finally:
                            if temporary.exists():
                                try:
                                    temporary.unlink()
                                except OSError:
                                    pass
                    local_updated += 1

                close_arr = field_arrays["close"]
                for pos in range(len(close_arr) - 1, -1, -1):
                    if not np.isnan(float(close_arr[pos])):
                        cal_idx = stock_idx + pos
                        if 0 <= cal_idx < len(cal):
                            local_end_dates[sym] = cal[cal_idx]
                        break

                # 定期更新进度
                with _write_lock:
                    _write_counter[0] += 1
                    if _write_counter[0] % max(1, len(update_symbols) // 100) == 0:
                        _set_sync_status(progressDone=_write_counter[0])

            _write_results.append((local_updated, local_start_dates, local_end_dates))

        num_write_workers = min(num_workers, max(1, len(update_symbols)))
        write_chunks = [update_symbols[i::num_write_workers] for i in range(num_write_workers)]
        with ThreadPoolExecutor(max_workers=num_write_workers) as pool:
            futs = [pool.submit(_write_worker, chunk) for chunk in write_chunks]
            while not all(f.done() for f in futs):
                time.sleep(0.3)
            # 工作线程完成收尾
            with _write_lock:
                if _write_counter[0] > 0:
                    _set_sync_status(progressDone=_write_counter[0])

        total_updated = sum(r[0] for r in _write_results)
        symbol_end_dates = {}
        symbol_start_dates = {}
        for _, start_dates, end_dates in _write_results:
            symbol_start_dates.update(start_dates)
            symbol_end_dates.update(end_dates)

        # 写入已完成，进入收尾阶段（重写 instruments、健康扫描、重建缓存），避免进度条卡在"写入 100%"
        _set_sync_status(
            progressPhase="finalizing", progressTotal=0, progressDone=0,
            progressLabel="正在收尾（更新覆盖范围、校验数据）...",
        )

        _update_instrument_end_dates(inst_path, symbol_end_dates)
        if symbol_start_dates:
            _update_instrument_start_dates(inst_path, symbol_start_dates)

        if data_obj:
            data_obj._calendar_cache.clear()

        final_calendar_last_date = cal[-1] if cal else calendar_last_date
        target_accepted = bool(target_sync_date and target_sync_date in accepted_new_dates)
        if target_accepted:
            latest_real_date = target_sync_date
            market_latest = target_sync_date
            equity_covered = len(equity_with_target_date)
            calendar_covered = len(equity_with_target_date) if final_calendar_last_date == target_sync_date else 0
        else:
            # A low-coverage target date is deliberately not added to the
            # calendar. Keep the previous complete snapshot until validation
            # confirms any earlier accepted dates.
            latest_real_date = max(
                accepted_new_dates,
                default=health_snapshot.get("effectiveLastDate") or last_date,
            )
            market_latest = health_snapshot.get("marketEffectiveLastDate") or latest_real_date
            equity_covered = health_snapshot.get("equityCoveredAtLastDate", 0)
            calendar_covered = health_snapshot.get("calendarCoveredEquities", 0)
        fast_health_snapshot = dict(
            health_snapshot,
            calendarLastDate=final_calendar_last_date,
            effectiveLastDate=latest_real_date,
            marketEffectiveLastDate=market_latest,
            equityCoveredAtLastDate=equity_covered,
            equityCoverageAtLastDate=(equity_covered / len(equity_symbols)) if equity_symbols else 0.0,
            calendarCoveredEquities=calendar_covered,
            calendarCoverage=(calendar_covered / len(equity_symbols)) if equity_symbols else 0.0,
            equityCount=len(equity_symbols),
        )
        stats = _build_sync_stats(
            dataDir=data_dir,
            **fast_health_snapshot,
            targetSyncDate=target_sync_date,
            acceptedNewDates=accepted_new_dates,
            rejectedNewDates=rejected_new_dates,
            totalSymbols=len(instruments),
            equitySymbols=len(equity_symbols),
            symbolsWithBars=len(symbols_with_bars),
            equitySymbolsWithBars=len(equity_with_bars),
            symbolsWithTargetDate=len(all_with_target_date),
            equitySymbolsWithTargetDate=len(equity_with_target_date),
            updatedSymbols=total_updated,
            staleSymbols=len(stale_equities) if target_sync_date else 0,
            connectionErrorCount=len(conn_failures),
            workerErrorCount=len(other_worker_errors),
            sampleTargetSymbols=all_with_target_date[:MAX_SAMPLE_SYMBOLS],
            sampleStaleSymbols=stale_equities[:MAX_SAMPLE_SYMBOLS],
            sampleErrors=worker_errors[:MAX_SAMPLE_SYMBOLS],
            invalidBarDateCount=len(invalid_bar_dates),
            sampleInvalidBarDates=invalid_bar_dates[:MAX_SAMPLE_SYMBOLS],
            partial=bool(partial_message),
        )
        latest_real_date = stats["effectiveLastDate"] or latest_real_date
        final_calendar_last_date = stats["calendarLastDate"] or final_calendar_last_date

        # Full validation still runs, but it no longer delays the completion
        # state seen by the browser.  The next status request uses the cached
        # snapshot and picks up the refreshed aggregate values when ready.
        health_key = str(Path(data_dir).expanduser().resolve())
        with _health_snapshot_lock:
            _health_snapshot_cache[health_key] = (time.monotonic(), dict(fast_health_snapshot))
        start_health_refresh(data_dir)

        elapsed = time.time() - t0
        if partial_message:
            _set_sync_status(lastSync=last_date, lastError=partial_message, lastStats=stats)
            print(
                f"Auto-sync: partial update after {elapsed:.1f}s, "
                f"updated={total_updated}, target={target_sync_date}, effective={latest_real_date}"
            )
            # 局部同步后仍然重建缓存，但不阻塞行情同步状态。
            start_cache_refresh(data_obj)
            return

        _set_sync_status(lastSync=latest_real_date, lastStats=stats)
        if total_updated > 0:
            print(f"Auto-sync: {total_updated} symbols in {elapsed:.1f}s, latest: {latest_real_date}")
        else:
            print(f"Auto-sync: no symbols updated ({elapsed:.1f}s)")
        # 股票摘要是派生数据，独立后台刷新，不延长行情同步的 running 状态。
        if total_updated > 0:
            start_cache_refresh(data_obj)

    except Exception as e:
        _set_sync_status(lastError=str(e))
        print(f"Auto-sync: error - {e}")
    finally:
        _set_sync_status(running=False, progressPhase=None, progressTotal=0, progressDone=0, progressLabel=None)
    return True


def schedule_daily_sync(data_dir, data_obj=None):
    """每个交易日 15:30 自动同步一次。"""
    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        time.sleep(wait_seconds)
        if _is_trading_day():
            print(f"Scheduled sync triggered at {now.strftime('%H:%M')}")
            auto_sync_daily(data_dir, data_obj)
