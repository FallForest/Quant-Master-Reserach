"""日线数据增量同步：多线程从 TDX 拉取最新 K 线，批量追加到 bin 文件。"""
import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .datadir import get_effective_data_dir
from .tdx_quote import TDXQuote


PARTIAL_SYNC_COVERAGE_THRESHOLD = 0.8
MAX_SAMPLE_SYMBOLS = 8

_sync_lock = threading.Lock()
_sync_status = {"running": False, "lastSync": None, "lastError": None, "lastStats": None}


def get_sync_status():
    """返回同步状态的快照副本（线程安全）。"""
    with _sync_lock:
        return dict(_sync_status)


def _set_sync_status(**kwargs):
    """原子更新同步状态字段。"""
    with _sync_lock:
        _sync_status.update(kwargs)


def _get_last_update_date(data_dir):
    """从 calendars/day.txt 读取最后一条日期。"""
    cal_path = Path(data_dir) / "calendars" / "day.txt"
    if not cal_path.exists():
        return None
    try:
        lines = cal_path.read_text(encoding="utf-8").strip().split("\n")
        return lines[-1].strip() if lines and lines[-1].strip() else None
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
    }



def _repair_overflow_tail(data_dir, calendar=None, fields=None, dry_run=False):
    """修复/裁剪超出 calendar 的尾部槽位。"""
    root = Path(data_dir)
    if calendar is None:
        cal_path = root / "calendars" / "day.txt"
        if not cal_path.exists():
            return {
                "repairedSymbols": 0,
                "overflowedSymbols": 0,
                "trimmedSymbols": 0,
                "shiftedSymbols": 0,
                "sampleRepairedSymbols": [],
                "sampleOverflowSymbols": [],
            }
        calendar = [line.strip() for line in cal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if fields is None:
        fields = ["open", "high", "low", "close", "volume", "amount", "adjclose", "change", "factor"]

    repaired_symbols = []
    overflow_symbols = []
    trimmed_symbols = []
    shifted_symbols = []
    for close_path in (root / "features").glob("*/close.day.bin"):
        raw = np.fromfile(str(close_path), dtype="<f4")
        if len(raw) < 2:
            continue
        start_idx = int(raw[0])
        arr = raw[1:]
        overflow = start_idx + len(arr) - len(calendar)
        if overflow <= 0:
            continue
        symbol = close_path.parent.name.upper()
        overflow_symbols.append(symbol)

        symbol_trimmed = False
        symbol_shifted = False
        symbol_changed = False
        for field in fields:
            bin_path = close_path.parent / f"{field}.day.bin"
            if not bin_path.exists() or bin_path.stat().st_size < 8:
                continue
            raw_field = np.fromfile(str(bin_path), dtype="<f4")
            field_start_idx = int(raw_field[0])
            field_arr = raw_field[1:].astype(np.float32)
            field_overflow = field_start_idx + len(field_arr) - len(calendar)
            changed = False

            if field_overflow > 0:
                field_last_valid_pos = len(calendar) - 1 - field_start_idx
                if 0 <= field_last_valid_pos < len(field_arr) - 1:
                    inrange_last = float(field_arr[field_last_valid_pos])
                    first_over = float(field_arr[field_last_valid_pos + 1])
                    rest = field_arr[field_last_valid_pos + 2:]
                    if np.isnan(inrange_last) and not np.isnan(first_over) and not np.any(~np.isnan(rest)):
                        if not dry_run:
                            field_arr[field_last_valid_pos] = field_arr[field_last_valid_pos + 1]
                            field_arr[field_last_valid_pos + 1:] = np.nan
                        changed = True
                        symbol_shifted = True
                if not dry_run:
                    field_arr = field_arr[:-field_overflow]
                changed = True
                symbol_trimmed = True

            if changed:
                symbol_changed = True
                if not dry_run:
                    payload = np.concatenate([
                        np.array([float(field_start_idx)], dtype=np.float32),
                        field_arr,
                    ])
                    payload.tofile(str(bin_path))

        if symbol_changed:
            repaired_symbols.append(symbol)
        if symbol_trimmed:
            trimmed_symbols.append(symbol)
        if symbol_shifted:
            shifted_symbols.append(symbol)

    return {
        "repairedSymbols": len(repaired_symbols),
        "overflowedSymbols": len(overflow_symbols),
        "trimmedSymbols": len(trimmed_symbols),
        "shiftedSymbols": len(shifted_symbols),
        "sampleRepairedSymbols": repaired_symbols[:MAX_SAMPLE_SYMBOLS],
        "sampleOverflowSymbols": overflow_symbols[:MAX_SAMPLE_SYMBOLS],
    }



def get_data_health_snapshot(data_dir):
    """扫描本地日线落盘情况，返回日历/全量/股票 universe 的最新覆盖信息。"""
    calendar_last_date = _get_last_update_date(data_dir)
    cal_path = Path(data_dir) / "calendars" / "day.txt"
    inst_path = Path(data_dir) / "instruments" / "all.txt"

    if not cal_path.exists():
        return _empty_health_snapshot(None)

    try:
        calendar = [line.strip() for line in cal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not calendar:
            return _empty_health_snapshot(None)

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
        return snapshot
    except Exception:
        return _empty_health_snapshot(calendar_last_date)



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
        "sampleTargetSymbols": [],
        "sampleStaleSymbols": [],
        "sampleErrors": [],
        "partial": False,
    }
    stats.update(kwargs)
    return stats


def auto_sync_daily(data_dir, data_obj=None):
    """增量同步日线数据。data_obj 用于清除 calendar 缓存。"""
    if get_sync_status()["running"]:
        return
    _set_sync_status(running=True, lastError=None)
    t0 = time.time()
    try:
        data_dir = get_effective_data_dir(data_obj, data_dir)
        pre_repair_snapshot = get_data_health_snapshot(data_dir)
        repair_result = {"repairedSymbols": 0, "overflowedSymbols": 0, "sampleRepairedSymbols": [], "sampleOverflowSymbols": []}
        if pre_repair_snapshot.get("overflowedSymbolCount"):
            repair_result = _repair_overflow_tail(data_dir)
            if repair_result.get("repairedSymbols"):
                print(
                    f"Auto-sync: repaired overflow tail for {repair_result['repairedSymbols']} symbols "
                    f"before sync"
                )
        health_snapshot = get_data_health_snapshot(data_dir)
        last_date = health_snapshot["effectiveLastDate"]
        calendar_last_date = health_snapshot["calendarLastDate"]
        today = datetime.date.today().strftime("%Y-%m-%d")
        now_hour = datetime.datetime.now().hour
        base_stats = _build_sync_stats(
            dataDir=data_dir,
            **health_snapshot,
        )

        if last_date and last_date >= today:
            print(f"Auto-sync: data already up to date ({last_date})")
            _set_sync_status(lastSync=last_date, lastStats=base_stats)
            return

        if not _is_trading_day():
            print(f"Auto-sync: today ({today}) is weekend, skipping")
            _set_sync_status(lastSync=last_date, lastStats=base_stats)
            return

        if now_hour < 15 and last_date and last_date >= (
            datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"):
            print(f"Auto-sync: market not closed yet ({now_hour}:00), skipping")
            _set_sync_status(lastSync=last_date, lastStats=base_stats)
            return

        num_bars = _bars_to_fetch(last_date)
        print(f"Auto-sync: fetching {num_bars} daily bars from TDX (gap since {last_date}) ...")
        data_root = Path(data_dir)
        features_dir = data_root / "features"
        cal_path = data_root / "calendars" / "day.txt"
        inst_path = data_root / "instruments" / "all.txt"

        instruments = []
        if inst_path.exists():
            with open(inst_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        instruments.append(parts[0])
        if not instruments:
            print("Auto-sync: no instruments found")
            _set_sync_status(lastError="no instruments", lastStats=base_stats)
            return

        equity_symbols = [sym for sym in instruments if _is_equity_symbol(sym)]

        cal = []
        if cal_path.exists():
            with open(cal_path, encoding="utf-8") as f:
                cal = [line.strip() for line in f.read().split("\n") if line.strip()]

        print(f"Auto-sync: fetching {len(instruments)} symbols from TDX ({len(equity_symbols)} equities) ...")

        NUM_WORKERS = 8
        fields = ["open", "high", "low", "close", "volume", "amount", "adjclose", "change", "factor"]

        def _connect():
            from pytdx.hq import TdxHq_API
            for ip, port in TDXQuote.TDX_HOSTS:
                try:
                    a = TdxHq_API()
                    a.connect(ip, port, time_out=5)
                    return a
                except Exception:
                    continue
            return None

        def _worker(sym_chunk, results_buf, error_buf):
            api = _connect()
            if api is None:
                error_buf.append("TDX connection failed: all hosts unreachable")
                return
            try:
                for sym in sym_chunk:
                    s = sym.upper()
                    if s.startswith("SZ") or s.startswith("BJ"):
                        mkt, code = 0, s[2:]
                    elif s.startswith("SH"):
                        mkt, code = 1, s[2:]
                    else:
                        continue
                    try:
                        bars = api.get_security_bars(9, mkt, code, 0, num_bars)
                        if not bars:
                            continue
                        for idx, bar in enumerate(bars):
                            bar_date = str(bar.get("datetime", ""))[:10]
                            if not bar_date:
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
                        continue
            finally:
                try:
                    api.disconnect()
                except Exception:
                    pass

        chunks = [instruments[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
        all_results = []
        all_errors = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futs = []
            for chunk in chunks:
                buf = []
                err_buf = []
                all_results.append(buf)
                all_errors.append(err_buf)
                futs.append(pool.submit(_worker, chunk, buf, err_buf))
            for f in futs:
                f.result(timeout=300)

        worker_errors = []
        for err_buf in all_errors:
            worker_errors.extend(err_buf)
        conn_failures = [e for e in worker_errors if "connection failed" in e.lower()]
        other_worker_errors = [e for e in worker_errors if "connection failed" not in e.lower()]

        if conn_failures:
            _set_sync_status(
                lastError="TDX connection failed",
                lastStats=_build_sync_stats(
                    **base_stats,
                    totalSymbols=len(instruments),
                    equitySymbols=len(equity_symbols),
                    connectionErrorCount=len(conn_failures),
                    workerErrorCount=len(other_worker_errors),
                    sampleErrors=worker_errors[:MAX_SAMPLE_SYMBOLS],
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
                **base_stats,
                totalSymbols=len(instruments),
                equitySymbols=len(equity_symbols),
                symbolsWithBars=0,
                equitySymbolsWithBars=0,
                connectionErrorCount=len(conn_failures),
                workerErrorCount=len(other_worker_errors),
                sampleErrors=worker_errors[:MAX_SAMPLE_SYMBOLS],
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
            if coverage_ratio < PARTIAL_SYNC_COVERAGE_THRESHOLD:
                accepted_new_dates = [d for d in new_date_list if d < target_sync_date]
                rejected_new_dates = [target_sync_date]
                partial_message = (
                    f"partial sync: equity coverage {len(equity_with_target_date)}/{len(equity_symbols)} "
                    f"for {target_sync_date} below threshold {PARTIAL_SYNC_COVERAGE_THRESHOLD:.0%}"
                )
                print(f"Auto-sync: {partial_message}")

        if accepted_new_dates:
            cal.extend(accepted_new_dates)
            cal_path.write_text("\n".join(cal) + "\n", encoding="utf-8")

        date_to_idx = {date_str: idx for idx, date_str in enumerate(cal)}
        accepted_date_set = set(cal)
        nan_vals = tuple([float("nan")] * len(fields))
        total_updated = 0
        symbol_end_dates = {}

        for sym in instruments:
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

            if wrote_any or bars_dict:
                for field in fields:
                    bin_path = sym_dir / f"{field}.day.bin"
                    payload = np.concatenate([
                        np.array([float(stock_idx)], dtype=np.float32),
                        field_arrays[field],
                    ])
                    payload.tofile(str(bin_path))
                total_updated += 1

            close_arr = field_arrays["close"]
            for pos in range(len(close_arr) - 1, -1, -1):
                if not np.isnan(float(close_arr[pos])):
                    cal_idx = stock_idx + pos
                    if 0 <= cal_idx < len(cal):
                        symbol_end_dates[sym] = cal[cal_idx]
                    break

        if inst_path.exists() and symbol_end_dates:
            lines = inst_path.read_text(encoding="utf-8").strip().split("\n")
            new_lines = []
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0] in symbol_end_dates:
                    parts[2] = symbol_end_dates[parts[0]]
                new_lines.append("\t".join(parts))
            inst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        if data_obj:
            data_obj._calendar_cache.clear()

        latest_real_date = max(symbol_end_dates.values(), default=last_date)
        final_calendar_last_date = cal[-1] if cal else calendar_last_date
        final_health_snapshot = get_data_health_snapshot(data_dir)
        stats = _build_sync_stats(
            dataDir=data_dir,
            **final_health_snapshot,
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
            partial=bool(partial_message),
        )
        latest_real_date = stats["effectiveLastDate"] or latest_real_date
        final_calendar_last_date = stats["calendarLastDate"] or final_calendar_last_date

        elapsed = time.time() - t0
        if partial_message:
            _set_sync_status(lastSync=last_date, lastError=partial_message, lastStats=stats)
            print(
                f"Auto-sync: partial update after {elapsed:.1f}s, "
                f"updated={total_updated}, target={target_sync_date}, effective={latest_real_date}"
            )
            return

        _set_sync_status(lastSync=latest_real_date, lastStats=stats)
        if total_updated > 0:
            print(f"Auto-sync: {total_updated} symbols in {elapsed:.1f}s, latest: {latest_real_date}")
        else:
            print(f"Auto-sync: no symbols updated ({elapsed:.1f}s)")

    except Exception as e:
        _set_sync_status(lastError=str(e))
        print(f"Auto-sync: error - {e}")
    finally:
        _set_sync_status(running=False)


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
