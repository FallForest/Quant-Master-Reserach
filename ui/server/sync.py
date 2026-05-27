"""日线数据增量同步：多线程从 TDX 拉取最新 K 线，批量追加到 bin 文件。"""
import datetime
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .tdx_quote import TDXQuote


_sync_status = {"running": False, "lastSync": None, "lastError": None}


def get_sync_status():
    return _sync_status


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
        # 每个日历日约 1 个 bar，取 1.5 倍余量再加 2
        need = max(min_bars, int(gap * 1.5) + 2)
        return min(need, max_bars)
    except Exception:
        return max_bars


def auto_sync_daily(data_dir, data_obj=None):
    """增量同步日线数据。data_obj 用于清除 calendar 缓存。"""
    if _sync_status["running"]:
        return
    _sync_status["running"] = True
    _sync_status["lastError"] = None
    t0 = time.time()
    try:
        last_date = _get_last_update_date(data_dir)
        today = datetime.date.today().strftime("%Y-%m-%d")
        now_hour = datetime.datetime.now().hour

        if last_date and last_date >= today:
            print(f"Auto-sync: data already up to date ({last_date})")
            _sync_status["lastSync"] = last_date
            return

        if not _is_trading_day():
            print(f"Auto-sync: today ({today}) is weekend, skipping")
            _sync_status["lastSync"] = last_date
            return

        if now_hour < 15 and last_date and last_date >= (
            datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"):
            print(f"Auto-sync: market not closed yet ({now_hour}:00), skipping")
            _sync_status["lastSync"] = last_date
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
            _sync_status["lastError"] = "no instruments"
            return

        cal = []
        if cal_path.exists():
            with open(cal_path, encoding="utf-8") as f:
                cal = f.read().strip().split("\n")
        date_idx = len(cal)

        print(f"Auto-sync: fetching {len(instruments)} stocks from TDX ...")

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
                        if not bars or len(bars) < 1:
                            continue
                        # 返回所有 bar，不只是最后一个
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
                                cl,      # adjclose
                                chg,     # change
                                1.0,     # factor
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

        # 检查是否有连接错误
        worker_errors = []
        for err_buf in all_errors:
            worker_errors.extend(err_buf)
        if worker_errors:
            conn_failures = [e for e in worker_errors if "connection failed" in e.lower()]
            if conn_failures:
                _sync_status["lastError"] = "TDX connection failed"
                print(f"Auto-sync: TDX connection failed, all hosts unreachable")
                return
            # 非连接错误只打印前 5 条
            if len(worker_errors) > 5:
                print(f"Auto-sync: {len(worker_errors)} worker errors (showing first 5):")
                for e in worker_errors[:5]:
                    print(f"  - {e}")
            else:
                for e in worker_errors:
                    print(f"Auto-sync: worker error - {e}")

        # 第一遍：收集所有 bar 日期，找出新日期
        all_bar_dates = set()
        cal_set = set(cal)
        for buf in all_results:
            for sym, bar_date, vals in buf:
                all_bar_dates.add(bar_date)
        new_date_list = sorted(d for d in all_bar_dates if d not in cal_set)

        if not new_date_list:
            elapsed = time.time() - t0
            _sync_status["lastSync"] = last_date
            print(f"Auto-sync: no new data ({elapsed:.1f}s)")
            return

        # 先扩展日历
        for d in new_date_list:
            cal.append(d)
        cal_path.write_text("\n".join(cal) + "\n", encoding="utf-8")

        # 第二遍：按股票聚合 bar 数据，写入二进制文件
        # 先按股票聚合
        stock_bars = {}  # sym -> {date: vals}
        for buf in all_results:
            for sym, bar_date, vals in buf:
                if sym not in stock_bars:
                    stock_bars[sym] = {}
                stock_bars[sym][bar_date] = vals

        nan_vals = tuple([float('nan')] * len(fields))
        total_updated = 0

        for sym, bars_dict in stock_bars.items():
            fname = sym.lower()
            sym_dir = features_dir / fname
            sym_dir.mkdir(parents=True, exist_ok=True)

            # 确定该股票的现有数据范围
            close_bin = sym_dir / "close.day.bin"
            stock_idx = date_idx
            existing_n = 0
            if close_bin.exists() and close_bin.stat().st_size >= 8:
                raw = np.fromfile(str(close_bin), dtype="<f4")
                stock_idx = int(raw[0])
                existing_n = len(raw) - 1

            existing_last_cal_idx = stock_idx + existing_n - 1 if existing_n > 0 else -1

            # 确定需要写入的日历位置范围
            # 从已有数据的下一个位置到新日历的末尾
            write_start = existing_last_cal_idx + 1
            write_end = len(cal) - 1

            if write_start > write_end:
                continue

            wrote_any = False
            for cal_idx in range(write_start, write_end + 1):
                date_str = cal[cal_idx]
                if date_str in bars_dict:
                    vals = bars_dict[date_str]
                else:
                    vals = nan_vals

                for i, field in enumerate(fields):
                    bin_path = sym_dir / f"{field}.day.bin"
                    raw = struct.pack("<f", vals[i])
                    if bin_path.exists() and bin_path.stat().st_size >= 8:
                        with open(bin_path, "ab") as fp:
                            fp.write(raw)
                    else:
                        with open(bin_path, "wb") as fp:
                            fp.write(struct.pack("<f", float(stock_idx)))
                            fp.write(raw)
                wrote_any = True

            if wrote_any:
                total_updated += 1

        if total_updated > 0:
            new_date = new_date_list[-1]

            # 更新 instruments（数据已确认写入）
            if inst_path.exists():
                lines = inst_path.read_text(encoding="utf-8").strip().split("\n")
                new_lines = []
                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3 and parts[0] in instruments:
                        parts[2] = new_date
                    new_lines.append("\t".join(parts))
                inst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

            if data_obj:
                data_obj._calendar_cache.clear()

            elapsed = time.time() - t0
            _sync_status["lastSync"] = new_date
            print(f"Auto-sync: {total_updated} stocks in {elapsed:.1f}s, latest: {new_date}")
        else:
            elapsed = time.time() - t0
            _sync_status["lastSync"] = last_date
            print(f"Auto-sync: no stocks updated ({elapsed:.1f}s)")

    except Exception as e:
        _sync_status["lastError"] = str(e)
        print(f"Auto-sync: error - {e}")
    finally:
        _sync_status["running"] = False


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
