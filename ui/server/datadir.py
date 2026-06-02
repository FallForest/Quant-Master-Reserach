"""Helpers for reading binary market data from the QuantMaster data directory."""

import logging
import warnings
from bisect import bisect_left
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("~/.quant_master/quant_master_data/tdx_cn_data")


def resolve_data_dir(data_dir: str | None = None) -> Path:
    candidate = Path(data_dir).expanduser() if data_dir else DEFAULT_DATA_DIR.expanduser()
    return candidate.resolve()


def get_effective_data_dir(data_dir_obj=None, data_dir: str | None = None) -> str:
    if data_dir_obj is not None and getattr(data_dir_obj, "data_dir", None):
        return str(resolve_data_dir(data_dir_obj.data_dir))
    return str(resolve_data_dir(data_dir))


def create_data_dir(data_dir: str | None = None):
    return DataDir(str(resolve_data_dir(data_dir)))


def ensure_data_dir(data_dir_obj=None, data_dir: str | None = None):
    if data_dir_obj is not None and getattr(data_dir_obj, "data_dir", None):
        if data_dir is None:
            return data_dir_obj
        if resolve_data_dir(data_dir_obj.data_dir) == resolve_data_dir(data_dir):
            return data_dir_obj
    return create_data_dir(data_dir)


def get_trading_calendar(data_dir_obj=None, freq="day", data_dir: str | None = None):
    data = ensure_data_dir(data_dir_obj, data_dir)
    return data.read_calendar(freq)


def describe_trading_day(target_date: str, trading_dates: list[str]):
    if not trading_dates:
        return {
            "ok": False,
            "requested": target_date,
            "latest": None,
            "previous": None,
            "next": None,
            "message": "No trading dates are available in the current data directory. Please sync data first.",
        }

    idx = bisect_left(trading_dates, target_date)
    is_exact = idx < len(trading_dates) and trading_dates[idx] == target_date
    previous_date = trading_dates[idx - 1] if idx > 0 else None
    next_date = trading_dates[idx] if idx < len(trading_dates) else None
    latest_date = trading_dates[-1]

    if is_exact:
        return {
            "ok": True,
            "requested": target_date,
            "latest": latest_date,
            "previous": previous_date,
            "next": trading_dates[idx + 1] if idx + 1 < len(trading_dates) else None,
            "message": "",
        }

    if target_date > latest_date:
        message = f"Target date {target_date} is beyond the current data range. Latest available trading day is {latest_date}."
    elif next_date and previous_date:
        message = (
            f"Target date {target_date} is not a supported trading day in the current data range. "
            f"Previous trading day is {previous_date}, next trading day is {next_date}."
        )
    elif next_date:
        message = (
            f"Target date {target_date} is earlier than the start of the current data range. "
            f"Earliest available trading day is {next_date}."
        )
    else:
        message = f"Target date {target_date} is not a supported trading day. Latest available trading day is {latest_date}."

    return {
        "ok": False,
        "requested": target_date,
        "latest": latest_date,
        "previous": previous_date,
        "next": next_date,
        "message": message,
    }


class DataDir:
    def __init__(self, data_dir: str):
        self.data_dir = str(resolve_data_dir(data_dir))
        self.root = Path(self.data_dir)
        self.features_dir = self.root / "features"
        self._calendar_cache = {}

    def read_calendar(self, freq="day"):
        if freq not in self._calendar_cache:
            path = self.root / "calendars" / f"{freq}.txt"
            if not path.exists():
                self._calendar_cache[freq] = []
            else:
                with open(path, encoding="utf-8") as f:
                    self._calendar_cache[freq] = f.read().strip().split("\n")
        return self._calendar_cache[freq]

    def _sym_to_dir(self, symbol):
        return self.features_dir / symbol.lower()

    def read_field(self, symbol, field, freq="day"):
        """Read one binary field file and return ``(date_index, values_array)``."""
        d = self._sym_to_dir(symbol)
        path = d / f"{field}.{freq}.bin"
        if not path.exists():
            if freq == "day":
                path = d / f"{field}.1min.bin"
                if path.exists():
                    _log.warning("Fallback: %s.day.bin not found for %s, using 1min data", field, symbol)
                    return self._read_bin(path)
            return None, None
        return self._read_bin(path)

    @staticmethod
    def _read_bin(path):
        data = np.fromfile(str(path), dtype="<f4")
        if len(data) < 2:
            return None, None
        return int(data[0]), data[1:]

    def get_instruments(self):
        """Read ``instruments/all.txt`` and return ``[(symbol, start, end), ...]``."""
        path = self.root / "instruments" / "all.txt"
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    result.append((parts[0], parts[1], parts[2]))
        return result

    def get_names(self):
        """Read ``instruments/names.txt`` and return ``{code: name}``."""
        path = self.root / "instruments" / "names.txt"
        mapping = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        mapping[parts[0]] = parts[1]
        return mapping

    def get_kline(self, symbol, freq="day", start=None, end=None):
        """Read K-line data and return ``[{date, open, high, low, close, volume}, ...]``."""
        cal = self.read_calendar(freq)
        if not cal:
            return []

        open_idx, open_vals = self.read_field(symbol, "open", freq)
        if open_idx is None:
            return []

        _, close_vals = self.read_field(symbol, "close", freq)
        _, high_vals = self.read_field(symbol, "high", freq)
        _, low_vals = self.read_field(symbol, "low", freq)
        _, vol_vals = self.read_field(symbol, "volume", freq)

        n = min(len(open_vals), len(close_vals), len(high_vals), len(low_vals))
        if vol_vals is not None:
            n = min(n, len(vol_vals))
        n = min(n, len(cal) - open_idx)

        result = []
        for i in range(n):
            c = float(close_vals[i])
            if np.isnan(c):
                continue
            o = float(open_vals[i])
            h = float(high_vals[i])
            l = float(low_vals[i])
            if np.isnan(o) or np.isnan(h) or np.isnan(l):
                continue

            date_str = cal[open_idx + i] if (open_idx + i) < len(cal) else ""
            if not date_str:
                continue
            if start and date_str < start:
                continue
            if end and date_str > end:
                continue

            v = int(vol_vals[i]) if vol_vals is not None and not np.isnan(vol_vals[i]) else 0
            item = {
                "date": date_str,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": v,
            }
            result.append(item)

        return result
