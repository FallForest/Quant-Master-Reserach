"""封装 quant_master 数据目录的二进制文件读取操作。"""
from pathlib import Path

import numpy as np


class DataDir:
    def __init__(self, data_dir: str):
        self.data_dir = str(Path(data_dir).expanduser().resolve())
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
        """读取某个字段的 bin 文件，返回 (date_index, values_array)。"""
        d = self._sym_to_dir(symbol)
        path = d / f"{field}.{freq}.bin"
        if not path.exists():
            if freq == "day":
                path = d / f"{field}.1min.bin"
                if path.exists():
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
        """读取 instruments/all.txt，返回 [(symbol, start, end), ...]。"""
        path = self.root / "instruments" / "all.txt"
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    result.append((parts[0], parts[1], parts[2]))
        return result

    def get_names(self):
        """读取 instruments/names.txt，返回 {code: name}。"""
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
        """读取 K 线数据，返回 [{date, open, high, low, close, volume}, ...]。"""
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
