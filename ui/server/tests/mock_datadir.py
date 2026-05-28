"""Fake DataDir and FakeTDXQuote for backend API tests."""
import numpy as np


class FakeDataDir:
    """Mimics DataDir with 3 fake stocks over 30 trading days."""

    SYMBOLS = ["sh600001", "sh600002", "sh600003"]
    NAMES = {"600001": "Test Stock A", "600002": "Test Stock B", "600003": "Test Stock C"}
    DATES = [
        "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-01-09", "2025-01-10", "2025-01-13", "2025-01-14", "2025-01-15",
        "2025-01-16", "2025-01-17", "2025-01-20", "2025-01-21", "2025-01-22",
        "2025-01-23", "2025-01-24", "2025-01-27", "2025-02-05", "2025-02-06",
        "2025-02-07", "2025-02-10", "2025-02-11", "2025-02-12",
        # pad to 30
        "2025-02-13", "2025-02-14", "2025-02-17", "2025-02-18",
        "2025-02-19", "2025-02-20",
    ]

    def __init__(self, tmp_path):
        from pathlib import Path
        self.root = Path(tmp_path)
        self.data_dir = str(self.root)
        self.features_dir = self.root / "features"
        self._calendar_cache = {}
        self._build()

    # -- DataDir-compatible attributes --
    @property
    def _calendar_cache_attr(self):
        return self._calendar_cache

    # -- Build fake file tree --
    def _build(self):
        cal_dir = self.root / "calendars"
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "day.txt").write_text("\n".join(self.DATES) + "\n", encoding="utf-8")

        inst_dir = self.root / "instruments"
        inst_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for sym in self.SYMBOLS:
            lines.append(f"{sym}\t{self.DATES[0]}\t{self.DATES[-1]}")
        (inst_dir / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        name_lines = []
        for code, name in self.NAMES.items():
            name_lines.append(f"{code}\t{name}")
        (inst_dir / "names.txt").write_text("\n".join(name_lines) + "\n", encoding="utf-8")

        # Write binary feature files for each stock
        np.random.seed(42)
        n = len(self.DATES)
        for i, sym in enumerate(self.SYMBOLS):
            sym_dir = self.features_dir / sym
            sym_dir.mkdir(parents=True, exist_ok=True)
            base = 10.0 + i * 2.0
            fields = {
                "open":    base + np.random.uniform(-0.5, 0.5, n).astype(np.float32),
                "high":    base + np.random.uniform(0.0, 1.0, n).astype(np.float32),
                "low":     base + np.random.uniform(-1.0, 0.0, n).astype(np.float32),
                "close":   base + np.random.uniform(-0.5, 0.5, n).astype(np.float32),
                "volume":  np.random.randint(100000, 999999, n).astype(np.float32),
            }
            for field, vals in fields.items():
                # First value is calendar index as f4, rest are values
                idx_val = np.array([0.0], dtype=np.float32)
                data = np.concatenate([idx_val, vals])
                data.tofile(str(sym_dir / f"{field}.day.bin"))

    # -- DataDir interface --
    def read_calendar(self, freq="day"):
        if freq not in self._calendar_cache:
            path = self.root / "calendars" / f"{freq}.txt"
            if not path.exists():
                self._calendar_cache[freq] = []
            else:
                with open(path, encoding="utf-8") as f:
                    self._calendar_cache[freq] = f.read().strip().split("\n")
        return self._calendar_cache[freq]

    def read_field(self, symbol, field, freq="day"):
        d = self.features_dir / symbol.lower()
        path = d / f"{field}.{freq}.bin"
        if not path.exists():
            return None, None
        data = np.fromfile(str(path), dtype="<f4")
        if len(data) < 2:
            return None, None
        return int(data[0]), data[1:]

    def get_instruments(self):
        path = self.root / "instruments" / "all.txt"
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    result.append((parts[0], parts[1], parts[2]))
        return result

    def get_names(self):
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
            result.append({
                "date": date_str,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": v,
            })
        return result


class FakeTDXQuote:
    """Canned TDX quote responses for testing."""

    def get_quote(self, symbol):
        return {
            "price": 10.50,
            "lastClose": 10.30,
            "open": 10.35,
            "high": 10.65,
            "low": 10.20,
            "vol": 1234567,
            "amount": 12962953.5,
            "bid1": 10.49,
            "ask1": 10.51,
            "time": "15:00:00",
        }

    def get_today_kline(self, symbol):
        items = []
        for i in range(5):
            items.append({
                "date": f"2025-01-02 {9 + i}:30",
                "open": round(10.30 + i * 0.05, 2),
                "high": round(10.40 + i * 0.05, 2),
                "low": round(10.25 + i * 0.05, 2),
                "close": round(10.35 + i * 0.05, 2),
                "volume": 100000 + i * 10000,
            })
        return items
