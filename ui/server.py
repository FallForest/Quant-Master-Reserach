"""轻量 API 服务器，为 Browser 页面提供真实行情数据。

用法:
    python server.py [--data_dir ~/.quant_master/quant_master_data/tdx_cn_data]

默认端口 5174，Vite dev server 通过 proxy 转发 /api 请求。
实时行情通过 pytdx 连接银河证券行情服务器。
"""
import argparse
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np


class DataDir:
    """封装 quant_master 数据目录的读取操作。"""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir).expanduser().resolve()
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
            # fallback: 尝试 1min
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
        # 确保不超出日历范围
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


# 全局数据目录实例（在 main 中初始化）
data: DataDir = None


class TDXQuote:
    """通过 pytdx 连接银河证券行情服务器，提供实时行情。"""

    TDX_HOSTS = [
        ("120.76.1.198", 7709),
        ("123.125.108.101", 7709),
        ("1.202.143.37", 7709),
        ("114.141.177.118", 7709),
    ]

    def __init__(self):
        self._api = None
        self._lock = threading.Lock()

    def _ensure_connected(self):
        if self._api is not None:
            return True
        from pytdx.hq import TdxHq_API
        for ip, port in self.TDX_HOSTS:
            try:
                api = TdxHq_API()
                api.connect(ip, port)
                self._api = api
                print(f"TDX connected: {ip}:{port}")
                return True
            except Exception:
                continue
        return False

    def _parse_symbol(self, symbol):
        """SZ000676 -> (0, '000676'), SH600519 -> (1, '600519')"""
        s = symbol.upper()
        if s.startswith("SZ") or s.startswith("BJ"):
            return 0, s[2:]
        if s.startswith("SH"):
            return 1, s[2:]
        # 纯数字：根据首位判断
        if s.startswith("6"):
            return 1, s
        return 0, s

    def get_quote(self, symbol):
        """获取实时行情。"""
        with self._lock:
            if not self._ensure_connected():
                return None
            try:
                market, code = self._parse_symbol(symbol)
                data = self._api.get_security_quotes([(market, code)])
                if data and len(data) > 0:
                    d = data[0]
                    return {
                        "price": d.get("price", 0),
                        "lastClose": d.get("last_close", 0),
                        "open": d.get("open", 0),
                        "high": d.get("high", 0),
                        "low": d.get("low", 0),
                        "vol": d.get("vol", 0),
                        "amount": d.get("amount", 0),
                        "bid1": d.get("bid1", 0),
                        "ask1": d.get("ask1", 0),
                        "time": d.get("servertime", ""),
                    }
            except Exception as e:
                self._api = None  # 标记需要重连
            return None

    def get_today_kline(self, symbol):
        """获取当天 1min K 线（240 根）。"""
        with self._lock:
            if not self._ensure_connected():
                return []
            try:
                market, code = self._parse_symbol(symbol)
                # freq=7: 1min, TDX 服务器单次最多返回 240 根（刚好一天）
                bars = self._api.get_security_bars(7, market, code, 0, 240)
                if not bars:
                    return []
                result = []
                for b in bars:
                    dt = str(b.get("datetime", ""))
                    date_str = dt[:16] if len(dt) >= 16 else dt
                    result.append({
                        "date": date_str,
                        "open": round(float(b["open"]), 2),
                        "high": round(float(b["high"]), 2),
                        "low": round(float(b["low"]), 2),
                        "close": round(float(b["close"]), 2),
                        "volume": int(float(b["vol"])),
                    })
                return result
            except Exception:
                self._api = None
                return []


# 全局实时行情实例
tdx_quote: TDXQuote = None


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 静默日志
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Content-Type", "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/api/browser/stocks":
                return self._handle_stocks()
            elif path.startswith("/api/browser/kline/"):
                symbol = path.split("/api/browser/kline/")[1]
                freq = params.get("freq", ["day"])[0]
                if freq == "1d":
                    freq = "day"
                start = params.get("start", [None])[0]
                end = params.get("end", [None])[0]
                return self._handle_kline(symbol, freq, start, end)
            elif path.startswith("/api/realtime/quote/"):
                symbol = path.split("/api/realtime/quote/")[1]
                return self._handle_realtime_quote(symbol)
            elif path.startswith("/api/realtime/kline/"):
                symbol = path.split("/api/realtime/kline/")[1]
                return self._handle_realtime_kline(symbol)
            elif path == "/api/pipeline/status":
                return self._json_response({"lastUpdate": "--"})
            else:
                self.send_error(404)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def _handle_stocks(self):
        names = data.get_names()
        instruments = data.get_instruments()
        stocks = []
        for sym, start, end in instruments:
            code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
            name = names.get(code6, "")

            close_idx, close_vals = data.read_field(sym, "close", "day")
            item = {"symbol": sym, "name": name, "startDate": start, "endDate": end}
            if close_vals is not None and len(close_vals) > 0:
                # 从后往前找第一个非 NaN 的 close
                c = None
                for j in range(len(close_vals) - 1, -1, -1):
                    v = float(close_vals[j])
                    if not np.isnan(v):
                        c = round(v, 2)
                        break
                if c is not None:
                    item["close"] = c
                    # 找倒数第二个非 NaN 算涨跌
                    for j in range((len(close_vals) - 2) if c is not None else len(close_vals) - 1, -1, -1):
                        prev = float(close_vals[j])
                        if not np.isnan(prev) and prev != 0:
                            item["change"] = round(c - prev, 2)
                            item["changePct"] = round((c - prev) / prev * 100, 2)
                            break
            vol_idx, vol_vals = data.read_field(sym, "volume", "day")
            if vol_vals is not None and len(vol_vals) > 0:
                for j in range(len(vol_vals) - 1, -1, -1):
                    v = float(vol_vals[j])
                    if not np.isnan(v):
                        item["volume"] = int(v)
                        break
            stocks.append(item)
        self._json_response({"stocks": stocks})

    def _handle_kline(self, symbol, freq, start, end):
        kline = data.get_kline(symbol, freq, start, end)
        if not kline:
            self._json_response({"kline": [], "quote": {}})
            return
        last = kline[-1]
        quote = {"close": last["close"], "change": 0, "changePct": "0"}
        if len(kline) > 1:
            prev = kline[-2]["close"]
            if prev != 0:
                quote["change"] = round(last["close"] - prev, 2)
                quote["changePct"] = round((last["close"] - prev) / prev * 100, 2)
        self._json_response({"kline": kline, "quote": quote})

    def _handle_realtime_quote(self, symbol):
        q = tdx_quote.get_quote(symbol)
        if q:
            self._json_response({"ok": True, "quote": q})
        else:
            self._json_response({"ok": False, "quote": {}})

    def _handle_realtime_kline(self, symbol):
        kline = tdx_quote.get_today_kline(symbol)
        if kline:
            last = kline[-1]
            quote = {"close": last["close"], "change": 0, "changePct": "0"}
            if len(kline) > 1:
                prev = kline[-2]["close"]
                if prev != 0:
                    quote["change"] = round(last["close"] - prev, 2)
                    quote["changePct"] = round((last["close"] - prev) / prev * 100, 2)
            self._json_response({"kline": kline, "quote": quote})
        else:
            self._json_response({"kline": [], "quote": {}})

    def _json_response(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global data, tdx_quote
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.expanduser(
        "~/.quant_master/quant_master_data/tdx_cn_data"))
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()

    data = DataDir(args.data_dir)
    tdx_quote = TDXQuote()
    server = HTTPServer(("127.0.0.1", args.port), APIHandler)
    print(f"API server running on http://127.0.0.1:{args.port}")
    print(f"Data dir: {args.data_dir}")
    print(f"Real-time quote: TDX (银河证券)")
    server.serve_forever()


if __name__ == "__main__":
    main()
