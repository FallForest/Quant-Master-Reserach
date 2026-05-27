"""通过 pytdx 连接通达信行情服务器，提供实时行情。

每次请求都新建连接、用完就断，避免持久连接被服务器踢掉。
"""
import time


class TDXQuote:
    TDX_HOSTS = [
        ("120.76.1.198", 7709),
        ("123.125.108.101", 7709),
        ("1.202.143.37", 7709),
        ("114.141.177.118", 7709),
    ]

    def _connect(self):
        """返回一个已连接的 TdxHq_API 实例，或 None。"""
        from pytdx.hq import TdxHq_API
        for ip, port in self.TDX_HOSTS:
            try:
                api = TdxHq_API()
                api.connect(ip, port, time_out=5)
                return api
            except Exception:
                continue
        return None

    @staticmethod
    def _disconnect(api):
        try:
            api.disconnect()
        except Exception:
            pass

    def _parse_symbol(self, symbol):
        """SZ000676 -> (0, '000676'), SH600519 -> (1, '600519')"""
        s = symbol.upper()
        if s.startswith("SZ") or s.startswith("BJ"):
            return 0, s[2:]
        if s.startswith("SH"):
            return 1, s[2:]
        if s.startswith("6"):
            return 1, s
        return 0, s

    def get_quote(self, symbol):
        """获取实时行情。"""
        api = self._connect()
        if api is None:
            return None
        try:
            market, code = self._parse_symbol(symbol)
            data = api.get_security_quotes([(market, code)])
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
        except Exception:
            pass
        finally:
            self._disconnect(api)
        return None

    def get_today_kline(self, symbol):
        """获取当天 1min K 线。"""
        api = self._connect()
        if api is None:
            return []
        try:
            market, code = self._parse_symbol(symbol)
            bars = api.get_security_bars(7, market, code, 0, 240)
            if not bars:
                return []
            last_dt = str(bars[-1].get("datetime", ""))
            today = last_dt[:10] if len(last_dt) >= 10 else ""
            if not today:
                return []
            result = []
            for b in bars:
                dt = str(b.get("datetime", ""))
                if not dt.startswith(today):
                    continue
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
            return []
        finally:
            self._disconnect(api)
