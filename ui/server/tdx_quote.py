"""通过 pytdx 连接通达信行情服务器，提供实时行情。

使用持久连接避免每次请求都 TCP 握手（~100-300ms 延迟）。
连接断开时自动重连。
"""
import logging
import threading
import time
import urllib.parse
import urllib.request

_log = logging.getLogger(__name__)


def _probe_host(api):
    """探针：确认这台服务器真的会返回行情数据。

    部分 TDX 服务器能建立连接、也能返回股票列表，但对行情/日线请求一律返回空，
    只凭 connect() 成功无法区分二者。用一次最小的日线请求判断。
    """
    try:
        return bool(api.get_security_bars(9, 0, "000001", 0, 1))
    except Exception:
        return False


def connect_first_working(hosts, time_out=2):
    """按顺序连接并校验，返回第一个真正能吐数据的 api；全部不可用返回 None。"""
    from pytdx.hq import TdxHq_API

    for ip, port in hosts:
        try:
            api = TdxHq_API()
            # pytdx 连接失败时 connect() 返回 False 而非抛异常。
            if not api.connect(ip, port, time_out=time_out):
                _log.warning("TDX 连接失败: %s:%s", ip, port)
                continue
        except Exception:
            continue
        if not _probe_host(api):
            _log.warning("TDX 主机无数据响应，跳过: %s:%s", ip, port)
            try:
                api.disconnect()
            except Exception:
                pass
            continue
        _log.info("TDX 持久连接已建立: %s:%s", ip, port)
        return api
    return None


class TDXQuote:
    TDX_HOSTS = [
        ("120.76.1.198", 7709),
        ("114.141.177.118", 7709),
        ("123.125.108.101", 7709),
        ("1.202.143.37", 7709),
    ]

    # 通达信服务器会主动断开长时间空闲的连接（通常 60s 左右）。
    # 若继续复用这条已被服务端断开的 socket，pytdx 会静默返回 None，
    # 导致实时行情/分钟线取不到数据且不触发重连。据此定期重建连接。
    STALE_IDLE_SECONDS = 30

    def __init__(self):
        self._api = None
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()  # 串行化所有 TDX 调用，防止 TCP 响应交叉
        self._last_connect_attempt = 0
        self._connect_cooldown = 2  # 连接失败冷却秒数
        self._last_used = 0.0  # 最近一次成功使用（实际 TDX 调用）的时间

    def _get_api(self):
        """获取持久连接，不存在/已空闲过久则重新创建。线程安全。"""
        if self._api is not None:
            # 上次实际使用距今过久——服务端可能已断开该连接，主动重建以防复用死 socket
            if time.time() - self._last_used > self.STALE_IDLE_SECONDS:
                _log.info("TDX 连接空闲超过 %ds，重建持久连接", self.STALE_IDLE_SECONDS)
                self._invalidate()
                return self._connect_fresh()
            return self._api
        # 冷却期内不重连（避免短时间内反复尝试连接）
        if time.time() - self._last_connect_attempt < self._connect_cooldown:
            return None
        return self._connect_fresh()

    def _connect_fresh(self):
        """强制创建新连接（调用方需确保必要时才调用）。"""
        with self._lock:
            if self._api is not None:
                return self._api
            self._last_connect_attempt = time.time()
            api = connect_first_working(self.TDX_HOSTS, time_out=2)
            if api is None:
                _log.warning("TDX 连接失败，%ds 后可重试", self._connect_cooldown)
                return None
            self._api = api
            self._last_used = time.time()
            return api

    def _invalidate(self):
        """标记连接失效，下次调用时自动重连。"""
        with self._lock:
            if self._api is not None:
                try:
                    self._api.disconnect()
                except Exception:
                    pass
                self._api = None

    def _retry(self, fn, default=None):
        """执行 fn(api)，失败时自动重连并重试。

        Args:
            fn: 接受 api 对象的 callable，返回结果。
            default: 最终失败时返回的默认值。
        Returns:
            fn(api) 的返回值，失败时返回 default。
        """
        api = self._get_api()
        if api is None:
            # 冷却期内拿不到连接，跳过冷却强制重连
            api = self._connect_fresh()
        if api is None:
            return default
        for attempt in range(3):
            try:
                result = fn(api)
                self._last_used = time.time()
                return result
            except Exception:
                self._invalidate()
                api = self._connect_fresh()
                if api is None:
                    return default
        return default

    # 保留旧接口供 browser.py 的 _fetch_tdx_quotes 兼容
    def _connect(self):
        return self._get_api()

    @staticmethod
    def _disconnect(api):
        # 不再断开——持久连接由 _invalidate 管理
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

    def _normalize_symbol_key(self, symbol):
        market, code = self._parse_symbol(symbol)
        return ("SH" if market == 1 else "SZ") + code

    def _eastmoney_secid(self, symbol):
        market, code = self._parse_symbol(symbol)
        return f"{1 if market == 1 else 0}.{code}"

    def _fetch_eastmoney_quotes(self, symbols):
        """TDX 不可用时，用东方财富未复权实时行情兜底。"""
        if not symbols:
            return {}
        secids = ",".join(self._eastmoney_secid(sym) for sym in symbols)
        params = urllib.parse.urlencode({
            "fltt": "2",
            "invt": "2",
            "secids": secids,
            "fields": "f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18",
        })
        url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?{params}"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                import json
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            _log.warning("EastMoney quote fallback failed", exc_info=True)
            return {}

        rows = payload.get("data", {}).get("diff", []) if isinstance(payload, dict) else []
        result = {}
        for row in rows:
            code = str(row.get("f12") or "").zfill(6)
            if not code:
                continue
            prefix = "SH" if code.startswith("6") else "SZ"
            price = row.get("f2")
            last_close = row.get("f18")
            try:
                price = float(price)
                last_close = float(last_close or 0)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            result[prefix + code] = {
                "price": price,
                "lastClose": last_close,
                "open": row.get("f17") or 0,
                "high": row.get("f15") or 0,
                "low": row.get("f16") or 0,
                "vol": row.get("f5") or 0,
                "amount": row.get("f6") or 0,
                "source": "eastmoney",
            }
        return result

    def fetch_quotes(self, symbols):
        """批量获取实时行情，返回 {symbol: {price, lastClose, ...}}。线程安全。"""
        pairs = []
        for sym in symbols:
            market, code = self._parse_symbol(sym)
            pairs.append((market, code))
        if not pairs:
            return {}

        def _do_fetch(api):
            result = {}
            # pytdx 每批最多约 80 只
            for i in range(0, len(pairs), 80):
                batch = pairs[i:i + 80]
                data = api.get_security_quotes(batch)
                if data:
                    for d in data:
                        code6 = d.get("code", "")
                        market_val = d.get("market", 1)
                        prefix = "SH" if market_val == 1 else "SZ"
                        sym_key = prefix + code6
                        result[sym_key] = {
                            "price": d.get("price", 0),
                            "lastClose": d.get("last_close", 0),
                            "open": d.get("open", 0),
                            "high": d.get("high", 0),
                            "low": d.get("low", 0),
                            "vol": d.get("vol", 0),
                            "amount": d.get("amount", 0),
                        }
            return result

        with self._call_lock:
            result = self._retry(_do_fetch, {})
        missing = [sym for sym in symbols if self._normalize_symbol_key(sym) not in result]
        if missing:
            result.update(self._fetch_eastmoney_quotes(missing))
        return result

    def fetch_instruments(self):
        """Discover the current A-share universe and names from TDX.

        The sync pipeline consumes this provider-level method instead of
        knowing pytdx pagination or connection details.
        """
        def _do_fetch(api):
            result = {}
            for market in (0, 1):
                count = int(api.get_security_count(market) or 0)
                for start in range(0, count, 1000):
                    rows = api.get_security_list(market, start) or []
                    for row in rows:
                        code = str(row.get("code") or "")
                        if len(code) != 6 or not code.isdigit():
                            continue
                        if market == 1 and code.startswith("6"):
                            prefix = "SH"
                        elif market == 0 and code.startswith(("0", "3")) and not code.startswith("39"):
                            prefix = "SZ"
                        elif market == 0 and code.startswith(("4", "8")):
                            prefix = "BJ"
                        else:
                            continue
                        result[prefix + code] = {
                            "code": code,
                            "name": str(row.get("name") or "").strip(),
                        }
            return result

        with self._call_lock:
            result = self._retry(_do_fetch, {})
        if not result:
            raise ConnectionError("TDX returned no A-share instruments")
        return result

    def get_quote(self, symbol):
        """获取实时行情。"""
        market, code = self._parse_symbol(symbol)

        def _do_fetch(api):
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
            return None

        with self._call_lock:
            result = self._retry(_do_fetch)
        if result and result.get("price", 0) > 0:
            return result
        return self._fetch_eastmoney_quotes([symbol]).get(self._normalize_symbol_key(symbol))

    def fetch_today_day_bar_from_eastmoney(self, symbol):
        """TDX 不可用时，用东方财富实时行情构造当日日 K 线数据。

        Returns:
            dict | None: {"date", "open", "high", "low", "close", "volume"} 或 None。
        """
        from datetime import datetime
        quotes = self._fetch_eastmoney_quotes([symbol])
        q = quotes.get(self._normalize_symbol_key(symbol))
        if not q or q.get("price", 0) <= 0:
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "date": today,
            "open": round(float(q.get("open", 0) or 0), 2),
            "high": round(float(q.get("high", 0) or 0), 2),
            "low": round(float(q.get("low", 0) or 0), 2),
            "close": round(float(q["price"]), 2),
            "volume": int(float(q.get("vol", 0) or 0)),
        }

    def get_today_kline(self, symbol):
        """获取当天 1min K 线。"""
        market, code = self._parse_symbol(symbol)

        def _do_fetch(api):
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

        with self._call_lock:
            return self._retry(_do_fetch, [])
