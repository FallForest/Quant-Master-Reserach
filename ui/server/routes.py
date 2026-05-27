"""HTTP 路由处理：所有 /api/* 端点。"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

from . import app
from .sync import get_sync_status, _get_last_update_date, auto_sync_daily
from .stock_select import (
    get_model_registry, get_handler_registry,
    get_runs as get_ss_runs, get_counter as next_ss_counter,
    run_stock_selection,
)


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Type", "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/pipeline/run":
                return self._handle_pipeline_run()
            elif path == "/api/stock-select/run":
                return self._handle_stock_select_run()
            elif path == "/api/sync/trigger":
                return self._handle_sync_trigger()
            else:
                self.send_error(404)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/api/browser/stocks":
                return self._handle_stocks()
            elif path == "/api/browser/quotes":
                return self._handle_quotes()
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
                return self._handle_pipeline_global_status()
            elif path.startswith("/api/pipeline/status/"):
                run_id = path.split("/api/pipeline/status/")[1]
                return self._handle_pipeline_status(run_id)
            elif path == "/api/models":
                return self._handle_models()
            elif path.startswith("/api/stock-select/status/"):
                run_id = path.split("/api/stock-select/status/")[1]
                return self._handle_stock_select_status(run_id)
            elif path.startswith("/api/stock-select/results/"):
                run_id = path.split("/api/stock-select/results/")[1]
                return self._handle_stock_select_results(run_id)
            else:
                self.send_error(404)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    # ---- Browser ----

    def _handle_stocks(self):
        data = app.data
        names = data.get_names()
        instruments = data.get_instruments()
        stocks = []
        for sym, start, end in instruments:
            code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
            name = names.get(code6, "")

            close_idx, close_vals = data.read_field(sym, "close", "day")
            item = {"symbol": sym, "name": name, "startDate": start, "endDate": end}
            if close_vals is not None and len(close_vals) > 0:
                c = None
                for j in range(len(close_vals) - 1, -1, -1):
                    v = float(close_vals[j])
                    if not np.isnan(v):
                        c = round(v, 2)
                        break
                if c is not None:
                    item["close"] = c
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

    def _handle_quotes(self):
        """轻量级端点：返回所有股票的最新行情，用于列表实时刷新。"""
        data = app.data
        instruments = data.get_instruments()
        quotes = {}
        for sym, start, end in instruments:
            close_idx, close_vals = data.read_field(sym, "close", "day")
            if close_vals is None or len(close_vals) == 0:
                continue
            c = None
            for j in range(len(close_vals) - 1, -1, -1):
                v = float(close_vals[j])
                if not np.isnan(v):
                    c = round(v, 2)
                    break
            if c is None:
                continue
            chg = 0
            chg_pct = 0.0
            for j in range(len(close_vals) - 2, -1, -1):
                prev = float(close_vals[j])
                if not np.isnan(prev) and prev != 0:
                    chg = round(c - prev, 2)
                    chg_pct = round((c - prev) / prev * 100, 2)
                    break
            vol = 0
            vol_idx, vol_vals = data.read_field(sym, "volume", "day")
            if vol_vals is not None and len(vol_vals) > 0:
                for j in range(len(vol_vals) - 1, -1, -1):
                    v = float(vol_vals[j])
                    if not np.isnan(v):
                        vol = int(v)
                        break
            quotes[sym] = {"close": c, "change": chg, "changePct": chg_pct, "volume": vol}
        self._json_response({"quotes": quotes})

    def _handle_kline(self, symbol, freq, start, end):
        kline = app.data.get_kline(symbol, freq, start, end)
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

    # ---- Realtime ----

    def _handle_realtime_quote(self, symbol):
        q = app.tdx_quote.get_quote(symbol)
        if q:
            self._json_response({"ok": True, "quote": q})
        else:
            self._json_response({"ok": False, "quote": {}})

    def _handle_realtime_kline(self, symbol):
        kline = app.tdx_quote.get_today_kline(symbol)
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

    # ---- Pipeline ----

    def _handle_pipeline_global_status(self):
        last_date = _get_last_update_date(app.data.data_dir if app.data else "")
        resp = {"lastUpdate": last_date or "--"}
        sync_st = get_sync_status()
        if sync_st["running"]:
            resp["syncing"] = True
        if sync_st["lastError"]:
            resp["syncError"] = sync_st["lastError"]
        return self._json_response(resp)

    def _handle_pipeline_run(self):
        run_id = f"run_{app.next_pipeline_counter()}"
        app.pipeline_runs[run_id] = {
            "startTime": time.time(),
            "done": False,
            "success": False,
        }
        self._json_response({"runId": run_id})

    def _handle_pipeline_status(self, run_id):
        run = app.pipeline_runs.get(run_id)
        if not run:
            return self._json_response({"error": "run not found"}, status=404)
        elapsed = time.time() - run["startTime"]
        steps = [
            (1.0,  10, "清理残留文件",   "清理 source/ 和 normalize/ 目录"),
            (2.5,  20, "下载数据",       "从 TDX 行情服务器下载日线数据..."),
            (5.0,  50, "标准化处理",     "对齐交易日历，计算复权因子"),
            (7.0,  70, "写入二进制",     "增量追加到 features/"),
            (8.5,  85, "校验数据",       "检查日历、instruments 完整性"),
            (9.5,  95, "清理中间文件",   "删除临时 CSV 文件"),
            (10.5, 100, "完成",          "数据更新完成!"),
        ]
        current = steps[0]
        for s in steps:
            if elapsed >= s[0]:
                current = s
        done = elapsed >= steps[-1][1] and elapsed >= steps[-1][0]
        resp = {
            "progress": current[1],
            "step": current[2],
            "logs": [{"level": "success" if done else "info", "msg": f"[{time.strftime('%H:%M:%S')}] {current[3]}"}],
            "done": done,
            "success": done,
        }
        self._json_response(resp)

    # ---- Sync ----

    def _handle_sync_trigger(self):
        """手动触发数据同步。"""
        st = get_sync_status()
        if st["running"]:
            self._json_response({"ok": False, "error": "同步正在进行中"})
            return
        t = threading.Thread(
            target=auto_sync_daily,
            args=(app.data.data_dir, app.data),
            daemon=True,
        )
        t.start()
        self._json_response({"ok": True, "msg": "同步已启动"})

    # ---- Models ----

    def _handle_models(self):
        model_reg = get_model_registry()
        handler_reg = get_handler_registry()
        models_list = []
        for mid, spec in model_reg.items():
            models_list.append({
                "id": mid,
                "label": spec["label"],
                "category": spec["category"],
                "desc": spec.get("desc", ""),
                "handler": spec["handler"],
            })
        handlers_list = [{"id": k, "label": v["label"]} for k, v in handler_reg.items()]
        self._json_response({"models": models_list, "handlers": handlers_list})

    # ---- Stock Selection ----

    def _handle_stock_select_run(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b"{}"
        config = json.loads(body)

        ss_runs = get_ss_runs()
        for rid, run in ss_runs.items():
            if not run.get("done"):
                return self._json_response({"error": "已有选股任务在运行中，请等待完成"}, status=409)

        run_id = f"ss_{next_ss_counter()}"
        ss_runs[run_id] = {
            "startTime": time.time(),
            "progress": 0,
            "step": "排队中",
            "logs": [],
            "done": False,
            "success": False,
            "error": None,
            "results": None,
            "config": config,
        }

        t = threading.Thread(target=run_stock_selection, args=(run_id, config, app.data), daemon=True)
        t.start()
        self._json_response({"runId": run_id})

    def _handle_stock_select_status(self, run_id):
        ss_runs = get_ss_runs()
        run = ss_runs.get(run_id)
        if not run:
            return self._json_response({"error": "run not found"}, status=404)
        resp = {
            "progress": run["progress"],
            "step": run["step"],
            "logs": run["logs"][-50:],
            "done": run["done"],
            "success": run["success"],
        }
        if run["error"]:
            resp["error"] = run["error"]
        self._json_response(resp)

    def _handle_stock_select_results(self, run_id):
        ss_runs = get_ss_runs()
        run = ss_runs.get(run_id)
        if not run:
            return self._json_response({"error": "run not found"}, status=404)
        if not run["done"]:
            return self._json_response({"error": "任务尚未完成"}, status=400)
        if not run["success"]:
            return self._json_response({"error": run.get("error", "任务失败")}, status=400)
        self._json_response({"results": run["results"], "config": run["config"]})

    # ---- Utils ----

    def _json_response(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
