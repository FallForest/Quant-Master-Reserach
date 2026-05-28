"""HTTP 路由处理：路由表驱动，实际逻辑委托给 handlers/ 模块。"""
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .handlers import browser, pipeline, backtest, analysis, static

# ---- 路由表 ----

_GET_ROUTES = {
    "/api/browser/stocks":          browser.stocks,
    "/api/browser/quotes":          browser.quotes,
    "/api/models":                  static.models,
    "/api/model-catalog":           static.model_catalog,
    "/api/strategies":              static.strategies,
    "/api/optimizer":               static.optimizer,
    "/api/overview":                browser.overview,
    "/api/pipeline/status":         pipeline.global_status,
    "/api/experiments":             analysis.experiments,
    "/api/portfolio":               analysis.portfolio,
    "/api/model-performance":       analysis.model_performance,
    "/api/attribution":             analysis.attribution,
    "/api/factor/analysis":         analysis.factor_analysis,
}

_GET_PREFIX_ROUTES = {
    "/api/browser/kline/":          browser.kline,
    "/api/realtime/quote/":         browser.realtime_quote,
    "/api/realtime/kline/":         browser.realtime_kline,
    "/api/pipeline/status/":        pipeline.status,
    "/api/stock-select/status/":    backtest.stock_select_status,
    "/api/stock-select/results/":   backtest.stock_select_results,
    "/api/backtest/status/":        backtest.status,
    "/api/backtest/results/":       backtest.results,
}

_POST_ROUTES = {
    "/api/pipeline/run":            pipeline.run,
    "/api/stock-select/run":        backtest.stock_select_run,
    "/api/sync/trigger":            pipeline.sync_trigger,
    "/api/backtest/run":            backtest.run,
}


class APIHandler(BaseHTTPRequestHandler):

    # ---- 基础方法 ----

    def log_message(self, format, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Type", "application/json; charset=utf-8")

    def _json_response(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        """读取 POST 请求体并解析为 JSON dict。"""
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len else b"{}"
        return json.loads(raw)

    def _query_params(self):
        """解析 URL query string，返回 {key: value} 单值字典。"""
        parsed = urlparse(self.path)
        multi = parse_qs(parsed.query)
        return {k: v[0] for k, v in multi.items()}

    # ---- HTTP 方法 ----

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path in _GET_ROUTES:
                return _GET_ROUTES[path](self)
            for prefix, handler in _GET_PREFIX_ROUTES.items():
                if path.startswith(prefix):
                    param = path[len(prefix):]
                    return handler(self, param)
            self.send_error(404)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path in _POST_ROUTES:
                return _POST_ROUTES[path](self)
            self.send_error(404)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)
