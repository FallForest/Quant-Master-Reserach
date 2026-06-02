"""HTTP route table."""

import json
import logging
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .handlers import browser, execution, model, pipeline, position, strategy

_log = logging.getLogger(__name__)

# 生产环境静态文件目录（ui/dist/）
_DIST_DIR = Path(__file__).resolve().parent.parent / "dist"

_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".map": "application/json",
}


class _BadRequest(Exception):
    """_read_body 中 JSON 解析失败时抛出，已发送 400 响应。"""

_GET_ROUTES = {
    "/api/browser/stocks": browser.stocks,
    "/api/browser/quotes": browser.quotes,
    "/api/browser/indices": browser.indices,
    "/api/overview": browser.overview,
    "/api/pipeline/status": pipeline.global_status,
    "/api/models": model.list_models,
    "/api/positions": position.current,
    "/api/positions/history": position.history,
    "/api/watchlist": browser.watchlist,
    "/api/strategy-buffered-rebalance": strategy.buffered_rebalance_preview,
    "/api/execution/config": execution.config,
    "/api/execution/history": execution.history,
}

_GET_PREFIX_ROUTES = {
    "/api/browser/kline/": browser.kline,
    "/api/realtime/quote/": browser.realtime_quote,
    "/api/realtime/kline/": browser.realtime_kline,
    "/api/models/": model.model_detail,
}

_POST_ROUTES = {
    "/api/sync/trigger": pipeline.sync_trigger,
    "/api/positions": position.add_or_update,
    "/api/positions/cash": position.set_cash,
    "/api/positions/account": position.set_account,
    "/api/watchlist": browser.watchlist_add,
    "/api/execution/preview": execution.preview,
    "/api/execution/submit": execution.submit,
}

_POST_PREFIX_ROUTES = {
    "/api/models/": model.run_prediction,
    "/api/positions/": position.update,
}

_DELETE_PREFIX_ROUTES = {
    "/api/positions/": position.remove,
    "/api/watchlist/": browser.watchlist_remove,
}


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        _log.info(format, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Content-Type", "application/json; charset=utf-8")

    def _json_response(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len else b"{}"
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
            raise _BadRequest(str(exc))

    def _query_params(self):
        parsed = urlparse(self.path)
        multi = parse_qs(parsed.query)
        return {k: v[0] for k, v in multi.items()}

    def _serve_static(self, path):
        """从 dist/ 提供静态文件，SPA fallback 到 index.html。"""
        if path == "" or path == "/":
            path = "/index.html"
        file_path = _DIST_DIR / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            # SPA fallback：客户端路由返回 index.html
            file_path = _DIST_DIR / "index.html"
        if not file_path.exists():
            return self.send_error(404)

        ext = file_path.suffix.lower()
        content_type = _MIME_TYPES.get(ext, "application/octet-stream")
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if ext == ".html" else "public, max-age=31536000")
        self.end_headers()
        self.wfile.write(body)

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
            # 生产环境 fallback：非 API 路径从 dist/ 提供静态文件
            if not path.startswith("/api") and _DIST_DIR.exists():
                return self._serve_static(path)
            self.send_error(404)
        except _BadRequest:
            pass  # 已发送 400 响应
        except Exception as exc:
            _log.exception("Unhandled error in GET %s", self.path)
            self._json_response({"error": str(exc)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path in _POST_ROUTES:
                return _POST_ROUTES[path](self)
            for prefix, handler in _POST_PREFIX_ROUTES.items():
                if path.startswith(prefix):
                    param = path[len(prefix):]
                    return handler(self, param)
            self.send_error(404)
        except _BadRequest:
            pass  # 已发送 400 响应
        except Exception as exc:
            _log.exception("Unhandled error in POST %s", self.path)
            self._json_response({"error": str(exc)}, status=500)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            for prefix, handler in _DELETE_PREFIX_ROUTES.items():
                if path.startswith(prefix):
                    param = path[len(prefix):]
                    return handler(self, param)
            self.send_error(404)
        except _BadRequest:
            pass
        except Exception as exc:
            _log.exception("Unhandled error in DELETE %s", self.path)
            self._json_response({"error": str(exc)}, status=500)
