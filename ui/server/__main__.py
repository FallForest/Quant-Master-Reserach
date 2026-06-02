"""入口: python -m server"""
import argparse
import logging
import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import quant_master

from . import app
from .datadir import DEFAULT_DATA_DIR, create_data_dir, get_effective_data_dir
from .model_service import ModelService
from .tdx_quote import TDXQuote
from .sync import auto_sync_daily, schedule_daily_sync
from .routes import APIHandler


def _write_runtime_pid():
    pid_file = Path(__file__).resolve().parent.parent / ".server-pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    app.data = create_data_dir(args.data_dir)
    app.tdx_quote = TDXQuote()
    # 预连接 TDX，避免首个请求阻塞
    threading.Thread(target=app.tdx_quote._get_api, daemon=True).start()

    # 初始化 quant_master 数据提供者（模型实时预测需要访问行情数据）
    provider_uri = str(Path(args.data_dir).expanduser().resolve())
    quant_master.init(provider_uri=provider_uri, region="cn")
    print(f"QuantMaster initialized with provider_uri={provider_uri}")

    app.model_service = ModelService()

    # 后台自动同步日线数据
    sync_thread = threading.Thread(target=auto_sync_daily, args=(None, app.data), daemon=True)
    sync_thread.start()

    # 定时同步：每个交易日 15:30
    timer_thread = threading.Thread(target=schedule_daily_sync, args=(None, app.data), daemon=True)
    timer_thread.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), APIHandler)
    _write_runtime_pid()
    print(f"API server running on http://127.0.0.1:{args.port}")
    print(f"Data dir: {get_effective_data_dir(app.data, args.data_dir)}")
    print(f"Real-time quote: TDX (银河证券)")
    print(f"Model registry: {len(app.model_service.list_models())} model(s) loaded")
    server.serve_forever()


if __name__ == "__main__":
    main()
