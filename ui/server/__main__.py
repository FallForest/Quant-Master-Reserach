"""入口: python -m server"""
import argparse
import os
import threading
from http.server import ThreadingHTTPServer

from . import app
from .datadir import DataDir
from .tdx_quote import TDXQuote
from .sync import auto_sync_daily, schedule_daily_sync
from .routes import APIHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.expanduser(
        "~/.quant_master/quant_master_data/tdx_cn_data"))
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()

    app.data = DataDir(args.data_dir)
    app.tdx_quote = TDXQuote()

    # 后台自动同步日线数据
    sync_thread = threading.Thread(target=auto_sync_daily, args=(app.data.data_dir, app.data), daemon=True)
    sync_thread.start()

    # 定时同步：每个交易日 15:30
    timer_thread = threading.Thread(target=schedule_daily_sync, args=(app.data.data_dir, app.data), daemon=True)
    timer_thread.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), APIHandler)
    print(f"API server running on http://127.0.0.1:{args.port}")
    print(f"Data dir: {args.data_dir}")
    print(f"Real-time quote: TDX (银河证券)")
    server.serve_forever()


if __name__ == "__main__":
    main()
