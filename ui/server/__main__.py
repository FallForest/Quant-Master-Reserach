"""入口: python -m server"""
import argparse
import logging

import uvicorn

from .datadir import DEFAULT_DATA_DIR
from .main import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    app = create_app(data_dir=args.data_dir, port=args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
