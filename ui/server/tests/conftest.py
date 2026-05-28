"""Pytest fixtures for the QuantMaster UI server tests."""
import sys
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

import pytest
import requests

# Add ui/ to sys.path so `import server` works
_ui_dir = str(Path(__file__).resolve().parent.parent.parent)
if _ui_dir not in sys.path:
    sys.path.insert(0, _ui_dir)

from server import app
from server.routes import APIHandler
from server.tests.mock_datadir import FakeDataDir, FakeTDXQuote


@pytest.fixture(scope="session")
def server_url(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    app.data = FakeDataDir(tmp)
    app.tdx_quote = FakeTDXQuote()

    server = ThreadingHTTPServer(("127.0.0.1", 0), APIHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
