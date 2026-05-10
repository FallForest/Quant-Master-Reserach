from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from quant_agent.config import LLMConfig
from quant_agent.llm import OpenAICompatibleClient


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class LLMRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = LLMConfig(api_base="https://example.com", api_key="k", model="m", timeout_seconds=1)
        self.client = OpenAICompatibleClient(cfg)
        self.client.base_backoff_seconds = 0

    def test_retry_on_502_then_success(self) -> None:
        err = urllib.error.HTTPError(
            url="https://example.com/chat/completions",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"upstream"}'),
        )
        ok = _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}',
                        }
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", side_effect=[err, ok]) as mocked, patch("time.sleep") as sleep_mock:
            payload = self.client.create_json_completion("s", "u")
        self.assertEqual(payload["ok"], True)
        self.assertEqual(mocked.call_count, 2)
        self.assertGreaterEqual(sleep_mock.call_count, 1)

    def test_no_retry_on_400(self) -> None:
        err = urllib.error.HTTPError(
            url="https://example.com/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad request"}'),
        )
        with patch("urllib.request.urlopen", side_effect=[err]) as mocked, patch("time.sleep") as sleep_mock:
            with self.assertRaises(RuntimeError):
                self.client.create_json_completion("s", "u")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(sleep_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
