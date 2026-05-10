from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from quant_agent.config import LLMConfig


class OpenAICompatibleClient:
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, config: LLMConfig):
        self.config = config
        self.max_retries = 2
        self.base_backoff_seconds = 1.0

    def create_json_completion(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        raw = self._request_chat_completion(payload)

        try:
            content = raw["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected LLM response payload: {raw}") from exc

    def create_text_completion(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        raw = self._request_chat_completion(payload)
        try:
            return str(raw["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response payload: {raw}") from exc

    def _request_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.config.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: RuntimeError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in self.RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}")
                break
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                last_error = RuntimeError(f"LLM request failed: {exc.reason}")
                break
        assert last_error is not None
        raise last_error

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.base_backoff_seconds * (2**attempt))
