from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_simple_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "LLMConfig | None":
        if dotenv_path is not None:
            _load_simple_dotenv(dotenv_path)
        api_base = os.getenv("QUANT_AGENT_API_BASE") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("QUANT_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("QUANT_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("CHAT_MODEL")
        if not (api_base and api_key and model):
            return None
        return cls(api_base=api_base.rstrip("/"), api_key=api_key, model=model)
