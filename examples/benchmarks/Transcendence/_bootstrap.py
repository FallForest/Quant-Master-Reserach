from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

from quant_master.config import resolve_provider_uri, resolve_provider_uri_in_config


DEFAULT_PROVIDER_URI = "~/.quant_master/quant_master_data/tdx_cn_data"

def ensure_repo_and_benchmark_on_path(anchor: str) -> None:
    anchor_path = Path(anchor).resolve()
    benchmark_root = anchor_path.parent
    repo_root = benchmark_root.parents[2]
    extra_dirs = (
        repo_root,
        benchmark_root,
        benchmark_root / "model",
        benchmark_root / "strategy",
        benchmark_root / "support",
    )
    for path in map(str, extra_dirs):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_default_provider_uri(base_dir: str | Path | None = None) -> str:
    return str(resolve_provider_uri(DEFAULT_PROVIDER_URI, base_dir=base_dir))


def load_config_with_resolved_provider(
    path: Path,
    *,
    loader: Callable[[Path], Any],
    binary_fallback: Callable[[Path], Any] | None = None,
) -> Any:
    try:
        config = loader(path)
    except UnicodeDecodeError:
        if binary_fallback is None:
            raise
        return binary_fallback(path)
    return resolve_provider_uri_in_config(config, base_dir=path.parent)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def init_quant_master_from_config(
    config: dict[str, Any],
    *,
    default_provider_uri: str = DEFAULT_PROVIDER_URI,
    base_dir: str | Path | None = None,
    **defaults: Any,
) -> dict[str, Any]:
    import quant_master

    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", default_provider_uri)
    for key, value in defaults.items():
        init_cfg.setdefault(key, value)
    resolve_provider_uri_in_config(init_cfg, base_dir=base_dir)
    quant_master.init(**init_cfg)
    return init_cfg
