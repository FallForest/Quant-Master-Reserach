from __future__ import annotations

import sys
from pathlib import Path


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
