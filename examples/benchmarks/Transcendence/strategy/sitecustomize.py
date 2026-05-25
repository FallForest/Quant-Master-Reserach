from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]

for path in map(
    str,
    (
        REPO_ROOT,
        BENCHMARK_ROOT,
        BENCHMARK_ROOT / "model",
        BENCHMARK_ROOT / "strategy",
        BENCHMARK_ROOT / "support",
    ),
):
    if path not in sys.path:
        sys.path.insert(0, path)
