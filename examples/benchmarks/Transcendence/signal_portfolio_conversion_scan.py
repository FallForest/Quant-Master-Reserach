import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks.Transcendence._bootstrap import ensure_repo_and_benchmark_on_path

ensure_repo_and_benchmark_on_path(__file__)

from examples.benchmarks.Transcendence.support.signal_portfolio_conversion_scan import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
