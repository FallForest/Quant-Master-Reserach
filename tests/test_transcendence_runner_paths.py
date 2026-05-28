from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_alpha158alpha360_runner_direct_execution_discovers_repo_root(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmarks"
        / "Transcendence"
        / "model"
        / "alpha158alpha360_regime_horizon_run.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Alpha158Alpha360 regime-horizon cost ensemble gate runner." in completed.stdout
    assert "{smoke,medium,full,verify}" in completed.stdout
    assert "--preserve-config-windows" in completed.stdout
