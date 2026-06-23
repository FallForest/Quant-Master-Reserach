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


def test_walk_forward_portfolio_scan_direct_execution_discovers_repo_root(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmarks"
        / "Transcendence"
        / "support"
        / "walk_forward_portfolio_scan.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Walk-forward and holdout portfolio scan on existing signal artifacts." in completed.stdout
    assert "--run-id" in completed.stdout
    assert "--config-path" in completed.stdout
    assert "--hold-thresh-grid" in completed.stdout
    assert "--enable-topk-derisk" in completed.stdout
    assert "--enable-confidence" in completed.stdout
