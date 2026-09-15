from pathlib import Path

import pandas as pd
import pytest

from scripts.check_data_health import DataHealthChecker


def _checker(tmp_path: Path, **kwargs) -> DataHealthChecker:
    checker = object.__new__(DataHealthChecker)
    checker.data = {}
    checker.quant_master_dir = str(tmp_path)
    checker.require_nonconstant_factor = kwargs.get("require_nonconstant_factor", False)
    checker.constant_factor_ratio_threshold = kwargs.get("constant_factor_ratio_threshold", 0.95)
    checker.universe = kwargs.get("universe")
    checker.universe_check_date = kwargs.get("universe_check_date")
    checker.max_active_universe_size = kwargs.get("max_active_universe_size")
    checker.fail_fast = kwargs.get("fail_fast", False)
    return checker


def test_constant_factor_guard_reports_unadjusted_data(tmp_path):
    checker = _checker(tmp_path, require_nonconstant_factor=True)
    checker.data = {"SH600000": pd.DataFrame({"factor": [1.0, 1.0, float("nan")]})}

    result = checker.check_constant_factor()

    assert result.loc["factor", "checked_instruments"] == 1
    assert result.loc["factor", "constant_one_ratio"] == 1.0


def test_universe_size_guard_counts_unique_active_instruments(tmp_path):
    instruments_dir = tmp_path / "instruments"
    instruments_dir.mkdir()
    (instruments_dir / "csi500.txt").write_text(
        "SH600000\t2024-01-01\t2024-12-31\n"
        "SH600001\t2024-01-01\t2024-12-31\n"
        "SH600002\t2024-02-01\t2024-12-31\n",
        encoding="utf-8",
    )
    checker = _checker(
        tmp_path,
        universe="csi500",
        universe_check_date="2024-03-01",
        max_active_universe_size=2,
    )

    result = checker.check_active_universe_size()

    assert result.loc["csi500", "active_instruments"] == 3


def test_fail_fast_raises_when_a_guard_fails(tmp_path, monkeypatch):
    checker = _checker(tmp_path, fail_fast=True)
    failed = pd.DataFrame({"constant_one_ratio": [1.0]}, index=["factor"])
    for method in (
        "check_missing_data",
        "check_large_step_changes",
        "check_required_columns",
        "check_missing_factor",
        "check_features_dir_lowercase",
        "check_active_universe_size",
    ):
        monkeypatch.setattr(checker, method, lambda: None)
    monkeypatch.setattr(checker, "check_constant_factor", lambda: failed)

    with pytest.raises(RuntimeError, match="constant_factor"):
        checker.check_data()
