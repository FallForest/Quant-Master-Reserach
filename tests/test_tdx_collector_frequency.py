from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.data_collector.tdx import collector as tdx_collector


@pytest.mark.parametrize(
    ("interval", "expected_code"),
    [
        ("1d", 9),
        ("1min", 8),
        ("15min", 1),
        ("30min", 2),
        ("60min", 3),
    ],
)
def test_tdx_frequency_codes_match_protocol(interval, expected_code):
    assert tdx_collector._get_tdx_freq_code(interval) == expected_code


@pytest.mark.parametrize(
    ("interval", "expected_freq"),
    [
        ("1d", "day"),
        ("1min", "1min"),
        ("15min", "15min"),
        ("30min", "30min"),
        ("60min", "60min"),
    ],
)
def test_collector_interval_maps_to_quant_master_storage_frequency(interval, expected_freq):
    assert tdx_collector._get_quant_master_freq(interval) == expected_freq


def test_unknown_tdx_interval_is_rejected():
    with pytest.raises(ValueError, match="Unsupported TDX interval 'weekly'"):
        tdx_collector._get_tdx_freq_code("weekly")

    with pytest.raises(ValueError, match="Unsupported TDX interval 'weekly'"):
        tdx_collector._get_quant_master_freq("weekly")


@pytest.mark.parametrize(
    ("interval", "expected_code", "expected_date"),
    [
        ("1d", 9, pd.Timestamp("2024-01-02 00:00:00")),
        ("1min", 8, pd.Timestamp("2024-01-02 09:31:00")),
        ("15min", 1, pd.Timestamp("2024-01-02 09:31:00")),
        ("30min", 2, pd.Timestamp("2024-01-02 09:31:00")),
        ("60min", 3, pd.Timestamp("2024-01-02 09:31:00")),
    ],
)
def test_get_data_passes_correct_frequency_to_pytdx(monkeypatch, interval, expected_code, expected_date):
    bar = {
        "datetime": "2024-01-02 09:31:00",
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "vol": 1000.0,
        "amount": 10100.0,
    }

    class FakeApi:
        def __init__(self):
            self.calls = []
            self.disconnected = False

        def get_security_bars(self, *args):
            self.calls.append(args)
            return [bar]

        def disconnect(self):
            self.disconnected = True

    api = FakeApi()
    monkeypatch.setattr(tdx_collector, "_connect_any", lambda: (api, "127.0.0.1", 7709))

    collector = object.__new__(tdx_collector.TdxCollector)
    collector.delay = 0
    result = collector.get_data(
        "SZ000001",
        interval,
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )

    assert api.calls == [(expected_code, 0, "000001", 0, tdx_collector.MAX_BARS_PER_REQUEST)]
    assert api.disconnected
    assert result.loc[0, "date"] == expected_date


def test_daily_update_uses_day_storage_frequency(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    normalize_dir = tmp_path / "normalize"
    data_dir = tmp_path / "data"
    source_dir.mkdir()
    normalize_dir.mkdir()
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    (data_dir / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")

    calls = {}

    def capture_download(**kwargs):
        calls["download"] = kwargs

    def capture_normalize(**kwargs):
        calls["normalize"] = kwargs

    class FakeDumpDataUpdate:
        def __init__(self, **kwargs):
            calls["dump"] = kwargs

        def dump(self):
            calls["dump_called"] = True

    def fail_dump_all(**kwargs):
        raise AssertionError("existing day calendar should select DumpDataUpdate")

    def capture_verify(quant_master_dir, expected_end_date=None, freq="day"):
        calls["verify"] = {
            "quant_master_dir": quant_master_dir,
            "expected_end_date": expected_end_date,
            "freq": freq,
        }
        return True

    monkeypatch.setattr(tdx_collector, "_download_all", capture_download)
    monkeypatch.setattr(tdx_collector, "_normalize_all", capture_normalize)
    monkeypatch.setattr(tdx_collector, "DumpDataUpdate", FakeDumpDataUpdate)
    monkeypatch.setattr(tdx_collector, "DumpDataAll", fail_dump_all)
    monkeypatch.setattr(tdx_collector, "verify_dump", capture_verify)
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_list", lambda: ["SZ000001"])
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_names", dict)

    run = object.__new__(tdx_collector.Run)
    run.source_dir = source_dir
    run.normalize_dir = normalize_dir
    run.interval = "1d"
    run.update_data_to_bin(
        str(data_dir),
        end_date="2024-01-05",
        allow_unadjusted=True,
    )

    assert calls["download"]["interval"] == "1d"
    assert calls["download"]["start"] == "2024-01-03"
    assert calls["normalize"]["interval"] == "1d"
    assert calls["dump"]["freq"] == "day"
    assert calls["dump_called"]
    assert calls["verify"] == {
        "quant_master_dir": str(data_dir),
        "expected_end_date": "2024-01-05",
        "freq": "day",
    }
    assert Path(data_dir / "instruments" / "names.txt").exists()


def test_daily_update_can_skip_derived_tasks_and_writes_manifest(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    normalize_dir = tmp_path / "normalize"
    data_dir = tmp_path / "data"
    source_dir.mkdir()
    normalize_dir.mkdir()
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    (data_dir / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(tdx_collector, "_download_all", lambda **kwargs: calls.setdefault("download", kwargs))
    monkeypatch.setattr(tdx_collector, "_normalize_all", lambda **kwargs: calls.setdefault("normalize", kwargs))
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_list", lambda: ["SZ000001"])
    monkeypatch.setattr(tdx_collector, "verify_dump", lambda *args, **kwargs: True)

    class FakeDump:
        def __init__(self, **kwargs):
            calls["dump"] = kwargs

        def dump(self):
            calls["dump_called"] = True

    monkeypatch.setattr(tdx_collector, "DumpDataUpdate", FakeDump)
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_names", lambda: (_ for _ in ()).throw(AssertionError()))

    run = object.__new__(tdx_collector.Run)
    run.source_dir = source_dir
    run.normalize_dir = normalize_dir
    run.interval = "1d"
    manifest = run.update_data_to_bin(
        str(data_dir),
        end_date="2024-01-05",
        allow_unadjusted=True,
        skip_names=True,
        skip_index=True,
        skip_pool=True,
    )

    assert manifest["source"] == "tdx"
    assert manifest["stages"]["names"] == "skipped"
    assert manifest["stages"]["index"] == "skipped"
    assert manifest["stages"]["pool"] == "skipped"
    assert manifest["options"]["skip_names"] is True
    assert (data_dir / "update_manifest.json").exists()


def test_minute_update_writes_child_frequency_dir_and_restricts_instruments(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    normalize_dir = tmp_path / "normalize"
    data_dir = tmp_path / "data"
    source_dir.mkdir()
    normalize_dir.mkdir()
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    (data_dir / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    (data_dir / "instruments" / "personal_a_share_pit_v1.txt").write_text(
        "SH600000\t2024-01-02\t2024-01-03\nSZ000001\t2024-01-02\t2024-01-03\n",
        encoding="utf-8",
    )

    calls = {}

    def capture_download(**kwargs):
        calls["download"] = kwargs

    def capture_normalize(**kwargs):
        calls["normalize"] = kwargs

    class FakeDumpDataAll:
        def __init__(self, **kwargs):
            calls["dump"] = kwargs

        def dump(self):
            calls["dump_called"] = True

    def fail_dump_update(**kwargs):
        raise AssertionError("missing 1min calendar should select DumpDataAll")

    def capture_verify(quant_master_dir, expected_end_date=None, freq="day"):
        calls["verify"] = {
            "quant_master_dir": quant_master_dir,
            "expected_end_date": expected_end_date,
            "freq": freq,
        }
        return True

    monkeypatch.setattr(tdx_collector, "_download_all", capture_download)
    monkeypatch.setattr(tdx_collector, "_normalize_all", capture_normalize)
    monkeypatch.setattr(tdx_collector, "DumpDataAll", FakeDumpDataAll)
    monkeypatch.setattr(tdx_collector, "DumpDataUpdate", fail_dump_update)
    monkeypatch.setattr(tdx_collector, "verify_dump", capture_verify)
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_names", dict)

    run = object.__new__(tdx_collector.Run)
    run.source_dir = source_dir
    run.normalize_dir = normalize_dir
    run.interval = "1min"
    run.update_data_to_bin(
        str(data_dir),
        end_date="2024-01-05",
        download_instruments="personal_a_share_pit_v1",
        download_workers=2,
    )

    minute_dir = data_dir / "1min"
    assert calls["download"]["symbols"] == ["SH600000", "SZ000001"]
    assert calls["download"]["interval"] == "1min"
    assert calls["download"]["num_workers"] == 2
    assert calls["normalize"]["interval"] == "1min"
    assert calls["dump"]["quant_master_dir"] == str(minute_dir)
    assert calls["dump"]["freq"] == "1min"
    assert calls["dump_called"]
    assert calls["verify"] == {
        "quant_master_dir": str(minute_dir),
        "expected_end_date": "2024-01-05",
        "freq": "1min",
    }


def test_minute_rebuild_output_removes_existing_child_frequency_dir(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    normalize_dir = tmp_path / "normalize"
    data_dir = tmp_path / "data"
    minute_dir = data_dir / "1min"
    source_dir.mkdir()
    normalize_dir.mkdir()
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    (data_dir / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    (data_dir / "instruments" / "personal_a_share_pit_v1.txt").write_text(
        "SH600000\t2024-01-02\t2024-01-03\n",
        encoding="utf-8",
    )
    (minute_dir / "calendars").mkdir(parents=True)
    (minute_dir / "calendars" / "1min.txt").write_text("2024-01-03 15:00:00\n", encoding="utf-8")
    sentinel = minute_dir / "old.txt"
    sentinel.write_text("stale minute data", encoding="utf-8")

    calls = {}

    monkeypatch.setattr(tdx_collector, "_download_all", lambda **kwargs: calls.setdefault("download", kwargs))
    monkeypatch.setattr(tdx_collector, "_normalize_all", lambda **kwargs: calls.setdefault("normalize", kwargs))
    monkeypatch.setattr(tdx_collector, "verify_dump", lambda *args, **kwargs: True)
    monkeypatch.setattr(tdx_collector, "_get_cached_stock_names", dict)

    class FakeDumpDataAll:
        def __init__(self, **kwargs):
            calls["dump"] = kwargs

        def dump(self):
            calls["dump_called"] = True

    def fail_dump_update(**kwargs):
        raise AssertionError("rebuild_output should remove calendar and select DumpDataAll")

    monkeypatch.setattr(tdx_collector, "DumpDataAll", FakeDumpDataAll)
    monkeypatch.setattr(tdx_collector, "DumpDataUpdate", fail_dump_update)

    run = object.__new__(tdx_collector.Run)
    run.source_dir = source_dir
    run.normalize_dir = normalize_dir
    run.interval = "1min"
    run.update_data_to_bin(
        str(data_dir),
        end_date="2024-01-05",
        download_instruments="personal_a_share_pit_v1",
        rebuild_output=True,
    )

    assert not sentinel.exists()
    assert calls["download"]["start"] != "2024-01-03 15:00:00"
    assert calls["dump"]["quant_master_dir"] == str(minute_dir)
    assert calls["dump_called"]


def test_daily_rebuild_output_is_rejected(tmp_path):
    run = object.__new__(tdx_collector.Run)
    run.source_dir = tmp_path / "source"
    run.normalize_dir = tmp_path / "normalize"
    run.interval = "1d"

    with pytest.raises(ValueError, match="non-daily child frequency directories"):
        run.update_data_to_bin(str(tmp_path / "data"), allow_unadjusted=True, rebuild_output=True)


def test_verify_dump_accepts_high_frequency_end_upper_bound(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    feature_dir = data_dir / "features" / "sh600000"
    feature_dir.mkdir(parents=True)
    (data_dir / "calendars" / "1min.txt").write_text(
        "2026-07-22 14:59:00\n2026-07-22 15:00:00\n",
        encoding="utf-8",
    )
    (data_dir / "instruments" / "all.txt").write_text(
        "SH600000\t2026-07-22 14:59:00\t2026-07-22 15:00:00\n",
        encoding="utf-8",
    )
    np.array([0, 10.0, 10.1], dtype="<f4").tofile(feature_dir / "close.1min.bin")

    assert tdx_collector.verify_dump(str(data_dir), expected_end_date="2026-07-23", freq="1min")
