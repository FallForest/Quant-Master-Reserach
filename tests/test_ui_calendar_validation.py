import pytest

from ui.server.calendar_validation import InvalidCalendarError, load_calendar_file, parse_calendar_date, validate_calendar_lines
from ui.server.datadir import DataDir
from ui.server.sync import get_data_health_snapshot


def test_parse_calendar_date_is_strict_iso_date():
    assert parse_calendar_date("2026-06-02") == "2026-06-02"
    assert parse_calendar_date("793-00-31") is None
    assert parse_calendar_date("2026-13-01") is None
    assert parse_calendar_date("2026/06/02") is None
    assert parse_calendar_date("2026-06-02 15:00:00") is None


def test_validate_calendar_lines_reports_invalid_duplicate_and_order():
    result = validate_calendar_lines([
        "2026-06-01",
        "793-00-31",
        "2026-06-01",
        "2026-05-31",
    ])

    assert result.valid_dates == ["2026-06-01", "2026-06-01", "2026-05-31"]
    assert result.invalid_line_count == 1
    assert result.invalid_lines[0] == {"line": 2, "value": "793-00-31"}
    assert result.duplicate_count == 1
    assert result.ordered is False
    assert result.healthy is False


def test_load_calendar_file_strict_raises_on_invalid_line(tmp_path):
    cal_path = tmp_path / "day.txt"
    cal_path.write_text("2026-06-01\n793-00-31\n", encoding="utf-8")

    with pytest.raises(InvalidCalendarError) as exc_info:
        load_calendar_file(cal_path, strict=True)

    assert "793-00-31" in str(exc_info.value)


def test_datadir_read_calendar_rejects_invalid_calendar(tmp_path):
    root = tmp_path / "data"
    (root / "calendars").mkdir(parents=True)
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2026-06-01\n793-00-31\n", encoding="utf-8")

    data_dir = DataDir(str(root))

    with pytest.raises(InvalidCalendarError):
        data_dir.read_calendar("day")


def test_health_snapshot_reports_invalid_calendar_lines(tmp_path):
    root = tmp_path / "data"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text(
        "2026-06-01\n2026-06-02\n793-00-31\n", encoding="utf-8"
    )
    (root / "instruments" / "all.txt").write_text("SH600000\t2026-06-01\t2026-06-02\n", encoding="utf-8")

    snapshot = get_data_health_snapshot(str(root))

    assert snapshot["calendarLastDate"] == "2026-06-02"
    assert snapshot["calendarHealthy"] is False
    assert snapshot["calendarInvalidLineCount"] == 1
    assert snapshot["sampleInvalidCalendarLines"] == [{"line": 3, "value": "793-00-31"}]

