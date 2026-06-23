"""Calendar validation helpers for UI data directories."""

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class CalendarValidationResult:
    """Validation result for a calendar text file or line list."""

    valid_dates: list[str] = field(default_factory=list)
    invalid_lines: list[dict] = field(default_factory=list)
    duplicate_count: int = 0
    ordered: bool = True

    @property
    def healthy(self) -> bool:
        return not self.invalid_lines and self.duplicate_count == 0 and self.ordered

    @property
    def invalid_line_count(self) -> int:
        return len(self.invalid_lines)

    def diagnostics(self) -> dict:
        return {
            "calendarHealthy": self.healthy,
            "calendarInvalidLineCount": self.invalid_line_count,
            "sampleInvalidCalendarLines": self.invalid_lines[:8],
            "calendarDuplicateCount": self.duplicate_count,
            "calendarOrdered": self.ordered,
        }


class InvalidCalendarError(ValueError):
    """Raised when a calendar file contains malformed dates."""

    def __init__(self, path: Path, result: CalendarValidationResult):
        self.path = Path(path)
        self.result = result
        sample = result.invalid_lines[:3]
        sample_text = ", ".join(
            f"line {item['line']}: {item['value']!r}" for item in sample
        ) or "no invalid sample"
        super().__init__(
            f"Invalid calendar file {self.path}: "
            f"{result.invalid_line_count} invalid date line(s); {sample_text}"
        )

    def to_user_message(self) -> str:
        sample = self.result.invalid_lines[:3]
        sample_text = "、".join(f"第 {item['line']} 行 {item['value']}" for item in sample)
        if not sample_text:
            sample_text = "无样例"
        return (
            f"当前数据日历文件已损坏：{self.path}。"
            f"发现 {self.result.invalid_line_count} 条非法日期（{sample_text}）。"
            "请修复/重新同步日历后再运行选股。"
        )


def parse_calendar_date(value) -> str | None:
    """Return normalized YYYY-MM-DD if value is a strict valid ISO date, otherwise None."""
    text = str(value or "").strip()
    if not _DATE_RE.match(text):
        return None
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def validate_calendar_lines(lines: Iterable[str]) -> CalendarValidationResult:
    """Validate calendar lines and collect valid dates plus diagnostics."""
    result = CalendarValidationResult()
    seen = set()
    previous = None

    for line_no, raw in enumerate(lines, 1):
        text = str(raw).strip()
        if not text:
            continue
        parsed = parse_calendar_date(text)
        if parsed is None:
            result.invalid_lines.append({"line": line_no, "value": text})
            continue
        if parsed in seen:
            result.duplicate_count += 1
        seen.add(parsed)
        if previous is not None and parsed < previous:
            result.ordered = False
        previous = parsed
        result.valid_dates.append(parsed)

    return result


def load_calendar_file(path: Path, strict: bool = True) -> tuple[list[str], CalendarValidationResult]:
    """Load a calendar file, returning valid dates and validation details."""
    path = Path(path)
    if not path.exists():
        result = CalendarValidationResult()
        return [], result

    lines = path.read_text(encoding="utf-8").splitlines()
    result = validate_calendar_lines(lines)
    if strict and result.invalid_lines:
        raise InvalidCalendarError(path, result)
    return result.valid_dates, result
