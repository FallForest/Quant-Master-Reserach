"""Build and incrementally update a personal point-in-time A-share training universe.

The universe is calculated independently for every trading day from the local
``all`` market.  It does not use CSI index constituents.  Daily membership is
compressed into QuantMaster/Qlib instrument intervals before it is written to
``instruments/<output_name>.txt``.

The default contract is intended for a small personal account while keeping a
large cross-section for model training:

* Shanghai/Shenzhen main board, ChiNext and STAR Market; BSE is excluded.
* Actual close price between CNY 2 and CNY 40, inclusive.
* Positive current-day volume and amount.
* At least 120 completed trading observations before the selection date.
* Previous 20-day average amount of at least CNY 50 million.
* Previous 20-day average absolute daily return of at least 0.8%, a simple
  activity proxy that avoids selecting completely dormant names.

The pool is formed after the close.  Rolling eligibility fields use ``Ref(...,
1)`` so that they contain only information available before the selection day.
Current close/volume/amount are known at the after-close signal time.  Live
execution must still re-check price, account permissions, suspension, limit
status, ST status and board-lot affordability.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROVIDER_URI = Path.home() / ".quant_master" / "quant_master_data" / "tdx_cn_data"
DEFAULT_OUTPUT_NAME = "personal_dynamic_pool"
SCHEMA_VERSION = 1

BOARD_PATTERNS = {
    "main": re.compile(r"^(?:SH60\d{4}|SZ00\d{4})$"),
    "chinext": re.compile(r"^SZ30\d{4}$"),
    "star": re.compile(r"^SH68\d{4}$"),
}


@dataclass(frozen=True)
class PersonalDynamicPoolConfig:
    """Selection contract for the daily training universe."""

    min_price: float = 2.0
    max_price: float = 40.0
    min_history_days: int = 120
    amount_lookback_days: int = 20
    min_average_amount: float = 50_000_000.0
    activity_lookback_days: int = 20
    min_average_abs_return: float = 0.008
    boards: Tuple[str, ...] = ("main", "chinext", "star")
    min_pool_size: int = 500
    max_pool_size: int = 5_500

    def validate(self) -> None:
        unsupported = set(self.boards).difference(BOARD_PATTERNS)
        if unsupported:
            raise ValueError(f"Unsupported boards: {sorted(unsupported)}")
        if not self.boards:
            raise ValueError("At least one board must be enabled.")
        if self.min_price <= 0 or self.min_price > self.max_price:
            raise ValueError("Price bounds must satisfy 0 < min_price <= max_price.")
        if self.min_history_days <= 0 or self.amount_lookback_days <= 0 or self.activity_lookback_days <= 1:
            raise ValueError("History and lookback days must be positive (activity lookback must exceed one).")
        if self.min_average_amount < 0:
            raise ValueError("min_average_amount must be non-negative.")
        if self.min_average_abs_return < 0:
            raise ValueError("min_average_abs_return must be non-negative.")
        if self.min_pool_size < 0 or self.max_pool_size < self.min_pool_size:
            raise ValueError("Pool size bounds are invalid.")


@dataclass(frozen=True)
class PoolUpdateResult:
    output_path: Path
    metadata_path: Path
    update_start: str
    update_end: str
    trading_days: int
    min_daily_size: int
    max_daily_size: int
    unique_instruments: int
    interval_count: int
    dry_run: bool
    backup_paths: Tuple[Path, ...] = ()


InstrumentSpans = Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]
DailyStates = Dict[pd.Timestamp, Set[str]]


def classify_board(symbol: str) -> str:
    """Return the supported A-share board name or ``unknown``."""

    normalized = str(symbol).strip().upper()
    for board, pattern in BOARD_PATTERNS.items():
        if pattern.fullmatch(normalized):
            return board
    return "unknown"


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    calendar_path = provider_uri / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Daily calendar does not exist: {calendar_path}")
    values = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    calendar = pd.DatetimeIndex(pd.to_datetime(values)).normalize().unique().sort_values()
    if calendar.empty:
        raise ValueError(f"Daily calendar is empty: {calendar_path}")
    return calendar


def _resolve_update_dates(
    calendar: pd.DatetimeIndex,
    *,
    update_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DatetimeIndex:
    single_mode = update_date is not None
    range_mode = start_date is not None or end_date is not None
    if single_mode == range_mode:
        raise ValueError("Specify either update_date or both start_date and end_date.")

    if single_mode:
        date = pd.Timestamp(update_date).normalize()
        if date not in calendar:
            raise ValueError(f"update_date is not present in the local trading calendar: {date.date()}")
        return pd.DatetimeIndex([date])

    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date are required for a range update.")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be later than end_date.")
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if dates.empty:
        raise ValueError(f"No local trading dates exist in range {start.date()}..{end.date()}.")
    return dates


def _read_spans(path: Path) -> InstrumentSpans:
    if not path.exists():
        return {}
    spans: InstrumentSpans = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 3:
            raise ValueError(f"Invalid instrument interval at {path}:{line_number}")
        symbol, start_text, end_text = parts
        start = pd.Timestamp(start_text).normalize()
        end = pd.Timestamp(end_text).normalize()
        if start > end:
            raise ValueError(f"Inverted instrument interval at {path}:{line_number}")
        spans[symbol.upper()].append((start, end))
    return dict(spans)


def _calendar_positions(calendar: pd.DatetimeIndex) -> Dict[pd.Timestamp, int]:
    return {pd.Timestamp(date): index for index, date in enumerate(calendar)}


def _merge_spans(spans: InstrumentSpans, calendar: pd.DatetimeIndex) -> InstrumentSpans:
    positions = _calendar_positions(calendar)
    merged: InstrumentSpans = {}
    for symbol, values in spans.items():
        output: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, end in sorted(values):
            if start not in positions or end not in positions:
                raise ValueError(
                    f"Instrument interval for {symbol} is outside the local trading calendar: {start}..{end}"
                )
            if not output:
                output.append((start, end))
                continue
            prior_start, prior_end = output[-1]
            overlaps = start <= prior_end
            adjacent = positions[start] == positions[prior_end] + 1
            if overlaps or adjacent:
                output[-1] = (prior_start, max(prior_end, end))
            else:
                output.append((start, end))
        if output:
            merged[symbol] = output
    return merged


def _states_to_spans(states: Mapping[pd.Timestamp, Set[str]], dates: pd.DatetimeIndex) -> InstrumentSpans:
    spans: InstrumentSpans = defaultdict(list)
    active: Dict[str, pd.Timestamp] = {}
    previous_date: Optional[pd.Timestamp] = None
    for date in dates:
        current = states.get(pd.Timestamp(date), set())
        for symbol in set(active).difference(current):
            if previous_date is None:
                raise RuntimeError("Cannot close an interval before processing a trading date.")
            spans[symbol].append((active.pop(symbol), previous_date))
        for symbol in current.difference(active):
            active[symbol] = pd.Timestamp(date)
        previous_date = pd.Timestamp(date)
    if previous_date is None:
        return {}
    for symbol, start in active.items():
        spans[symbol].append((start, previous_date))
    return dict(spans)


def _trim_spans_for_range(
    spans: InstrumentSpans,
    calendar: pd.DatetimeIndex,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> InstrumentSpans:
    positions = _calendar_positions(calendar)
    start_position = positions[range_start]
    end_position = positions[range_end]
    prior_date = calendar[start_position - 1] if start_position > 0 else None
    next_date = calendar[end_position + 1] if end_position + 1 < len(calendar) else None
    trimmed: InstrumentSpans = defaultdict(list)
    for symbol, values in spans.items():
        for start, end in values:
            if end < range_start or start > range_end:
                trimmed[symbol].append((start, end))
                continue
            if start < range_start and prior_date is not None:
                trimmed[symbol].append((start, pd.Timestamp(prior_date)))
            if end > range_end and next_date is not None:
                trimmed[symbol].append((pd.Timestamp(next_date), end))
    return dict(trimmed)


def _replace_spans_for_range(
    existing: InstrumentSpans,
    replacement: InstrumentSpans,
    calendar: pd.DatetimeIndex,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> InstrumentSpans:
    combined = _trim_spans_for_range(existing, calendar, range_start, range_end)
    for symbol, values in replacement.items():
        combined.setdefault(symbol, []).extend(values)
    return _merge_spans(combined, calendar)


def _selection_expressions(config: PersonalDynamicPoolConfig) -> List[str]:
    return [
        "$close/($factor+1e-12)",
        "$volume",
        "$amount",
        f"Mean(Ref($amount,1),{config.amount_lookback_days})",
        f"Count(Ref($close,1),{config.min_history_days})",
        f"Mean(Abs(Ref($close,1)/Ref($close,2)-1),{config.activity_lookback_days})",
    ]


def _select_chunk(
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: PersonalDynamicPoolConfig,
) -> Tuple[DailyStates, List[dict]]:
    columns = [
        "price",
        "volume",
        "amount",
        "prior_average_amount",
        "prior_history_count",
        "prior_average_abs_return",
    ]
    if data.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} selection fields, got {data.shape[1]}.")
    data = data.copy()
    data.columns = columns
    if not isinstance(data.index, pd.MultiIndex) or "datetime" not in data.index.names:
        raise ValueError("Feature data must have a MultiIndex containing datetime.")
    if "instrument" not in data.index.names:
        raise ValueError("Feature data must have a MultiIndex containing instrument.")

    date_index = pd.DatetimeIndex(data.index.get_level_values("datetime")).normalize()
    instrument_index = data.index.get_level_values("instrument").astype(str).str.upper()
    enabled_boards = set(config.boards)
    board_mask = np.fromiter(
        (classify_board(symbol) in enabled_boards for symbol in instrument_index),
        dtype=bool,
        count=len(data),
    )
    complete_mask = np.isfinite(data[columns].to_numpy(dtype=float)).all(axis=1)
    price_mask = data["price"].between(config.min_price, config.max_price, inclusive="both").to_numpy()
    traded_mask = ((data["volume"] > 0) & (data["amount"] > 0)).to_numpy()
    history_mask = (data["prior_history_count"] >= config.min_history_days).to_numpy()
    liquid_mask = (data["prior_average_amount"] >= config.min_average_amount).to_numpy()
    activity_mask = (data["prior_average_abs_return"] >= config.min_average_abs_return).to_numpy()

    stage_board = complete_mask & board_mask
    stage_price = stage_board & price_mask
    stage_traded = stage_price & traded_mask
    stage_history = stage_traded & history_mask
    stage_liquid = stage_history & liquid_mask
    eligible_mask = stage_liquid & activity_mask

    states: DailyStates = {}
    diagnostics: List[dict] = []
    for date in dates:
        date = pd.Timestamp(date)
        rows = date_index == date
        eligible_rows = rows & eligible_mask
        symbols = set(instrument_index[eligible_rows])
        if len(symbols) < config.min_pool_size or len(symbols) > config.max_pool_size:
            raise ValueError(
                f"Pool size {len(symbols)} on {date.date()} is outside "
                f"{config.min_pool_size}..{config.max_pool_size}."
            )
        states[date] = symbols
        diagnostics.append(
            {
                "date": str(date.date()),
                "source_rows": int(rows.sum()),
                "supported_board_rows": int((rows & stage_board).sum()),
                "price_eligible_rows": int((rows & stage_price).sum()),
                "currently_traded_rows": int((rows & stage_traded).sum()),
                "history_eligible_rows": int((rows & stage_history).sum()),
                "liquidity_eligible_rows": int((rows & stage_liquid).sum()),
                "activity_eligible_rows": int((rows & eligible_mask).sum()),
                "eligible_rows": len(symbols),
            }
        )
    return states, diagnostics


def _load_daily_states(
    dates: pd.DatetimeIndex,
    config: PersonalDynamicPoolConfig,
    *,
    chunk_size: int,
) -> Tuple[DailyStates, List[dict]]:
    from quant_master.data import D  # pylint: disable=import-outside-toplevel

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    states: DailyStates = {}
    diagnostics: List[dict] = []
    instruments = D.instruments("all")
    expressions = _selection_expressions(config)
    for offset in range(0, len(dates), chunk_size):
        chunk_dates = dates[offset : offset + chunk_size]
        data = D.features(
            instruments,
            expressions,
            start_time=chunk_dates[0],
            end_time=chunk_dates[-1],
            freq="day",
            disk_cache=0,
        )
        chunk_states, chunk_diagnostics = _select_chunk(data, chunk_dates, config)
        states.update(chunk_states)
        diagnostics.extend(chunk_diagnostics)
    return states, diagnostics


def _spans_to_text(spans: InstrumentSpans) -> str:
    lines = [
        f"{symbol}\t{start.date()}\t{end.date()}"
        for symbol in sorted(spans)
        for start, end in spans[symbol]
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _backup_existing(paths: Iterable[Path], stamp: str) -> Tuple[Path, ...]:
    backups = []
    for path in paths:
        if not path.exists():
            continue
        backup_path = path.with_name(f"{path.name}.bak.{stamp}")
        shutil.copy2(str(path), str(backup_path))
        backups.append(backup_path)
    return tuple(backups)


def _read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read existing pool metadata: {path}") from error
    return value if isinstance(value, dict) else {}


def update_personal_dynamic_pool(
    *,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    update_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    config: Optional[PersonalDynamicPoolConfig] = None,
    chunk_size: int = 64,
    kernels: int = 4,
    dry_run: bool = False,
) -> PoolUpdateResult:
    """Replace one day or an inclusive date range in the personal PIT pool.

    A single-day update requires an exact local trading date.  Range endpoints
    may be calendar dates; all trading dates within the inclusive range are
    updated.  Existing intervals outside the requested range are preserved.
    """

    import quant_master  # pylint: disable=import-outside-toplevel

    provider_uri = Path(provider_uri).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
        raise ValueError("output_name may contain only letters, digits, underscore, dot and hyphen.")
    config = config or PersonalDynamicPoolConfig()
    config.validate()
    calendar = _read_calendar(provider_uri)
    dates = _resolve_update_dates(
        calendar,
        update_date=update_date,
        start_date=start_date,
        end_date=end_date,
    )

    quant_master.init(provider_uri=str(provider_uri), region="cn", kernels=kernels)
    states, diagnostics = _load_daily_states(dates, config, chunk_size=chunk_size)
    replacement = _states_to_spans(states, dates)

    instruments_dir = (provider_uri / "instruments").resolve()
    output_path = (instruments_dir / f"{output_name}.txt").resolve()
    metadata_path = output_path.with_suffix(".meta.json")
    if output_path.parent != instruments_dir or metadata_path.parent != instruments_dir:
        raise ValueError("Output paths must remain inside the provider instruments directory.")

    existing = _read_spans(output_path)
    combined = _replace_spans_for_range(existing, replacement, calendar, dates[0], dates[-1])
    output_text = _spans_to_text(combined)
    if not output_text:
        raise ValueError("Refusing to write an empty personal dynamic pool.")

    daily_sizes = [len(states[date]) for date in dates]
    interval_count = sum(len(values) for values in combined.values())
    all_starts = [start for values in combined.values() for start, _ in values]
    all_ends = [end for values in combined.values() for _, end in values]
    old_metadata = _read_metadata(metadata_path)
    update_history = old_metadata.get("update_history", [])
    if not isinstance(update_history, list):
        update_history = []
    updated_at = datetime.now(timezone.utc).isoformat()
    update_history = (update_history + [
        {
            "updated_at_utc": updated_at,
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "trading_days": len(dates),
            "min_daily_size": min(daily_sizes),
            "max_daily_size": max(daily_sizes),
        }
    ])[-200:]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output_path),
        "definition": "daily PIT filter of local all-market A-shares; no CSI constituent dependency",
        "signal_time": "after_close",
        "source_market": "all",
        "source_calendar_latest": str(calendar[-1].date()),
        "filter": asdict(config),
        "rolling_fields_use_prior_data": True,
        "current_day_fields": ["actual_close", "volume", "amount"],
        "excluded_boards": ["bse"],
        "historical_status_limitations": [
            "no exact point-in-time ST-name field",
            "no next-session suspension or price-limit knowledge",
        ],
        "execution_rechecks_required": [
            "actual buy price <= 40 CNY",
            "board permission",
            "ST and delisting-risk status",
            "suspension and price-limit status",
            "100-share lot affordability and fees",
        ],
        "start_time": str(min(all_starts).date()),
        "end_time": str(max(all_ends).date()),
        "latest_update_start": str(dates[0].date()),
        "latest_update_end": str(dates[-1].date()),
        "latest_update_states": diagnostics,
        "unique_instruments": len(combined),
        "interval_count": interval_count,
        "min_latest_update_size": min(daily_sizes),
        "max_latest_update_size": max(daily_sizes),
        "output_sha256": _sha256_text(output_text),
        "updated_at_utc": updated_at,
        "update_history": update_history,
    }

    backups: Tuple[Path, ...] = ()
    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        backups = _backup_existing((output_path, metadata_path), stamp)
        _atomic_write_text(output_path, output_text)
        _atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        from quant_master.data.cache import H  # pylint: disable=import-outside-toplevel

        H["i"].clear()

    return PoolUpdateResult(
        output_path=output_path,
        metadata_path=metadata_path,
        update_start=str(dates[0].date()),
        update_end=str(dates[-1].date()),
        trading_days=len(dates),
        min_daily_size=min(daily_sizes),
        max_daily_size=max(daily_sizes),
        unique_instruments=len(combined),
        interval_count=interval_count,
        dry_run=dry_run,
        backup_paths=backups,
    )


def update_latest_personal_dynamic_pool(
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    **kwargs,
) -> PoolUpdateResult:
    """Update the latest trading date currently present in the local calendar."""

    provider_uri = Path(provider_uri).expanduser().resolve()
    latest = _read_calendar(provider_uri)[-1]
    return update_personal_dynamic_pool(
        provider_uri=provider_uri,
        update_date=str(latest.date()),
        **kwargs,
    )


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_PROVIDER_URI",
    "PersonalDynamicPoolConfig",
    "PoolUpdateResult",
    "classify_board",
    "update_latest_personal_dynamic_pool",
    "update_personal_dynamic_pool",
]
