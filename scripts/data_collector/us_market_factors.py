from pathlib import Path
from typing import Dict, Iterable, List, Optional

import fire
import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = ["SPY", "QQQ", "DIA", "US_INX", "US_IXIC", "US_NDX", "US_DJI"]
ALIAS_MAP = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DIA": "DIA",
    ".INX": "US_INX",
    "^GSPC": "US_INX",
    "INX": "US_INX",
    "US_INX": "US_INX",
    ".IXIC": "US_IXIC",
    "IXIC": "US_IXIC",
    "US_IXIC": "US_IXIC",
    ".NDX": "US_NDX",
    "^NDX": "US_NDX",
    "NDX": "US_NDX",
    "US_NDX": "US_NDX",
    ".DJI": "US_DJI",
    "^DJI": "US_DJI",
    "DJI": "US_DJI",
    "US_DJI": "US_DJI",
}


def _parse_symbols(symbols) -> List[str]:
    if symbols is None:
        return list(DEFAULT_SYMBOLS)
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]
    return [str(s).strip() for s in symbols if str(s).strip()]


def _read_calendar(data_dir: Path, freq: str = "day") -> pd.DatetimeIndex:
    path = data_dir / "calendars" / f"{freq}.txt"
    if not path.exists():
        raise FileNotFoundError(f"calendar file does not exist: {path}")
    values = pd.read_csv(path, header=None).iloc[:, 0]
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def _feature_dir(data_dir: Path, symbol: str) -> Path:
    return data_dir / "features" / symbol.lower()


def _read_feature(data_dir: Path, calendar: pd.DatetimeIndex, symbol: str, field: str) -> pd.Series:
    path = _feature_dir(data_dir, symbol) / f"{field.lower()}.day.bin"
    if not path.exists():
        return pd.Series(dtype=float, name=field)
    raw = np.fromfile(path, dtype="<f")
    if len(raw) == 0:
        return pd.Series(dtype=float, name=field)
    start_idx = int(raw[0])
    values = raw[1:]
    idx = calendar[start_idx : start_idx + len(values)]
    return pd.Series(values, index=idx, name=field)


def _load_symbol_frame(data_dir: Path, calendar: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    fields = ["adjclose", "close", "open", "high", "low", "volume", "change"]
    cols = {field: _read_feature(data_dir, calendar, symbol, field) for field in fields}
    df = pd.concat(cols.values(), axis=1)
    if df.empty:
        return df
    return df.sort_index()


def _storage_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return ALIAS_MAP.get(normalized, normalized.replace("^", "US_").replace("-", "_").replace(".", "US_"))


def _factor_alias(storage_symbol: str) -> str:
    storage_symbol = storage_symbol.strip().upper()
    if storage_symbol.startswith("US_"):
        return storage_symbol[3:]
    return storage_symbol


def _price_series(df: pd.DataFrame) -> pd.Series:
    if "adjclose" in df and df["adjclose"].notna().any():
        return df["adjclose"].astype(float)
    return df["close"].astype(float)


def _build_symbol_features(alias: str, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if df.empty:
        return out

    price = _price_series(df)
    ret1 = price.pct_change(fill_method=None)
    for window in [1, 3, 5, 20]:
        out[f"US_{alias}_RET_{window}D"] = price / price.shift(window) - 1.0
    for window in [5, 20]:
        out[f"US_{alias}_RVOL_{window}D"] = ret1.rolling(window, min_periods=max(2, window // 2)).std()

    denom = df["close"].replace(0, np.nan)
    out[f"US_{alias}_RANGE_1D"] = (df["high"] - df["low"]) / denom
    return out


def _align_to_cn_calendar(features: pd.DataFrame, cn_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=cn_calendar)
    right = features.copy()
    right["US_SESSION_DATE"] = pd.DatetimeIndex(right.index)
    right = right.sort_index().reset_index(names="us_datetime")
    left = pd.DataFrame({"datetime": cn_calendar}).sort_values("datetime")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="datetime",
        right_on="us_datetime",
        direction="backward",
        allow_exact_matches=False,
    )
    session_date = pd.to_datetime(aligned.pop("US_SESSION_DATE"))
    aligned.drop(columns=["us_datetime"], inplace=True)
    aligned.set_index("datetime", inplace=True)
    stale_days = pd.Series(aligned.index, index=aligned.index) - session_date.values
    aligned["US_SESSION_STALE_DAYS"] = stale_days.dt.days.astype(float)
    return aligned


def build_us_market_factors(
    us_data_dir: str,
    cn_data_dir: str,
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    us_data_dir = Path(us_data_dir).expanduser().resolve()
    cn_data_dir = Path(cn_data_dir).expanduser().resolve()
    us_calendar = _read_calendar(us_data_dir)
    cn_calendar = _read_calendar(cn_data_dir)

    symbol_frames: Dict[str, pd.DataFrame] = {}
    factor_parts = []
    for symbol in _parse_symbols(symbols):
        storage_symbol = _storage_symbol(symbol)
        alias = _factor_alias(storage_symbol)
        df = _load_symbol_frame(us_data_dir, us_calendar, storage_symbol)
        symbol_frames[alias] = df
        if not df.empty:
            factor_parts.append(_build_symbol_features(alias, df))

    factors = pd.concat(factor_parts, axis=1).sort_index() if factor_parts else pd.DataFrame(index=us_calendar)

    if {"QQQ", "SPY"}.issubset(symbol_frames):
        factors["US_QQQ_MINUS_SPY_RET_1D"] = (
            _price_series(symbol_frames["QQQ"]).pct_change(fill_method=None)
            - _price_series(symbol_frames["SPY"]).pct_change(fill_method=None)
        )
    if {"IXIC", "INX"}.issubset(symbol_frames):
        factors["US_IXIC_MINUS_INX_RET_1D"] = (
            _price_series(symbol_frames["IXIC"]).pct_change(fill_method=None)
            - _price_series(symbol_frames["INX"]).pct_change(fill_method=None)
        )
    if {"NDX", "INX"}.issubset(symbol_frames):
        factors["US_NDX_MINUS_INX_RET_1D"] = (
            _price_series(symbol_frames["NDX"]).pct_change(fill_method=None)
            - _price_series(symbol_frames["INX"]).pct_change(fill_method=None)
        )

    return _align_to_cn_calendar(factors.replace([np.inf, -np.inf], np.nan), cn_calendar)


class Run:
    def build(
        self,
        us_data_dir: str,
        cn_data_dir: str,
        output: str,
        symbols: Optional[str] = None,
    ):
        factors = build_us_market_factors(us_data_dir=us_data_dir, cn_data_dir=cn_data_dir, symbols=symbols)
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        factors.to_parquet(output_path)
        print(f"saved {len(factors)} rows x {len(factors.columns)} columns to {output_path}")


if __name__ == "__main__":
    fire.Fire(Run)
