import datetime
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import fire
import pandas as pd
from loguru import logger

try:
    import akshare as ak
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    ak = None


CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataAll, verify_dump


DEFAULT_ETF_SYMBOLS = ["SPY", "QQQ", "DIA"]
DEFAULT_INDEX_SYMBOLS = [".INX", ".IXIC", ".NDX", ".DJI"]
DEFAULT_SYMBOLS = DEFAULT_ETF_SYMBOLS + DEFAULT_INDEX_SYMBOLS
SYMBOL_ALIASES = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DIA": "DIA",
    ".INX": "US_INX",
    ".IXIC": "US_IXIC",
    ".NDX": "US_NDX",
    ".DJI": "US_DJI",
}
SOURCE_SYMBOLS = {alias: symbol for symbol, alias in SYMBOL_ALIASES.items()}
OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "adjclose", "factor", "change", "symbol"]


def parse_symbols(symbols: Optional[Iterable[str]]) -> List[str]:
    if symbols is None:
        return list(DEFAULT_SYMBOLS)
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    parsed = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    return parsed or list(DEFAULT_SYMBOLS)


def alias_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    return SYMBOL_ALIASES.get(symbol, symbol.replace(".", "US_").replace("^", "US_").replace("-", "_"))


def _require_akshare():
    if ak is None:
        raise ImportError("akshare is required for the US market collector. Install it with `pip install akshare`.")
    return ak


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {}
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in {"date", "日期", "时间"}:
            column_map[col] = "date"
        elif normalized in {"open", "开盘", "开盘价"}:
            column_map[col] = "open"
        elif normalized in {"high", "最高", "最高价"}:
            column_map[col] = "high"
        elif normalized in {"low", "最低", "最低价"}:
            column_map[col] = "low"
        elif normalized in {"close", "收盘", "收盘价"}:
            column_map[col] = "close"
        elif normalized in {"volume", "成交量"}:
            column_map[col] = "volume"
    return df.rename(columns=column_map)


def _filter_dates(df: pd.DataFrame, start: str = None, end_date: str = None) -> pd.DataFrame:
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df


def normalize_akshare_frame(df: pd.DataFrame, symbol: str, start: str = None, end_date: str = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _standardize_columns(df.copy())
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{symbol} AkShare data is missing required columns: {missing}")

    df = df.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = _filter_dates(df, start=start, end_date=end_date)
    df = df.drop_duplicates("date").sort_values("date")

    for field in ["open", "high", "low", "close", "volume"]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["adjclose"] = df["close"]
    df["factor"] = 1.0
    df["change"] = df["close"].pct_change(fill_method=None)
    df["symbol"] = alias_symbol(symbol)
    return df.loc[:, OUTPUT_COLUMNS]


def fetch_symbol(symbol: str, start: str = None, end_date: str = None) -> pd.DataFrame:
    client = _require_akshare()
    symbol = str(symbol).strip().upper()
    source_symbol = SOURCE_SYMBOLS.get(symbol, symbol)
    if source_symbol in DEFAULT_ETF_SYMBOLS:
        raw = client.stock_us_daily(symbol=source_symbol, adjust="")
    elif source_symbol in DEFAULT_INDEX_SYMBOLS:
        raw = client.index_us_stock_sina(symbol=source_symbol)
    else:
        raise ValueError(
            f"Unsupported US market symbol: {symbol}. "
            f"Supported symbols are: {','.join(DEFAULT_SYMBOLS + list(SOURCE_SYMBOLS))}"
        )
    return normalize_akshare_frame(raw, symbol=source_symbol, start=start, end_date=end_date)


class Run:
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        if interval.lower() != "1d":
            raise ValueError(f"US market collector only supports 1d data, got interval={interval}")
        self.source_dir = Path(source_dir or CUR_DIR / "source").expanduser().resolve()
        self.normalize_dir = Path(normalize_dir or CUR_DIR / "normalize").expanduser().resolve()
        self.max_workers = max_workers
        self.interval = "day"

    @staticmethod
    def _cleanup_dir(path: Path):
        if not path.exists():
            return
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    @staticmethod
    def _cleanup_quant_master_dir(path: Path):
        for dirname in ["calendars", "instruments", "features"]:
            target = path / dirname
            if target.exists():
                shutil.rmtree(target)

    def download_data(self, symbols: Optional[str] = None, start: str = "2000-01-01", end_date: str = None):
        self.source_dir.mkdir(parents=True, exist_ok=True)
        selected_symbols = parse_symbols(symbols)
        for symbol in selected_symbols:
            logger.info(f"fetch AkShare US market data: {symbol}")
            df = fetch_symbol(symbol=symbol, start=start, end_date=end_date)
            if df.empty:
                logger.warning(f"{symbol} returned no data")
                continue
            output = self.source_dir / f"{alias_symbol(symbol).lower()}.csv"
            df.to_csv(output, index=False, date_format="%Y-%m-%d")
            logger.info(f"saved {len(df)} rows to {output}")

    def normalize_data(self):
        self.normalize_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.source_dir.glob("*.csv")):
            df = pd.read_csv(path)
            missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"{path} is missing normalized columns: {missing}")
            df = df.loc[:, OUTPUT_COLUMNS].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df.to_csv(self.normalize_dir / path.name, index=False)

    def update_data_to_bin(
        self,
        quant_master_data_1d_dir: str,
        start: str = "2000-01-01",
        end_date: str = None,
        symbols: Optional[str] = None,
        overwrite: bool = True,
    ):
        if end_date is None:
            end_date = pd.Timestamp(datetime.datetime.now()).strftime("%Y-%m-%d")

        quant_master_dir = Path(quant_master_data_1d_dir).expanduser().resolve()
        self._cleanup_dir(self.source_dir)
        self._cleanup_dir(self.normalize_dir)
        self.download_data(symbols=symbols, start=start, end_date=end_date)
        self.normalize_data()

        if overwrite:
            self._cleanup_quant_master_dir(quant_master_dir)
        quant_master_dir.mkdir(parents=True, exist_ok=True)

        dump = DumpDataAll(
            data_path=str(self.normalize_dir),
            quant_master_dir=str(quant_master_dir),
            freq="day",
            exclude_fields="symbol,date",
            max_workers=max(int(self.max_workers), 1),
        )
        dump.dump()
        verify_dump(str(quant_master_dir), expected_end_date=None, freq="day")
        logger.info(f"US market QuantMaster data saved to {quant_master_dir}")


if __name__ == "__main__":
    fire.Fire(Run)
