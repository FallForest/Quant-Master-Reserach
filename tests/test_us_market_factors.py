from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_master.contrib.data.processor import USMarketFeatureJoin
from scripts.data_collector import us_market_factors
from scripts.data_collector.us_market import collector as us_market_collector


def _write_calendar(root: Path, dates):
    cal_dir = root / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")


def _write_symbol(root: Path, symbol: str, dates, fields):
    inst_dir = root / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    all_path = inst_dir / "all.txt"
    with all_path.open("a", encoding="utf-8") as fp:
        fp.write(f"{symbol}\t{dates[0]}\t{dates[-1]}\n")

    feat_dir = root / "features" / symbol.lower()
    feat_dir.mkdir(parents=True, exist_ok=True)
    for field, values in fields.items():
        np.hstack([0, np.asarray(values, dtype=np.float32)]).astype("<f").tofile(feat_dir / f"{field}.day.bin")


def _akshare_frame(closes):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000, 1100, 1200],
        }
    )


def test_akshare_us_market_collector_routes_and_dumps(monkeypatch, tmp_path):
    calls = []

    class FakeAkShare:
        @staticmethod
        def stock_us_daily(symbol, adjust=""):
            calls.append(("stock_us_daily", symbol, adjust))
            return _akshare_frame([100.0, 110.0, 121.0])

        @staticmethod
        def index_us_stock_sina(symbol):
            calls.append(("index_us_stock_sina", symbol))
            return _akshare_frame([200.0, 220.0, 242.0])

    monkeypatch.setattr(us_market_collector, "ak", FakeAkShare)

    run = us_market_collector.Run(
        source_dir=tmp_path / "source",
        normalize_dir=tmp_path / "normalize",
        max_workers=1,
    )
    output_dir = tmp_path / "us_market_data"
    run.update_data_to_bin(
        quant_master_data_1d_dir=str(output_dir),
        symbols="SPY,.INX",
        start="2024-01-01",
        end_date="2024-01-03",
    )

    assert calls == [("stock_us_daily", "SPY", ""), ("index_us_stock_sina", ".INX")]

    csv_df = pd.read_csv(tmp_path / "normalize" / "spy.csv")
    assert list(csv_df.columns) == us_market_collector.OUTPUT_COLUMNS
    assert csv_df["factor"].dropna().eq(1.0).all()
    assert csv_df.loc[1, "change"] == pytest.approx(0.1)
    assert csv_df["symbol"].unique().tolist() == ["SPY"]

    assert (output_dir / "calendars" / "day.txt").exists()
    assert (output_dir / "instruments" / "all.txt").exists()
    assert (output_dir / "features" / "spy" / "close.day.bin").exists()
    assert (output_dir / "features" / "us_inx" / "close.day.bin").exists()


def test_build_us_market_factors_uses_prior_us_session_only(tmp_path):
    us_root = tmp_path / "us"
    cn_root = tmp_path / "cn"
    _write_calendar(us_root, ["2024-01-01", "2024-01-02", "2024-01-03"])
    _write_calendar(cn_root, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06"])

    base_fields = {
        "adjclose": [100.0, 110.0, 220.0],
        "close": [100.0, 110.0, 220.0],
        "high": [101.0, 111.0, 222.0],
        "low": [99.0, 109.0, 218.0],
    }
    _write_symbol(us_root, "SPY", ["2024-01-01", "2024-01-03"], base_fields)

    factors = us_market_factors.build_us_market_factors(
        us_data_dir=str(us_root),
        cn_data_dir=str(cn_root),
        symbols=["SPY"],
    )

    assert pd.isna(factors.loc[pd.Timestamp("2024-01-02"), "US_SPY_RET_1D"])
    assert factors.loc[pd.Timestamp("2024-01-03"), "US_SPY_RET_1D"] == pytest.approx(0.1)
    assert factors.loc[pd.Timestamp("2024-01-04"), "US_SPY_RET_1D"] == pytest.approx(1.0)
    assert factors.loc[pd.Timestamp("2024-01-06"), "US_SESSION_STALE_DAYS"] == pytest.approx(3.0)


def test_build_us_market_factors_uses_akshare_aliases_without_vix(tmp_path):
    us_root = tmp_path / "us"
    cn_root = tmp_path / "cn"
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _write_calendar(us_root, dates)
    _write_calendar(cn_root, ["2024-01-04"])

    for symbol, close in {
        "SPY": [100.0, 101.0, 102.0],
        "QQQ": [100.0, 102.0, 104.0],
        "US_INX": [100.0, 101.0, 103.0],
        "US_IXIC": [100.0, 103.0, 106.0],
        "US_NDX": [100.0, 104.0, 108.0],
    }.items():
        _write_symbol(
            us_root,
            symbol,
            dates,
            {
                "adjclose": close,
                "close": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
            },
        )

    factors = us_market_factors.build_us_market_factors(
        us_data_dir=str(us_root),
        cn_data_dir=str(cn_root),
    )

    assert "US_QQQ_MINUS_SPY_RET_1D" in factors.columns
    assert "US_IXIC_MINUS_INX_RET_1D" in factors.columns
    assert "US_NDX_MINUS_INX_RET_1D" in factors.columns
    assert "US_INX_RET_1D" in factors.columns
    assert not any("VIX" in column for column in factors.columns)


def test_us_market_feature_join_broadcasts_date_factors(tmp_path):
    factor_path = tmp_path / "us_market.parquet"
    pd.DataFrame(
        {
            "US_SPY_RET_1D": [0.01, -0.02],
            "US_SESSION_STALE_DAYS": [1.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ).to_parquet(factor_path)

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["SH600000", "SH600004"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {("feature", "BASE"): [1.0, 2.0, 3.0, 4.0], ("label", "LABEL0"): [0.1, 0.2, 0.3, 0.4]},
        index=index,
    )

    out = USMarketFeatureJoin(factor_path=str(factor_path))(df)

    assert ("feature", "BASE") in out.columns
    assert ("label", "LABEL0") in out.columns
    assert ("feature", "US_SPY_RET_1D") in out.columns
    assert out.loc[(pd.Timestamp("2024-01-02"), "SH600004"), ("feature", "US_SPY_RET_1D")] == pytest.approx(0.01)
    assert out.loc[(pd.Timestamp("2024-01-03"), "SH600000"), ("feature", "US_SESSION_STALE_DAYS")] == pytest.approx(3.0)
