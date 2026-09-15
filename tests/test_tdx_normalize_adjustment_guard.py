import pandas as pd
import pytest

from scripts.data_collector.tdx.collector import (
    Run,
    TdxNormalizeCN1d,
    TdxNormalizeCN1min,
    UnadjustedPriceError,
    _normalize_all,
)


def _bars():
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000.0, 1100.0],
            "amount": [1010000.0, 1122000.0],
        }
    )


def _minute_bars():
    return _bars().assign(amount=[10100.0, 11220.0])


def test_daily_normalizer_rejects_synthetic_factor_by_default():
    with pytest.raises(UnadjustedPriceError, match="Refusing to emit daily factor=1"):
        TdxNormalizeCN1d().adjusted_price(_bars())


def test_daily_normalizer_requires_explicit_unadjusted_opt_in():
    result = TdxNormalizeCN1d(allow_unadjusted=True).adjusted_price(_bars())

    assert result["factor"].eq(1.0).all()
    assert result["adjclose"].equals(result["close"])
    assert result["vwap"].tolist() == pytest.approx([10.1, 10.2])


def test_minute_normalizer_preserves_backward_compatible_default():
    result = TdxNormalizeCN1min().adjusted_price(_minute_bars())

    assert result["factor"].eq(1.0).all()
    assert result["vwap"].tolist() == pytest.approx([10.1, 10.2])


def test_xdxr_mode_uses_adjclose_factor_and_reweights_ohlcv():
    bars = _bars().assign(adjclose=[20.2, 20.4])
    result = TdxNormalizeCN1d(adjustment_mode="xdxr").adjusted_price(bars)

    assert result["factor"].tolist() == pytest.approx([2.0, 2.0])
    assert result["open"].tolist() == pytest.approx([20.0, 20.2])
    assert result["volume"].tolist() == pytest.approx([500.0, 550.0])
    assert result["vwap"].tolist() == pytest.approx([20.2, 20.4])
    assert result["adjclose"].equals(result["close"])


def test_strict_mode_accepts_explicit_adjclose_source():
    bars = _bars().assign(adjclose=[20.2, 20.4])
    result = TdxNormalizeCN1d().adjusted_price(bars)
    assert result["factor"].eq(2.0).all()


@pytest.mark.parametrize(
    "bars, message",
    [
        (_bars().assign(adjclose=[None, 20.4]), "missing leading"),
        (_bars().assign(close=[0.0, 10.2], adjclose=[20.2, 20.4]), "non-positive close"),
        (_bars().assign(adjclose=[-20.2, 20.4]), "invalid adjustment factor"),
    ],
)
def test_xdxr_mode_fails_closed_for_invalid_or_missing_factors(bars, message):
    with pytest.raises(UnadjustedPriceError, match=message):
        TdxNormalizeCN1d(adjustment_mode="xdxr").adjusted_price(bars)


def test_bulk_daily_normalize_propagates_adjustment_guard(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "normalized"
    source_dir.mkdir()
    _bars().to_csv(source_dir / "000001.csv", index=False)

    with pytest.raises(UnadjustedPriceError):
        _normalize_all(source_dir, target_dir, interval="1d")

    assert not list(target_dir.glob("*.csv"))


def test_daily_binary_update_fails_before_download_or_cleanup(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sentinel = source_dir / "keep.csv"
    sentinel.write_text("existing temporary input", encoding="utf-8")
    run = object.__new__(Run)
    run.source_dir = source_dir
    run.normalize_dir = tmp_path / "normalized"
    run.interval = "1d"
    download_called = False

    def fail_if_downloaded(*args, **kwargs):
        nonlocal download_called
        download_called = True

    monkeypatch.setattr("scripts.data_collector.tdx.collector._download_all", fail_if_downloaded)

    with pytest.raises(UnadjustedPriceError):
        run.update_data_to_bin(str(tmp_path / "data"))

    assert not download_called
    assert sentinel.exists()


def test_daily_xdxr_binary_update_explains_upstream_source_requirement(tmp_path, monkeypatch):
    run = object.__new__(Run)
    run.source_dir = tmp_path / "source"
    run.normalize_dir = tmp_path / "normalized"
    run.interval = "1d"
    monkeypatch.setattr("scripts.data_collector.tdx.collector._download_all", pytest.fail)

    with pytest.raises(UnadjustedPriceError, match="already enriched with a reliable PIT adjclose/XDXR"):
        run.update_data_to_bin(str(tmp_path / "data"), adjustment_mode="xdxr")
