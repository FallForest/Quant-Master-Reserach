from datetime import date as _date

from server.helpers import calc_trade_fee, fee_settings_from_raw
from server.position_service import enrich_positions, load_positions_file


def test_enrich_positions_matches_plain_code_with_prefixed_quote_and_name():
    raw = {
        "cash": 998183.0,
        "positions": {
            "000676": {"shares": 200, "price": 9.085},
        },
        "date": "2026-06-01",
    }
    names = {"000676": "智度股份"}
    quotes = {"SZ000676": {"price": 9.37}}

    result = enrich_positions(raw, names=names, quotes=quotes)

    assert result["positionCount"] == 1
    assert result["totalMarketValue"] == 1874.0
    assert result["positions"][0]["name"] == "智度股份"
    assert result["positions"][0]["currentPrice"] == 9.37
    assert result["positions"][0]["marketValue"] == 1874.0
    assert result["positions"][0]["pnlPct"] == 3.14


def test_enrich_positions_prefers_valid_realtime_quote_over_local_close(monkeypatch):
    class _FakeData:
        def get_names(self):
            return {"002572": "索菲亚"}

        def get_kline(self, symbol, freq="day"):
            return [{"close": 6.06}]

    raw = {
        "cash": 100000.0,
        "positions": {"002572": {"shares": 100, "price": 7.0}},
        "date": "2026-06-02",
    }

    result = enrich_positions(raw, data=_FakeData(), quotes={"SZ002572": {"price": 9.33}})

    assert result["positions"][0]["name"] == "索菲亚"
    assert result["positions"][0]["currentPrice"] == 9.33
    assert result["positions"][0]["marketValue"] == 933.0


def test_enrich_positions_falls_back_to_local_close_when_realtime_quote_is_zero():
    class _FakeData:
        def get_names(self):
            return {"600001": "Test Stock"}

        def get_kline(self, symbol, freq="day"):
            return [{"close": 12.34}]

    raw = {
        "cash": 100000.0,
        "positions": {"600001": {"shares": 100, "price": 10.0}},
        "date": "2026-06-01",
    }

    result = enrich_positions(raw, data=_FakeData(), quotes={"SH600001": {"price": 0}})

    assert result["positions"][0]["currentPrice"] == 12.34
    assert result["positions"][0]["marketValue"] == 1234.0
    assert result["positions"][0]["pnlPct"] == 23.4
    assert result["totalPnl"] == 234.0


def test_enrich_positions_includes_account_settings_fields():
    raw = {
        "cash": 1000000.0,
        "positions": {},
        "date": "2026-06-01",
        "stock_commission_rate": 0.0001,
        "etf_commission_rate": 0.00005,
        "stamp_duty_rate": 0.0005,
        "sh_transfer_fee_rate": 0.00001,
        "capital_amount": 1200000.0,
    }

    result = enrich_positions(raw, names={}, quotes={})

    assert result["capitalAmount"] == 1200000.0
    assert result["feeSettings"]["stockCommissionRate"] == 0.0001
    assert result["feeSettings"]["etfCommissionRate"] == 0.00005
    assert result["feeSettings"]["stampDutyRate"] == 0.0005
    assert result["feeSettings"]["shTransferFeeRate"] == 0.00001
    assert "股票万1" in result["feeRuleSummary"]


def test_load_positions_file_backfills_account_settings_defaults(tmp_path, monkeypatch):
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir(parents=True)
    (positions_dir / "latest.json").write_text(
        '{"cash": 888888.0, "positions": {}, "date": "2026-06-01", "total_assets": 999999.0}',
        encoding="utf-8",
    )
    monkeypatch.setattr("server.config.LIVE_DATA_DIR", tmp_path)

    result = load_positions_file()

    assert result["stock_commission_rate"] == 0.0001
    assert result["etf_commission_rate"] == 0.00005
    assert result["stamp_duty_rate"] == 0.0005
    assert result["sh_transfer_fee_rate"] == 0.00001
    assert result["capital_amount"] == 999999.0


def test_calc_trade_fee_uses_sell_stamp_duty_and_sh_transfer_fee():
    fee_settings = fee_settings_from_raw({})

    sell_fee = calc_trade_fee("SH600000", 100000, "sell", fee_settings)
    buy_fee = calc_trade_fee("SZ159915", 100000, "buy", fee_settings)

    assert sell_fee == {
        "commission": 10.0,
        "stampDuty": 50.0,
        "transferFee": 1.0,
        "total": 61.0,
    }
    assert buy_fee == {
        "commission": 5.0,
        "stampDuty": 0.0,
        "transferFee": 0.0,
        "total": 5.0,
    }


def test_set_account_updates_positions_file_and_returns_enriched_payload(client, monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.LIVE_DATA_DIR", tmp_path)

    r = client.post("/api/positions/account", json={
        "capitalAmount": 1500000,
        "stockCommissionRate": 0.00012,
        "etfCommissionRate": 0.00006,
        "stampDutyRate": 0.00045,
        "shTransferFeeRate": 0.00002,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["capitalAmount"] == 1500000.0
    assert data["feeSettings"]["stockCommissionRate"] == 0.00012
    assert data["feeSettings"]["etfCommissionRate"] == 0.00006
    assert data["feeSettings"]["stampDutyRate"] == 0.00045
    assert data["feeSettings"]["shTransferFeeRate"] == 0.00002
    assert data["date"] == str(_date.today())

    saved = load_positions_file()
    assert saved["capital_amount"] == 1500000.0
    assert saved["stock_commission_rate"] == 0.00012
    assert saved["etf_commission_rate"] == 0.00006
    assert saved["stamp_duty_rate"] == 0.00045
    assert saved["sh_transfer_fee_rate"] == 0.00002


def test_set_account_rejects_invalid_fee_rate(client):
    r = client.post("/api/positions/account", json={
        "capitalAmount": 1000000,
        "stockCommissionRate": -0.01,
    })
    assert r.status_code == 400
    assert r.json()["error"] == "stockCommissionRate 必须 >= 0"


def test_set_account_rejects_invalid_capital_amount(client):
    r = client.post("/api/positions/account", json={"capitalAmount": "oops"})
    assert r.status_code == 422  # Pydantic validation error
