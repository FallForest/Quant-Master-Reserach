from datetime import date as _date

from server.handlers import position


class _DummyHandler:
    def __init__(self, body):
        self.body = body
        self.response = None
        self.status = None

    def _read_body(self):
        return self.body

    def _json_response(self, obj, status=200):
        self.response = obj
        self.status = status


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

    result = position._enrich_positions(raw, names=names, quotes=quotes)

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

    monkeypatch.setattr(position.app, "data", _FakeData())
    monkeypatch.setattr(position.app, "tdx_quote", None)

    raw = {
        "cash": 100000.0,
        "positions": {"002572": {"shares": 100, "price": 7.0}},
        "date": "2026-06-02",
    }

    result = position._enrich_positions(raw, quotes={"SZ002572": {"price": 9.33}})

    assert result["positions"][0]["name"] == "索菲亚"
    assert result["positions"][0]["currentPrice"] == 9.33
    assert result["positions"][0]["marketValue"] == 933.0


def test_enrich_positions_falls_back_to_local_close_when_realtime_quote_is_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(position, "_LIVE_DATA_DIR", tmp_path)

    class _FakeData:
        def get_names(self):
            return {"600001": "Test Stock"}

        def get_kline(self, symbol, freq="day"):
            return [{"close": 12.34}]

    class _FakeTDX:
        def fetch_quotes(self, instruments):
            return {"SH600001": {"price": 0}}

    monkeypatch.setattr(position.app, "data", _FakeData())
    monkeypatch.setattr(position.app, "tdx_quote", _FakeTDX())

    raw = {
        "cash": 100000.0,
        "positions": {"600001": {"shares": 100, "price": 10.0}},
        "date": "2026-06-01",
    }

    result = position._enrich_positions(raw)

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

    result = position._enrich_positions(raw, names={}, quotes={})

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
    monkeypatch.setattr(position, "_LIVE_DATA_DIR", tmp_path)

    result = position._load_positions_file()

    assert result["stock_commission_rate"] == 0.0001
    assert result["etf_commission_rate"] == 0.00005
    assert result["stamp_duty_rate"] == 0.0005
    assert result["sh_transfer_fee_rate"] == 0.00001
    assert result["capital_amount"] == 999999.0


def test_calc_trade_fee_uses_sell_stamp_duty_and_sh_transfer_fee():
    fee_settings = position._fee_settings_from_raw({})

    sell_fee = position._calc_trade_fee("SH600000", 100000, "sell", fee_settings)
    buy_fee = position._calc_trade_fee("SZ159915", 100000, "buy", fee_settings)

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


def test_set_account_updates_positions_file_and_returns_enriched_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(position, "_LIVE_DATA_DIR", tmp_path)
    handler = _DummyHandler({
        "capitalAmount": 1500000,
        "stockCommissionRate": 0.00012,
        "etfCommissionRate": 0.00006,
        "stampDutyRate": 0.00045,
        "shTransferFeeRate": 0.00002,
    })

    position.set_account(handler)

    assert handler.status == 200
    assert handler.response["capitalAmount"] == 1500000.0
    assert handler.response["feeSettings"]["stockCommissionRate"] == 0.00012
    assert handler.response["feeSettings"]["etfCommissionRate"] == 0.00006
    assert handler.response["feeSettings"]["stampDutyRate"] == 0.00045
    assert handler.response["feeSettings"]["shTransferFeeRate"] == 0.00002
    assert handler.response["date"] == str(_date.today())

    saved = position._load_positions_file()
    assert saved["capital_amount"] == 1500000.0
    assert saved["stock_commission_rate"] == 0.00012
    assert saved["etf_commission_rate"] == 0.00006
    assert saved["stamp_duty_rate"] == 0.00045
    assert saved["sh_transfer_fee_rate"] == 0.00002


def test_set_account_rejects_invalid_fee_rate():
    handler = _DummyHandler({"capitalAmount": 1000000, "stockCommissionRate": -0.01})

    position.set_account(handler)

    assert handler.status == 400
    assert handler.response == {"error": "stockCommissionRate 必须 >= 0"}


def test_set_account_rejects_invalid_capital_amount():
    handler = _DummyHandler({"capitalAmount": "oops"})

    position.set_account(handler)

    assert handler.status == 400
    assert handler.response == {"error": "capitalAmount 必须为数字"}
