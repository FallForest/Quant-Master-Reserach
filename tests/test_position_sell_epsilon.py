import pytest

from quant_master.backtest.position import Position


def test_sell_micro_oversell_is_clipped_and_passes():
    position = Position(cash=0, position_dict={"SH603296": {"amount": 5161.309634, "price": 10.0}})
    requested_amount = 5161.397462
    position._sell_stock(
        "SH603296",
        trade_val=requested_amount * 10.0,
        cost=0.0,
        trade_price=10.0,
    )

    assert not position.check_stock("SH603296")
    assert position.get_cash() == pytest.approx(5161.309634 * 10.0)


def test_sell_large_position_micro_oversell_is_clipped_and_passes():
    position = Position(cash=0, position_dict={"SZ300661": {"amount": 107734.1514283342, "price": 10.0}})
    requested_amount = 107734.34572806784
    position._sell_stock(
        "SZ300661",
        trade_val=requested_amount * 10.0,
        cost=0.0,
        trade_price=10.0,
    )

    assert not position.check_stock("SZ300661")
    assert position.get_cash() == pytest.approx(107734.1514283342 * 10.0)


def test_sell_conversion_boundary_oversell_is_clipped_and_passes():
    position = Position(cash=0, position_dict={"SZ300502": {"amount": 65498.11258266295, "price": 10.0}})
    requested_amount = 65498.37539069897
    position._sell_stock(
        "SZ300502",
        trade_val=requested_amount * 10.0,
        cost=0.0,
        trade_price=10.0,
    )

    assert not position.check_stock("SZ300502")
    assert position.get_cash() == pytest.approx(65498.11258266295 * 10.0)


@pytest.mark.parametrize(
    ("stock_id", "current_amount", "requested_amount"),
    [
        ("SH605499", 391089.92241522664, 391091.5051663436),
        ("SH688111", 1011695.4183003235, 1011699.3018626958),
    ],
)
def test_sell_audit_relative_drift_is_clipped_and_passes(stock_id, current_amount, requested_amount):
    position = Position(cash=0, position_dict={stock_id: {"amount": current_amount, "price": 10.0}})
    position._sell_stock(
        stock_id,
        trade_val=requested_amount * 10.0,
        cost=0.0,
        trade_price=10.0,
    )

    assert not position.check_stock(stock_id)
    assert position.get_cash() == pytest.approx(current_amount * 10.0)


def test_sell_clear_oversell_still_raises():
    position = Position(cash=0, position_dict={"SH603296": {"amount": 100.0, "price": 10.0}})
    with pytest.raises(ValueError, match="only have"):
        position._sell_stock(
            "SH603296",
            trade_val=100.2 * 10.0,
            cost=0.0,
            trade_price=10.0,
        )


def test_sell_above_sub_share_cap_still_raises():
    position = Position(cash=0, position_dict={"SZ300502": {"amount": 65498.11258266295, "price": 10.0}})
    with pytest.raises(ValueError, match="only have"):
        position._sell_stock(
            "SZ300502",
            trade_val=(65498.11258266295 + 5.001) * 10.0,
            cost=0.0,
            trade_price=10.0,
        )


def test_sell_multi_share_oversell_still_raises():
    position = Position(cash=0, position_dict={"SH688041": {"amount": 161749.7, "price": 10.0}})
    with pytest.raises(ValueError, match="only have"):
        position._sell_stock(
            "SH688041",
            trade_val=(161749.7 + 5.848) * 10.0,
            cost=0.0,
            trade_price=10.0,
        )


def test_sell_normal_path_unchanged():
    position = Position(cash=0, position_dict={"SH603296": {"amount": 100.0, "price": 10.0}})
    position._sell_stock("SH603296", trade_val=40.0 * 10.0, cost=1.0, trade_price=10.0)

    assert position.check_stock("SH603296")
    assert position.get_stock_amount("SH603296") == pytest.approx(60.0)
    assert position.get_cash() == pytest.approx(399.0)
