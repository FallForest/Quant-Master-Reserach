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


def test_sell_clear_oversell_still_raises():
    position = Position(cash=0, position_dict={"SH603296": {"amount": 100.0, "price": 10.0}})
    with pytest.raises(ValueError, match="only have"):
        position._sell_stock(
            "SH603296",
            trade_val=100.2 * 10.0,
            cost=0.0,
            trade_price=10.0,
        )


def test_sell_normal_path_unchanged():
    position = Position(cash=0, position_dict={"SH603296": {"amount": 100.0, "price": 10.0}})
    position._sell_stock("SH603296", trade_val=40.0 * 10.0, cost=1.0, trade_price=10.0)

    assert position.check_stock("SH603296")
    assert position.get_stock_amount("SH603296") == pytest.approx(60.0)
    assert position.get_cash() == pytest.approx(399.0)
