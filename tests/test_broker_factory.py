from unittest.mock import patch

from quant_master.contrib.broker import EasytraderBroker, PaperBroker, TcDllBroker, XiadanBroker, create_broker


def test_create_broker_default_returns_paper_adapter():
    broker = create_broker()
    assert isinstance(broker, PaperBroker)


def test_create_broker_xiadan_returns_direct_adapter():
    broker = create_broker("xiadan")
    assert isinstance(broker, XiadanBroker)


def test_create_broker_tcdll_returns_tcdll_adapter():
    broker = create_broker("tcdll")
    assert isinstance(broker, TcDllBroker)


def test_create_broker_easytrader_aliases():
    with patch.object(EasytraderBroker, "__init__", return_value=None):
        broker = create_broker("easytrader")
        assert isinstance(broker, EasytraderBroker)

        broker_alias = create_broker("yh_client")
        assert isinstance(broker_alias, EasytraderBroker)


def test_create_broker_tdx_requires_host():
    try:
        create_broker("tdx")
    except ValueError as exc:
        assert "host" in str(exc)
    else:
        raise AssertionError("create_broker('tdx') should require host")
