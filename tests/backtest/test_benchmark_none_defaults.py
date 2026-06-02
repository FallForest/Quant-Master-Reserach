from quant_master.backtest import create_account_instance


def test_create_account_instance_keeps_explicit_benchmark_disable():
    account = create_account_instance(
        start_time="2024-01-01",
        end_time="2024-01-31",
        benchmark=None,
        account=1000000,
    )

    assert account.benchmark_config == {"benchmark": None}
