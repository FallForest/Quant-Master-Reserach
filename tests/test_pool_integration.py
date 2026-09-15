import pytest

from scripts.data_collector.pool_integration import update_personal_pool_after_data_update


def test_pool_integration_is_noop_when_disabled():
    assert update_personal_pool_after_data_update(
        provider_uri="unused",
        region="cn",
        storage_freq="day",
        enabled=False,
    ) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"region": "us", "storage_freq": "day", "enabled": True},
        {"region": "cn", "storage_freq": "1min", "enabled": True},
        {"region": "cn", "storage_freq": "day", "enabled": True, "update_date": "2024-01-01", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        {"region": "cn", "storage_freq": "day", "enabled": True, "start_date": "2024-01-01"},
    ],
)
def test_pool_integration_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        update_personal_pool_after_data_update(provider_uri="unused", **kwargs)
