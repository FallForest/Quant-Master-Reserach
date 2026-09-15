import pytest

from scripts.check_example_provider_uris import audit_provider_uris, scan_provider_uris


def _write_yaml(path, provider_uri):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'quant_master_init:\n  provider_uri: "{provider_uri}"\n', encoding="utf-8")


def test_scan_provider_uris_skips_intraday_by_default(tmp_path):
    _write_yaml(tmp_path / "daily.yaml", "~/.quant_master/quant_master_data/tdx_cn_data")
    _write_yaml(tmp_path / "minute.yaml", "~/.quant_master/quant_master_data/cn_data_1min")

    records = scan_provider_uris(tmp_path)

    assert records == [(tmp_path / "daily.yaml", "~/.quant_master/quant_master_data/tdx_cn_data")]


def test_provider_uri_audit_can_gate_against_environment_value(tmp_path, monkeypatch):
    expected = "~/.quant_master/quant_master_data/approved_cn_data"
    _write_yaml(tmp_path / "matching.yaml", expected)
    _write_yaml(tmp_path / "mismatch.yaml", "~/.quant_master/quant_master_data/tdx_cn_data")
    monkeypatch.setenv("QUANT_MASTER_PROVIDER_URI", expected)

    with pytest.raises(RuntimeError, match="1 mismatched workflow files"):
        audit_provider_uris(tmp_path, fail_on_mismatch=True)


def test_provider_uri_gate_requires_explicit_expected_value(tmp_path, monkeypatch):
    _write_yaml(tmp_path / "daily.yaml", "~/.quant_master/quant_master_data/tdx_cn_data")
    monkeypatch.delenv("QUANT_MASTER_PROVIDER_URI", raising=False)

    with pytest.raises(ValueError, match="QUANT_MASTER_PROVIDER_URI"):
        audit_provider_uris(tmp_path, fail_on_mismatch=True)
