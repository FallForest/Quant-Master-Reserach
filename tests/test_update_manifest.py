import json

from scripts.data_collector.update_manifest import write_update_manifest


def test_write_update_manifest_is_json_and_replaces_previous(tmp_path):
    first = write_update_manifest(
        tmp_path,
        source="tdx",
        interval="1d",
        storage_freq="day",
        start_date="2024-01-01",
        end_date="2024-01-02",
        stages={"dump": "success"},
        options={"skip_pool": True},
    )
    second = write_update_manifest(
        tmp_path,
        source="yahoo",
        interval="1d",
        storage_freq="day",
        start_date="2024-01-02",
        end_date="2024-01-03",
        stages={"dump": "success", "pool": "skipped"},
        options={"skip_pool": False},
    )

    assert first["source"] == "tdx"
    assert second["source"] == "yahoo"
    payload = json.loads((tmp_path / "update_manifest.json").read_text(encoding="utf-8"))
    assert payload == second
    assert not (tmp_path / "update_manifest.json.tmp").exists()
