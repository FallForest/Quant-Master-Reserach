import threading

import pandas as pd
import pytest

from server.model_service import ModelService, drop_invalid_live_predictions
from server.sync import _update_instrument_end_dates


def test_sync_updates_all_txt_without_touching_index_membership(tmp_path):
    instruments_dir = tmp_path / "instruments"
    instruments_dir.mkdir(exist_ok=True)
    all_path = instruments_dir / "all.txt"
    csi300_path = instruments_dir / "csi300.txt"

    all_path.write_text(
        "SH600000\t2020-01-01\t2026-06-01\n"
        "SZ000001\t2020-01-01\t2026-06-01\n",
        encoding="utf-8",
    )
    original_index = "SH600000\t2020-01-01\t2021-12-31\n"
    csi300_path.write_text(original_index, encoding="utf-8")

    updated = _update_instrument_end_dates(all_path, {"SH600000": "2026-06-10"})

    assert updated == 1
    assert "SH600000\t2020-01-01\t2026-06-10" in all_path.read_text(encoding="utf-8")
    assert csi300_path.read_text(encoding="utf-8") == original_index


def test_live_prediction_universe_guard_rejects_polluted_csi300_count():
    svc = ModelService.__new__(ModelService)
    svc._registry = {"rhce_v1": {"run_id": "run", "instruments": "csi300"}}
    svc._lock = threading.Lock()
    svc._dataset_cache = {}
    svc._pred_cache = {}
    svc._get_run = lambda alias: (None, svc._registry[alias])

    polluted_index = [f"SH{i:06d}" for i in range(697)]
    day_data = pd.DataFrame({"score": range(697)}, index=polluted_index)

    with pytest.raises(ValueError, match="Universe guard failed"):
        svc._validate_live_prediction_universe("rhce_v1", day_data, pd.Timestamp("2026-06-10"))


def test_drop_invalid_live_predictions_removes_polluted_live_groups():
    historical_index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-06-02")], [f"SH{i:06d}" for i in range(300)]],
        names=["date", "instrument"],
    )
    live_index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-06-10")], [f"SH{i:06d}" for i in range(697)]],
        names=["date", "instrument"],
    )
    historical = pd.DataFrame({"score": 1.0, "source": "historical"}, index=historical_index)
    live = pd.DataFrame({"score": 2.0, "source": "live"}, index=live_index)
    df = pd.concat([historical, live])

    cleaned, dropped = drop_invalid_live_predictions(df, "csi300")

    assert dropped == 697
    assert len(cleaned) == 300
    assert cleaned.index.get_level_values(0).unique().tolist() == [pd.Timestamp("2026-06-02")]
