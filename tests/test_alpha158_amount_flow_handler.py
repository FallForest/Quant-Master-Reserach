import re

import numpy as np
import pandas as pd

from quant_master.contrib.data.amount_flow_handler import (
    Alpha158AmountFlow,
    Alpha158AmountFlowCoreBaseline,
    Alpha158AmountFlowCore,
    Alpha158TDX,
    Alpha360TDX,
)
from quant_master.contrib.data.handler import Alpha158
from quant_master.data.dataset.loader import StaticDataLoader
from quant_master.workflow.record_temp import SignalRecord
from scripts.discover_amount_flow_features import evaluate_features, select_low_redundancy_features


def test_amount_flow_handler_replaces_missing_vwap_and_appends_features():
    base_fields, base_names = Alpha158(init_data=False).get_feature_config()
    fields, names = Alpha158AmountFlow(init_data=False).get_feature_config()
    candidate_fields, candidate_names = Alpha158AmountFlow.get_amount_flow_feature_config()
    candidates = dict(zip(candidate_names, candidate_fields))
    extra_names = list(Alpha158AmountFlow.RECOMMENDED_FEATURES)
    extra_fields = [candidates[name] for name in extra_names]

    assert len(fields) == len(base_fields) + len(extra_fields)
    assert len(names) == len(base_names) + len(extra_names)
    assert "$vwap" not in " ".join(fields).lower()
    assert names[-len(extra_names) :] == extra_names
    assert all(name.startswith(Alpha158AmountFlow.EXTRA_FEATURE_PREFIX) for name in extra_names)
    assert len(candidate_names) == 21
    assert len(extra_names) == 10


def test_amount_flow_features_only_use_current_or_past_observations():
    fields, _ = Alpha158AmountFlow.get_amount_flow_feature_config()
    future_ref = re.compile(r"Ref\s*\([^)]*,\s*-", re.IGNORECASE)

    assert all(future_ref.search(expression) is None for expression in fields)


def test_alpha158_tdx_only_replaces_the_native_vwap_feature():
    base_fields, base_names = Alpha158(init_data=False).get_feature_config()
    fields, names = Alpha158TDX(init_data=False).get_feature_config()

    assert names == base_names
    assert len(fields) == len(names) == 158
    assert "$vwap" not in " ".join(fields).lower()
    assert sum(left != right for left, right in zip(base_fields, fields)) == 1


def test_amount_flow_core_is_a_two_feature_ablation():
    fields, names = Alpha158AmountFlowCore(init_data=False).get_feature_config()

    assert len(fields) == len(names) == 160
    assert names[-2:] == list(Alpha158AmountFlowCore.RECOMMENDED_FEATURES)


def test_amount_flow_core_baseline_only_appends_two_features():
    base_fields, base_names = Alpha158(init_data=False).get_feature_config()
    fields, names = Alpha158AmountFlowCoreBaseline(init_data=False).get_feature_config()

    assert fields[:158] == base_fields
    assert names[:158] == base_names
    assert names[-2:] == list(Alpha158AmountFlowCoreBaseline.RECOMMENDED_FEATURES)


def test_alpha360_tdx_replaces_every_native_vwap_expression():
    fields, names = Alpha360TDX.get_feature_config()

    assert len(fields) == len(names) == 360
    assert len(set(names)) == 360
    assert all("$vwap" not in field for field in fields)
    assert sum("$amount/($volume*100+1e-12)" in field for field in fields) == 60


def test_alpha360_tdx_can_load_raw_data_before_processing():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-01-03", "2023-01-04"]), ["A"]], names=["datetime", "instrument"]
    )
    columns = pd.MultiIndex.from_tuples([("feature", "F0"), ("label", "LABEL0")])
    raw = pd.DataFrame([[1.0, 0.1], [2.0, 0.2]], index=index, columns=columns)
    handler = Alpha360TDX(
        instruments=None,
        start_time="2023-01-03",
        end_time="2023-01-04",
        data_loader=StaticDataLoader(config=raw),
        infer_processors=[],
        learn_processors=[],
        load_only=True,
    )

    assert handler._data.equals(raw)
    assert not hasattr(handler, "_infer")
    handler.fit_process_data()
    assert handler._infer.equals(raw)
    assert handler._learn.equals(raw)


def test_feature_evaluation_requires_stable_rank_ic_sign():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2021-01-01"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {
            "stable": [1, 2, 3, 1, 2, 3],
            "flipped": [1, 2, 3, 3, 2, 1],
            "label": [1, 2, 3, 1, 2, 3],
        },
        index=index,
        dtype=float,
    )

    rows = evaluate_features(
        frame,
        ["stable", "flipped"],
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )

    assert rows[0]["feature"] == "stable"
    assert rows[0]["same_sign"] is True
    assert np.isclose(rows[0]["stable_rank_ic"], 1.0)
    assert rows[1]["same_sign"] is False
    assert rows[1]["stable_rank_ic"] == 0.0


def test_feature_selection_removes_highly_correlated_candidates():
    dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-02"])
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["datetime", "instrument"])
    frame = pd.DataFrame(index=index)
    frame["first"] = [1, 2, 3, 4] * len(dates)
    frame["duplicate"] = frame["first"] * 2
    frame["distinct"] = [1, 4, 2, 3] * len(dates)
    frame["label"] = frame["first"]
    rows = evaluate_features(
        frame,
        ["first", "duplicate", "distinct"],
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )

    selected = select_low_redundancy_features(
        frame,
        rows,
        ("2021-01-01", "2021-12-31"),
        min_stable_rank_ic=0.0,
        max_abs_rank_corr=0.8,
    )

    assert [item["feature"] for item in selected] == ["first", "distinct"]


def test_signal_record_list_supports_dependency_class_casting():
    record = SignalRecord.__new__(SignalRecord)

    assert record.list() == ["pred.pkl", "label.pkl"]
