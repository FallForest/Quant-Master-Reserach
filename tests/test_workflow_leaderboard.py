# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_master.workflow.leaderboard import (
    build_leaderboard,
    build_leaderboard_from_records,
    build_round_artifacts,
    build_round_artifacts_from_records,
    coerce_leaderboard_records,
    compare_to_baseline,
    log_round_artifacts,
    normalize_leaderboard_row,
    recommend_next_action,
    sota_snapshot,
    write_round_artifacts,
)


def test_normalize_canonical_recorder_metrics() -> None:
    row = normalize_leaderboard_row(
        {
            "IC": "0.031",
            "ICIR": 1.5,
            "Rank IC": 0.041,
            "Rank ICIR": 1.8,
            "1day.excess_return_with_cost.annualized_return": 0.25,
            "1day.excess_return_with_cost.information_ratio": 2.8,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
            "1day.turnover": 0.12,
            "recorder_id": "rec-1",
            "name": "baseline",
        }
    )

    assert row["ic"] == 0.031
    assert row["rank_ic"] == 0.041
    assert row["costed_annret"] == 0.25
    assert row["costed_ir"] == 2.8
    assert row["max_drawdown"] == -0.05
    assert row["turnover"] == 0.12
    assert row["recorder_id"] == "rec-1"
    assert row["name"] == "baseline"
    assert row["sources"]["ic"] == "IC"
    assert row["sources"]["costed_ir"] == "1day.excess_return_with_cost.information_ratio"
    assert row["missing"] == []


def test_normalize_search_record_metrics_shape_and_identity_fields() -> None:
    row = normalize_leaderboard_row(
        {
            "run_id": "run-1",
            "id": "row-1",
            "metrics.IC": 0.02,
            "metrics.Rank IC": 0.03,
            "metrics.1day.excess_return_with_cost.annualized_return": 0.21,
            "metrics.1day.excess_return_with_cost.information_ratio": 2.1,
            "metrics.1day.excess_return_with_cost.max_drawdown": -0.07,
            "metrics.1day.turnover": 0.22,
            "params.model.class": "LGBModel",
            "tags.task_hash": "abc123",
        }
    )

    assert row["run_id"] == "run-1"
    assert row["id"] == "row-1"
    assert row["model_class"] == "LGBModel"
    assert row["task_hash"] == "abc123"
    assert row["sources"]["rank_ic"] == "metrics.Rank IC"
    assert row["sources"]["costed_annret"] == "metrics.1day.excess_return_with_cost.annualized_return"
    assert "icir" in row["missing"]
    assert "rank_icir" in row["missing"]


def test_normalize_run_data_metrics_object() -> None:
    run = SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "IC": 0.01,
                "ICIR": 0.5,
                "costed_ir": 1.2,
                "costed_annret": 0.08,
                "max_drawdown": -0.09,
            },
            tags={"task_hash": "run-task"},
        ),
        info=SimpleNamespace(run_id="mlflow-run", run_name="mlflow-name"),
    )

    row = normalize_leaderboard_row(run)

    assert row["run_id"] == "mlflow-run"
    assert row["name"] == "mlflow-name"
    assert row["task_hash"] == "run-task"
    assert row["icir"] == 0.5
    assert row["costed_ir"] == 1.2
    assert row["sources"]["max_drawdown"] == "max_drawdown"


def test_normalize_transcendence_style_aliases_from_nested_summary() -> None:
    row = normalize_leaderboard_row(
        {
            "validation_metrics": {
                "ic_mean": 0.025,
                "rankic_mean": 0.026,
                "rankicir": 1.1,
            },
            "test_metrics": {
                "annret": 0.24,
                "ir": 2.81,
                "mdd": -0.048,
                "turnover": 0.11,
            },
            "model": {"class": "TranscendenceSignalEnsemble"},
        }
    )

    assert row["ic"] == 0.025
    assert row["rank_ic"] == 0.026
    assert row["rank_icir"] == 1.1
    assert row["costed_annret"] == 0.24
    assert row["costed_ir"] == 2.81
    assert row["max_drawdown"] == -0.048
    assert row["model_class"] == "TranscendenceSignalEnsemble"
    assert row["sources"]["ic"] == "validation_metrics.ic_mean"
    assert row["sources"]["costed_ir"] == "test_metrics.ir"


def test_build_leaderboard_sorts_eligible_before_ineligible_with_tiebreak() -> None:
    leaderboard = build_leaderboard(
        [
            {"name": "missing-annret", "costed_ir": 9.0, "max_drawdown": -0.01},
            {"name": "lower", "costed_ir": 2.0, "costed_annret": 0.3, "max_drawdown": -0.06},
            {"name": "tie-better-annret", "costed_ir": 3.0, "costed_annret": 0.4, "max_drawdown": -0.05},
            {"name": "tie-worse-annret", "costed_ir": 3.0, "costed_annret": 0.2, "max_drawdown": -0.04},
        ]
    )

    assert [row["name"] for row in leaderboard] == [
        "tie-better-annret",
        "tie-worse-annret",
        "lower",
        "missing-annret",
    ]
    assert [row["eligible"] for row in leaderboard] == [True, True, True, False]
    assert "missing or nonfinite required field: costed_annret" in leaderboard[-1]["failures"]


def test_coerce_leaderboard_records_accepts_dataframe_to_dict_records() -> None:
    class FakeFrame:
        def to_dict(self, orient=None):
            assert orient == "records"
            return [
                {"name": "frame-best", "costed_ir": 3.2, "costed_annret": 0.32, "max_drawdown": -0.04},
                {"name": "frame-low", "costed_ir": 2.1, "costed_annret": 0.21, "max_drawdown": -0.06},
            ]

    records = coerce_leaderboard_records(FakeFrame())
    leaderboard = build_leaderboard_from_records(FakeFrame())

    assert [record["name"] for record in records] == ["frame-best", "frame-low"]
    assert [row["name"] for row in leaderboard] == ["frame-best", "frame-low"]


def test_coerce_leaderboard_records_accepts_dataframe_iterrows_fallback() -> None:
    class FakeFrame:
        def iterrows(self):
            return iter(
                [
                    (0, {"name": "row-low", "costed_ir": 1.1, "costed_annret": 0.11, "max_drawdown": -0.08}),
                    (1, {"name": "row-best", "costed_ir": 2.4, "costed_annret": 0.24, "max_drawdown": -0.05}),
                ]
            )

    records = coerce_leaderboard_records(FakeFrame())
    leaderboard = build_leaderboard_from_records(FakeFrame())

    assert [record["name"] for record in records] == ["row-low", "row-best"]
    assert [row["name"] for row in leaderboard] == ["row-best", "row-low"]


def test_coerce_leaderboard_records_accepts_recorder_metrics_and_info_identity() -> None:
    class FakeRecorder:
        info = SimpleNamespace(id="recorder-info-id", name="recorder-info-name")

        def list_metrics(self):
            return {"costed_ir": 2.9, "costed_annret": 0.29, "max_drawdown": -0.05}

    records = coerce_leaderboard_records(FakeRecorder())
    leaderboard = build_leaderboard_from_records(FakeRecorder())

    assert records == [
        {
            "costed_ir": 2.9,
            "costed_annret": 0.29,
            "max_drawdown": -0.05,
            "id": "recorder-info-id",
            "name": "recorder-info-name",
        }
    ]
    assert leaderboard[0]["id"] == "recorder-info-id"
    assert leaderboard[0]["name"] == "recorder-info-name"
    assert leaderboard[0]["eligible"] is True


def test_coerce_leaderboard_records_rejects_text_and_unsupported_sources() -> None:
    with pytest.raises(TypeError, match="must not be text"):
        coerce_leaderboard_records("candidate")

    with pytest.raises(TypeError, match="unsupported leaderboard record source"):
        coerce_leaderboard_records(object())

    with pytest.raises(TypeError, match="unsupported record"):
        coerce_leaderboard_records([1, 2, 3])


def test_build_round_artifacts_from_records_uses_adapter() -> None:
    class FakeFrame:
        def to_dict(self, orient=None):
            assert orient == "records"
            return [{"name": "candidate", "costed_ir": 3.5, "costed_annret": 0.35, "max_drawdown": -0.04}]

    bundle = build_round_artifacts_from_records(
        FakeFrame(),
        round_name="adapted-round",
        gates={"costed_ir": 3.0, "costed_annret": 0.3, "max_drawdown": -0.05},
    )

    assert bundle["round_summary"]["status"] == "pass"
    assert bundle["round_summary"]["round_name"] == "adapted-round"
    assert bundle["round_summary"]["counts"] == {"records": 1, "eligible": 1, "ineligible": 0}
    assert bundle["leaderboard"][0]["name"] == "candidate"


def test_sota_snapshot_passes_and_fails_gates() -> None:
    leaderboard = build_leaderboard(
        [
            {
                "name": "best",
                "costed_ir": 3.1,
                "costed_annret": 0.31,
                "max_drawdown": -0.04,
                "turnover": 0.18,
                "ic": 0.03,
            }
        ]
    )

    passed = sota_snapshot(
        leaderboard,
        gates={"costed_ir": 3.0, "costed_annret": 0.3, "max_drawdown": -0.05, "turnover": 0.2, "ic": 0.02},
    )
    failed = sota_snapshot(leaderboard, gates={"costed_ir": 3.1, "max_drawdown": -0.03, "turnover": 0.1})

    assert passed["passed"] is True
    assert passed["best"]["name"] == "best"
    assert passed["metrics"]["costed_ir"] == 3.1
    assert failed["passed"] is False
    assert "costed_ir 3.1 must be > 3.1" in failed["failures"]
    assert "max_drawdown -0.04 must be >= -0.03" in failed["failures"]
    assert "turnover 0.18 must be <= 0.1" in failed["failures"]


def test_sota_snapshot_fails_closed_on_missing_required_or_gated_fields() -> None:
    leaderboard = build_leaderboard(
        [{"name": "no-drawdown", "costed_ir": 2.0, "costed_annret": 0.2}],
    )

    result = sota_snapshot(leaderboard, gates={"ic": 0.01})

    assert result["passed"] is False
    assert "missing required field: max_drawdown" in result["failures"]
    assert "missing gated field: ic" in result["failures"]


def test_compare_to_baseline_passes_for_improvements_and_risk_margins() -> None:
    result = compare_to_baseline(
        {
            "costed_ir": 2.6,
            "costed_annret": 0.28,
            "ic": 0.035,
            "rank_ic": 0.041,
            "max_drawdown": -0.052,
            "turnover": 0.115,
        },
        {
            "costed_ir": 2.4,
            "costed_annret": 0.25,
            "ic": 0.03,
            "rank_ic": 0.038,
            "max_drawdown": -0.05,
            "turnover": 0.10,
        },
        fields=("costed_ir", "costed_annret", "ic", "rank_ic", "max_drawdown", "turnover"),
        margins={"costed_ir": 0.1, "costed_annret": 0.02, "max_drawdown": 0.005, "turnover": 0.02},
    )

    assert result["passed"] is True
    assert result["margins"]["ic"] == 0.0
    assert result["comparisons"]["costed_ir"]["direction"] == ">"
    assert result["comparisons"]["costed_ir"]["threshold"] == 2.5
    assert result["comparisons"]["max_drawdown"]["direction"] == ">="
    assert result["comparisons"]["max_drawdown"]["threshold"] == -0.055
    assert result["comparisons"]["turnover"]["direction"] == "<="
    assert result["comparisons"]["turnover"]["threshold"] == pytest.approx(0.12)


def test_compare_to_baseline_fails_closed_on_missing_or_nonfinite_values() -> None:
    result = compare_to_baseline(
        {"costed_ir": math.nan, "costed_annret": 0.25},
        {"costed_annret": 0.24},
        fields=("costed_ir", "costed_annret"),
    )

    assert result["passed"] is False
    assert "missing or nonfinite candidate field: costed_ir" in result["failures"]
    assert "missing or nonfinite baseline field: costed_ir" in result["failures"]
    assert result["comparisons"]["costed_ir"]["candidate"] is None
    assert result["comparisons"]["costed_ir"]["baseline"] is None


def test_compare_to_baseline_drawdown_and_turnover_semantics() -> None:
    drawdown_failed = compare_to_baseline(
        {"max_drawdown": -0.061},
        {"max_drawdown": -0.05},
        fields=("max_drawdown",),
        margins={"max_drawdown": 0.01},
    )
    turnover_failed = compare_to_baseline(
        {"turnover": 0.131},
        {"turnover": 0.10},
        fields=("turnover",),
        margins={"turnover": 0.03},
    )

    assert drawdown_failed["passed"] is False
    assert "max_drawdown -0.061 must be >= -0.060000000000000005 versus baseline -0.05" in drawdown_failed[
        "failures"
    ]
    assert turnover_failed["passed"] is False
    assert "turnover 0.131 must be <= 0.13 versus baseline 0.1" in turnover_failed["failures"]


def test_bool_nan_and_inf_values_fail_closed() -> None:
    row = normalize_leaderboard_row(
        {
            "IC": True,
            "ICIR": math.nan,
            "Rank IC": math.inf,
            "Rank ICIR": "-inf",
            "costed_ir": False,
            "costed_annret": "not-a-number",
            "max_drawdown": -0.05,
        }
    )

    assert "ic" not in row
    assert "icir" not in row
    assert "rank_ic" not in row
    assert "rank_icir" not in row
    assert "costed_ir" not in row
    assert "costed_annret" not in row
    assert row["missing"] == ["ic", "icir", "rank_ic", "rank_icir", "costed_annret", "costed_ir", "turnover"]

    leaderboard = build_leaderboard([{"name": "bad", "costed_ir": math.inf, "costed_annret": True}])

    assert leaderboard[0]["eligible"] is False
    assert "missing or nonfinite required field: costed_ir" in leaderboard[0]["failures"]
    assert "missing or nonfinite required field: costed_annret" in leaderboard[0]["failures"]


def test_build_round_artifacts_returns_required_keys_and_summary_status() -> None:
    bundle = build_round_artifacts(
        [
            {
                "name": "candidate",
                "costed_ir": 3.2,
                "costed_annret": 0.33,
                "max_drawdown": -0.04,
            }
        ],
        round_name="round-1",
        gates={"costed_ir": 3.0, "costed_annret": 0.3, "max_drawdown": -0.05},
        metadata={"worker": "T1"},
    )

    assert set(bundle) == {"round_summary", "trace_out", "leaderboard", "sota"}
    assert bundle["round_summary"]["status"] == "pass"
    assert bundle["round_summary"]["passed"] is True
    assert bundle["round_summary"]["round_name"] == "round-1"
    assert bundle["round_summary"]["counts"] == {"records": 1, "eligible": 1, "ineligible": 0}
    assert bundle["round_summary"]["best"]["identity"]["name"] == "candidate"
    assert bundle["round_summary"]["best"]["metrics"]["costed_ir"] == 3.2
    assert bundle["round_summary"]["metadata"] == {"worker": "T1"}
    assert bundle["trace_out"]["decisions"]["status"] == "pass"

    failed = build_round_artifacts(
        [{"name": "candidate", "costed_ir": 2.9, "costed_annret": 0.33, "max_drawdown": -0.04}],
        gates={"costed_ir": 3.0},
    )

    assert failed["round_summary"]["status"] == "fail"
    assert failed["round_summary"]["passed"] is False
    assert "costed_ir 2.9 must be > 3.0" in failed["round_summary"]["failures"]


def test_build_round_artifacts_with_baseline_requires_dual_gate_pass() -> None:
    passed = build_round_artifacts(
        [
            {
                "name": "candidate",
                "costed_ir": 3.2,
                "costed_annret": 0.33,
                "ic": 0.04,
                "rank_ic": 0.05,
                "max_drawdown": -0.045,
            }
        ],
        gates={"costed_ir": 3.0, "costed_annret": 0.3, "max_drawdown": -0.05},
        baseline={"costed_ir": 3.0, "costed_annret": 0.31, "ic": 0.03, "rank_ic": 0.04, "max_drawdown": -0.04},
        baseline_margins={"costed_ir": 0.1, "costed_annret": 0.01, "max_drawdown": 0.01},
    )
    failed = build_round_artifacts(
        [
            {
                "name": "candidate",
                "costed_ir": 3.2,
                "costed_annret": 0.33,
                "ic": 0.04,
                "rank_ic": 0.05,
                "max_drawdown": -0.045,
            }
        ],
        gates={"costed_ir": 3.0, "costed_annret": 0.3, "max_drawdown": -0.05},
        baseline={"costed_ir": 3.15, "costed_annret": 0.31, "ic": 0.03, "rank_ic": 0.04, "max_drawdown": -0.05},
        baseline_margins={"costed_ir": 0.1},
    )

    assert passed["sota"]["passed"] is True
    assert passed["sota"]["baseline_comparison"]["passed"] is True
    assert passed["round_summary"]["status"] == "pass"
    assert passed["round_summary"]["passed"] is True
    assert passed["round_summary"]["baseline_passed"] is True
    assert passed["trace_out"]["decisions"]["absolute_passed"] is True
    assert passed["trace_out"]["decisions"]["baseline_passed"] is True

    assert failed["sota"]["passed"] is True
    assert failed["sota"]["baseline_comparison"]["passed"] is False
    assert failed["round_summary"]["status"] == "fail"
    assert failed["round_summary"]["passed"] is False
    assert failed["round_summary"]["baseline_passed"] is False
    assert {
        "baseline_comparison": ["costed_ir 3.2 must be > 3.25 versus baseline 3.15"]
    } in failed["round_summary"]["failures"]


def test_build_round_artifacts_without_baseline_keeps_existing_shape() -> None:
    bundle = build_round_artifacts(
        [{"name": "candidate", "costed_ir": 3.2, "costed_annret": 0.33, "max_drawdown": -0.04}],
        gates={"costed_ir": 3.0},
    )

    assert "baseline_comparison" not in bundle["sota"]
    assert "baseline_passed" not in bundle["round_summary"]
    assert "baseline_comparison" not in bundle["trace_out"]
    assert "absolute_passed" not in bundle["trace_out"]["decisions"]
    assert bundle["round_summary"]["status"] == "pass"
    assert bundle["round_summary"]["passed"] is True


def test_write_round_artifacts_writes_expected_json_files(tmp_path) -> None:
    bundle = build_round_artifacts(
        [{"name": "candidate", "costed_ir": 3.2, "costed_annret": 0.33, "max_drawdown": -0.04}],
        trace_out={"provided": True},
    )

    paths = write_round_artifacts(bundle, tmp_path)

    assert set(paths) == {"round_summary", "trace_out", "leaderboard", "sota"}
    for artifact_name in paths:
        path = tmp_path / f"{artifact_name}.json"
        assert paths[artifact_name] == str(path)
        assert path.exists()
        with path.open(encoding="utf-8") as fp:
            assert json.load(fp) == bundle[artifact_name]

    assert json.loads((tmp_path / "trace_out.json").read_text(encoding="utf-8")) == {"provided": True}


def test_log_round_artifacts_uses_fake_recorder_and_artifact_path() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.calls = []

        def log_artifact(self, path, artifact_path=None):
            with open(path, encoding="utf-8") as fp:
                payload = json.load(fp)
            self.calls.append((path, artifact_path, payload))
            return f"logged:{artifact_path}:{path}"

    bundle = build_round_artifacts(
        [{"name": "candidate", "costed_ir": 3.2, "costed_annret": 0.33, "max_drawdown": -0.04}]
    )
    recorder = FakeRecorder()

    results = log_round_artifacts(recorder, bundle, artifact_path="round")

    assert set(results) == {"round_summary", "trace_out", "leaderboard", "sota"}
    assert [call[1] for call in recorder.calls] == ["round", "round", "round", "round"]
    assert {Path(call[0]).name for call in recorder.calls} == {
        "round_summary.json",
        "trace_out.json",
        "leaderboard.json",
        "sota.json",
    }
    assert recorder.calls[0][2] == bundle["round_summary"]
    assert all(item["logged"] is True for item in results.values())
    assert all("path" not in item for item in results.values())
    assert all(item["artifact_path"] == "round" for item in results.values())
    assert {item["filename"] for item in results.values()} == {
        "round_summary.json",
        "trace_out.json",
        "leaderboard.json",
        "sota.json",
    }


def test_log_round_artifacts_captures_recorder_failures_best_effort() -> None:
    class FailingRecorder:
        def log_artifact(self, path, artifact_path=None):
            if path.endswith("sota.json"):
                raise RuntimeError("cannot log sota")
            return None

    bundle = build_round_artifacts(
        [{"name": "candidate", "costed_ir": 3.2, "costed_annret": 0.33, "max_drawdown": -0.04}]
    )

    results = log_round_artifacts(FailingRecorder(), bundle, artifact_path="round")

    assert results["sota"]["logged"] is False
    assert results["sota"]["error"] == "cannot log sota"
    assert results["sota"]["filename"] == "sota.json"
    assert results["sota"]["artifact_path"] == "round"
    assert "path" not in results["sota"]
    assert "failure" not in results["sota"]
    assert results["round_summary"]["logged"] is True
    assert results["trace_out"]["logged"] is True
    assert results["leaderboard"]["logged"] is True


def test_round_artifacts_fail_closed_when_required_or_gated_metrics_are_missing_or_nonfinite() -> None:
    bundle = build_round_artifacts(
        [
            {
                "name": "bad",
                "costed_ir": math.inf,
                "costed_annret": 0.2,
            }
        ],
        gates={"ic": 0.01},
    )

    assert bundle["round_summary"]["status"] == "fail"
    assert bundle["round_summary"]["passed"] is False
    assert bundle["round_summary"]["counts"] == {"records": 1, "eligible": 0, "ineligible": 1}
    assert bundle["sota"]["best"] is None
    assert "no eligible rows" in bundle["round_summary"]["failures"]
    assert {
        "index": 0,
        "identity": {"name": "bad"},
        "failures": [
            "missing or nonfinite required field: costed_ir",
            "missing or nonfinite sort field: costed_ir",
        ],
    } in bundle["round_summary"]["failures"][-1]["leaderboard"]


def _decision_candidate(**overrides):
    candidate = {
        "name": "candidate",
        "costed_ir": 3.2,
        "costed_annret": 0.33,
        "ic": 0.04,
        "icir": 1.4,
        "rank_ic": 0.05,
        "rank_icir": 1.6,
        "max_drawdown": -0.045,
        "turnover": 0.12,
    }
    candidate.update(overrides)
    return candidate


def _decision_baseline(**overrides):
    baseline = {
        "costed_ir": 3.0,
        "costed_annret": 0.30,
        "ic": 0.03,
        "rank_ic": 0.04,
        "max_drawdown": -0.05,
    }
    baseline.update(overrides)
    return baseline


def _decision_bundle(records=None, *, baseline=None, gates=None):
    return build_round_artifacts(
        [_decision_candidate()] if records is None else records,
        gates=gates
        or {
            "costed_ir": 3.0,
            "costed_annret": 0.3,
            "max_drawdown": -0.05,
        },
        baseline=baseline,
    )


def test_recommend_next_action_smoke_pass_promotes_full_run() -> None:
    decision = recommend_next_action(_decision_bundle(), stage="smoke")

    assert decision["action"] == "promote_full_run"
    assert decision["passed"] is True
    assert decision["stage"] == "smoke"
    assert decision["confidence"] == "medium"
    assert decision["required_followups"] == ["run full experiment"]


def test_recommend_next_action_full_pass_with_baseline_reruns_verify_by_default() -> None:
    decision = recommend_next_action(_decision_bundle(baseline=_decision_baseline()), stage="full")

    assert decision["action"] == "rerun_verify"
    assert decision["passed"] is True
    assert decision["confidence"] == "high"
    assert decision["required_followups"] == ["run verification experiment"]


def test_recommend_next_action_full_pass_with_verification_margin_accepts_candidate() -> None:
    bundle = _decision_bundle(
        records=[_decision_candidate(costed_ir=3.125, costed_annret=0.375, ic=0.25, rank_ic=0.375)],
        baseline=_decision_baseline(costed_ir=3.0, costed_annret=0.25, ic=0.125, rank_ic=0.25),
    )

    decision = recommend_next_action(bundle, stage="full", verification_margin=0.125)

    assert decision["action"] == "accept_candidate"
    assert decision["passed"] is True
    assert decision["reasons"] == ["full gates passed and verification margin cleared"]
    assert decision["required_followups"] == []


def test_recommend_next_action_full_verification_margin_not_met_reruns_verify() -> None:
    bundle = _decision_bundle(baseline=_decision_baseline(costed_ir=3.0, costed_annret=0.30))

    decision = recommend_next_action(bundle, stage="full", verification_margin=0.25)

    assert decision["action"] == "rerun_verify"
    assert decision["passed"] is True
    assert "costed_ir 3.2 must be >= 3.25 for verification margin" in decision["failures"]
    assert decision["required_followups"] == ["run verification experiment"]


def test_recommend_next_action_full_verification_margin_nonfinite_comparison_reruns_verify() -> None:
    bundle = _decision_bundle(baseline=_decision_baseline())
    bundle["sota"]["baseline_comparison"]["comparisons"]["costed_ir"]["candidate"] = None

    decision = recommend_next_action(bundle, stage="full", verification_margin=0.01)

    assert decision["action"] == "rerun_verify"
    assert decision["passed"] is True
    assert "costed_ir needs finite candidate and baseline for verification margin" in decision["failures"]
    assert "full gates passed and verification margin cleared" not in decision["reasons"]


def test_recommend_next_action_verify_pass_accepts_candidate() -> None:
    decision = recommend_next_action(_decision_bundle(baseline=_decision_baseline()), stage="verify")

    assert decision["action"] == "accept_candidate"
    assert decision["passed"] is True
    assert decision["confidence"] == "high"
    assert decision["required_followups"] == []


def test_recommend_next_action_missing_metrics_or_no_eligible_row_needs_metrics() -> None:
    missing_metrics = _decision_bundle(records=[{"name": "bad", "costed_ir": math.inf, "costed_annret": 0.2}])
    no_eligible = _decision_bundle(records=[])

    missing_decision = recommend_next_action(missing_metrics, stage="smoke")
    no_eligible_decision = recommend_next_action(no_eligible, stage="full")

    assert missing_decision["action"] == "needs_metrics"
    assert missing_decision["passed"] is False
    assert missing_decision["confidence"] == "low"
    assert no_eligible_decision["action"] == "needs_metrics"
    assert no_eligible_decision["failures"] == ["no eligible rows"]


def test_recommend_next_action_sota_or_absolute_gate_failure_rejects() -> None:
    bundle = _decision_bundle(gates={"costed_ir": 3.5, "costed_annret": 0.3, "max_drawdown": -0.05})

    decision = recommend_next_action(bundle, stage="smoke")

    assert decision["action"] == "reject"
    assert decision["passed"] is False
    assert "absolute or SOTA gates failed" in decision["reasons"]
    assert "costed_ir 3.2 must be > 3.5" in decision["failures"]


def test_recommend_next_action_require_baseline_missing_comparison_needs_baseline() -> None:
    decision = recommend_next_action(_decision_bundle(), stage="full", require_baseline=True)

    assert decision["action"] == "needs_baseline"
    assert decision["passed"] is False
    assert decision["confidence"] == "low"
    assert decision["failures"] == ["missing baseline comparison"]


def test_recommend_next_action_baseline_comparison_failure_rejects() -> None:
    bundle = _decision_bundle(baseline=_decision_baseline(costed_ir=3.25))

    decision = recommend_next_action(bundle, stage="full")

    assert decision["action"] == "reject"
    assert decision["passed"] is False
    assert "baseline comparison failed" in decision["reasons"]
    assert "costed_ir 3.2 must be > 3.25 versus baseline 3.25" in decision["failures"]


def test_recommend_next_action_unknown_stage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown experiment decision stage"):
        recommend_next_action(_decision_bundle(), stage="pilot")
