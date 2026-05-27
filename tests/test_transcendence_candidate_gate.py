from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from examples.benchmarks.Transcendence.support.candidate_gate import evaluate_candidate_gate


def _passing_metrics() -> dict:
    return {
        "ic": 0.031,
        "rank_ic": 0.041,
        "costed_ir": 2.81,
        "costed_annret": 0.25,
        "max_drawdown": -0.05,
        "turnover": 0.12,
        "finite_rows": 562,
        "nonfinite_rows": 0,
        "leakage_check": "pass",
    }


def test_candidate_gate_passes_strict_baseline_without_requiring_sota() -> None:
    result = evaluate_candidate_gate(_passing_metrics())

    assert result["verdict"] == "PASS"
    assert result["passed"] is True
    assert result["checks"]["strict_model_baseline"]["passed"] is True
    assert result["checks"]["aspirational_portfolio_sota"]["passed"] is False
    assert result["checks"]["aspirational_portfolio_sota"]["required_for_pass"] is False


def test_candidate_gate_fails_baseline_comparison() -> None:
    metrics = _passing_metrics()
    metrics["costed_ir"] = 2.7999836767

    result = evaluate_candidate_gate(metrics)

    assert result["verdict"] == "NO_GO"
    assert any("costed_ir" in failure and "strict baseline" in failure for failure in result["failures"])


def test_candidate_gate_fails_missing_cost_field() -> None:
    metrics = _passing_metrics()
    del metrics["costed_annret"]

    result = evaluate_candidate_gate(metrics)

    assert result["verdict"] == "NO_GO"
    assert "missing required field: costed_annret" in result["failures"]
    assert any("cost fields present" in failure for failure in result["failures"])


def test_candidate_gate_fails_nonfinite_rows() -> None:
    metrics = _passing_metrics()
    metrics["nonfinite_rows"] = 1

    result = evaluate_candidate_gate(metrics)

    assert result["verdict"] == "NO_GO"
    assert "nonfinite_rows 1 must equal 0" in result["failures"]


def test_candidate_gate_accepts_summary_json_shape() -> None:
    summary = {
        "leakage_guardrails": {
            "full_selection_uses_test_metrics": False,
            "full_does_not_tune_using_2024_2026": True,
        },
        "test_metrics": {
            "costed_ir": 2.81,
            "costed_annret": 0.25,
            "max_drawdown": -0.048,
            "turnover": 0.11,
            "finite_rows": 562,
            "nonfinite_rows": 0,
        },
        "metrics": {
            "IC": 0.025,
            "RankIC": 0.026,
        },
    }

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "PASS"
    assert result["sources"]["costed_ir"] == "test_metrics.costed_ir"
    assert result["sources"]["leakage_check"] == "leakage_guardrails"


def test_candidate_gate_accepts_summary_alias_metrics() -> None:
    summary = {
        "protocol": {
            "no_2024_2026_tuning": True,
        },
        "test_metrics": {
            "costed_ir": 2.81,
            "costed_annret": 0.25,
            "max_drawdown": -0.048,
            "turnover": 0.11,
            "finite_rows": 562,
            "nonfinite_rows": 0,
        },
        "validation_metrics": {
            "ic_mean": 0.025,
            "rankic_mean": 0.026,
        },
    }

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "PASS"
    assert result["metrics"]["ic"] == 0.025
    assert result["metrics"]["rank_ic"] == 0.026
    assert result["sources"]["ic"] == "validation_metrics.ic_mean"
    assert result["sources"]["rank_ic"] == "validation_metrics.rankic_mean"


def test_candidate_gate_accepts_nested_signal_alias_metrics() -> None:
    summary = {
        "protocol": {
            "no_2024_2026_tuning": True,
        },
        "test_metrics": {
            "costed_ir": 2.81,
            "costed_annret": 0.25,
            "max_drawdown": -0.048,
            "turnover": 0.11,
            "finite_rows": 562,
            "nonfinite_rows": 0,
        },
        "signal": {
            "ic_mean": 0.025,
            "rank_ic_mean": 0.026,
        },
    }

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "PASS"
    assert result["metrics"]["ic"] == 0.025
    assert result["metrics"]["rank_ic"] == 0.026
    assert result["sources"]["ic"] == "signal.ic_mean"
    assert result["sources"]["rank_ic"] == "signal.rank_ic_mean"


def test_candidate_gate_missing_alias_metrics_still_fails_closed() -> None:
    summary = {
        "protocol": {
            "no_2024_2026_tuning": True,
        },
        "test_metrics": {
            "costed_ir": 2.81,
            "costed_annret": 0.25,
            "max_drawdown": -0.048,
            "turnover": 0.11,
            "finite_rows": 562,
            "nonfinite_rows": 0,
        },
    }

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "NO_GO"
    assert "missing required field: ic" in result["failures"]
    assert "missing required field: rank_ic" in result["failures"]


def _overlay_summary() -> dict:
    return {
        "signal_coverage": {
            "rows": 168600,
            "unique_trade_days": 562,
        },
        "leakage_boundary": {
            "guardrails": [
                "All regime features are shifted to t-1 (no t-day realized return/cost/bench in decisions).",
                "Quantile thresholds use expanding history ending at t-1 only.",
                "Strategy universe and rule coefficients are fixed before evaluating 2026_ytd.",
                "No per-slice future pick: single rule runs through full timeline.",
            ],
        },
        "continuous_regime_metrics_full": {
            "costed_annret": 0.27620091509314804,
            "costed_ir": 2.9565999209261453,
            "max_drawdown": -0.04812598495553835,
        },
        "baseline_signal_metrics": {
            "ic": 0.031,
            "rank_ic": 0.041,
        },
        "breakthrough_checks": {
            "beats_7406_full_ir_and_annret": True,
            "has_breakthrough": True,
        },
    }


def test_candidate_gate_accepts_overlay_summary_with_baseline_signal_ic() -> None:
    result = evaluate_candidate_gate(_overlay_summary())

    assert result["verdict"] == "PASS"
    assert result["metrics"]["finite_rows"] == 562
    assert result["metrics"]["nonfinite_rows"] == 0
    assert result["metrics"]["turnover"] is None
    assert result["metrics"]["turnover_explained"] is False
    assert result["sources"]["costed_ir"] == "continuous_regime_metrics_full.costed_ir"
    assert result["sources"]["finite_rows"] == "signal_coverage.unique_trade_days"
    assert result["sources"]["leakage_check"] == "leakage_boundary.guardrails"
    assert result["sources"]["ic"] == "baseline_signal_metrics.ic"
    assert result["sources"]["rank_ic"] == "baseline_signal_metrics.rank_ic"
    assert result["metrics"]["risk_notes"] == [
        "overlay modifies exposure on existing signal; turnover not emitted by source summary"
    ]


def test_candidate_gate_overlay_require_sota_still_blocks_missing_turnover() -> None:
    result = evaluate_candidate_gate(_overlay_summary(), require_sota=True)

    assert result["verdict"] == "NO_GO"
    assert "turnover explained check failed: provide turnover or turnover_explained" in result["failures"]


def test_candidate_gate_overlay_fails_missing_guardrail_closed() -> None:
    summary = _overlay_summary()
    del summary["leakage_boundary"]

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "NO_GO"
    assert "leakage_check 'fail' must be pass" in result["failures"]
    assert result["sources"]["leakage_check"] == "leakage_boundary"


def test_candidate_gate_overlay_fails_missing_ic_closed() -> None:
    summary = _overlay_summary()
    del summary["baseline_signal_metrics"]

    result = evaluate_candidate_gate(summary)

    assert result["verdict"] == "NO_GO"
    assert "missing required field: ic" in result["failures"]
    assert "missing required field: rank_ic" in result["failures"]


def test_candidate_gate_require_sota_blocks_when_sota_not_met() -> None:
    result = evaluate_candidate_gate(_passing_metrics(), require_sota=True)

    assert result["verdict"] == "NO_GO"
    assert any("SOTA" in failure for failure in result["failures"])


def test_candidate_gate_cli_outputs_no_go_and_exit_code(tmp_path: Path) -> None:
    metrics = _passing_metrics()
    metrics["finite_rows"] = 561
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "examples/benchmarks/Transcendence/validate_candidate_gate.py",
            str(metrics_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "NO_GO"
    assert "finite_rows 561 must equal 562" in payload["failures"]
