from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STRICT_BASELINE_COSTED_IR = 2.7999836767
STRICT_BASELINE_COSTED_ANNRET = 0.2446646361
REQUIRED_FINITE_ROWS = 562
REQUIRED_NONFINITE_ROWS = 0
MIN_IC = 0.020
MIN_RANK_IC = 0.020

SOTA_COSTED_IR = 3.0230019402
SOTA_COSTED_ANNRET = 0.3878544155
SOTA_MAX_DRAWDOWN = -0.047723
DEFAULT_MDD_MATERIAL_MARGIN = 0.0005

PASS = "PASS"
NO_GO = "NO_GO"


@dataclass(frozen=True)
class CandidateGateThresholds:
    strict_costed_ir_gt: float = STRICT_BASELINE_COSTED_IR
    strict_costed_annret_gt: float = STRICT_BASELINE_COSTED_ANNRET
    finite_rows_eq: int = REQUIRED_FINITE_ROWS
    nonfinite_rows_eq: int = REQUIRED_NONFINITE_ROWS
    min_ic: float = MIN_IC
    min_rank_ic: float = MIN_RANK_IC
    sota_costed_ir_gt: float = SOTA_COSTED_IR
    sota_costed_annret_gt: float = SOTA_COSTED_ANNRET
    sota_max_drawdown_floor: float = SOTA_MAX_DRAWDOWN
    mdd_material_margin: float = DEFAULT_MDD_MATERIAL_MARGIN


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_int(value: Any) -> Optional[int]:
    numeric = _to_float(value)
    if numeric is None:
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def _get_path(data: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for part in path:
        if not _is_mapping(current) or part not in current:
            return None
        current = current[part]
    return current


def _first_path(data: Dict[str, Any], paths: Iterable[Sequence[str]]) -> Tuple[Any, str]:
    for path in paths:
        value = _get_path(data, path)
        if value is not None:
            return value, ".".join(path)
    return None, ""


def _first_numeric(data: Dict[str, Any], paths: Iterable[Sequence[str]]) -> Tuple[Optional[float], str]:
    for path in paths:
        raw = _get_path(data, path)
        value = _to_float(raw)
        if value is not None:
            return value, ".".join(path)
    return None, ""


def _first_int(data: Dict[str, Any], paths: Iterable[Sequence[str]]) -> Tuple[Optional[int], str]:
    for path in paths:
        raw = _get_path(data, path)
        value = _to_int(raw)
        if value is not None:
            return value, ".".join(path)
    return None, ""


def _has_overlay_summary_schema(data: Dict[str, Any]) -> bool:
    return _is_mapping(data.get("continuous_regime_metrics_full"))


def _normalize_overlay_row_counts(data: Dict[str, Any]) -> Tuple[Optional[int], str, Optional[int], str]:
    if not _has_overlay_summary_schema(data):
        return None, "", None, ""

    coverage = data.get("signal_coverage")
    if not _is_mapping(coverage):
        return None, "", None, ""

    finite_rows = _to_int(coverage.get("unique_trade_days"))
    total_rows = _to_int(coverage.get("rows"))
    if finite_rows is None or total_rows is None or finite_rows <= 0 or total_rows <= 0:
        return None, "", None, ""

    return finite_rows, "signal_coverage.unique_trade_days", 0, "signal_coverage.rows+unique_trade_days"


def _flatten_text(value: Any) -> str:
    parts: List[str] = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            parts.append(_flatten_text(item))
    elif _is_mapping(value):
        for item in value.values():
            parts.append(_flatten_text(item))
    elif value is not None:
        parts.append(str(value))
    return " ".join(part for part in parts if part)


def _text_has_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _normalize_overlay_leakage_check(data: Dict[str, Any]) -> Tuple[Optional[str], str]:
    if "leakage_boundary" not in data and "breakthrough_checks" not in data:
        return None, ""

    boundary = data.get("leakage_boundary")
    if not _is_mapping(boundary):
        return "fail", "leakage_boundary"

    guardrails = boundary.get("guardrails")
    if not isinstance(guardrails, list) or not guardrails:
        return "fail", "leakage_boundary.guardrails"

    guard_text = _flatten_text(guardrails)
    no_same_day = _text_has_any(
        guard_text,
        (
            "t-1",
            "previous-day",
            "previous day",
            "shifted",
            "no t-day",
            "future",
        ),
    )
    historical_thresholds = _text_has_any(
        guard_text,
        (
            "expanding history ending at t-1",
            "history ending at t-1",
            "ending at t-1",
            "prev-day only",
            "previous-day diagnostics only",
        ),
    )
    fixed_rule = _text_has_any(
        guard_text,
        (
            "fixed before evaluating",
            "fixed before",
            "no per-slice future pick",
            "single rule",
            "predeclared",
        ),
    )

    explicit_no_snooping, no_snooping_source = _first_path(
        boundary,
        (
            ("no_snooping",),
            ("no_snooping_check",),
            ("no_snooping_condition",),
            ("guardrails_no_snooping",),
        ),
    )
    if explicit_no_snooping is not None:
        text = str(explicit_no_snooping).strip().lower()
        no_snooping = explicit_no_snooping is True or text in {"pass", "passed", "ok", "true", "yes"}
        return ("pass" if no_snooping else "fail"), f"leakage_boundary.{no_snooping_source}"

    if no_same_day and historical_thresholds and fixed_rule:
        return "pass", "leakage_boundary.guardrails"

    return "fail", "leakage_boundary.guardrails"


def _normalize_leakage_check(data: Dict[str, Any]) -> Tuple[Optional[str], str]:
    raw, source = _first_path(
        data,
        (
            ("leakage_check",),
            ("leakage", "check"),
            ("leakage", "status"),
            ("verification", "leakage_check"),
            ("verification", "status"),
            ("protocol", "leakage_check"),
        ),
    )
    if raw is not None:
        return str(raw).strip().lower(), source

    overlay_leakage, overlay_source = _normalize_overlay_leakage_check(data)
    if overlay_leakage is not None:
        return overlay_leakage, overlay_source

    guard = data.get("leakage_guardrails")
    if _is_mapping(guard):
        negative_flag_failures = (
            "does_not_tune",
            "no_2024_2026_tuning",
            "selection_locked",
            "selection_locked_from_validation_only",
        )
        positive_flag_failures = (
            "uses_test_metrics",
            "selection_uses_test",
            "tune_using_2024_2026",
            "leakage_detected",
        )
        for key, value in guard.items():
            key_l = str(key).lower()
            text = str(value).strip().lower()
            if any(flag in key_l for flag in negative_flag_failures):
                if value is False:
                    return "fail", f"leakage_guardrails.{key}"
                continue
            if any(flag in key_l for flag in positive_flag_failures):
                if value is True:
                    return "fail", f"leakage_guardrails.{key}"
                continue
            if ("status" in key_l or "check" in key_l) and text in {"fail", "failed", "false", "no"}:
                return "fail", f"leakage_guardrails.{key}"
        return "pass", "leakage_guardrails"

    protocol = data.get("protocol")
    if _is_mapping(protocol):
        locked = protocol.get("full_does_not_tune_using_2024_2026")
        no_tuning = protocol.get("no_2024_2026_tuning")
        if locked is True or no_tuning is True:
            return "pass", "protocol"

    return None, ""


def _normalize_turnover_explanation(data: Dict[str, Any]) -> Tuple[bool, str]:
    raw, source = _first_path(
        data,
        (
            ("turnover_explained",),
            ("turnover", "explained"),
            ("sota", "turnover_explained"),
            ("portfolio_sota", "turnover_explained"),
            ("turnover_explanation",),
        ),
    )
    if isinstance(raw, bool):
        return raw, source
    if raw is not None:
        text = str(raw).strip().lower()
        return bool(text) and text not in {"false", "fail", "failed", "no", "none", "missing"}, source

    turnover, turnover_source = _first_numeric(
        data,
        (
            ("turnover",),
            ("test_metrics", "turnover"),
            ("metrics", "test_2024_2026", "turnover"),
            ("selected_strategy", "test_2024_2026", "turnover"),
            ("selected_strategy", "valid_2023", "turnover"),
        ),
    )
    return turnover is not None, turnover_source


def normalize_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    costed_ir, costed_ir_source = _first_numeric(
        data,
        (
            ("costed_ir",),
            ("test_metrics", "costed_ir"),
            ("test_metrics", "ir"),
            ("metrics", "test_2024_2026", "costed_ir"),
            ("metrics", "test_2024_2026", "ir"),
            ("metrics", "verification_2024_2026", "costed_ir"),
            ("metrics", "verification_2024_2026", "ir"),
            ("verification_metrics", "costed_ir"),
            ("verification_metrics", "ir"),
            ("continuous_regime_metrics_full", "costed_ir"),
            ("continuous_regime_metrics_full", "ir"),
        ),
    )
    costed_annret, costed_annret_source = _first_numeric(
        data,
        (
            ("costed_annret",),
            ("test_metrics", "costed_annret"),
            ("test_metrics", "annret"),
            ("metrics", "test_2024_2026", "costed_annret"),
            ("metrics", "test_2024_2026", "annret"),
            ("metrics", "verification_2024_2026", "costed_annret"),
            ("metrics", "verification_2024_2026", "annret"),
            ("verification_metrics", "costed_annret"),
            ("verification_metrics", "annret"),
            ("continuous_regime_metrics_full", "costed_annret"),
            ("continuous_regime_metrics_full", "annret"),
        ),
    )
    max_drawdown, max_drawdown_source = _first_numeric(
        data,
        (
            ("max_drawdown",),
            ("mdd",),
            ("MDD",),
            ("test_metrics", "max_drawdown"),
            ("test_metrics", "mdd"),
            ("test_metrics", "MDD"),
            ("metrics", "test_2024_2026", "max_drawdown"),
            ("metrics", "test_2024_2026", "mdd"),
            ("metrics", "test_2024_2026", "MDD"),
            ("verification_metrics", "max_drawdown"),
            ("verification_metrics", "mdd"),
            ("verification_metrics", "MDD"),
            ("continuous_regime_metrics_full", "max_drawdown"),
            ("continuous_regime_metrics_full", "mdd"),
            ("continuous_regime_metrics_full", "MDD"),
        ),
    )
    turnover, turnover_source = _first_numeric(
        data,
        (
            ("turnover",),
            ("test_metrics", "turnover"),
            ("metrics", "test_2024_2026", "turnover"),
            ("verification_metrics", "turnover"),
        ),
    )
    finite_rows, finite_rows_source = _first_int(
        data,
        (
            ("finite_rows",),
            ("test_finite_rows",),
            ("test_metrics", "finite_rows"),
            ("full_hard_gate", "finite_rows_actual"),
            ("metrics", "test_2024_2026", "finite_rows"),
            ("verification_metrics", "finite_rows"),
        ),
    )
    nonfinite_rows, nonfinite_rows_source = _first_int(
        data,
        (
            ("nonfinite_rows",),
            ("test_nonfinite_rows",),
            ("test_metrics", "nonfinite_rows"),
            ("full_hard_gate", "nonfinite_rows_actual"),
            ("metrics", "test_2024_2026", "nonfinite_rows"),
            ("verification_metrics", "nonfinite_rows"),
        ),
    )
    overlay_finite_rows, overlay_finite_source, overlay_nonfinite_rows, overlay_nonfinite_source = (
        _normalize_overlay_row_counts(data)
    )
    if finite_rows is None:
        finite_rows = overlay_finite_rows
        finite_rows_source = overlay_finite_source
    if nonfinite_rows is None:
        nonfinite_rows = overlay_nonfinite_rows
        nonfinite_rows_source = overlay_nonfinite_source

    ic, ic_source = _first_numeric(
        data,
        (
            ("ic",),
            ("ic_mean",),
            ("IC",),
            ("baseline_signal_ic",),
            ("baseline_signal_ic_mean",),
            ("metrics", "ic"),
            ("metrics", "ic_mean"),
            ("metrics", "IC"),
            ("signal", "ic"),
            ("signal", "ic_mean"),
            ("signal", "IC"),
            ("baseline_signal", "ic"),
            ("baseline_signal", "ic_mean"),
            ("baseline_signal", "IC"),
            ("baseline_signal_metrics", "ic"),
            ("baseline_signal_metrics", "ic_mean"),
            ("baseline_signal_metrics", "IC"),
            ("test_metrics", "ic"),
            ("test_metrics", "ic_mean"),
            ("test_metrics", "IC"),
            ("validation_metrics", "ic"),
            ("validation_metrics", "ic_mean"),
            ("validation_metrics", "IC"),
            ("verification_metrics", "ic"),
            ("verification_metrics", "ic_mean"),
            ("verification_metrics", "IC"),
            ("metrics", "test_2024_2026", "ic"),
            ("metrics", "test_2024_2026", "ic_mean"),
            ("metrics", "test_2024_2026", "IC"),
            ("metrics", "verification_2024_2026", "ic"),
            ("metrics", "verification_2024_2026", "ic_mean"),
            ("metrics", "verification_2024_2026", "IC"),
        ),
    )
    rank_ic, rank_ic_source = _first_numeric(
        data,
        (
            ("rank_ic",),
            ("rank_ic_mean",),
            ("rankic",),
            ("rankic_mean",),
            ("rankIC",),
            ("rankIC_mean",),
            ("RankIC",),
            ("RankIC_mean",),
            ("Rank IC",),
            ("baseline_signal_rank_ic",),
            ("baseline_signal_rank_ic_mean",),
            ("baseline_signal_rankic",),
            ("baseline_signal_rankic_mean",),
            ("metrics", "rank_ic"),
            ("metrics", "rank_ic_mean"),
            ("metrics", "rankic"),
            ("metrics", "rankic_mean"),
            ("metrics", "rankIC"),
            ("metrics", "rankIC_mean"),
            ("metrics", "RankIC"),
            ("metrics", "RankIC_mean"),
            ("metrics", "Rank IC"),
            ("signal", "rank_ic"),
            ("signal", "rank_ic_mean"),
            ("signal", "rankic"),
            ("signal", "rankic_mean"),
            ("signal", "rankIC"),
            ("signal", "rankIC_mean"),
            ("signal", "RankIC"),
            ("signal", "RankIC_mean"),
            ("signal", "Rank IC"),
            ("baseline_signal", "rank_ic"),
            ("baseline_signal", "rank_ic_mean"),
            ("baseline_signal", "rankic"),
            ("baseline_signal", "rankic_mean"),
            ("baseline_signal", "rankIC"),
            ("baseline_signal", "rankIC_mean"),
            ("baseline_signal", "RankIC"),
            ("baseline_signal", "RankIC_mean"),
            ("baseline_signal", "Rank IC"),
            ("baseline_signal_metrics", "rank_ic"),
            ("baseline_signal_metrics", "rank_ic_mean"),
            ("baseline_signal_metrics", "rankic"),
            ("baseline_signal_metrics", "rankic_mean"),
            ("baseline_signal_metrics", "rankIC"),
            ("baseline_signal_metrics", "rankIC_mean"),
            ("baseline_signal_metrics", "RankIC"),
            ("baseline_signal_metrics", "RankIC_mean"),
            ("baseline_signal_metrics", "Rank IC"),
            ("test_metrics", "rank_ic"),
            ("test_metrics", "rank_ic_mean"),
            ("test_metrics", "rankic"),
            ("test_metrics", "rankic_mean"),
            ("test_metrics", "rankIC"),
            ("test_metrics", "rankIC_mean"),
            ("test_metrics", "RankIC"),
            ("test_metrics", "RankIC_mean"),
            ("test_metrics", "Rank IC"),
            ("validation_metrics", "rank_ic"),
            ("validation_metrics", "rank_ic_mean"),
            ("validation_metrics", "rankic"),
            ("validation_metrics", "rankic_mean"),
            ("validation_metrics", "rankIC"),
            ("validation_metrics", "rankIC_mean"),
            ("validation_metrics", "RankIC"),
            ("validation_metrics", "RankIC_mean"),
            ("validation_metrics", "Rank IC"),
            ("verification_metrics", "rank_ic"),
            ("verification_metrics", "rank_ic_mean"),
            ("verification_metrics", "rankic"),
            ("verification_metrics", "rankic_mean"),
            ("verification_metrics", "rankIC"),
            ("verification_metrics", "rankIC_mean"),
            ("verification_metrics", "RankIC"),
            ("verification_metrics", "RankIC_mean"),
            ("verification_metrics", "Rank IC"),
            ("metrics", "test_2024_2026", "rank_ic"),
            ("metrics", "test_2024_2026", "rank_ic_mean"),
            ("metrics", "test_2024_2026", "rankic"),
            ("metrics", "test_2024_2026", "rankic_mean"),
            ("metrics", "test_2024_2026", "rankIC"),
            ("metrics", "test_2024_2026", "rankIC_mean"),
            ("metrics", "test_2024_2026", "RankIC"),
            ("metrics", "test_2024_2026", "RankIC_mean"),
            ("metrics", "test_2024_2026", "Rank IC"),
            ("metrics", "verification_2024_2026", "rank_ic"),
            ("metrics", "verification_2024_2026", "rank_ic_mean"),
            ("metrics", "verification_2024_2026", "rankic"),
            ("metrics", "verification_2024_2026", "rankic_mean"),
            ("metrics", "verification_2024_2026", "rankIC"),
            ("metrics", "verification_2024_2026", "rankIC_mean"),
            ("metrics", "verification_2024_2026", "RankIC"),
            ("metrics", "verification_2024_2026", "RankIC_mean"),
            ("metrics", "verification_2024_2026", "Rank IC"),
        ),
    )
    leakage_check, leakage_check_source = _normalize_leakage_check(data)
    turnover_explained, turnover_explained_source = _normalize_turnover_explanation(data)
    risk_notes: List[str] = []
    if _has_overlay_summary_schema(data) and turnover is None and not turnover_explained:
        risk_notes.append(
            "overlay modifies exposure on existing signal; turnover not emitted by source summary"
        )

    cost_fields_present = costed_ir is not None and costed_annret is not None
    if cost_fields_present and max_drawdown is None:
        # MDD is part of the aspirational SOTA comparison, not the strict model baseline.
        pass

    return {
        "costed_ir": costed_ir,
        "costed_annret": costed_annret,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "finite_rows": finite_rows,
        "nonfinite_rows": nonfinite_rows,
        "ic": ic,
        "rank_ic": rank_ic,
        "leakage_check": leakage_check,
        "turnover_explained": turnover_explained,
        "risk_notes": risk_notes,
        "cost_fields_present": cost_fields_present,
        "sources": {
            "costed_ir": costed_ir_source,
            "costed_annret": costed_annret_source,
            "max_drawdown": max_drawdown_source,
            "turnover": turnover_source,
            "finite_rows": finite_rows_source,
            "nonfinite_rows": nonfinite_rows_source,
            "ic": ic_source,
            "rank_ic": rank_ic_source,
            "leakage_check": leakage_check_source,
            "turnover_explained": turnover_explained_source,
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _missing_failures(metrics: Dict[str, Any], required_fields: Sequence[str]) -> List[str]:
    failures: List[str] = []
    for field in required_fields:
        if metrics.get(field) is None:
            failures.append(f"missing required field: {field}")
    return failures


def evaluate_candidate_gate(
    data: Dict[str, Any],
    *,
    thresholds: CandidateGateThresholds = CandidateGateThresholds(),
    require_sota: bool = False,
) -> Dict[str, Any]:
    metrics = normalize_metrics(data)
    failures: List[str] = []
    baseline_failures: List[str] = []
    minimum_failures: List[str] = []
    integrity_failures: List[str] = []
    sota_failures: List[str] = []

    minimum_failures.extend(_missing_failures(metrics, ("ic", "rank_ic", "costed_ir", "costed_annret")))
    integrity_failures.extend(_missing_failures(metrics, ("finite_rows", "nonfinite_rows", "leakage_check")))

    ic = metrics.get("ic")
    if ic is not None and not ic >= thresholds.min_ic:
        minimum_failures.append(f"IC {_fmt(ic)} must be >= {_fmt(thresholds.min_ic)}")
    rank_ic = metrics.get("rank_ic")
    if rank_ic is not None and not rank_ic >= thresholds.min_rank_ic:
        minimum_failures.append(f"RankIC {_fmt(rank_ic)} must be >= {_fmt(thresholds.min_rank_ic)}")
    if not metrics.get("cost_fields_present"):
        minimum_failures.append("cost fields present check failed: costed_ir and costed_annret are required")

    costed_ir = metrics.get("costed_ir")
    if costed_ir is not None and not costed_ir > thresholds.strict_costed_ir_gt:
        baseline_failures.append(
            f"costed_ir {_fmt(costed_ir)} must be > strict baseline {_fmt(thresholds.strict_costed_ir_gt)}"
        )
    costed_annret = metrics.get("costed_annret")
    if costed_annret is not None and not costed_annret > thresholds.strict_costed_annret_gt:
        baseline_failures.append(
            "costed_annret "
            f"{_fmt(costed_annret)} must be > strict baseline {_fmt(thresholds.strict_costed_annret_gt)}"
        )

    finite_rows = metrics.get("finite_rows")
    if finite_rows is not None and finite_rows != thresholds.finite_rows_eq:
        integrity_failures.append(f"finite_rows {finite_rows} must equal {thresholds.finite_rows_eq}")
    nonfinite_rows = metrics.get("nonfinite_rows")
    if nonfinite_rows is not None and nonfinite_rows != thresholds.nonfinite_rows_eq:
        integrity_failures.append(f"nonfinite_rows {nonfinite_rows} must equal {thresholds.nonfinite_rows_eq}")
    leakage_check = metrics.get("leakage_check")
    if leakage_check is not None and leakage_check not in {"pass", "passed", "ok", "true"}:
        integrity_failures.append(f"leakage_check {leakage_check!r} must be pass")

    if costed_ir is None:
        sota_failures.append("missing SOTA comparison field: costed_ir")
    elif not costed_ir > thresholds.sota_costed_ir_gt:
        sota_failures.append(f"costed_ir {_fmt(costed_ir)} must be > SOTA {_fmt(thresholds.sota_costed_ir_gt)}")

    if costed_annret is None:
        sota_failures.append("missing SOTA comparison field: costed_annret")
    elif not costed_annret > thresholds.sota_costed_annret_gt:
        sota_failures.append(
            f"costed_annret {_fmt(costed_annret)} must be > SOTA {_fmt(thresholds.sota_costed_annret_gt)}"
        )

    max_drawdown = metrics.get("max_drawdown")
    mdd_floor = thresholds.sota_max_drawdown_floor - thresholds.mdd_material_margin
    if max_drawdown is None:
        sota_failures.append("missing SOTA comparison field: max_drawdown")
    elif max_drawdown < mdd_floor:
        sota_failures.append(
            f"max_drawdown {_fmt(max_drawdown)} is materially worse than "
            f"{_fmt(thresholds.sota_max_drawdown_floor)} with margin {_fmt(thresholds.mdd_material_margin)}"
        )

    if not metrics.get("turnover_explained"):
        sota_failures.append("turnover explained check failed: provide turnover or turnover_explained")

    failures.extend(minimum_failures)
    failures.extend(baseline_failures)
    failures.extend(integrity_failures)
    if require_sota:
        failures.extend(sota_failures)

    verdict = PASS if not failures else NO_GO
    return {
        "verdict": verdict,
        "passed": verdict == PASS,
        "failures": failures,
        "checks": {
            "hard_minimum": {
                "passed": not minimum_failures,
                "failures": minimum_failures,
                "thresholds": {
                    "ic_gte": thresholds.min_ic,
                    "rank_ic_gte": thresholds.min_rank_ic,
                    "cost_fields_present": True,
                },
            },
            "strict_model_baseline": {
                "passed": not baseline_failures and not integrity_failures,
                "failures": baseline_failures + integrity_failures,
                "thresholds": {
                    "costed_ir_gt": thresholds.strict_costed_ir_gt,
                    "costed_annret_gt": thresholds.strict_costed_annret_gt,
                    "finite_rows_eq": thresholds.finite_rows_eq,
                    "nonfinite_rows_eq": thresholds.nonfinite_rows_eq,
                    "leakage_check": "pass",
                },
            },
            "aspirational_portfolio_sota": {
                "passed": not sota_failures,
                "required_for_pass": require_sota,
                "failures": sota_failures,
                "thresholds": {
                    "costed_ir_gt": thresholds.sota_costed_ir_gt,
                    "costed_annret_gt": thresholds.sota_costed_annret_gt,
                    "max_drawdown_no_worse_than": thresholds.sota_max_drawdown_floor,
                    "mdd_material_margin": thresholds.mdd_material_margin,
                    "turnover_explained": True,
                },
            },
        },
        "metrics": {key: value for key, value in metrics.items() if key != "sources"},
        "sources": metrics["sources"],
    }


def load_metrics_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"metrics JSON must be an object: {path}")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Transcendence candidate metrics/summary JSON against the reproducible "
            "candidate verification gate."
        )
    )
    parser.add_argument("metrics_json", help="Path to metrics JSON, summary JSON, or hard_gate JSON.")
    parser.add_argument(
        "--require-sota",
        action="store_true",
        help="Make aspirational portfolio SOTA checks blocking instead of informational.",
    )
    parser.add_argument(
        "--mdd-material-margin",
        type=float,
        default=DEFAULT_MDD_MATERIAL_MARGIN,
        help="Allowed material degradation beyond SOTA max_drawdown before SOTA fails.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    thresholds = CandidateGateThresholds(mdd_material_margin=args.mdd_material_margin)
    data = load_metrics_json(Path(args.metrics_json).expanduser().resolve())
    result = evaluate_candidate_gate(data, thresholds=thresholds, require_sota=args.require_sota)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
