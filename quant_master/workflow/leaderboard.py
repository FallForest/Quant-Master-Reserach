# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pure helpers for comparing workflow recorder metrics.

The functions in this module intentionally avoid recorder or MLflow lifecycle
calls. They accept already-fetched metric rows, normalize common metric aliases,
and produce small dictionaries suitable for dashboards, gates, and reports.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CANONICAL_METRICS: Tuple[str, ...] = (
    "ic",
    "icir",
    "rank_ic",
    "rank_icir",
    "costed_annret",
    "costed_ir",
    "max_drawdown",
    "turnover",
)

ROUND_ARTIFACT_NAMES: Tuple[str, ...] = ("round_summary", "trace_out", "leaderboard", "sota")

BASELINE_BENEFIT_METRICS: Tuple[str, ...] = (
    "costed_ir",
    "costed_annret",
    "ic",
    "icir",
    "rank_ic",
    "rank_icir",
)

DECISION_STAGES: Tuple[str, ...] = ("smoke", "full", "verify")
OPTIONAL_CONFIDENCE_METRICS: Tuple[str, ...] = ("ic", "icir", "rank_ic", "rank_icir", "turnover")

DEFAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ic": ("IC", "metrics.IC", "ic", "ic_mean"),
    "icir": ("ICIR", "metrics.ICIR", "icir", "ic_ir"),
    "rank_ic": (
        "Rank IC",
        "metrics.Rank IC",
        "rank_ic",
        "rank_ic_mean",
        "rankic",
        "rankic_mean",
        "RankIC",
    ),
    "rank_icir": ("Rank ICIR", "metrics.Rank ICIR", "rank_icir", "rankicir", "rank_ic_ir"),
    "costed_annret": (
        "1day.excess_return_with_cost.annualized_return",
        "metrics.1day.excess_return_with_cost.annualized_return",
        "costed_annret",
        "annualized_return_with_cost",
        "annret",
    ),
    "costed_ir": (
        "1day.excess_return_with_cost.information_ratio",
        "metrics.1day.excess_return_with_cost.information_ratio",
        "costed_ir",
        "information_ratio_with_cost",
        "ir",
    ),
    "max_drawdown": (
        "1day.excess_return_with_cost.max_drawdown",
        "metrics.1day.excess_return_with_cost.max_drawdown",
        "max_drawdown",
        "mdd",
        "MDD",
    ),
    "turnover": ("1day.turnover", "metrics.1day.turnover", "turnover"),
}


def _is_mapping_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    return hasattr(value, "items") and not isinstance(value, (str, bytes))


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if _is_mapping_like(value):
        try:
            return dict(value.items())
        except (TypeError, ValueError):
            return None
    return None


def _get_attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return None


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return str(value)


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        full_key = f"{prefix}.{key}" if prefix else key
        flattened[full_key] = value
        nested = _as_mapping(value)
        if nested is not None:
            flattened.update(_flatten_mapping(nested, full_key))
    return flattened


def _extract_input(record_or_metrics: Any) -> Dict[str, Any]:
    data = _as_mapping(record_or_metrics)
    if data is not None:
        return _flatten_mapping(data)

    extracted: Dict[str, Any] = {}
    run_data = _get_attr(record_or_metrics, "data")
    metrics = _as_mapping(_get_attr(run_data, "metrics")) if run_data is not None else None
    if metrics is None:
        metrics = _as_mapping(_get_attr(record_or_metrics, "metrics"))
    if metrics is not None:
        extracted.update(_flatten_mapping(metrics))

    tags = _as_mapping(_get_attr(run_data, "tags")) if run_data is not None else None
    if tags is not None:
        extracted.update(_flatten_mapping(tags, "tags"))

    info = _get_attr(record_or_metrics, "info")
    for key in ("run_id", "id", "name", "run_name", "recorder_id"):
        value = _get_attr(info, key) if info is not None else None
        if value is None:
            value = _get_attr(record_or_metrics, key)
        if value is not None:
            extracted[key] = value
    return extracted


def _merge_aliases(aliases: Optional[Mapping[str, Iterable[str]]]) -> Dict[str, Tuple[str, ...]]:
    merged = {key: tuple(values) for key, values in DEFAULT_ALIASES.items()}
    if aliases is None:
        return merged

    for canonical, values in aliases.items():
        existing = merged.get(canonical, ())
        custom = tuple(str(value) for value in values)
        merged[canonical] = custom + tuple(value for value in existing if value not in custom)
    return merged


def _source_for_alias(flattened: Mapping[str, Any], alias: str) -> Optional[str]:
    if alias in flattened:
        return alias

    prefixed = f"metrics.{alias}"
    if prefixed in flattened:
        return prefixed

    suffix = f".{alias}"
    for key in sorted(flattened):
        if key.endswith(suffix):
            return key
    return None


def _first_numeric(flattened: Mapping[str, Any], aliases: Sequence[str]) -> Tuple[Optional[float], str]:
    for alias in aliases:
        source = _source_for_alias(flattened, alias)
        if source is None:
            continue
        value = _to_float(flattened[source])
        if value is not None:
            return value, source
    return None, ""


def _copy_identity(row: Dict[str, Any], flattened: Mapping[str, Any]) -> None:
    direct_fields = ("recorder_id", "run_id", "id", "name", "model_class", "task_hash")
    for field in direct_fields:
        if field in flattened and flattened[field] is not None:
            row[field] = flattened[field]

    alias_fields = (
        ("run_id", ("info.run_id",)),
        ("name", ("run_name", "info.run_name", "info.name")),
        ("model_class", ("model.class", "params.model.class", "tags.model.class")),
        ("task_hash", ("tags.task_hash", "params.task_hash")),
    )
    for target, sources in alias_fields:
        if target in row:
            continue
        for source in sources:
            if source in flattened and flattened[source] is not None:
                row[target] = flattened[source]
                break


def _is_record_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if _as_mapping(value) is not None:
        return True

    metrics = _as_mapping(_get_attr(value, "metrics"))
    if metrics is not None:
        return True

    data = _get_attr(value, "data")
    if data is not None and (
        _as_mapping(_get_attr(data, "metrics")) is not None or _as_mapping(_get_attr(data, "tags")) is not None
    ):
        return True

    return False


def _coerce_record_list(records: Any, *, source_label: str) -> List[Any]:
    if isinstance(records, (str, bytes, bytearray)):
        raise TypeError(f"{source_label} must yield records, not text")

    try:
        coerced = list(records)
    except TypeError as exc:
        raise TypeError(f"{source_label} must be iterable") from exc

    for index, record in enumerate(coerced):
        if not _is_record_like(record):
            raise TypeError(
                f"{source_label} yielded unsupported record at index {index}: {type(record).__name__}"
            )
    return coerced


def _lookup_identity_value(container: Any, names: Sequence[str]) -> Any:
    data = _as_mapping(container)
    if data is not None:
        for name in names:
            value = data.get(name)
            if value is not None:
                return value

    for name in names:
        value = _get_attr(container, name)
        if value is not None:
            return value
    return None


def _recorder_metrics_record(recorder: Any) -> dict:
    list_metrics = _get_attr(recorder, "list_metrics")
    if not callable(list_metrics):
        raise TypeError(f"unsupported leaderboard record source: {type(recorder).__name__}")

    metrics = _as_mapping(list_metrics())
    if metrics is None:
        raise TypeError("recorder list_metrics() must return a mapping")

    record = dict(metrics)
    info = _get_attr(recorder, "info")
    identity_sources = (
        ("recorder_id", ("recorder_id",)),
        ("run_id", ("run_id",)),
        ("id", ("id",)),
        ("name", ("name", "run_name")),
    )
    for target, names in identity_sources:
        if record.get(target) is not None:
            continue
        value = _lookup_identity_value(info, names)
        if value is None:
            value = _lookup_identity_value(recorder, names)
        if value is not None:
            record[target] = value
    return record


def _row_from_iterrows_item(item: Any) -> Any:
    if isinstance(item, tuple) and len(item) == 2:
        row = item[1]
    else:
        row = item

    if _as_mapping(row) is not None:
        return row

    to_dict = _get_attr(row, "to_dict")
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return row
    return row


def coerce_leaderboard_records(source: Any) -> list:
    """Adapt common experiment query outputs into leaderboard record inputs.

    This helper is intentionally pure: it only reads already-fetched objects and
    never imports pandas, MLflow, or mutates recorder lifecycle state.
    """

    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError("leaderboard record source must not be text")

    if _as_mapping(source) is not None:
        return [source]

    list_metrics = _get_attr(source, "list_metrics")
    if callable(list_metrics):
        return [_recorder_metrics_record(source)]

    to_dict = _get_attr(source, "to_dict")
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
        except Exception:
            records = None
        else:
            return _coerce_record_list(records, source_label="DataFrame-like source")

    iterrows = _get_attr(source, "iterrows")
    if callable(iterrows):
        rows = [_row_from_iterrows_item(item) for item in iterrows()]
        return _coerce_record_list(rows, source_label="DataFrame-like source")

    if isinstance(source, Iterable):
        return _coerce_record_list(source, source_label="leaderboard record source")

    raise TypeError(f"unsupported leaderboard record source: {type(source).__name__}")


def normalize_leaderboard_row(record_or_metrics: Any, aliases: Optional[Mapping[str, Iterable[str]]] = None) -> dict:
    """Normalize recorder/search/run metrics into canonical leaderboard fields.

    Accepted inputs include plain metric dictionaries, row-like mappings with
    ``metrics.*`` columns, nested metric summaries, and MLflow run-like objects
    whose metrics live at ``run.data.metrics``. Boolean, nonnumeric, NaN, and
    infinite metric values are treated as missing.
    """

    flattened = _extract_input(record_or_metrics)
    merged_aliases = _merge_aliases(aliases)

    row: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    missing: List[str] = []
    _copy_identity(row, flattened)

    for metric in CANONICAL_METRICS:
        value, source = _first_numeric(flattened, merged_aliases.get(metric, (metric,)))
        if value is None:
            missing.append(metric)
            continue
        row[metric] = value
        sources[metric] = source

    row["sources"] = sources
    row["missing"] = missing
    return row


def _is_finite_metric(row: Mapping[str, Any], field: str) -> bool:
    return _to_float(row.get(field)) is not None


def compare_to_baseline(
    candidate: Any,
    baseline: Any,
    *,
    margins: Optional[Mapping[str, Any]] = None,
    fields: Sequence[str] = ("costed_ir", "costed_annret", "ic", "rank_ic", "max_drawdown"),
) -> dict:
    """Compare a candidate row against a baseline using relative gates.

    Benefit metrics must be strictly greater than ``baseline + margin``.
    ``max_drawdown`` is commonly represented as a negative value, so the
    candidate must be at least ``baseline - margin`` to avoid materially worse
    drawdown. ``turnover`` must be no more than ``baseline + margin`` to avoid
    materially worse trading cost pressure. Missing, nonnumeric, NaN, and
    infinite candidate, baseline, or margin values fail closed.
    """

    candidate_row = normalize_leaderboard_row(candidate)
    baseline_row = normalize_leaderboard_row(baseline)
    margin_values = dict(margins or {})
    resolved_margins: Dict[str, float] = {}
    comparisons: Dict[str, dict] = {}
    failures: List[str] = []

    for raw_field in fields:
        field = str(raw_field)
        candidate_value = _to_float(candidate_row.get(field))
        baseline_value = _to_float(baseline_row.get(field))
        margin = _to_float(margin_values.get(field, 0.0))

        comparison = {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "margin": margin,
            "passed": False,
        }
        comparisons[field] = comparison

        if margin is None:
            failures.append(f"nonfinite baseline margin: {field}")
            continue
        resolved_margins[field] = margin

        missing_value = False
        if candidate_value is None:
            failures.append(f"missing or nonfinite candidate field: {field}")
            missing_value = True
        if baseline_value is None:
            failures.append(f"missing or nonfinite baseline field: {field}")
            missing_value = True
        if missing_value:
            continue

        if field == "max_drawdown":
            threshold = baseline_value - margin
            passed = candidate_value >= threshold
            comparison["threshold"] = threshold
            comparison["direction"] = ">="
            if not passed:
                failures.append(f"{field} {candidate_value} must be >= {threshold} versus baseline {baseline_value}")
        elif field == "turnover":
            threshold = baseline_value + margin
            passed = candidate_value <= threshold
            comparison["threshold"] = threshold
            comparison["direction"] = "<="
            if not passed:
                failures.append(f"{field} {candidate_value} must be <= {threshold} versus baseline {baseline_value}")
        elif field in BASELINE_BENEFIT_METRICS:
            threshold = baseline_value + margin
            passed = candidate_value > threshold
            comparison["threshold"] = threshold
            comparison["direction"] = ">"
            if not passed:
                failures.append(f"{field} {candidate_value} must be > {threshold} versus baseline {baseline_value}")
        else:
            failures.append(f"unsupported baseline comparison field: {field}")
            continue

        comparison["passed"] = passed

    return {
        "passed": not failures,
        "comparisons": comparisons,
        "failures": failures,
        "margins": resolved_margins,
    }


def build_leaderboard(
    records: Iterable[Any],
    *,
    sort_by: str = "costed_ir",
    descending: bool = True,
    required: Sequence[str] = ("costed_ir", "costed_annret"),
) -> List[dict]:
    """Normalize and sort records into a comparable leaderboard.

    Rows are eligible only when all required fields and the sort field are
    present as finite numeric values. Eligible rows sort before ineligible rows.
    """

    rows = [normalize_leaderboard_row(record) for record in records]
    required_fields = tuple(required)

    for row in rows:
        failures: List[str] = []
        for field in required_fields:
            if not _is_finite_metric(row, field):
                failures.append(f"missing or nonfinite required field: {field}")
        if not _is_finite_metric(row, sort_by):
            failures.append(f"missing or nonfinite sort field: {sort_by}")
        row["eligible"] = not failures
        row["failures"] = failures

    def sort_key(item: Tuple[int, dict]) -> Tuple[Any, ...]:
        index, row = item
        if not row["eligible"]:
            return (1, index)

        primary = _to_float(row.get(sort_by))
        annret = _to_float(row.get("costed_annret"))
        primary_key = -primary if descending else primary
        annret_key = -annret if annret is not None else math.inf
        identity = str(row.get("run_id") or row.get("recorder_id") or row.get("id") or row.get("name") or "")
        return (0, primary_key, annret_key, identity, index)

    return [row for _, row in sorted(enumerate(rows), key=sort_key)]


def build_leaderboard_from_records(source: Any, **kwargs: Any) -> List[dict]:
    """Build a leaderboard from experiment query, recorder, or raw record output."""

    return build_leaderboard(coerce_leaderboard_records(source), **kwargs)


def sota_snapshot(
    leaderboard: Iterable[Mapping[str, Any]],
    *,
    gates: Optional[Mapping[str, Any]] = None,
    required: Sequence[str] = ("costed_ir", "costed_annret", "max_drawdown"),
) -> dict:
    """Return a SOTA-style pass/fail snapshot for the best eligible row.

    Gate comparators are intentionally simple: most metrics must be greater
    than their threshold. ``max_drawdown`` must be at least the threshold because
    drawdown is commonly represented as a negative value, so ``-0.04`` is no
    worse than ``-0.05``. ``turnover`` must be less than or equal to its
    threshold.
    """

    gate_values = dict(gates or {})
    best = next((row for row in leaderboard if row.get("eligible")), None)
    failures: List[str] = []

    if best is None:
        return {
            "passed": False,
            "best": None,
            "metrics": {},
            "failures": ["no eligible rows"],
            "gates": gate_values,
        }

    for field in required:
        if not _is_finite_metric(best, field):
            failures.append(f"missing required field: {field}")

    for field, raw_threshold in gate_values.items():
        threshold = _to_float(raw_threshold)
        if threshold is None:
            failures.append(f"nonfinite gate threshold: {field}")
            continue

        value = _to_float(best.get(field))
        if value is None:
            failures.append(f"missing gated field: {field}")
            continue

        if field == "max_drawdown":
            if value < threshold:
                failures.append(f"{field} {value} must be >= {threshold}")
        elif field == "turnover":
            if value > threshold:
                failures.append(f"{field} {value} must be <= {threshold}")
        elif value <= threshold:
            failures.append(f"{field} {value} must be > {threshold}")

    metrics = {field: best[field] for field in CANONICAL_METRICS if _is_finite_metric(best, field)}
    return {
        "passed": not failures,
        "best": best,
        "metrics": metrics,
        "failures": failures,
        "gates": gate_values,
    }


def _identity_from_row(row: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if row is None:
        return None

    identity_fields = ("recorder_id", "run_id", "id", "name", "model_class", "task_hash")
    identity = {field: row[field] for field in identity_fields if row.get(field) is not None}
    return identity or None


def _row_failure_summary(leaderboard: Sequence[Mapping[str, Any]]) -> List[dict]:
    failures: List[dict] = []
    for index, row in enumerate(leaderboard):
        row_failures = row.get("failures")
        if not row_failures:
            continue
        failures.append(
            {
                "index": index,
                "identity": _identity_from_row(row),
                "failures": list(row_failures) if isinstance(row_failures, (list, tuple)) else [str(row_failures)],
            }
        )
    return failures


def build_round_artifacts(
    records: Iterable[Any],
    *,
    round_name: Optional[str] = None,
    gates: Optional[Mapping[str, Any]] = None,
    baseline: Optional[Any] = None,
    baseline_margins: Optional[Mapping[str, Any]] = None,
    sort_by: str = "costed_ir",
    required: Sequence[str] = ("costed_ir", "costed_annret"),
    sota_required: Sequence[str] = ("costed_ir", "costed_annret", "max_drawdown"),
    trace_out: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a JSON-safe round artifact bundle from already-fetched records."""

    required_fields = tuple(required)
    sota_required_fields = tuple(sota_required)
    gate_values = dict(gates or {})
    leaderboard = build_leaderboard(records, sort_by=sort_by, required=required_fields)
    sota = sota_snapshot(leaderboard, gates=gate_values, required=sota_required_fields)
    best = sota.get("best")
    absolute_passed = bool(sota.get("passed"))
    baseline_comparison = None
    if baseline is not None:
        baseline_comparison = compare_to_baseline(best or {}, baseline, margins=baseline_margins)
        sota["baseline_comparison"] = baseline_comparison

    row_failures = _row_failure_summary(leaderboard)
    failures = list(sota.get("failures", []))
    if baseline_comparison is not None and not baseline_comparison.get("passed"):
        failures.append({"baseline_comparison": list(baseline_comparison.get("failures", []))})
    if row_failures:
        failures.append({"leaderboard": row_failures})

    baseline_passed = baseline_comparison is None or bool(baseline_comparison.get("passed"))
    round_passed = absolute_passed and baseline_passed

    round_summary = {
        "status": "pass" if round_passed else "fail",
        "passed": round_passed,
        "round_name": round_name,
        "counts": {
            "records": len(leaderboard),
            "eligible": sum(1 for row in leaderboard if row.get("eligible")),
            "ineligible": sum(1 for row in leaderboard if not row.get("eligible")),
        },
        "sort_by": sort_by,
        "required": list(required_fields),
        "sota_required": list(sota_required_fields),
        "gates": gate_values,
        "best": {
            "identity": _identity_from_row(best),
            "metrics": dict(sota.get("metrics", {})),
        }
        if best is not None
        else None,
        "failures": failures,
    }
    if baseline_comparison is not None:
        round_summary["baseline_passed"] = bool(baseline_comparison.get("passed"))
    if metadata is not None:
        round_summary["metadata"] = dict(metadata)

    if trace_out is None:
        trace = {
            "normalization": {
                "canonical_metrics": list(CANONICAL_METRICS),
                "required": list(required_fields),
                "sota_required": list(sota_required_fields),
            },
            "sort": {"sort_by": sort_by, "descending": True},
            "gates": gate_values,
            "decisions": {
                "passed": round_passed,
                "status": round_summary["status"],
                "failures": failures,
            },
        }
        if baseline_comparison is not None:
            trace["baseline_comparison"] = baseline_comparison
            trace["decisions"]["absolute_passed"] = absolute_passed
            trace["decisions"]["baseline_passed"] = bool(baseline_comparison.get("passed"))
    else:
        trace = trace_out

    return _json_safe(
        {
            "round_summary": round_summary,
            "trace_out": trace,
            "leaderboard": leaderboard,
            "sota": sota,
        }
    )


def build_round_artifacts_from_records(source: Any, **kwargs: Any) -> dict:
    """Build round artifacts from experiment query, recorder, or raw record output."""

    return build_round_artifacts(coerce_leaderboard_records(source), **kwargs)


def _eligible_count(bundle: Mapping[str, Any]) -> int:
    round_summary = _as_mapping(bundle.get("round_summary")) or {}
    counts = _as_mapping(round_summary.get("counts")) or {}
    value = counts.get("eligible")
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    leaderboard = bundle.get("leaderboard")
    if isinstance(leaderboard, Sequence) and not isinstance(leaderboard, (str, bytes, bytearray)):
        return sum(1 for row in leaderboard if _as_mapping(row) is not None and row.get("eligible"))
    return 0


def _failure_indicates_missing_evidence(failure: Any) -> bool:
    if isinstance(failure, str):
        lowered = failure.lower()
        return (
            lowered == "no eligible rows"
            or lowered.startswith("missing ")
            or lowered.startswith("nonfinite ")
            or "missing or nonfinite" in lowered
            or "missing required field" in lowered
            or "missing gated field" in lowered
            or "nonfinite gate threshold" in lowered
        )
    failure_map = _as_mapping(failure)
    if failure_map is not None:
        return any(_failure_indicates_missing_evidence(item) for item in failure_map.values())
    if isinstance(failure, (list, tuple)):
        return any(_failure_indicates_missing_evidence(item) for item in failure)
    return False


def _sota_failures(bundle: Mapping[str, Any]) -> List[Any]:
    sota = _as_mapping(bundle.get("sota")) or {}
    failures = sota.get("failures", [])
    if isinstance(failures, (list, tuple)):
        return list(failures)
    return [failures]


def _round_failures(bundle: Mapping[str, Any]) -> List[Any]:
    round_summary = _as_mapping(bundle.get("round_summary")) or {}
    failures = round_summary.get("failures", [])
    if isinstance(failures, (list, tuple)):
        return list(failures)
    return [failures]


def _baseline_comparison(bundle: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    sota = _as_mapping(bundle.get("sota")) or {}
    comparison = _as_mapping(sota.get("baseline_comparison"))
    return comparison


def _verification_margin_passed(
    baseline_comparison: Mapping[str, Any],
    verification_margin: Any,
) -> Tuple[bool, List[str]]:
    """Return whether all available benefit comparisons meet verification margin."""

    margin = _to_float(verification_margin)
    if margin is None:
        return False, ["verification_margin must be finite"]

    comparisons = _as_mapping(baseline_comparison.get("comparisons")) or {}
    checked: List[str] = []
    failures: List[str] = []
    for field in BASELINE_BENEFIT_METRICS:
        comparison = _as_mapping(comparisons.get(field))
        if comparison is None:
            continue
        candidate = _to_float(comparison.get("candidate"))
        baseline = _to_float(comparison.get("baseline"))
        if candidate is None or baseline is None:
            failures.append(f"{field} needs finite candidate and baseline for verification margin")
            continue
        checked.append(field)
        threshold = baseline + margin
        if candidate < threshold:
            failures.append(f"{field} {candidate} must be >= {threshold} for verification margin")

    if not checked:
        return False, ["verification_margin needs at least one finite baseline benefit comparison"]
    return not failures, failures


def _decision_confidence(stage: str, bundle: Mapping[str, Any], baseline_comparison: Optional[Mapping[str, Any]]) -> str:
    sota = _as_mapping(bundle.get("sota")) or {}
    metrics = _as_mapping(sota.get("metrics")) or {}
    if any(_to_float(metrics.get(field)) is None for field in OPTIONAL_CONFIDENCE_METRICS):
        return "low"
    if stage in ("full", "verify") and baseline_comparison is not None and baseline_comparison.get("passed"):
        return "high"
    if stage == "smoke":
        return "medium"
    return "medium"


def recommend_next_action(
    round_artifacts: Mapping[str, Any],
    *,
    stage: str = "smoke",
    require_baseline: bool = False,
    verification_margin: Optional[Any] = None,
) -> dict:
    """Translate round artifacts into a deterministic next experiment action.

    The helper is deliberately fail-closed: missing required evidence returns
    ``needs_metrics`` or ``needs_baseline`` instead of inferring unreported
    values. For ``stage="full"``, a candidate normally moves to verification;
    when ``verification_margin`` is finite and every available baseline benefit
    comparison meets or exceeds that margin, the candidate can be accepted immediately.
    """

    normalized_stage = str(stage)
    if normalized_stage not in DECISION_STAGES:
        raise ValueError(f"unknown experiment decision stage: {stage}")

    bundle = _as_mapping(round_artifacts)
    if bundle is None:
        raise TypeError("round_artifacts must be a mapping")

    sota = _as_mapping(bundle.get("sota")) or {}
    baseline_comparison = _baseline_comparison(bundle)
    sota_failures = _sota_failures(bundle)
    round_failures = _round_failures(bundle)
    confidence = _decision_confidence(normalized_stage, bundle, baseline_comparison)

    decision: Dict[str, Any] = {
        "action": "",
        "passed": False,
        "stage": normalized_stage,
        "reasons": [],
        "failures": [],
        "required_followups": [],
        "confidence": confidence,
    }

    if _eligible_count(bundle) <= 0 or sota.get("best") is None or any(
        _failure_indicates_missing_evidence(failure) for failure in sota_failures
    ):
        decision.update(
            {
                "action": "needs_metrics",
                "reasons": ["required leaderboard or SOTA evidence is incomplete"],
                "failures": round_failures or sota_failures,
                "required_followups": ["collect required round metrics and rebuild artifacts"],
                "confidence": "low",
            }
        )
        return _json_safe(decision)

    if not sota.get("passed"):
        decision.update(
            {
                "action": "reject",
                "reasons": ["absolute or SOTA gates failed"],
                "failures": sota_failures,
                "required_followups": ["revise candidate before another gated run"],
            }
        )
        return _json_safe(decision)

    if require_baseline and baseline_comparison is None:
        decision.update(
            {
                "action": "needs_baseline",
                "reasons": ["baseline comparison is required but missing"],
                "failures": ["missing baseline comparison"],
                "required_followups": ["rerun round artifacts with a baseline comparison"],
                "confidence": "low",
            }
        )
        return _json_safe(decision)

    if baseline_comparison is not None and not baseline_comparison.get("passed"):
        decision.update(
            {
                "action": "reject",
                "reasons": ["baseline comparison failed"],
                "failures": list(baseline_comparison.get("failures", [])),
                "required_followups": ["revise candidate before another baseline-gated run"],
            }
        )
        return _json_safe(decision)

    if normalized_stage == "smoke":
        decision.update(
            {
                "action": "promote_full_run",
                "passed": True,
                "reasons": ["smoke gates passed"],
                "required_followups": ["run full experiment"],
            }
        )
        return _json_safe(decision)

    if normalized_stage == "verify":
        decision.update(
            {
                "action": "accept_candidate",
                "passed": True,
                "reasons": ["verification gates passed"],
                "required_followups": [],
            }
        )
        return _json_safe(decision)

    if verification_margin is not None and baseline_comparison is not None:
        margin_passed, margin_failures = _verification_margin_passed(baseline_comparison, verification_margin)
        if margin_passed:
            decision.update(
                {
                    "action": "accept_candidate",
                    "passed": True,
                    "reasons": ["full gates passed and verification margin cleared"],
                    "required_followups": [],
                }
            )
            return _json_safe(decision)
        decision["failures"] = margin_failures

    decision.update(
        {
            "action": "rerun_verify",
            "passed": True,
            "reasons": ["full gates passed"],
            "required_followups": ["run verification experiment"],
        }
    )
    return _json_safe(decision)


def _write_json_artifact(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_json_safe(payload), fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")


def write_round_artifacts(bundle: Mapping[str, Any], output_dir: Any, *, suffix: str = ".json") -> dict:
    """Write round artifact JSON files without binding to a recorder lifecycle."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for name in ROUND_ARTIFACT_NAMES:
        path = output_path / f"{name}{suffix}"
        _write_json_artifact(path, bundle.get(name))
        paths[name] = str(path)

    return paths


def log_round_artifacts(recorder: Any, bundle: Mapping[str, Any], *, artifact_path: str = "round") -> dict:
    """Best-effort recorder artifact logging for a prebuilt round bundle."""

    results: Dict[str, Any] = {}
    log_artifact = getattr(recorder, "log_artifact", None)
    if not callable(log_artifact):
        return {
            name: {
                "logged": False,
                "filename": f"{name}.json",
                "artifact_path": artifact_path,
                "error": "recorder does not provide log_artifact",
            }
            for name in ROUND_ARTIFACT_NAMES
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = write_round_artifacts(bundle, temp_dir)
        for name in ROUND_ARTIFACT_NAMES:
            path = paths[name]
            entry: Dict[str, Any] = {
                "logged": False,
                "filename": f"{name}.json",
                "artifact_path": artifact_path,
            }
            try:
                logged = log_artifact(path, artifact_path=artifact_path)
            except Exception as exc:
                entry["error"] = str(exc)
            else:
                entry["logged"] = True
                entry["result"] = _json_safe(logged)
            results[name] = entry

    return results
