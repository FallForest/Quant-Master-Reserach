from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMPORTANT_METRICS = (
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
)


@dataclass
class TargetGate:
    metric: str
    threshold: float
    direction: str = "gte"  # "gte" means current >= threshold passes; "lte" means current <= threshold passes


@dataclass
class TargetProfile:
    gates: list[TargetGate] = field(default_factory=list)


FIN_QUANT_DEFAULT_TARGETS = TargetProfile(
    gates=[
        TargetGate(metric="1day.excess_return_with_cost.annualized_return", threshold=0.25, direction="gte"),
        TargetGate(metric="IC", threshold=0.03, direction="gte"),
        TargetGate(metric="1day.excess_return_with_cost.information_ratio", threshold=0.8, direction="gte"),
        TargetGate(metric="1day.excess_return_with_cost.max_drawdown", threshold=-0.12, direction="gte"),
    ]
)


def build_target_profile(
    target_arr: float | None = None,
    target_ic: float | None = None,
    target_ir: float | None = None,
    target_max_drawdown: float | None = None,
    base: TargetProfile | None = None,
) -> TargetProfile:
    if base is not None:
        override_map = {
            "1day.excess_return_with_cost.annualized_return": target_arr,
            "IC": target_ic,
            "1day.excess_return_with_cost.information_ratio": target_ir,
            "1day.excess_return_with_cost.max_drawdown": target_max_drawdown,
        }
        gates: list[TargetGate] = []
        for gate in base.gates:
            override = override_map.get(gate.metric)
            gates.append(TargetGate(
                metric=gate.metric,
                threshold=override if override is not None else gate.threshold,
                direction=gate.direction,
            ))
        return TargetProfile(gates=gates)
    gates = []
    if target_arr is not None:
        gates.append(TargetGate(metric="1day.excess_return_with_cost.annualized_return", threshold=target_arr, direction="gte"))
    if target_ic is not None:
        gates.append(TargetGate(metric="IC", threshold=target_ic, direction="gte"))
    if target_ir is not None:
        gates.append(TargetGate(metric="1day.excess_return_with_cost.information_ratio", threshold=target_ir, direction="gte"))
    if target_max_drawdown is not None:
        gates.append(TargetGate(metric="1day.excess_return_with_cost.max_drawdown", threshold=target_max_drawdown, direction="gte"))
    return TargetProfile(gates=gates)


def check_targets(
    metrics: dict[str, float],
    profile: TargetProfile,
) -> dict[str, Any]:
    target_thresholds: dict[str, float] = {}
    target_metrics: dict[str, float] = {}
    target_gaps: dict[str, float] = {}
    missing: list[str] = []
    all_satisfied = True

    for gate in profile.gates:
        target_thresholds[gate.metric] = gate.threshold
        current = metrics.get(gate.metric)
        if current is None:
            missing.append(gate.metric)
            all_satisfied = False
            continue
        target_metrics[gate.metric] = current
        gap = current - gate.threshold
        target_gaps[gate.metric] = gap
        if gate.direction == "gte" and current < gate.threshold:
            all_satisfied = False
        elif gate.direction == "lte" and current > gate.threshold:
            all_satisfied = False

    return {
        "target_reached": all_satisfied and len(missing) == 0,
        "target_thresholds": target_thresholds,
        "target_metrics": target_metrics,
        "target_gaps": target_gaps,
        "missing_target_metrics": missing,
    }


@dataclass
class RuntimeResult:
    return_code: int
    log_tail: str
    metrics: dict[str, float]


def extract_metrics_from_workspace(workspace: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metrics_dir in workspace.rglob("metrics"):
        if not metrics_dir.is_dir():
            continue
        for metric_file in metrics_dir.rglob("*"):
            if not metric_file.is_file():
                continue
            value = _read_last_metric_value(metric_file)
            if value is None:
                continue
            name = metric_file.relative_to(metrics_dir).as_posix().replace("/", ".")
            metrics[name] = value
    return metrics


def build_feedback(
    action: str,
    runtime: RuntimeResult,
    trace: list[dict[str, Any]],
    target_profile: TargetProfile | None = None,
) -> dict[str, Any]:
    if runtime.return_code != 0:
        return {
            "observations": f"Runtime failed with return code {runtime.return_code}. Log tail:\n{runtime.log_tail}",
            "hypothesis_evaluation": "Execution failed before reliable Quant-Master metrics were produced.",
            "decision": False,
            "new_hypothesis": "Revise the experiment for runtime compatibility and rerun.",
            "reason": "runtime_failed",
            "metrics": runtime.metrics,
        }
    if _has_runtime_error_signal(runtime.log_tail):
        return {
            "observations": f"Runtime log contains error signal. Log tail:\n{runtime.log_tail}",
            "hypothesis_evaluation": "Execution log indicates a runtime error before reliable metrics were produced.",
            "decision": False,
            "new_hypothesis": "Fix runtime/data issues first, then continue evolution.",
            "reason": "runtime_log_error",
            "metrics": runtime.metrics,
        }

    score = score_metrics(runtime.metrics)
    baseline = best_action_baseline(action, trace)
    best_score = baseline["score"] if baseline is not None else best_trace_score(trace)
    baseline_metrics = baseline["metrics"] if baseline is not None else {}
    if score is None:
        score_improved = best_score is None
        evaluation = "Execution succeeded, but no comparable Quant-Master metrics were found."
    else:
        score_improved = best_score is None or score > best_score
        if score_improved:
            evaluation = "Execution succeeded and the result replaces the current best result."
        else:
            evaluation = "Execution succeeded, but the result does not improve on the current best result."

    target_result: dict[str, Any] = {}
    decision = score_improved
    if target_profile is not None and target_profile.gates:
        target_result = check_targets(runtime.metrics, target_profile)
        if not target_result["target_reached"]:
            decision = False
            missing = target_result.get("missing_target_metrics", [])
            gaps = target_result.get("target_gaps", {})
            gap_parts = []
            for metric_name, gap in gaps.items():
                threshold = target_result["target_thresholds"].get(metric_name, 0.0)
                current = target_result["target_metrics"].get(metric_name)
                if current is not None:
                    gap_parts.append(f"{metric_name}: current={current:.6f} threshold={threshold:.6f} gap={gap:+.6f}")
            missing_parts = [f"{m}: metric missing" for m in missing]
            detail = "; ".join(gap_parts + missing_parts)
            if score_improved:
                evaluation = (
                    f"Execution succeeded and score improved, but target gates are not satisfied. {detail}"
                )
            else:
                evaluation = (
                    f"Execution succeeded, but score did not improve and target gates are not satisfied. {detail}"
                )

    metric_delta = diff_metrics(runtime.metrics, baseline_metrics)
    observations = _format_observations(action, runtime.metrics, baseline_metrics, metric_delta, runtime.log_tail)
    if target_profile is not None and target_profile.gates:
        target_obs = _format_target_observations(target_result)
        if target_obs:
            observations = f"{observations}\n{target_obs}"

    result: dict[str, Any] = {
        "observations": observations,
        "hypothesis_evaluation": evaluation,
        "decision": decision,
        "new_hypothesis": "" if decision else "Use the metric gaps to revise the next hypothesis.",
        "reason": "metric_improved" if decision else ("improved_below_target" if score_improved else "metric_not_improved"),
        "metrics": runtime.metrics,
        "sota_metrics": baseline_metrics,
        "metric_delta": metric_delta,
        "sota_score": best_score,
        "performance_score": score,
    }
    if target_profile is not None and target_profile.gates:
        result["target_thresholds"] = target_result.get("target_thresholds", {})
        result["target_metrics"] = target_result.get("target_metrics", {})
        result["target_gaps"] = target_result.get("target_gaps", {})
        result["target_reached"] = target_result.get("target_reached", False)
        missing_metrics = target_result.get("missing_target_metrics", [])
        if missing_metrics:
            result["missing_target_metrics"] = missing_metrics
    return result


def score_metrics(metrics: dict[str, float]) -> float | None:
    ic = metrics.get("IC")
    annualized_return = metrics.get("1day.excess_return_with_cost.annualized_return")
    max_drawdown = metrics.get("1day.excess_return_with_cost.max_drawdown")
    if ic is None and annualized_return is None:
        return None
    score = 0.0
    if ic is not None:
        score += 10.0 * ic
    if annualized_return is not None:
        score += annualized_return
    if max_drawdown is not None:
        score += max_drawdown
    return score


def best_trace_score(trace: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for item in trace:
        feedback = item.get("feedback", {}) if isinstance(item, dict) else {}
        if not isinstance(feedback, dict):
            continue
        score = feedback.get("performance_score")
        if not isinstance(score, (int, float)):
            metrics = feedback.get("metrics")
            score = score_metrics(metrics) if isinstance(metrics, dict) else None
        if score is None:
            continue
        best = float(score) if best is None else max(best, float(score))
    return best


def best_action_baseline(action: str, trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted: list[tuple[float, dict[str, float]]] = []
    all_results: list[tuple[float, dict[str, float]]] = []

    for item in trace:
        if not isinstance(item, dict):
            continue
        experiment = item.get("experiment", {})
        hypothesis = experiment.get("hypothesis", {}) if isinstance(experiment, dict) else {}
        if hypothesis.get("action") != action:
            continue
        feedback = item.get("feedback", {})
        if not isinstance(feedback, dict):
            continue
        metrics = feedback.get("metrics")
        if not isinstance(metrics, dict):
            continue
        score = feedback.get("performance_score")
        if not isinstance(score, (int, float)):
            score = score_metrics(metrics)
        if score is None:
            continue
        entry = (float(score), metrics)
        all_results.append(entry)
        if bool(feedback.get("decision", False)):
            accepted.append(entry)

    source = accepted if accepted else all_results
    if not source:
        return None
    score, metrics = max(source, key=lambda item: item[0])
    return {"score": score, "metrics": metrics}


def diff_metrics(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in IMPORTANT_METRICS:
        cur = current.get(name)
        base = baseline.get(name)
        if cur is None or base is None:
            continue
        deltas[name] = cur - base
    return deltas


def _read_last_metric_value(metric_file: Path) -> float | None:
    try:
        lines = [line.strip() for line in metric_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        return None
    if not lines:
        return None
    parts = lines[-1].split()
    candidates = parts[1:2] + parts[-1:]
    for item in candidates:
        try:
            return float(item)
        except ValueError:
            continue
    return None


def _format_observations(
    action: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    metric_delta: dict[str, float],
    log_tail: str,
) -> str:
    if metrics:
        metric_text = "; ".join(
            f"{name}={metrics[name]:.6f}" for name in IMPORTANT_METRICS if name in metrics
        )
        if not metric_text:
            metric_text = "; ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
        if baseline_metrics:
            baseline_text = "; ".join(
                f"{name}={baseline_metrics[name]:.6f}" for name in IMPORTANT_METRICS if name in baseline_metrics
            )
            delta_text = "; ".join(f"{name}={metric_delta[name]:+.6f}" for name in IMPORTANT_METRICS if name in metric_delta)
            return (
                f"{action} runtime succeeded. Current metrics: {metric_text}. "
                f"SOTA metrics: {baseline_text or 'n/a'}. "
                f"Delta(current-SOTA): {delta_text or 'n/a'}. Log tail:\n{log_tail}"
            )
        return f"{action} runtime succeeded. Metrics: {metric_text}. Log tail:\n{log_tail}"
    return f"{action} runtime succeeded. No MLflow metric files were found. Log tail:\n{log_tail}"


def _has_runtime_error_signal(log_tail: str) -> bool:
    if not log_tail:
        return False
    lowered = log_tail.lower()
    strong_patterns = (
        "traceback (most recent call last)",
        "valueerror:",
        "raise valueerror(",
        "does not contain data for",
        "runtimeerror:",
        "scannererror:",
        "exception has been raised",
        "quant_master.workflow - [utils.py:41] - an exception has been raised",
    )
    return any(pattern in lowered for pattern in strong_patterns)


def _format_target_observations(target_result: dict[str, Any]) -> str:
    if not target_result:
        return ""
    if target_result.get("target_reached"):
        return "All target gates satisfied."
    parts: list[str] = []
    thresholds = target_result.get("target_thresholds", {})
    metrics = target_result.get("target_metrics", {})
    gaps = target_result.get("target_gaps", {})
    missing = target_result.get("missing_target_metrics", [])
    for metric_name, threshold in thresholds.items():
        current = metrics.get(metric_name)
        if current is not None:
            gap = gaps.get(metric_name, 0.0)
            parts.append(f"{metric_name}: current={current:.6f} threshold={threshold:.6f} gap={gap:+.6f}")
        else:
            parts.append(f"{metric_name}: metric missing (threshold={threshold:.6f})")
    for m in missing:
        if m not in thresholds:
            parts.append(f"{m}: metric missing")
    return "Target gate status: " + "; ".join(parts)
