from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

from quant_agent.models import TraceRecord
from quant_agent.prompts_data import (
    MODEL_ONE_SHOT_PROMPTS,
    PROPOSAL_PROMPTS,
    QUANT_MASTER_EXPERIMENT_PROMPTS,
    QUANT_MASTER_PROMPTS,
)


_jinja_env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render_template(template: str, **context: Any) -> str:
    return _jinja_env.from_string(template).render(**context).strip()


def render_q_prompt(key: str, **context: Any) -> str:
    return render_template(str(QUANT_MASTER_PROMPTS[key]), **context)


def render_q_section(key: str, section: str, **context: Any) -> str:
    return render_template(str(QUANT_MASTER_PROMPTS[key][section]), **context)


def render_exp_prompt(key: str, **context: Any) -> str:
    return render_template(str(QUANT_MASTER_EXPERIMENT_PROMPTS[key]), **context)


def render_component_prompt(section: str, key: str, **context: Any) -> str:
    return render_template(str(PROPOSAL_PROMPTS[section][key]), **context)


def render_model_one_shot_prompt(key: str, **context: Any) -> str:
    return render_template(str(MODEL_ONE_SHOT_PROMPTS[key]), **context)


def summarize_trace(trace: TraceRecord) -> str:
    if not trace.hist:
        return "No previous hypothesis and feedback available since it's the first round."
    lines: list[str] = []
    for idx, item in enumerate(trace.hist, start=1):
        hypothesis = item.get("experiment", {}).get("hypothesis", {})
        feedback = item.get("feedback", {})
        action = hypothesis.get("action")
        if action:
            lines.append(f"# Trial {idx} ({action})")
        else:
            lines.append(f"# Trial {idx}")
        lines.append(f"Hypothesis: {hypothesis.get('hypothesis', '')}")
        lines.append(f"Reason: {hypothesis.get('reason', '')}")
        lines.append(f"Observations: {feedback.get('observations', '')}")
        lines.append(f"Evaluation: {feedback.get('hypothesis_evaluation', '')}")
        lines.append(f"Decision: {feedback.get('decision', False)}")
        if feedback.get("new_hypothesis"):
            lines.append(f"Suggested Next Hypothesis: {feedback['new_hypothesis']}")
        lines.append("=" * 40)
    return "\n".join(lines)


def summarize_trace_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No previous hypothesis and feedback available since it's the first round."
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        hypothesis = item.get("experiment", {}).get("hypothesis", {})
        feedback = item.get("feedback", {})
        action = hypothesis.get("action")
        if action:
            lines.append(f"# Trial {idx} ({action})")
        else:
            lines.append(f"# Trial {idx}")
        lines.append(f"Hypothesis: {hypothesis.get('hypothesis', '')}")
        lines.append(f"Reason: {hypothesis.get('reason', '')}")
        lines.append(f"Observations: {feedback.get('observations', '')}")
        lines.append(f"Evaluation: {feedback.get('hypothesis_evaluation', '')}")
        lines.append(f"Decision: {feedback.get('decision', False)}")
        if feedback.get("new_hypothesis"):
            lines.append(f"Suggested Next Hypothesis: {feedback['new_hypothesis']}")
        lines.append("=" * 40)
    return "\n".join(lines)


def last_trace_for_action(trace: TraceRecord, action: str | None = None) -> str:
    if not trace.hist:
        return "No previous hypothesis and feedback available since it's the first round."
    for item in reversed(trace.hist):
        hypothesis = item.get("experiment", {}).get("hypothesis", {})
        if action is None or hypothesis.get("action") == action or hypothesis.get("action") is None:
            feedback = item.get("feedback", {})
            return "\n".join(
                [
                    f"Hypothesis: {hypothesis.get('hypothesis', '')}",
                    f"Reason: {hypothesis.get('reason', '')}",
                    f"Observations: {feedback.get('observations', '')}",
                    f"Evaluation: {feedback.get('hypothesis_evaluation', '')}",
                    f"Decision: {feedback.get('decision', False)}",
                    f"New Hypothesis: {feedback.get('new_hypothesis', '')}",
                ]
            )
    return "No previous hypothesis and feedback available."


def filtered_trace_for_action(trace: TraceRecord, action: str) -> list[dict[str, Any]]:
    if action not in {"factor", "model"}:
        return list(trace.hist)

    opposite = "model" if action == "factor" else "factor"
    same_action_items: list[dict[str, Any]] = []
    latest_accepted_opposite: dict[str, Any] | None = None

    for item in trace.hist:
        hypothesis = item.get("experiment", {}).get("hypothesis", {})
        item_action = hypothesis.get("action")
        feedback = item.get("feedback", {})
        accepted = bool(feedback.get("decision", False)) if isinstance(feedback, dict) else False
        if item_action == action:
            same_action_items.append(item)
        elif item_action == opposite and accepted:
            latest_accepted_opposite = item

    if latest_accepted_opposite is not None:
        return [*same_action_items, latest_accepted_opposite]
    return same_action_items


def sota_trace_for_action(trace: TraceRecord, action: str) -> str:
    for item in reversed(trace.hist):
        hypothesis = item.get("experiment", {}).get("hypothesis", {})
        feedback = item.get("feedback", {})
        if hypothesis.get("action") == action and bool(feedback.get("decision", False)):
            return "\n".join(
                [
                    f"Hypothesis: {hypothesis.get('hypothesis', '')}",
                    f"Reason: {hypothesis.get('reason', '')}",
                    f"Observations: {feedback.get('observations', '')}",
                    f"Evaluation: {feedback.get('hypothesis_evaluation', '')}",
                    f"Decision: {feedback.get('decision', False)}",
                    f"New Hypothesis: {feedback.get('new_hypothesis', '')}",
                ]
            )
    return ""


def quant_master_scenario_description(action: str) -> str:
    runtime_environment = "Windows native Python environment."
    if action == "factor":
        background = render_exp_prompt("quant_master_factor_background", runtime_environment=runtime_environment)
        interface = render_exp_prompt("quant_master_factor_interface")
        output = render_exp_prompt("quant_master_factor_output_format")
        simulator = render_exp_prompt("quant_master_factor_simulator")
    else:
        background = render_exp_prompt("quant_master_model_background", runtime_environment=runtime_environment)
        interface = render_exp_prompt("quant_master_model_interface")
        output = render_exp_prompt("quant_master_model_output_format")
        simulator = render_exp_prompt("quant_master_model_simulator")
    return "\n".join(
        [
            "------Background of the scenario------",
            background,
            "------The interface you should follow to write the runnable code------",
            interface,
            "------The output of your code should be in the format------",
            output,
            "------The simulator user can use to test your solution------",
            simulator,
        ]
    )
