from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quant_agent.config import LLMConfig
from quant_agent.llm import OpenAICompatibleClient
from quant_agent.models import TraceRecord
from quant_agent.prompting import (
    filtered_trace_for_action,
    last_trace_for_action,
    quant_master_scenario_description,
    render_component_prompt,
    render_exp_prompt,
    render_model_one_shot_prompt,
    render_q_prompt,
    render_q_section,
    sota_trace_for_action,
    summarize_trace,
    summarize_trace_items,
)
from quant_agent.scheduler import decide_bandit_action, decide_random_action


class QuantAgent:
    def __init__(self, config: LLMConfig | None = None):
        self.client = OpenAICompatibleClient(config) if config is not None else None

    def generate_round(
        self,
        action: str,
        trace: TraceRecord,
        instruction: str | None = None,
        mock_hypothesis: Path | None = None,
        mock_experiment: Path | None = None,
    ) -> dict[str, Any]:
        filtered_items = filtered_trace_for_action(trace, action)
        history = summarize_trace_items(filtered_items)
        last_history = last_trace_for_action(trace, action=action)
        sota_history = sota_trace_for_action(trace, action=action) if action == "model" else ""
        scenario = quant_master_scenario_description(action)

        if action == "factor":
            hypothesis_format = render_q_prompt("factor_hypothesis_output_format")
            hypothesis_specification = render_q_prompt("factor_hypothesis_specification")
            experiment_output_format = render_q_prompt("factor_experiment_output_format")
            targets = "factors"
            rag = (
                "Try the easiest and fastest factors to experiment with from various perspectives first."
                if len(trace.hist) < 15
                else "Now, you need to try factors that can achieve high IC (for example, machine learning based factors)."
            )
        else:
            hypothesis_format = render_q_prompt("hypothesis_output_format")
            hypothesis_specification = render_q_prompt("model_hypothesis_specification")
            experiment_output_format = render_q_prompt("model_experiment_output_format")
            targets = "model tuning"
            rag = (
                "Design a compact, trainable model first. If training was previously unstable, prioritize fixing that before adding novelty."
            )

        hypothesis_system = render_component_prompt(
            "hypothesis_gen",
            "system_prompt",
            targets=targets,
            scenario=scenario,
            hypothesis_output_format=hypothesis_format,
            hypothesis_specification=hypothesis_specification,
            user_instruction=instruction,
        )
        hypothesis_user = render_component_prompt(
            "hypothesis_gen",
            "user_prompt",
            targets=targets,
            hypothesis_and_feedback=history,
            last_hypothesis_and_feedback=last_history,
            sota_hypothesis_and_feedback=sota_history,
            RAG=rag,
        )
        hypothesis_payload = self._complete(
            hypothesis_system,
            hypothesis_user,
            mock_file=mock_hypothesis,
        )
        if action == "quant" and "action" not in hypothesis_payload:
            raise RuntimeError("Quant hypothesis response must include an action field.")

        experiment_system = render_component_prompt(
            "hypothesis2experiment",
            "system_prompt",
            targets=targets,
            scenario=scenario,
            experiment_output_format=experiment_output_format,
        )
        experiment_user = render_component_prompt(
            "hypothesis2experiment",
            "user_prompt",
            targets=targets,
            target_hypothesis=json.dumps(hypothesis_payload, ensure_ascii=False, indent=2),
            hypothesis_and_feedback=history,
            last_hypothesis_and_feedback=last_history,
            sota_hypothesis_and_feedback=sota_history,
            target_list=[],
            RAG=rag,
        )
        experiment_payload = self._complete(
            experiment_system,
            experiment_user,
            mock_file=mock_experiment,
        )

        return {
            "hypothesis": hypothesis_payload,
            "experiment": experiment_payload,
            "prompts": {
                "hypothesis_system": hypothesis_system,
                "hypothesis_user": hypothesis_user,
                "experiment_system": experiment_system,
                "experiment_user": experiment_user,
            },
        }

    def generate_model_code(
        self,
        experiment_payload: dict[str, Any],
        mock_code: Path | None = None,
    ) -> dict[str, Any]:
        if not experiment_payload:
            raise RuntimeError("Model experiment payload is empty.")
        model_name, spec = next(iter(experiment_payload.items()))
        system_prompt = render_model_one_shot_prompt("code_implement_sys")
        user_prompt = render_model_one_shot_prompt(
            "code_implement_user",
            name=model_name,
            description=spec.get("description", ""),
            formulation=spec.get("formulation", ""),
            variables=spec.get("variables", {}),
        ) + (
            "\n\nAdditional hard constraints for this native Quant-Master runtime:\n"
            "1. Use only `torch` and Python standard library modules.\n"
            "2. Do not import `torch_geometric`, `dgl`, `xgboost`, `catboost`, or any package not already guaranteed by PyTorch.\n"
            "3. The model must expose `model_cls` and support Quant-Master's tabular or time-series tensor inputs.\n"
            "4. If the idea is graph-like, approximate it with plain torch tensor operations instead of graph libraries.\n"
        )
        if mock_code is not None:
            payload = json.loads(mock_code.read_text(encoding="utf-8"))
        else:
            if self.client is None:
                raise RuntimeError(
                    "No LLM config found. Set QUANT_AGENT_API_BASE, QUANT_AGENT_API_KEY, and QUANT_AGENT_MODEL, "
                    "or pass a mock code response for offline generation."
                )
            raw = self.client.create_text_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = self.extract_code_payload(raw)
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RuntimeError("Model code generation did not return a non-empty `code` field.")
        return {
            "model_name": model_name,
            "code": code,
            "prompts": {
                "codegen_system": system_prompt,
                "codegen_user": user_prompt,
            },
        }

    def choose_quant_action(
        self,
        trace: TraceRecord,
        forced_action: str | None = None,
        mock_action: Path | None = None,
        action_selection: str = "bandit",
    ) -> str:
        if forced_action in {"factor", "model"}:
            return forced_action
        if not trace.hist:
            return "factor"
        if mock_action is not None:
            payload = json.loads(mock_action.read_text(encoding="utf-8"))
            return str(payload["action"])
        if action_selection == "bandit":
            return decide_bandit_action(trace.hist)
        if action_selection == "random":
            return decide_random_action()
        if self.client is None:
            return "factor"
        system_prompt = render_q_section("action_gen", "system")
        user_prompt = render_q_section(
            "action_gen",
            "user",
            hypothesis_and_feedback=summarize_trace(trace),
            last_hypothesis_and_feedback=last_trace_for_action(trace, action=None),
        )
        payload = self.client.create_json_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        action = str(payload.get("action", "factor"))
        if action not in {"factor", "model"}:
            return "factor"
        return action

    def _complete(self, system_prompt: str, user_prompt: str, mock_file: Path | None) -> dict[str, Any]:
        if mock_file is not None:
            return json.loads(mock_file.read_text(encoding="utf-8"))
        if self.client is None:
            raise RuntimeError(
                "No LLM config found. Set QUANT_AGENT_API_BASE, QUANT_AGENT_API_KEY, and QUANT_AGENT_MODEL, "
                "or pass mock response files for offline generation."
            )
        return self.client.create_json_completion(system_prompt=system_prompt, user_prompt=user_prompt)

    @staticmethod
    def extract_code_payload(raw_content: str) -> dict[str, Any]:
        match = re.search(r"```(?:python)?\s*(.*?)```", raw_content, re.DOTALL | re.IGNORECASE)
        if match:
            return {"code": match.group(1).strip()}
        return {"code": raw_content.strip()}
