from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_agent.agent import QuantAgent
from quant_agent.evaluation import RuntimeResult, TargetProfile, build_feedback, extract_metrics_from_workspace
from quant_agent.models import TraceRecord
from quant_agent.workspace import create_workspace


@dataclass
class LoopRoundResult:
    loop_id: int
    action: str
    workspace: Path
    hypothesis_file: Path
    experiment_file: Path
    feedback: dict[str, Any] | None


class QuantRDLoop:
    """
    A lightweight RD-Agent style loop:
    direct_exp_gen -> coding(optional for model) -> feedback -> record(trace)
    """

    def __init__(self, agent: QuantAgent, trace: TraceRecord, target_profile: TargetProfile | None = None):
        self.agent = agent
        self.trace = trace
        self.target_profile = target_profile

    @staticmethod
    def load_feedback_sequence(path: Path | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("hist"), list):
            seq: list[dict[str, Any]] = []
            for item in payload["hist"]:
                if not isinstance(item, dict):
                    continue
                feedback = item.get("feedback")
                if isinstance(feedback, dict):
                    seq.append(feedback)
            return seq
        raise RuntimeError("Feedback file must be a JSON list or a trace JSON object with `hist` entries.")

    def run(
        self,
        workspace_root: Path,
        instruction: str | None,
        forced_action: str | None,
        max_rounds: int,
        action_selection: str = "bandit",
        feedback_sequence: list[dict[str, Any]] | None = None,
        auto_run: bool = False,
        run_timeout_seconds: int = 1800,
        quick_smoke: bool = False,
        mock_hypothesis: Path | None = None,
        mock_experiment: Path | None = None,
        mock_code: Path | None = None,
    ) -> dict[str, Any]:
        workspace_root = workspace_root.expanduser().resolve()
        feedback_sequence = feedback_sequence or []
        rounds: list[LoopRoundResult] = []
        initial_hist_len = len(self.trace.hist)
        initial_round_no = max(initial_hist_len, self._max_existing_round_no(workspace_root))

        for loop_id in range(max_rounds):
            round_no = initial_round_no + loop_id + 1
            action = self.agent.choose_quant_action(
                trace=self.trace,
                forced_action=forced_action,
                action_selection=action_selection,
            )
            round_result = self.agent.generate_round(
                action=action,
                trace=self.trace,
                instruction=instruction,
                mock_hypothesis=mock_hypothesis,
                mock_experiment=mock_experiment,
            )
            model_code_payload = None
            if action == "model":
                model_code_payload = self.agent.generate_model_code(
                    experiment_payload=round_result["experiment"],
                    mock_code=mock_code,
                )
                round_result["prompts"].update(model_code_payload["prompts"])

            round_workspace = workspace_root / f"round_{round_no:03d}"
            workspace = create_workspace(
                root=round_workspace,
                action=action,
                hypothesis_payload=round_result["hypothesis"],
                experiment_payload=round_result["experiment"],
                prompt_dump=round_result["prompts"],
                model_code_payload=model_code_payload,
                quick_smoke=quick_smoke,
            )

            feedback = feedback_sequence[loop_id] if loop_id < len(feedback_sequence) else None
            if feedback is None and auto_run:
                feedback = self._auto_feedback_from_run(action, workspace, run_timeout_seconds)
            self.trace.hist.append(
                {
                    "loop_id": round_no,
                    "workspace": str(workspace),
                    "experiment": {
                        "hypothesis": {
                            **round_result["hypothesis"],
                            "action": action,
                        }
                    },
                    "feedback": feedback or {},
                }
            )
            rounds.append(
                LoopRoundResult(
                    loop_id=round_no,
                    action=action,
                    workspace=workspace,
                    hypothesis_file=workspace / "hypothesis.json",
                    experiment_file=workspace / "experiment.json",
                    feedback=feedback,
                )
            )
            round_summary = {
                "loop_id": round_no,
                "action": action,
                "workspace": str(workspace),
                "hypothesis_file": str(workspace / "hypothesis.json"),
                "experiment_file": str(workspace / "experiment.json"),
                "feedback": feedback,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (workspace / "round_summary.json").write_text(
                json.dumps(round_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if feedback and bool(feedback.get("decision", False)):
                break

        workspace_root.mkdir(parents=True, exist_ok=True)
        trace_path = workspace_root / "trace_out.json"
        trace_path.write_text(json.dumps(self.trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        leaderboard = self._build_leaderboard(
            trace_hist=self.trace.hist,
            current_rounds=rounds,
            initial_hist_len=initial_hist_len,
        )
        leaderboard_path = workspace_root / "leaderboard.json"
        leaderboard_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
        sota_factor_path = workspace_root / "sota_factor.json"
        sota_model_path = workspace_root / "sota_model.json"
        sota_factor_path.write_text(
            json.dumps(leaderboard.get("sota", {}).get("factor"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sota_model_path.write_text(
            json.dumps(leaderboard.get("sota", {}).get("model"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "rounds": [
                {
                    "loop_id": item.loop_id,
                    "action": item.action,
                    "workspace": str(item.workspace),
                    "hypothesis_file": str(item.hypothesis_file),
                    "experiment_file": str(item.experiment_file),
                    "feedback": item.feedback,
                }
                for item in rounds
            ],
            "trace_out_file": str(trace_path),
            "leaderboard_file": str(leaderboard_path),
            "sota_factor_file": str(sota_factor_path),
            "sota_model_file": str(sota_model_path),
        }

    def _auto_feedback_from_run(self, action: str, workspace: Path, run_timeout_seconds: int) -> dict[str, Any]:
        script = workspace / "run_experiment.bat"
        if not script.exists():
            return {
                "observations": "run_experiment.bat is missing.",
                "hypothesis_evaluation": "Runtime preparation failed.",
                "decision": False,
                "new_hypothesis": "Fix workspace generation and rerun.",
                "reason": "missing_runtime_script",
            }
        try:
            completed = subprocess.run(
                ["cmd.exe", "/c", str(script)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=max(run_timeout_seconds, 1),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "observations": f"Runtime timed out after {run_timeout_seconds} seconds.",
                "hypothesis_evaluation": "Execution timeout.",
                "decision": False,
                "new_hypothesis": "Reduce complexity or verify data/runtime dependencies.",
                "reason": "runtime_timeout",
            }

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        tail = "\n".join((stdout + "\n" + stderr).strip().splitlines()[-20:])
        runtime = RuntimeResult(
            return_code=completed.returncode,
            log_tail=tail,
            metrics=extract_metrics_from_workspace(workspace),
        )
        (workspace / "runtime_stdout.log").write_text(stdout, encoding="utf-8")
        (workspace / "runtime_stderr.log").write_text(stderr, encoding="utf-8")
        runtime_payload = {
            "return_code": runtime.return_code,
            "metrics": runtime.metrics,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (workspace / "runtime_result.json").write_text(
            json.dumps(runtime_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return build_feedback(action=action, runtime=runtime, trace=self.trace.hist, target_profile=self.target_profile)

    @staticmethod
    def _max_existing_round_no(workspace_root: Path) -> int:
        if not workspace_root.exists():
            return 0
        max_no = 0
        for path in workspace_root.glob("round_*"):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix("round_")
            if not suffix.isdigit():
                continue
            max_no = max(max_no, int(suffix))
        return max_no

    @staticmethod
    def _build_leaderboard(
        trace_hist: list[dict[str, Any]],
        current_rounds: list[LoopRoundResult],
        initial_hist_len: int,
    ) -> dict[str, Any]:
        by_action: dict[str, list[dict[str, Any]]] = {"factor": [], "model": []}
        all_rounds: list[dict[str, Any]] = []
        current_round_map = {item.loop_id: item for item in current_rounds}
        for idx, item in enumerate(trace_hist, start=1):
            experiment = item.get("experiment", {}) if isinstance(item, dict) else {}
            hypothesis = experiment.get("hypothesis", {}) if isinstance(experiment, dict) else {}
            action = str(hypothesis.get("action", ""))
            feedback = item.get("feedback", {}) if isinstance(item, dict) else {}
            if not isinstance(feedback, dict):
                feedback = {}
            loop_id = item.get("loop_id", idx) if isinstance(item, dict) else idx
            if not isinstance(loop_id, int):
                loop_id = idx
            round_workspace = None
            if loop_id in current_round_map:
                round_workspace = str(current_round_map[loop_id].workspace)
            elif isinstance(item, dict) and item.get("workspace"):
                round_workspace = str(item["workspace"])
            summary = {
                "loop_id": loop_id,
                "action": action,
                "workspace": round_workspace,
                "hypothesis": hypothesis.get("hypothesis"),
                "reasoning": hypothesis.get("reason"),
                "decision": bool(feedback.get("decision", False)),
                "performance_score": feedback.get("performance_score"),
                "sota_score": feedback.get("sota_score"),
                "metrics": feedback.get("metrics", {}),
                "metric_delta": feedback.get("metric_delta", {}),
                "reason": feedback.get("reason"),
            }
            all_rounds.append(summary)
            if action in by_action:
                by_action[action].append(summary)

        def _pick_sota(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
            accepted = [entry for entry in entries if entry.get("decision")]
            source = accepted if accepted else entries
            source = [entry for entry in source if isinstance(entry.get("performance_score"), (int, float))]
            if not source:
                return None
            return max(source, key=lambda entry: float(entry["performance_score"]))

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_rounds": len(trace_hist),
            "run_window": {
                "start_loop_id": current_rounds[0].loop_id if current_rounds else None,
                "end_loop_id": current_rounds[-1].loop_id if current_rounds else None,
                "rounds_generated": len(current_rounds),
            },
            "all_rounds": all_rounds,
            "by_action": by_action,
            "sota": {
                "factor": _pick_sota(by_action["factor"]),
                "model": _pick_sota(by_action["model"]),
            },
        }
