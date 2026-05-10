from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from quant_agent import __version__
from quant_agent.agent import QuantAgent
from quant_agent.config import LLMConfig
from quant_agent.evaluation import FIN_QUANT_DEFAULT_TARGETS, TargetProfile, build_target_profile
from quant_agent.loop import QuantRDLoop
from quant_agent.models import TraceRecord
from quant_agent.workspace import create_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-agent",
        description="Quant-agent that runs native Quant-Master workflows for automated quant R&D.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("fin_factor", "fin_model", "fin_quant"):
        sub = subparsers.add_parser(name, help=f"Generate a {name} research round.")
        sub.add_argument("--instruction", default=None, help="Optional user instruction passed into the reused prompt.")
        sub.add_argument("--trace-file", default=None, help="Optional trace JSON file from prior rounds.")
        sub.add_argument("--workspace-dir", required=True, help="Output workspace directory for this round.")
        sub.add_argument("--dotenv-path", default=".env", help="Optional dotenv file for API configuration.")
        sub.add_argument("--mock-hypothesis", default=None, help="Offline hypothesis JSON response.")
        sub.add_argument("--mock-experiment", default=None, help="Offline experiment JSON response.")
        sub.add_argument("--mock-code", default=None, help="Offline model code JSON response.")
        sub.add_argument("--action", choices=["factor", "model"], default=None, help="Only for fin_quant.")
        sub.add_argument(
            "--action-selection",
            choices=["bandit", "llm", "random"],
            default="bandit",
            help="Only for fin_quant: action selection strategy.",
        )
        sub.add_argument("--max-rounds", type=int, default=1, help="Only for fin_quant: max evolving loops.")
        sub.add_argument(
            "--feedback-file",
            default=None,
            help="Only for fin_quant: feedback sequence JSON (list or trace with hist).",
        )
        sub.add_argument("--auto-run", action="store_true", help="Only for fin_quant: execute run_experiment.bat each round.")
        sub.add_argument(
            "--run-timeout-seconds",
            type=int,
            default=1800,
            help="Only for fin_quant: timeout for each runtime execution.",
        )
        sub.add_argument(
            "--quick-smoke",
            action="store_true",
            help="Use a smaller time window and lighter training defaults for faster validation.",
        )
        if name == "fin_quant":
            sub.add_argument(
                "--target-arr",
                type=float,
                default=None,
                help="Annualized excess return target gate (default: 0.25).",
            )
            sub.add_argument(
                "--target-ic",
                type=float,
                default=None,
                help="IC floor target gate (default: 0.03).",
            )
            sub.add_argument(
                "--target-ir",
                type=float,
                default=None,
                help="Information ratio floor target gate (default: 0.8).",
            )
            sub.add_argument(
                "--target-max-drawdown",
                type=float,
                default=None,
                help="Max drawdown floor target gate (default: -0.12, i.e. must be >= -0.12).",
            )

    return parser


def _load_trace(trace_file: str | None, workspace_dir: str | None = None) -> TraceRecord:
    if trace_file is None and workspace_dir is not None:
        inferred = Path(workspace_dir) / "trace_out.json"
        if inferred.exists():
            trace_file = str(inferred)
    if trace_file is None:
        return TraceRecord()
    payload = json.loads(Path(trace_file).read_text(encoding="utf-8"))
    return TraceRecord.from_dict(payload)


def _run_generation(args: argparse.Namespace) -> int:
    trace = _load_trace(args.trace_file, getattr(args, "workspace_dir", None))
    config = LLMConfig.from_env(Path(args.dotenv_path))
    agent = QuantAgent(config)
    loop_runner = QuantRDLoop(agent=agent, trace=trace)

    if args.command == "fin_factor":
        action = "factor"
    elif args.command == "fin_model":
        action = "model"
    else:
        any_target_override = any(
            getattr(args, attr, None) is not None
            for attr in ("target_arr", "target_ic", "target_ir", "target_max_drawdown")
        )
        if any_target_override:
            target_profile = build_target_profile(
                target_arr=getattr(args, "target_arr", None),
                target_ic=getattr(args, "target_ic", None),
                target_ir=getattr(args, "target_ir", None),
                target_max_drawdown=getattr(args, "target_max_drawdown", None),
                base=FIN_QUANT_DEFAULT_TARGETS,
            )
        else:
            target_profile = FIN_QUANT_DEFAULT_TARGETS
        loop_runner = QuantRDLoop(agent=agent, trace=trace, target_profile=target_profile)
        feedback_sequence = QuantRDLoop.load_feedback_sequence(Path(args.feedback_file) if args.feedback_file else None)
        loop_result = loop_runner.run(
            workspace_root=Path(args.workspace_dir),
            instruction=args.instruction,
            forced_action=args.action,
            max_rounds=max(int(args.max_rounds), 1),
            action_selection=args.action_selection,
            feedback_sequence=feedback_sequence,
            auto_run=bool(args.auto_run),
            run_timeout_seconds=max(int(args.run_timeout_seconds), 1),
            quick_smoke=bool(args.quick_smoke),
            mock_hypothesis=Path(args.mock_hypothesis) if args.mock_hypothesis else None,
            mock_experiment=Path(args.mock_experiment) if args.mock_experiment else None,
            mock_code=Path(args.mock_code) if args.mock_code else None,
        )
        output = {
            "command": args.command,
            "mode": "rd_loop",
            "round_count": len(loop_result["rounds"]),
            "rounds": loop_result["rounds"],
            "trace_out_file": loop_result["trace_out_file"],
            "leaderboard_file": loop_result["leaderboard_file"],
            "sota_factor_file": loop_result["sota_factor_file"],
            "sota_model_file": loop_result["sota_model_file"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    round_result = agent.generate_round(
        action=action,
        trace=trace,
        instruction=args.instruction,
        mock_hypothesis=Path(args.mock_hypothesis) if args.mock_hypothesis else None,
        mock_experiment=Path(args.mock_experiment) if args.mock_experiment else None,
    )
    model_code_payload = None
    if action == "model":
        model_code_payload = agent.generate_model_code(
            experiment_payload=round_result["experiment"],
            mock_code=Path(args.mock_code) if args.mock_code else None,
        )
        round_result["prompts"].update(model_code_payload["prompts"])
    workspace = create_workspace(
        root=Path(args.workspace_dir),
        action=action,
        hypothesis_payload=round_result["hypothesis"],
        experiment_payload=round_result["experiment"],
        prompt_dump=round_result["prompts"],
        model_code_payload=model_code_payload,
        quick_smoke=bool(args.quick_smoke),
    )
    output = {
        "command": args.command,
        "action": action,
        "workspace": str(workspace),
        "hypothesis_file": str(workspace / "hypothesis.json"),
        "experiment_file": str(workspace / "experiment.json"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
