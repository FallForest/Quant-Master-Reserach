from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from quant_agent.agent import QuantAgent
from quant_agent.cli import main
from quant_agent.evaluation import (
    FIN_QUANT_DEFAULT_TARGETS,
    RuntimeResult,
    TargetGate,
    TargetProfile,
    best_action_baseline,
    build_feedback,
    check_targets,
    extract_metrics_from_workspace,
)
from quant_agent.models import TraceRecord
from quant_agent.prompting import filtered_trace_for_action, summarize_trace_items
from quant_agent.scheduler import decide_bandit_action


class QuantAgentCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="quant-agent-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fin_factor_creates_workspace_from_rd_agent_assets(self) -> None:
        hypothesis = self.tmpdir / "hypothesis.json"
        experiment = self.tmpdir / "experiment.json"
        hypothesis.write_text(
            json.dumps(
                {
                    "hypothesis": "Test a simple momentum factor family.",
                    "reason": "Start with a small, interpretable baseline.",
                }
            ),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "Momentum_5D": {
                        "description": "[Momentum Factor] 5-day return momentum.",
                        "formulation": "MOM_5 = close_t / close_{t-5} - 1",
                        "variables": {"close_t": "close price on day t"},
                    }
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "factor_round"
        result = main(
            [
                "fin_factor",
                "--workspace-dir",
                str(workspace),
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue((workspace / "prompts" / "hypothesis_system.txt").exists())
        self.assertIn("quant_master.cli.run", (workspace / "run_experiment.bat").read_text(encoding="utf-8"))
        self.assertNotIn("quant_agent.runner", (workspace / "run_experiment.bat").read_text(encoding="utf-8"))
        workflow = (workspace / "rendered_factor_workflow.yaml").read_text(encoding="utf-8")
        self.assertIn("LGBModel", workflow)
        self.assertIn("quant_master.contrib.model.gbdt", workflow)
        self.assertIn("DataHandlerLP", workflow)
        self.assertIn("quant_master.contrib.data.handler", workflow)
        self.assertIn("$close/Ref($close,5)-1", workflow)
        self.assertNotIn("~/.quant_master/quant_master_data/cn_data", workflow)
        self.assertNotIn("QuantAgentLGBModel", workflow)
        self.assertNotIn("QuantAgentFactorHandler", workflow)

    def test_fin_model_creates_gpu_only_workspace(self) -> None:
        hypothesis = self.tmpdir / "model_hypothesis.json"
        experiment = self.tmpdir / "model_experiment.json"
        code = self.tmpdir / "model_code.json"
        hypothesis.write_text(
            json.dumps(
                {
                    "hypothesis": "Test a compact residual temporal network.",
                    "reason": "A small trainable baseline is enough for local validation.",
                }
            ),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "CompactResidualTemporalGatedMLP": {
                        "description": "A compact temporal residual model.",
                        "model_type": "TimeSeries",
                        "formulation": "Use residual temporal mixing blocks with gated pooling.",
                        "variables": {"x": "alpha158 feature sequence"},
                        "training_hyperparameters": {"n_epochs": 4, "batch_size": 4096, "lr": 0.001},
                    }
                }
            ),
            encoding="utf-8",
        )
        code.write_text(
            json.dumps(
                {
                    "code": "import torch\\nimport torch.nn as nn\\n\\nclass Tiny(nn.Module):\\n    def __init__(self, input_dim):\\n        super().__init__()\\n        self.linear = nn.Linear(input_dim, 1)\\n\\n    def forward(self, x):\\n        if x.dim() == 3:\\n            x = x[:, -1, :]\\n        return self.linear(x).squeeze(-1)\\n\\nmodel_cls = Tiny\\n"
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "model_round"
        result = main(
            [
                "fin_model",
                "--workspace-dir",
                str(workspace),
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
                "--mock-code",
                str(code),
            ]
        )
        self.assertEqual(result, 0)
        workflow = (workspace / "rendered_model_workflow.yaml").read_text(encoding="utf-8")
        self.assertIn("GeneralPTNN", workflow)
        self.assertIn("quant_master.contrib.model.pytorch_general_nn", workflow)
        self.assertIn("GPU: 0", workflow)
        self.assertIn('pt_model_uri: "model.model_cls"', workflow)
        self.assertIn("pt_model_kwargs:", workflow)
        self.assertIn("num_timesteps: 20", workflow)
        self.assertNotIn("step_len: 20    record:", workflow)
        self.assertNotIn("~/.quant_master/quant_master_data/cn_data", workflow)
        self.assertNotIn("QuantAgentGeneralPTNN", workflow)

    def test_fin_model_tabular_quick_smoke_uses_rd_template_workflow(self) -> None:
        hypothesis = self.tmpdir / "model_tab_hypothesis.json"
        experiment = self.tmpdir / "model_tab_experiment.json"
        code = self.tmpdir / "model_tab_code.json"
        hypothesis.write_text(
            json.dumps(
                {
                    "hypothesis": "Test a compact tabular residual model in quick smoke mode.",
                    "reason": "Fast validation first.",
                }
            ),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "CompactResidualTabularMLP": {
                        "description": "A compact tabular residual model.",
                        "model_type": "Tabular",
                        "formulation": "Use residual feed-forward blocks for tabular alpha features.",
                        "variables": {"x": "alpha feature vector"},
                        "training_hyperparameters": {"n_epochs": 100, "batch_size": 4096, "lr": 0.001},
                    }
                }
            ),
            encoding="utf-8",
        )
        code.write_text(
            json.dumps(
                {
                    "code": "import torch\\nimport torch.nn as nn\\n\\nclass Tiny(nn.Module):\\n    def __init__(self, num_features):\\n        super().__init__()\\n        self.linear = nn.Linear(num_features, 1)\\n\\n    def forward(self, x):\\n        return self.linear(x).squeeze(-1)\\n\\nmodel_cls = Tiny\\n"
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "model_tab_quick_round"
        result = main(
            [
                "fin_model",
                "--workspace-dir",
                str(workspace),
                "--quick-smoke",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
                "--mock-code",
                str(code),
            ]
        )
        self.assertEqual(result, 0)
        workflow = (workspace / "rendered_model_workflow.yaml").read_text(encoding="utf-8")
        self.assertIn("GeneralPTNN", workflow)
        self.assertIn("class: DatasetH", workflow)
        self.assertIn("train: [2019-01-01, 2019-06-30]", workflow)
        self.assertIn("valid: [2019-07-01, 2019-09-30]", workflow)
        self.assertIn("test: [2019-10-01, 2019-12-31]", workflow)
        self.assertIn("num_features: 20", workflow)
        self.assertIn("n_jobs: 1", workflow)
        self.assertIn("topk: 10", workflow)
        self.assertIn("n_drop: 1", workflow)
        self.assertNotIn('pt_model_kwargs: {"num_features"', workflow)
        self.assertNotIn("step_len: 20    record:", workflow)

    def test_fin_quant_runs_rd_style_loop_and_stops_on_positive_feedback(self) -> None:
        hypothesis = self.tmpdir / "quant_hypothesis.json"
        experiment = self.tmpdir / "quant_experiment.json"
        feedback = self.tmpdir / "quant_feedback.json"
        hypothesis.write_text(
            json.dumps(
                {
                    "hypothesis": "Iteratively test compact momentum factors.",
                    "reason": "Start small and stop once validated by feedback.",
                    "action": "factor",
                }
            ),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "Momentum_5D": {
                        "description": "[Momentum Factor] 5-day return momentum.",
                        "formulation": "MOM_5 = close_t / close_{t-5} - 1",
                        "variables": {"close_t": "close price on day t"},
                    }
                }
            ),
            encoding="utf-8",
        )
        feedback.write_text(
            json.dumps(
                [
                    {
                        "observations": "IC is weak in the first loop.",
                        "hypothesis_evaluation": "Need another trial.",
                        "decision": False,
                        "new_hypothesis": "Adjust to more robust momentum variations.",
                        "reason": "keep evolving",
                    },
                    {
                        "observations": "IC and stability improved.",
                        "hypothesis_evaluation": "Accept current direction.",
                        "decision": True,
                        "new_hypothesis": "",
                        "reason": "stop",
                    },
                ]
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "quant_rounds"
        result = main(
            [
                "fin_quant",
                "--workspace-dir",
                str(workspace),
                "--max-rounds",
                "5",
                "--action",
                "factor",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
                "--feedback-file",
                str(feedback),
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue((workspace / "round_001" / "hypothesis.json").exists())
        self.assertTrue((workspace / "round_002" / "hypothesis.json").exists())
        self.assertTrue((workspace / "round_001" / "round_summary.json").exists())
        self.assertTrue((workspace / "round_002" / "round_summary.json").exists())
        self.assertFalse((workspace / "round_003" / "hypothesis.json").exists())
        self.assertTrue((workspace / "leaderboard.json").exists())
        self.assertTrue((workspace / "sota_factor.json").exists())
        self.assertTrue((workspace / "sota_model.json").exists())
        trace_out = json.loads((workspace / "trace_out.json").read_text(encoding="utf-8"))
        self.assertEqual(len(trace_out["hist"]), 2)
        leaderboard = json.loads((workspace / "leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(leaderboard["total_rounds"], 2)
        self.assertEqual(leaderboard["run_window"]["rounds_generated"], 2)
        self.assertIn("hypothesis", leaderboard["all_rounds"][0])

    def test_metric_feedback_uses_rd_agent_important_metrics(self) -> None:
        run_dir = self.tmpdir / "round_001" / "mlruns" / "0" / "run-id" / "metrics"
        run_dir.mkdir(parents=True)
        (run_dir / "IC").write_text("0 0.030 1\n", encoding="utf-8")
        nested = run_dir / "1day" / "excess_return_with_cost"
        nested.mkdir(parents=True)
        (nested / "annualized_return").write_text("0 0.120 1\n", encoding="utf-8")
        (nested / "max_drawdown").write_text("0 -0.050 1\n", encoding="utf-8")

        metrics = extract_metrics_from_workspace(self.tmpdir / "round_001")
        self.assertEqual(metrics["IC"], 0.03)
        self.assertEqual(metrics["1day.excess_return_with_cost.annualized_return"], 0.12)
        self.assertEqual(metrics["1day.excess_return_with_cost.max_drawdown"], -0.05)

        first_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=metrics),
            trace=[],
        )
        self.assertTrue(first_feedback["decision"])
        weaker_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.01,
                    "1day.excess_return_with_cost.annualized_return": 0.01,
                    "1day.excess_return_with_cost.max_drawdown": -0.20,
                },
            ),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": first_feedback}],
        )
        self.assertFalse(weaker_feedback["decision"])
        self.assertIn("sota_metrics", weaker_feedback)
        self.assertIn("metric_delta", weaker_feedback)
        self.assertIn("Delta(current-SOTA)", weaker_feedback["observations"])

    def test_bandit_action_selection_uses_trace_metrics(self) -> None:
        self.assertEqual(decide_bandit_action([]), "factor")
        trace_hist = [
            {
                "experiment": {"hypothesis": {"action": "factor"}},
                "feedback": {
                    "metrics": {
                        "IC": 0.03,
                        "ICIR": 0.2,
                        "Rank IC": 0.02,
                        "Rank ICIR": 0.1,
                        "1day.excess_return_with_cost.annualized_return": 0.12,
                        "1day.excess_return_with_cost.information_ratio": 0.8,
                        "1day.excess_return_with_cost.max_drawdown": -0.05,
                    }
                },
            },
            {
                "experiment": {"hypothesis": {"action": "model"}},
                "feedback": {
                    "metrics": {
                        "IC": 0.02,
                        "ICIR": 0.1,
                        "Rank IC": 0.01,
                        "Rank ICIR": 0.05,
                        "1day.excess_return_with_cost.annualized_return": 0.05,
                        "1day.excess_return_with_cost.information_ratio": 0.4,
                        "1day.excess_return_with_cost.max_drawdown": -0.08,
                    }
                },
            },
        ]
        self.assertIn(decide_bandit_action(trace_hist), {"factor", "model"})

    def test_trace_filtering_matches_quant_cross_action_context(self) -> None:
        trace_hist = [
            {
                "experiment": {"hypothesis": {"action": "factor", "hypothesis": "f1", "reason": "r1"}},
                "feedback": {"decision": False, "observations": "o1", "hypothesis_evaluation": "e1"},
            },
            {
                "experiment": {"hypothesis": {"action": "model", "hypothesis": "m1", "reason": "r2"}},
                "feedback": {"decision": True, "observations": "o2", "hypothesis_evaluation": "e2"},
            },
            {
                "experiment": {"hypothesis": {"action": "model", "hypothesis": "m2", "reason": "r3"}},
                "feedback": {"decision": False, "observations": "o3", "hypothesis_evaluation": "e3"},
            },
            {
                "experiment": {"hypothesis": {"action": "factor", "hypothesis": "f2", "reason": "r4"}},
                "feedback": {"decision": True, "observations": "o4", "hypothesis_evaluation": "e4"},
            },
        ]
        trace = TraceRecord(hist=trace_hist)
        factor_items = filtered_trace_for_action(trace, action="factor")
        model_items = filtered_trace_for_action(trace, action="model")

        factor_text = summarize_trace_items(factor_items)
        model_text = summarize_trace_items(model_items)

        self.assertIn("f1", factor_text)
        self.assertIn("f2", factor_text)
        self.assertIn("m1", factor_text)
        self.assertNotIn("m2", factor_text)

        self.assertIn("m1", model_text)
        self.assertIn("m2", model_text)
        self.assertIn("f2", model_text)
        self.assertNotIn("f1", model_text)

    def test_generate_round_uses_filtered_history_for_model_action(self) -> None:
        hypothesis = self.tmpdir / "model_round_hypothesis.json"
        experiment = self.tmpdir / "model_round_experiment.json"
        hypothesis.write_text(json.dumps({"hypothesis": "h", "reason": "r"}), encoding="utf-8")
        experiment.write_text(
            json.dumps(
                {
                    "TinyModel": {
                        "description": "d",
                        "formulation": "f",
                        "variables": {"x": "input"},
                    }
                }
            ),
            encoding="utf-8",
        )
        trace = TraceRecord(
            hist=[
                {
                    "experiment": {"hypothesis": {"action": "factor", "hypothesis": "factor-rejected", "reason": "a"}},
                    "feedback": {"decision": False, "observations": "o", "hypothesis_evaluation": "e"},
                },
                {
                    "experiment": {"hypothesis": {"action": "factor", "hypothesis": "factor-accepted", "reason": "b"}},
                    "feedback": {"decision": True, "observations": "o", "hypothesis_evaluation": "e"},
                },
                {
                    "experiment": {"hypothesis": {"action": "model", "hypothesis": "model-history", "reason": "c"}},
                    "feedback": {"decision": False, "observations": "o", "hypothesis_evaluation": "e"},
                },
            ]
        )
        result = QuantAgent(config=None).generate_round(
            action="model",
            trace=trace,
            mock_hypothesis=hypothesis,
            mock_experiment=experiment,
        )
        hyp_prompt = result["prompts"]["hypothesis_user"]
        self.assertIn("model-history", hyp_prompt)
        self.assertIn("factor-accepted", hyp_prompt)
        self.assertNotIn("factor-rejected", hyp_prompt)

    def test_best_action_baseline_prefers_accepted_same_action(self) -> None:
        trace_hist = [
            {
                "experiment": {"hypothesis": {"action": "factor"}},
                "feedback": {
                    "decision": False,
                    "performance_score": 9.0,
                    "metrics": {"IC": 0.08},
                },
            },
            {
                "experiment": {"hypothesis": {"action": "factor"}},
                "feedback": {
                    "decision": True,
                    "performance_score": 3.0,
                    "metrics": {"IC": 0.03},
                },
            },
        ]
        baseline = best_action_baseline("factor", trace_hist)
        self.assertIsNotNone(baseline)
        assert baseline is not None
        self.assertEqual(baseline["score"], 3.0)
        self.assertEqual(baseline["metrics"]["IC"], 0.03)

    def test_feedback_detects_error_signal_even_with_zero_return_code(self) -> None:
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="Traceback (most recent call last)\nValueError: missing data",
                metrics={},
            ),
            trace=[],
        )
        self.assertFalse(feedback["decision"])
        self.assertEqual(feedback["reason"], "runtime_log_error")

    def test_fin_quant_resumes_round_index_from_existing_trace(self) -> None:
        hypothesis = self.tmpdir / "resume_hypothesis.json"
        experiment = self.tmpdir / "resume_experiment.json"
        trace_file = self.tmpdir / "trace_in.json"
        hypothesis.write_text(
            json.dumps({"hypothesis": "h", "reason": "r", "action": "factor"}),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "Momentum_5D": {
                        "description": "d",
                        "formulation": "f",
                        "variables": {"x": "v"},
                    }
                }
            ),
            encoding="utf-8",
        )
        trace_file.write_text(
            json.dumps(
                {
                    "hist": [
                        {
                            "experiment": {"hypothesis": {"action": "factor", "hypothesis": "old", "reason": "old"}},
                            "feedback": {"decision": False},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "resume_rounds"
        result = main(
            [
                "fin_quant",
                "--workspace-dir",
                str(workspace),
                "--trace-file",
                str(trace_file),
                "--max-rounds",
                "1",
                "--action",
                "factor",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue((workspace / "round_002" / "hypothesis.json").exists())
        self.assertFalse((workspace / "round_001" / "hypothesis.json").exists())
        leaderboard = json.loads((workspace / "leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(leaderboard["total_rounds"], 2)
        self.assertEqual(leaderboard["run_window"]["start_loop_id"], 2)
        self.assertEqual(leaderboard["run_window"]["rounds_generated"], 1)

    def test_fin_quant_avoids_existing_round_directories(self) -> None:
        hypothesis = self.tmpdir / "disk_hypothesis.json"
        experiment = self.tmpdir / "disk_experiment.json"
        hypothesis.write_text(
            json.dumps({"hypothesis": "h", "reason": "r", "action": "factor"}),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "Momentum_5D": {
                        "description": "d",
                        "formulation": "f",
                        "variables": {"x": "v"},
                    }
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "disk_rounds"
        (workspace / "round_001").mkdir(parents=True)
        result = main(
            [
                "fin_quant",
                "--workspace-dir",
                str(workspace),
                "--max-rounds",
                "1",
                "--action",
                "factor",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue((workspace / "round_002" / "hypothesis.json").exists())
        leaderboard = json.loads((workspace / "leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(leaderboard["run_window"]["start_loop_id"], 2)

    def test_fin_quant_auto_loads_trace_from_workspace(self) -> None:
        hypothesis = self.tmpdir / "autoload_hypothesis.json"
        experiment = self.tmpdir / "autoload_experiment.json"
        hypothesis.write_text(
            json.dumps({"hypothesis": "h", "reason": "r", "action": "factor"}),
            encoding="utf-8",
        )
        experiment.write_text(
            json.dumps(
                {
                    "Momentum_5D": {
                        "description": "d",
                        "formulation": "f",
                        "variables": {"x": "v"},
                    }
                }
            ),
            encoding="utf-8",
        )
        workspace = self.tmpdir / "autoload_rounds"

        first = main(
            [
                "fin_quant",
                "--workspace-dir",
                str(workspace),
                "--max-rounds",
                "1",
                "--action",
                "factor",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
            ]
        )
        self.assertEqual(first, 0)
        self.assertTrue((workspace / "trace_out.json").exists())
        self.assertTrue((workspace / "round_001" / "hypothesis.json").exists())

        second = main(
            [
                "fin_quant",
                "--workspace-dir",
                str(workspace),
                "--max-rounds",
                "1",
                "--action",
                "factor",
                "--mock-hypothesis",
                str(hypothesis),
                "--mock-experiment",
                str(experiment),
            ]
        )
        self.assertEqual(second, 0)
        self.assertTrue((workspace / "round_002" / "hypothesis.json").exists())

    def test_improved_but_below_target_gives_decision_false(self) -> None:
        baseline_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.01,
                    "1day.excess_return_with_cost.annualized_return": 0.01,
                    "1day.excess_return_with_cost.max_drawdown": -0.20,
                },
            ),
            trace=[],
        )
        better_metrics = {
            "IC": 0.04,
            "1day.excess_return_with_cost.annualized_return": 0.10,
            "1day.excess_return_with_cost.information_ratio": 0.5,
            "1day.excess_return_with_cost.max_drawdown": -0.08,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=better_metrics),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": baseline_feedback}],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertFalse(feedback["decision"])
        self.assertFalse(feedback["target_reached"])
        self.assertIn("target_thresholds", feedback)
        self.assertIn("target_metrics", feedback)
        self.assertIn("target_gaps", feedback)
        self.assertEqual(feedback["target_thresholds"]["1day.excess_return_with_cost.annualized_return"], 0.25)
        self.assertIn("target gates are not satisfied", feedback["hypothesis_evaluation"])
        self.assertIn("improved_below_target", feedback["reason"])

    def test_improved_and_meets_target_gives_decision_true(self) -> None:
        baseline_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.01,
                    "1day.excess_return_with_cost.annualized_return": 0.01,
                    "1day.excess_return_with_cost.max_drawdown": -0.20,
                },
            ),
            trace=[],
        )
        strong_metrics = {
            "IC": 0.05,
            "1day.excess_return_with_cost.annualized_return": 0.30,
            "1day.excess_return_with_cost.information_ratio": 1.0,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=strong_metrics),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": baseline_feedback}],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertTrue(feedback["decision"])
        self.assertTrue(feedback["target_reached"])
        self.assertIn("target_thresholds", feedback)
        self.assertIn("target_gaps", feedback)
        self.assertGreaterEqual(feedback["target_gaps"]["1day.excess_return_with_cost.annualized_return"], 0.0)
        self.assertGreaterEqual(feedback["target_gaps"]["IC"], 0.0)

    def test_missing_metric_for_target_gate_gives_decision_false(self) -> None:
        metrics_no_ir = {
            "IC": 0.05,
            "1day.excess_return_with_cost.annualized_return": 0.30,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=metrics_no_ir),
            trace=[],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertFalse(feedback["decision"])
        self.assertFalse(feedback["target_reached"])
        self.assertIn("1day.excess_return_with_cost.information_ratio", feedback.get("missing_target_metrics", []))
        self.assertIn("metric missing", feedback["hypothesis_evaluation"])
        self.assertIn("1day.excess_return_with_cost.information_ratio", feedback["observations"])

    def test_no_target_profile_preserves_backward_compatible_decision(self) -> None:
        metrics = {
            "IC": 0.04,
            "1day.excess_return_with_cost.annualized_return": 0.10,
            "1day.excess_return_with_cost.max_drawdown": -0.08,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=metrics),
            trace=[],
            target_profile=None,
        )
        self.assertTrue(feedback["decision"])
        self.assertNotIn("target_thresholds", feedback)
        self.assertNotIn("target_reached", feedback)

    def test_check_targets_all_satisfied(self) -> None:
        metrics = {
            "IC": 0.05,
            "1day.excess_return_with_cost.annualized_return": 0.30,
            "1day.excess_return_with_cost.information_ratio": 1.0,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
        }
        result = check_targets(metrics, FIN_QUANT_DEFAULT_TARGETS)
        self.assertTrue(result["target_reached"])
        self.assertEqual(result["missing_target_metrics"], [])
        self.assertGreaterEqual(result["target_gaps"]["1day.excess_return_with_cost.annualized_return"], 0.0)

    def test_check_targets_with_missing_metric(self) -> None:
        metrics = {"IC": 0.05}
        result = check_targets(metrics, FIN_QUANT_DEFAULT_TARGETS)
        self.assertFalse(result["target_reached"])
        self.assertIn("1day.excess_return_with_cost.annualized_return", result["missing_target_metrics"])
        self.assertIn("1day.excess_return_with_cost.information_ratio", result["missing_target_metrics"])

    def test_build_target_profile_custom_values(self) -> None:
        from quant_agent.evaluation import build_target_profile

        profile = build_target_profile(target_arr=0.15, target_ic=0.02)
        self.assertEqual(len(profile.gates), 2)
        arr_gate = next(g for g in profile.gates if g.metric == "1day.excess_return_with_cost.annualized_return")
        self.assertEqual(arr_gate.threshold, 0.15)
        ic_gate = next(g for g in profile.gates if g.metric == "IC")
        self.assertEqual(ic_gate.threshold, 0.02)

    def test_build_target_profile_with_base_merges_overrides(self) -> None:
        from quant_agent.evaluation import build_target_profile

        profile = build_target_profile(target_arr=0.20, base=FIN_QUANT_DEFAULT_TARGETS)
        self.assertEqual(len(profile.gates), 4)
        arr_gate = next(g for g in profile.gates if g.metric == "1day.excess_return_with_cost.annualized_return")
        self.assertEqual(arr_gate.threshold, 0.20)
        ic_gate = next(g for g in profile.gates if g.metric == "IC")
        self.assertEqual(ic_gate.threshold, 0.03)
        ir_gate = next(g for g in profile.gates if g.metric == "1day.excess_return_with_cost.information_ratio")
        self.assertEqual(ir_gate.threshold, 0.8)
        mdd_gate = next(g for g in profile.gates if g.metric == "1day.excess_return_with_cost.max_drawdown")
        self.assertEqual(mdd_gate.threshold, -0.12)

    def test_build_target_profile_with_base_all_overrides(self) -> None:
        from quant_agent.evaluation import build_target_profile

        profile = build_target_profile(
            target_arr=0.30,
            target_ic=0.05,
            target_ir=1.0,
            target_max_drawdown=-0.05,
            base=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertEqual(len(profile.gates), 4)
        thresholds = {g.metric: g.threshold for g in profile.gates}
        self.assertEqual(thresholds["1day.excess_return_with_cost.annualized_return"], 0.30)
        self.assertEqual(thresholds["IC"], 0.05)
        self.assertEqual(thresholds["1day.excess_return_with_cost.information_ratio"], 1.0)
        self.assertEqual(thresholds["1day.excess_return_with_cost.max_drawdown"], -0.05)

    def test_improved_below_target_reason_label(self) -> None:
        baseline_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.01,
                    "1day.excess_return_with_cost.annualized_return": 0.01,
                    "1day.excess_return_with_cost.max_drawdown": -0.20,
                },
            ),
            trace=[],
        )
        better_metrics = {
            "IC": 0.04,
            "1day.excess_return_with_cost.annualized_return": 0.10,
            "1day.excess_return_with_cost.information_ratio": 0.5,
            "1day.excess_return_with_cost.max_drawdown": -0.08,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=better_metrics),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": baseline_feedback}],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertFalse(feedback["decision"])
        self.assertEqual(feedback["reason"], "improved_below_target")

    def test_not_improved_reason_label(self) -> None:
        baseline_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.05,
                    "1day.excess_return_with_cost.annualized_return": 0.30,
                    "1day.excess_return_with_cost.information_ratio": 1.0,
                    "1day.excess_return_with_cost.max_drawdown": -0.05,
                },
            ),
            trace=[],
        )
        weaker_metrics = {
            "IC": 0.02,
            "1day.excess_return_with_cost.annualized_return": 0.05,
            "1day.excess_return_with_cost.information_ratio": 0.3,
            "1day.excess_return_with_cost.max_drawdown": -0.15,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=weaker_metrics),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": baseline_feedback}],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertFalse(feedback["decision"])
        self.assertEqual(feedback["reason"], "metric_not_improved")

    def test_improved_and_meets_target_reason_label(self) -> None:
        baseline_feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(
                return_code=0,
                log_tail="ok",
                metrics={
                    "IC": 0.01,
                    "1day.excess_return_with_cost.annualized_return": 0.01,
                    "1day.excess_return_with_cost.max_drawdown": -0.20,
                },
            ),
            trace=[],
        )
        strong_metrics = {
            "IC": 0.05,
            "1day.excess_return_with_cost.annualized_return": 0.30,
            "1day.excess_return_with_cost.information_ratio": 1.0,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
        }
        feedback = build_feedback(
            action="factor",
            runtime=RuntimeResult(return_code=0, log_tail="ok", metrics=strong_metrics),
            trace=[{"experiment": {"hypothesis": {"action": "factor"}}, "feedback": baseline_feedback}],
            target_profile=FIN_QUANT_DEFAULT_TARGETS,
        )
        self.assertTrue(feedback["decision"])
        self.assertEqual(feedback["reason"], "metric_improved")


if __name__ == "__main__":
    unittest.main()
