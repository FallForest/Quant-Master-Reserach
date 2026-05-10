# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for scripts/fin_quant_sweep.py -- parameter sweep utility."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Load the module from file path since scripts/ has no __init__.py
_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fin_quant_sweep.py"
_spec = importlib.util.spec_from_file_location("fin_quant_sweep", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["fin_quant_sweep"] = _mod
_spec.loader.exec_module(_mod)

_parse_csv_list = _mod._parse_csv_list
_cartesian_product = _mod._cartesian_product
_build_run_dir_name = _mod._build_run_dir_name
_extract_feedback_data = _mod._extract_feedback_data
_extract_normalized = _mod._extract_normalized
build_parser = _mod.build_parser
main = _mod.main
run_sweep = _mod.run_sweep


class ParseCsvListTests(unittest.TestCase):
    """Tests for _parse_csv_list."""

    def test_returns_none_for_none_input(self) -> None:
        self.assertIsNone(_parse_csv_list(None))

    def test_returns_none_for_empty_string(self) -> None:
        self.assertIsNone(_parse_csv_list(""))

    def test_parses_int_list(self) -> None:
        self.assertEqual(_parse_csv_list("20,30,50", int), [20, 30, 50])

    def test_parses_float_list(self) -> None:
        result = _parse_csv_list("0.01,0.02", float)
        self.assertEqual(result, [0.01, 0.02])

    def test_strips_whitespace(self) -> None:
        self.assertEqual(_parse_csv_list(" 1 , 3 , 5 ", int), [1, 3, 5])

    def test_skips_empty_tokens(self) -> None:
        self.assertEqual(_parse_csv_list("1,,3,", int), [1, 3])


class CartesianProductTests(unittest.TestCase):
    """Tests for _cartesian_product."""

    def test_single_list(self) -> None:
        result = _cartesian_product({"topk": [10, 20]})
        self.assertEqual(result, [{"topk": 10}, {"topk": 20}])

    def test_two_lists(self) -> None:
        result = _cartesian_product({"topk": [10, 20], "n_drop": [1, 3]})
        self.assertEqual(len(result), 4)
        self.assertIn({"topk": 10, "n_drop": 1}, result)
        self.assertIn({"topk": 20, "n_drop": 3}, result)

    def test_none_list_becomes_single_none(self) -> None:
        result = _cartesian_product({"topk": [10], "n_drop": None})
        self.assertEqual(result, [{"topk": 10, "n_drop": None}])

    def test_all_none(self) -> None:
        result = _cartesian_product({"a": None, "b": None})
        self.assertEqual(result, [{"a": None, "b": None}])

    def test_three_lists_product(self) -> None:
        result = _cartesian_product({"a": [1, 2], "b": [3], "c": [4, 5]})
        self.assertEqual(len(result), 4)


class BuildRunDirNameTests(unittest.TestCase):
    """Tests for _build_run_dir_name."""

    def test_basic_name(self) -> None:
        name = _build_run_dir_name(1, {"topk": 20, "n_drop": 3})
        self.assertEqual(name, "run_001_topk20_ndrop3")

    def test_with_all_params(self) -> None:
        name = _build_run_dir_name(5, {
            "topk": 50, "n_drop": 1, "open_cost": 0.01,
            "close_cost": 0.02, "min_cost": 5.0, "limit_threshold": 0.1,
        })
        self.assertIn("run_005", name)
        self.assertIn("topk50", name)
        self.assertIn("ndrop1", name)
        self.assertIn("ocost0.01", name)
        self.assertIn("ccost0.02", name)
        self.assertIn("mincost5.0", name)
        self.assertIn("lthresh0.1", name)

    def test_none_values_omitted(self) -> None:
        name = _build_run_dir_name(2, {"topk": 10, "n_drop": None})
        self.assertNotIn("ndrop", name)
        self.assertIn("topk10", name)


class ExtractFeedbackDataTests(unittest.TestCase):
    """Tests for _extract_feedback_data."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sweep-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extracts_all_feedback_fields(self) -> None:
        summary = {
            "loop_id": 1,
            "action": "factor",
            "feedback": {
                "metrics": {"IC": 0.05, "1day.excess_return_with_cost.annualized_return": 0.3},
                "target_metrics": {"arr": True},
                "target_gaps": {"arr": 0.05},
                "reason": "good performance",
                "decision": "accept",
            },
        }
        path = self.tmpdir / "round_summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = _extract_feedback_data(path)
        self.assertEqual(result["feedback.metrics"]["IC"], 0.05)
        self.assertEqual(result["feedback.decision"], "accept")
        self.assertEqual(result["feedback.reason"], "good performance")

    def test_handles_missing_file(self) -> None:
        result = _extract_feedback_data(self.tmpdir / "nonexistent.json")
        self.assertIn("error", result)

    def test_handles_missing_feedback_key(self) -> None:
        path = self.tmpdir / "round_summary.json"
        path.write_text(json.dumps({"loop_id": 1}), encoding="utf-8")
        result = _extract_feedback_data(path)
        self.assertIsNone(result["feedback.metrics"])
        self.assertIsNone(result["feedback.decision"])


class ExtractNormalizedTests(unittest.TestCase):
    """Tests for _extract_normalized."""

    def test_extracts_all_normalized_columns(self) -> None:
        metrics = {
            "1day.excess_return_with_cost.annualized_return": 0.25,
            "IC": 0.04,
            "1day.excess_return_with_cost.information_ratio": 1.2,
            "1day.excess_return_with_cost.max_drawdown": -0.08,
        }
        result = _extract_normalized(metrics)
        self.assertEqual(result["arr_with_cost"], 0.25)
        self.assertEqual(result["ic"], 0.04)
        self.assertEqual(result["ir_with_cost"], 1.2)
        self.assertEqual(result["mdd_with_cost"], -0.08)

    def test_handles_none_metrics(self) -> None:
        result = _extract_normalized(None)
        for col in ("arr_with_cost", "ic", "ir_with_cost", "mdd_with_cost"):
            self.assertIsNone(result[col])

    def test_handles_missing_keys(self) -> None:
        result = _extract_normalized({"IC": 0.03})
        self.assertEqual(result["ic"], 0.03)
        self.assertIsNone(result["arr_with_cost"])


class RunSweepIntegrationTests(unittest.TestCase):
    """Integration tests for run_sweep with mocked subprocess."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sweep-integ-test-"))
        self.hypothesis = self.tmpdir / "hypothesis.json"
        self.experiment = self.tmpdir / "experiment.json"
        self.hypothesis.write_text(json.dumps({"hypothesis": "test"}), encoding="utf-8")
        self.experiment.write_text(json.dumps({"experiment": "test"}), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _fake_subprocess_run(cmd, **kwargs):
        """Simulate a successful fin_quant run that writes round_summary.json."""
        # Extract workspace-dir from cmd
        ws_idx = cmd.index("--workspace-dir") + 1
        ws_dir = Path(cmd[ws_idx])
        round_dir = ws_dir / "round_001"
        round_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "loop_id": 1,
            "action": "factor",
            "feedback": {
                "metrics": {
                    "IC": 0.05,
                    "1day.excess_return_with_cost.annualized_return": 0.30,
                    "1day.excess_return_with_cost.information_ratio": 1.5,
                    "1day.excess_return_with_cost.max_drawdown": -0.05,
                },
                "target_metrics": {"arr": True},
                "target_gaps": {"arr": 0.0},
                "reason": "target met",
                "decision": "accept",
            },
        }
        (round_dir / "round_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    def test_produces_json_and_csv_outputs(self) -> None:
        workspace = self.tmpdir / "sweep_out"
        args = [
            "--workspace-root", str(workspace),
            "--mock-hypothesis", str(self.hypothesis),
            "--mock-experiment", str(self.experiment),
            "--topk-list", "10,20",
            "--n-drop-list", "1",
        ]
        with mock.patch("fin_quant_sweep.subprocess.run", side_effect=self._fake_subprocess_run):
            result = main(args)
        self.assertEqual(result, 0)
        self.assertTrue((workspace / "sweep_results.json").exists())
        self.assertTrue((workspace / "sweep_results.csv").exists())
        # Verify JSON content
        data = json.loads((workspace / "sweep_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)
        # Verify CSV has header + 2 rows
        with (workspace / "sweep_results.csv").open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        self.assertEqual(len(rows), 3)  # header + 2 data rows

    def test_normalized_columns_in_output(self) -> None:
        workspace = self.tmpdir / "sweep_norm"
        args = [
            "--workspace-root", str(workspace),
            "--mock-hypothesis", str(self.hypothesis),
            "--mock-experiment", str(self.experiment),
            "--topk-list", "30",
        ]
        with mock.patch("fin_quant_sweep.subprocess.run", side_effect=self._fake_subprocess_run):
            main(args)
        data = json.loads((workspace / "sweep_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        row = data[0]
        self.assertEqual(row["arr_with_cost"], 0.30)
        self.assertEqual(row["ic"], 0.05)
        self.assertEqual(row["ir_with_cost"], 1.5)
        self.assertEqual(row["mdd_with_cost"], -0.05)

    def test_error_handling_does_not_crash_sweep(self) -> None:
        """A failing run should be recorded with error fields, not crash the sweep."""
        workspace = self.tmpdir / "sweep_err"
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated failure")
            return self._fake_subprocess_run(cmd, **kwargs)

        args = [
            "--workspace-root", str(workspace),
            "--mock-hypothesis", str(self.hypothesis),
            "--mock-experiment", str(self.experiment),
            "--topk-list", "10,20",
        ]
        with mock.patch("fin_quant_sweep.subprocess.run", side_effect=side_effect):
            result = main(args)
        self.assertEqual(result, 0)
        data = json.loads((workspace / "sweep_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)
        self.assertIn("error", data[0])
        # Second run should have succeeded
        self.assertEqual(data[1].get("returncode"), 0)

    def test_ranking_by_arr_with_cost(self) -> None:
        """Verify top configs are sorted by arr_with_cost desc."""
        workspace = self.tmpdir / "sweep_rank"
        scores = iter([0.10, 0.40, 0.25])

        def side_effect(cmd, **kwargs):
            score = next(scores)
            ws_idx = cmd.index("--workspace-dir") + 1
            ws_dir = Path(cmd[ws_idx])
            round_dir = ws_dir / "round_001"
            round_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "loop_id": 1,
                "action": "factor",
                "feedback": {
                    "metrics": {
                        "IC": 0.05,
                        "1day.excess_return_with_cost.annualized_return": score,
                        "1day.excess_return_with_cost.information_ratio": 1.0,
                        "1day.excess_return_with_cost.max_drawdown": -0.1,
                    },
                    "target_metrics": {},
                    "target_gaps": {},
                    "reason": "",
                    "decision": "accept",
                },
            }
            (round_dir / "round_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        args = [
            "--workspace-root", str(workspace),
            "--mock-hypothesis", str(self.hypothesis),
            "--mock-experiment", str(self.experiment),
            "--topk-list", "10,20,30",
        ]
        with mock.patch("fin_quant_sweep.subprocess.run", side_effect=side_effect):
            main(args)
        data = json.loads((workspace / "sweep_results.json").read_text(encoding="utf-8"))
        # Sort same way as _print_summary
        scored = [r for r in data if r.get("arr_with_cost") is not None]
        scored.sort(key=lambda r: (-(r.get("arr_with_cost") or 0), -(r.get("ir_with_cost") or 0)))
        self.assertEqual(scored[0]["arr_with_cost"], 0.40)
        self.assertEqual(scored[1]["arr_with_cost"], 0.25)
        self.assertEqual(scored[2]["arr_with_cost"], 0.10)

    def test_run_dir_names_are_deterministic(self) -> None:
        workspace = self.tmpdir / "sweep_determ"
        args = [
            "--workspace-root", str(workspace),
            "--mock-hypothesis", str(self.hypothesis),
            "--mock-experiment", str(self.experiment),
            "--topk-list", "20,30",
            "--n-drop-list", "1,3",
        ]
        with mock.patch("fin_quant_sweep.subprocess.run", side_effect=self._fake_subprocess_run):
            main(args)
        expected_dirs = {"run_001_topk20_ndrop1", "run_002_topk20_ndrop3",
                         "run_003_topk30_ndrop1", "run_004_topk30_ndrop3"}
        actual_dirs = {p.name for p in workspace.iterdir() if p.is_dir()}
        self.assertEqual(actual_dirs, expected_dirs)


class BuildParserTests(unittest.TestCase):
    """Tests for CLI argument parser construction."""

    def test_required_args(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--workspace-root", "/tmp/ws",
            "--mock-hypothesis", "/tmp/h.json",
            "--mock-experiment", "/tmp/e.json",
        ])
        self.assertEqual(args.action, "factor")
        self.assertEqual(args.max_rounds, 1)
        self.assertFalse(args.auto_run)
        self.assertFalse(args.quick_smoke)
        self.assertEqual(args.run_timeout_seconds, 1800)
        self.assertIsNone(args.topk_list)
        self.assertIsNone(args.n_drop_list)


if __name__ == "__main__":
    unittest.main()
