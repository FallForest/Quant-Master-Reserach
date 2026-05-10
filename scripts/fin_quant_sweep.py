# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Reproducible parameter sweep utility for ``quant_agent.cli fin_quant``.

Runs the fin_quant subcommand across a Cartesian product of execution
overrides (topk, n-drop, costs, limit-threshold) and aggregates per-run
metrics into JSON / CSV for downstream analysis.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Column extraction helpers
# ---------------------------------------------------------------------------

_NORMALIZED_COLUMNS: dict[str, str] = {
    "arr_with_cost": "1day.excess_return_with_cost.annualized_return",
    "ic": "IC",
    "ir_with_cost": "1day.excess_return_with_cost.information_ratio",
    "mdd_with_cost": "1day.excess_return_with_cost.max_drawdown",
}


def _parse_csv_list(raw: str | None, cast: type = str) -> list[Any] | None:
    """Parse a comma-separated string into a typed list.

    Parameters
    ----------
    raw : str or None
        Comma-separated CLI value (e.g. ``"20,30,50"``).
    cast : type
        Target type for each element (``int``, ``float``, ``str``).

    Returns
    -------
    list or None
        Parsed list, or ``None`` when *raw* is empty / ``None``.
    """
    if raw is None:
        return None
    items = [t.strip() for t in raw.split(",") if t.strip()]
    if not items:
        return None
    return [cast(v) for v in items]


def _cartesian_product(sweep_lists: dict[str, list[Any] | None]) -> list[dict[str, Any]]:
    """Build the Cartesian product of sweep parameter lists.

    Parameters
    ----------
    sweep_lists : dict
        Mapping from parameter name to its value list.  ``None`` entries
        are replaced with ``[None]`` so the parameter is carried through
        but not varied.

    Returns
    -------
    list[dict]
        Each element is one combination, keyed by parameter name.
    """
    keys = list(sweep_lists.keys())
    pools: list[list[Any]] = [
        sweep_lists[k] if sweep_lists[k] is not None else [None] for k in keys
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*pools)]


def _build_run_dir_name(idx: int, config: dict[str, Any]) -> str:
    """Return a deterministic, human-readable run directory name."""
    parts = [f"run_{idx:03d}"]
    if config.get("topk") is not None:
        parts.append(f"topk{config['topk']}")
    if config.get("n_drop") is not None:
        parts.append(f"ndrop{config['n_drop']}")
    if config.get("open_cost") is not None:
        parts.append(f"ocost{config['open_cost']}")
    if config.get("close_cost") is not None:
        parts.append(f"ccost{config['close_cost']}")
    if config.get("min_cost") is not None:
        parts.append(f"mincost{config['min_cost']}")
    if config.get("limit_threshold") is not None:
        parts.append(f"lthresh{config['limit_threshold']}")
    return "_".join(parts)


def _extract_feedback_data(round_summary_path: Path) -> dict[str, Any]:
    """Extract relevant feedback fields from a round summary file.

    Returns a flat dict with ``feedback.metrics``, ``feedback.target_metrics``,
    ``feedback.target_gaps``, ``feedback.reason``, ``feedback.decision`` -- or
    error info when the file is missing / malformed.
    """
    result: dict[str, Any] = {}
    try:
        payload = json.loads(round_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = f"read_error: {exc}"
        return result

    fb = payload.get("feedback") or {}
    result["feedback.metrics"] = fb.get("metrics")
    result["feedback.target_metrics"] = fb.get("target_metrics")
    result["feedback.target_gaps"] = fb.get("target_gaps")
    result["feedback.reason"] = fb.get("reason")
    result["feedback.decision"] = fb.get("decision")
    return result


def _extract_normalized(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Pull normalized columns out of a metrics dict."""
    out: dict[str, Any] = {}
    if not isinstance(metrics, dict):
        for col in _NORMALIZED_COLUMNS:
            out[col] = None
        return out
    for norm_name, metric_key in _NORMALIZED_COLUMNS.items():
        out[norm_name] = metrics.get(metric_key)
    return out


def _build_cli_args(
    *,
    workspace_dir: Path,
    mock_hypothesis: str,
    mock_experiment: str,
    action: str,
    max_rounds: int,
    auto_run: bool,
    quick_smoke: bool,
    run_timeout_seconds: int,
    config: dict[str, Any],
) -> list[str]:
    """Build the argument list for a single ``quant_agent.cli fin_quant`` call."""
    args: list[str] = [
        "-m", "quant_agent.cli", "fin_quant",
        "--workspace-dir", str(workspace_dir),
        "--mock-hypothesis", mock_hypothesis,
        "--mock-experiment", mock_experiment,
        "--action", action,
        "--max-rounds", str(max_rounds),
        "--action-selection", "random",
        "--run-timeout-seconds", str(run_timeout_seconds),
    ]
    if auto_run:
        args.append("--auto-run")
    if quick_smoke:
        args.append("--quick-smoke")
    for key in ("topk", "n_drop", "open_cost", "close_cost", "min_cost", "limit_threshold"):
        val = config.get(key)
        if val is not None:
            cli_key = key.replace("_", "-")
            args.extend([f"--{cli_key}", str(val)])
    return args


def run_sweep(args: argparse.Namespace) -> int:
    """Execute the full parameter sweep.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    workspace_root = Path(args.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    sweep_lists: dict[str, list[Any] | None] = {
        "topk": _parse_csv_list(args.topk_list, int),
        "n_drop": _parse_csv_list(args.n_drop_list, int),
        "open_cost": _parse_csv_list(args.open_cost_list, float),
        "close_cost": _parse_csv_list(args.close_cost_list, float),
        "min_cost": _parse_csv_list(args.min_cost_list, float),
        "limit_threshold": _parse_csv_list(args.limit_threshold_list, float),
    }
    configs = _cartesian_product(sweep_lists)
    print(f"[sweep] {len(configs)} configurations to evaluate")

    all_results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs, start=1):
        run_name = _build_run_dir_name(idx, config)
        run_dir = workspace_root / run_name
        print(f"[sweep] ({idx}/{len(configs)}) {run_name}")

        cli_args = _build_cli_args(
            workspace_dir=run_dir,
            mock_hypothesis=args.mock_hypothesis,
            mock_experiment=args.mock_experiment,
            action=args.action,
            max_rounds=args.max_rounds,
            auto_run=args.auto_run,
            quick_smoke=args.quick_smoke,
            run_timeout_seconds=args.run_timeout_seconds,
            config=config,
        )

        row: dict[str, Any] = {"run_index": idx, "run_dir": str(run_dir)}
        row.update({k: v for k, v in config.items()})

        try:
            proc = subprocess.run(
                [sys.executable] + cli_args,
                capture_output=True,
                text=True,
                timeout=args.run_timeout_seconds + 120,
            )
            row["returncode"] = proc.returncode
            row["stdout"] = proc.stdout[-4000:] if proc.stdout else ""
            row["stderr"] = proc.stderr[-4000:] if proc.stderr else ""
        except subprocess.TimeoutExpired:
            row["returncode"] = -1
            row["error"] = "subprocess_timeout"
            all_results.append(row)
            continue
        except Exception as exc:
            row["returncode"] = -1
            row["error"] = f"subprocess_error: {exc}"
            all_results.append(row)
            continue

        # Try to read round_summary.json
        summary_path = run_dir / "round_001" / "round_summary.json"
        feedback_data = _extract_feedback_data(summary_path)
        row.update(feedback_data)

        normalized = _extract_normalized(feedback_data.get("feedback.metrics"))
        row.update(normalized)

        all_results.append(row)

    # Write outputs
    results_json_path = workspace_root / "sweep_results.json"
    results_json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _write_csv(workspace_root / "sweep_results.csv", all_results)

    _print_summary(all_results)
    print(f"\n[sweep] Results written to {results_json_path}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sweep results as CSV, flattening nested dicts/lists to JSON strings."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Collect all keys preserving order from first row, then extras
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    flat[k] = v
            writer.writerow(flat)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a concise table of top configs sorted by arr_with_cost then ir_with_cost."""
    scored = [
        r for r in rows
        if r.get("arr_with_cost") is not None
    ]
    if not scored:
        print("\n[sweep] No runs produced arr_with_cost metrics.")
        return
    scored.sort(key=lambda r: (-(r.get("arr_with_cost") or 0), -(r.get("ir_with_cost") or 0)))
    top = scored[:10]
    print(f"\n{'='*80}")
    print(f"Top {len(top)} configs by arr_with_cost desc, ir_with_cost desc:")
    print(f"{'='*80}")
    header = f"{'Rank':>4}  {'Run':>4}  {'topk':>6}  {'n_drop':>6}  {'arr_w_cost':>12}  {'ic':>8}  {'ir_w_cost':>10}  {'mdd_w_cost':>12}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(top, start=1):
        print(
            f"{rank:>4}  {r.get('run_index', '?'):>4}  "
            f"{str(r.get('topk', '-')):>6}  {str(r.get('n_drop', '-')):>6}  "
            f"{r.get('arr_with_cost', 'N/A'):>12}  {str(r.get('ic', 'N/A')):>8}  "
            f"{r.get('ir_with_cost', 'N/A'):>10}  {r.get('mdd_with_cost', 'N/A'):>12}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fin_quant_sweep",
        description="Parameter sweep over quant_agent.cli fin_quant execution overrides.",
    )
    parser.add_argument("--workspace-root", required=True, help="Root directory for all sweep runs and result files.")
    parser.add_argument("--mock-hypothesis", required=True, help="Path to offline hypothesis JSON.")
    parser.add_argument("--mock-experiment", required=True, help="Path to offline experiment JSON.")
    parser.add_argument("--action", default="factor", choices=["factor", "model"], help="Action for fin_quant (default: factor).")
    parser.add_argument("--max-rounds", type=int, default=1, help="Max rounds per run (default: 1).")
    parser.add_argument("--auto-run", action="store_true", help="Enable auto-run of run_experiment.bat.")
    parser.add_argument("--quick-smoke", action="store_true", help="Use quick-smoke mode.")
    parser.add_argument("--run-timeout-seconds", type=int, default=1800, help="Per-run timeout in seconds (default: 1800).")
    # Sweep lists
    parser.add_argument("--topk-list", default=None, help="Comma-separated topk values (e.g. 20,30,50).")
    parser.add_argument("--n-drop-list", default=None, help="Comma-separated n-drop values (e.g. 1,3).")
    parser.add_argument("--open-cost-list", default=None, help="Comma-separated open-cost values.")
    parser.add_argument("--close-cost-list", default=None, help="Comma-separated close-cost values.")
    parser.add_argument("--min-cost-list", default=None, help="Comma-separated min-cost values.")
    parser.add_argument("--limit-threshold-list", default=None, help="Comma-separated limit-threshold values.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
