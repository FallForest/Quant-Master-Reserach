#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

import robust_backtest_constraint_diagnostics as diag

TARGET_IR = 2.90
TARGET_ANNRET = 0.27


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _find_latest(trans_dir: Path, pattern: str) -> Path:
    candidates = sorted(trans_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing artifact: {pattern}")
    return candidates[-1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _gap_to_gate(metrics: Dict[str, Any] | None) -> Dict[str, float | None]:
    if not metrics:
        return {"ir_shortfall": None, "annret_shortfall": None}
    ir = _safe_float(metrics.get("ir"))
    annret = _safe_float(metrics.get("annret"))
    return {
        "ir_shortfall": None if not np.isfinite(ir) else float(TARGET_IR - ir),
        "annret_shortfall": None if not np.isfinite(annret) else float(TARGET_ANNRET - annret),
    }


def _series_metrics(rs: Dict[str, Any]) -> Dict[str, Any]:
    return rs["metrics"] if rs.get("ok") else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict lockstep replay after sell epsilon fix, with diagnostic wrapper comparison.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=diag.nrs.DEFAULT_BASE_RUN)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--test-start", default=diag.DEFAULT_TEST_START)
    p.add_argument("--test-end", default=diag.DEFAULT_TEST_END)
    p.add_argument("--lockstep-summary-json", default="")
    p.add_argument("--lockstep-audit-json", default="")
    p.add_argument("--nonlinear-results-csv", default="")
    p.add_argument("--output-prefix", default="execution_epsilon_fix_recheck")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    ctx = diag._load_context(args)
    trans_dir: Path = ctx["trans_dir"]
    stamp = _timestamp()

    lockstep_summary_json = (
        Path(args.lockstep_summary_json)
        if args.lockstep_summary_json
        else _find_latest(trans_dir, "robust_regime_lockstep_summary_*.json")
    )
    lockstep_audit_json = (
        Path(args.lockstep_audit_json)
        if args.lockstep_audit_json
        else _find_latest(trans_dir, "robust_regime_lockstep_audit_*.json")
    )
    nonlinear_results_csv = (
        Path(args.nonlinear_results_csv)
        if args.nonlinear_results_csv
        else _find_latest(trans_dir, "nonlinear_regime_results_*.csv")
    )

    summary_json_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    period_csv_path = trans_dir / f"{args.output_prefix}_periods_{stamp}.csv"
    parse_smoke_path = trans_dir / f"{args.output_prefix}_parse_smoke_{stamp}.json"

    lockstep_summary = _read_json(lockstep_summary_json)
    lockstep_audit = _read_json(lockstep_audit_json)
    candidate_map = diag._load_candidate_map(nonlinear_results_csv)

    base_port_cfg = ctx["base_port_cfg"]
    base_strategy_kwargs = ctx["base_strategy_kwargs"]
    run_map = ctx["run_map"]
    anchor_key = ctx["anchor_key"]

    selected_candidates = lockstep_summary.get("selected_candidates", [])
    period_rows: List[Dict[str, Any]] = []
    baseline_excess_parts: List[pd.Series] = []
    safe_excess_parts: List[pd.Series] = []

    for row in selected_candidates:
        cid = str(row.get("candidate_id", "")).strip()
        select_tag = str(row.get("select_tag", "")).strip()
        cand = diag._candidate_from_obj(row.get("candidate", {})) if row.get("candidate") else candidate_map.get(cid)
        if cand is None:
            continue

        if select_tag == "2024H1_degraded":
            apply_start, apply_end = "2024-07-01", "2024-12-31"
        elif select_tag == "2024":
            apply_start, apply_end = "2025-01-01", "2025-12-31"
        elif select_tag == "up_to_2025":
            apply_start, apply_end = "2026-01-01", args.test_end
        else:
            continue

        baseline_rs = diag._eval_candidate_with_runtime_capture(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_date=apply_start,
            end_date=apply_end,
            safe_mode=False,
        )
        safe_rs = diag._eval_candidate_with_runtime_capture(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_date=apply_start,
            end_date=apply_end,
            safe_mode=True,
        )

        if baseline_rs.get("ok") and baseline_rs.get("excess") is not None:
            baseline_excess_parts.append(baseline_rs["excess"])
        if safe_rs.get("ok") and safe_rs.get("excess") is not None:
            safe_excess_parts.append(safe_rs["excess"])

        baseline_metrics = _series_metrics(baseline_rs)
        safe_metrics = _series_metrics(safe_rs)
        period_rows.append(
            {
                "select_tag": select_tag,
                "apply_start": apply_start,
                "apply_end": apply_end,
                "candidate_id": cid,
                "baseline_ok": bool(baseline_rs.get("ok")),
                "baseline_error": None if baseline_rs.get("ok") else json.dumps(baseline_rs.get("error"), ensure_ascii=False),
                "baseline_ir": None if not baseline_metrics else baseline_metrics.get("ir"),
                "baseline_annret": None if not baseline_metrics else baseline_metrics.get("annret"),
                "baseline_max_drawdown": None if not baseline_metrics else baseline_metrics.get("max_drawdown"),
                "safe_ok": bool(safe_rs.get("ok")),
                "safe_error": None if safe_rs.get("ok") else json.dumps(safe_rs.get("error"), ensure_ascii=False),
                "safe_ir": None if not safe_metrics else safe_metrics.get("ir"),
                "safe_annret": None if not safe_metrics else safe_metrics.get("annret"),
                "safe_max_drawdown": None if not safe_metrics else safe_metrics.get("max_drawdown"),
            }
        )

    strict_after_fix_metrics = None
    if baseline_excess_parts:
        strict_after_fix_metrics = diag._metrics_from_excess(pd.concat(baseline_excess_parts).sort_index())
    safe_wrapper_metrics = None
    if safe_excess_parts:
        safe_wrapper_metrics = diag._metrics_from_excess(pd.concat(safe_excess_parts).sort_index())

    strict_before_fix_metrics = lockstep_summary.get("results", {}).get("stitched_metrics")
    strict_gap = _gap_to_gate(strict_after_fix_metrics)
    safe_gap = _gap_to_gate(safe_wrapper_metrics)

    failed_events = diag._extract_failed_events(lockstep_audit)
    sell_fix_events = [
        x
        for x in failed_events
        if str(x.get("symbol")) == "SH603296" and abs(_safe_float(x.get("diff"))) <= 0.1
    ]
    unresolved_events = [
        x
        for x in failed_events
        if not (str(x.get("symbol")) == "SH603296" and abs(_safe_float(x.get("diff"))) <= 0.1)
    ]

    summary = {
        "timestamp_utc": _now_utc(),
        "source_artifacts": {
            "lockstep_summary_json": str(lockstep_summary_json).replace("\\", "/"),
            "lockstep_audit_json": str(lockstep_audit_json).replace("\\", "/"),
            "nonlinear_results_csv": str(nonlinear_results_csv).replace("\\", "/"),
        },
        "hard_gate": {"ir_gt": TARGET_IR, "annret_gt": TARGET_ANNRET},
        "baseline_strict_before_fix": {
            "stitched_metrics": strict_before_fix_metrics,
            "note": "previous lockstep summary; 2025 segment failed before the sell epsilon fix was available to the replay chain",
        },
        "baseline_strict_after_fix": {
            "stitched_metrics": strict_after_fix_metrics,
            "gate_gap": strict_gap,
            "hard_gate_pass": bool(
                strict_after_fix_metrics
                and _safe_float(strict_after_fix_metrics.get("ir")) > TARGET_IR
                and _safe_float(strict_after_fix_metrics.get("annret")) > TARGET_ANNRET
            ),
        },
        "diagnostic_safe_wrapper": {
            "stitched_metrics": safe_wrapper_metrics,
            "gate_gap": safe_gap,
            "hard_gate_pass": bool(
                safe_wrapper_metrics
                and _safe_float(safe_wrapper_metrics.get("ir")) > TARGET_IR
                and _safe_float(safe_wrapper_metrics.get("annret")) > TARGET_ANNRET
            ),
            "wrapper_only": True,
        },
        "per_period_comparison": period_rows,
        "attribution": {
            "real_bug_fix_effect": {
                "resolved_events": sell_fix_events,
                "explanation": "The production Position._sell_stock epsilon tolerance removes the micro-oversell crash and restores the 2025 baseline segment to a real strict replay result.",
            },
            "wrapper_only_effect": {
                "unresolved_events": unresolved_events,
                "guards": [
                    "min_names_guard",
                    "dynamic_topk_clamp",
                    "fallback_mode=cash",
                    "safe order-generator sell epsilon clip",
                ],
                "explanation": "These guards change runtime portfolio construction behavior in the diagnostic script and therefore cannot be counted as production-ready lockstep performance.",
            },
        },
        "conclusion": {
            "epsilon_fix_sufficient": False,
            "summary": (
                "Sell epsilon fix removes the 2025 crash, but the strict stitched lockstep result remains far below the hard gate; "
                "the large uplift appears only under diagnostic wrapper guards."
            ),
        },
        "artifacts": {
            "period_csv": str(period_csv_path).replace("\\", "/"),
            "summary_json": str(summary_json_path).replace("\\", "/"),
            "summary_md": str(summary_md_path).replace("\\", "/"),
            "parse_smoke_json": str(parse_smoke_path).replace("\\", "/"),
        },
    }

    _write_csv(period_csv_path, period_rows)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Lockstep After Epsilon Recheck ({stamp})",
        "",
        f"- Strict baseline before fix artifact: `{json.dumps(strict_before_fix_metrics, ensure_ascii=False)}`",
        f"- Strict baseline after fix: `{json.dumps(strict_after_fix_metrics, ensure_ascii=False)}`",
        f"- Diagnostic safe wrapper: `{json.dumps(safe_wrapper_metrics, ensure_ascii=False)}`",
        "",
        "## Gate Distance",
        f"- strict after fix IR shortfall: `{strict_gap['ir_shortfall']}`; AnnRet shortfall: `{strict_gap['annret_shortfall']}`",
        f"- safe wrapper IR shortfall: `{safe_gap['ir_shortfall']}`; AnnRet shortfall: `{safe_gap['annret_shortfall']}`",
        "",
        "## Attribution",
        "- Real bug fix: production `Position._sell_stock` now clips tiny floating-point oversell and restores the 2025 strict replay instead of throwing.",
        "- Wrapper only: `min_names_guard`, dynamic `topk` clamp, cash fallback, and safe order-generator clipping alter diagnostic runtime behavior and are not production lockstep results.",
        "",
        "## Conclusion",
        "- Epsilon fix alone is not enough to get close to the hard gate.",
        "- Any apparent near-recovery that depends on the safe wrapper should be treated as diagnostic evidence, not a new approved model result.",
    ]
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    smoke = {
        "timestamp_utc": _now_utc(),
        "summary_json_exists": summary_json_path.exists(),
        "summary_md_exists": summary_md_path.exists(),
        "period_csv_exists": period_csv_path.exists(),
        "summary_json_parse_ok": False,
        "period_rows": 0,
    }
    try:
        json.loads(summary_json_path.read_text(encoding="utf-8"))
        smoke["summary_json_parse_ok"] = True
    except Exception as exc:  # noqa: BLE001
        smoke["summary_json_parse_error"] = f"{type(exc).__name__}: {exc}"
    try:
        with period_csv_path.open("r", encoding="utf-8", newline="") as f:
            smoke["period_rows"] = max(0, sum(1 for _ in f) - 1)
    except Exception as exc:  # noqa: BLE001
        smoke["period_csv_parse_error"] = f"{type(exc).__name__}: {exc}"
    parse_smoke_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
