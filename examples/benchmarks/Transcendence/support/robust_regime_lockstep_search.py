#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import nonlinear_regime_portfolio_search as nrs

TARGET_IR = 2.90
TARGET_ANNRET = 0.27
DEFAULT_TEST_START = "2024-01-01"
DEFAULT_TEST_END = "2026-04-30"


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


def _parse_pipe_str_list(x: Any) -> Tuple[str, ...]:
    s = str(x or "").strip()
    return tuple([t for t in s.split("|") if t])


def _parse_pipe_int_list(x: Any) -> Tuple[int, ...]:
    vals = []
    for t in _parse_pipe_str_list(x):
        vals.append(int(float(t)))
    return tuple(vals)


def _parse_pipe_float_list(x: Any) -> Tuple[float, ...]:
    vals = []
    for t in _parse_pipe_str_list(x):
        vals.append(float(t))
    return tuple(vals)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _candidate_from_row(row: Dict[str, Any]) -> nrs.CandidateSpec:
    return nrs.CandidateSpec(
        stage=str(row.get("stage", "stage2")),
        members=_parse_pipe_str_list(row.get("members", "")),
        signs=_parse_pipe_int_list(row.get("signs", "")),
        w_normal=_parse_pipe_float_list(row.get("w_normal", "")),
        w_stress=_parse_pipe_float_list(row.get("w_stress", "")),
        rank_power=float(row.get("rank_power", 1.0)),
        winsor_q=float(row.get("winsor_q", 0.01)),
        neutralize=_as_bool(row.get("neutralize", True)),
        interaction_strength=float(row.get("interaction_strength", 0.0)),
        tanh_temp=float(row.get("tanh_temp", 1.0)),
        regime_lookback=int(float(row.get("regime_lookback", 40))),
        regime_z=float(row.get("regime_z", 0.8)),
        threshold_normal=float(row.get("threshold_normal", 0.0)),
        threshold_stress=float(row.get("threshold_stress", 0.0)),
        topk_normal=int(float(row.get("topk_normal", 40))),
        topk_stress=int(float(row.get("topk_stress", 35))),
        n_drop=int(float(row.get("n_drop", 2))),
        conversion_family=str(row.get("conversion_family", "convex_softmax")),
        rebalance_mode=str(row.get("rebalance_mode", "weekly")),
        hold_buffer=int(float(row.get("hold_buffer", 10))),
        softmax_temp=float(row.get("softmax_temp", 8.0)),
        softmax_power=float(row.get("softmax_power", 1.0)),
        max_weight=float(row.get("max_weight", 0.07)),
        vol_target=float(row.get("vol_target", 0.0)),
        lambda_turnover=float(row.get("lambda_turnover", 0.25)),
        lambda_dd=float(row.get("lambda_dd", 0.35)),
        parent_id=str(row.get("parent_id", "")),
    )


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:  # noqa: BLE001
        return float("nan")
    return x


def _year_subwindows(start_date: str, end_date: str) -> List[Tuple[str, str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    years = []
    for y in range(start.year, end.year + 1):
        st = max(start, pd.Timestamp(f"{y}-01-01"))
        ed = min(end, pd.Timestamp(f"{y}-12-31"))
        if st <= ed:
            years.append((str(y), str(st.date()), str(ed.date())))
    return years


def _stable_score(overall: Dict[str, float], yearly_rows: Sequence[Dict[str, Any]]) -> float:
    ir = float(overall["ir"])
    ann = float(overall["annret"])
    mdd = float(overall["max_drawdown"])
    to = float(overall["turnover"])
    y_ir = [float(r["ir"]) for r in yearly_rows if np.isfinite(float(r.get("ir", float("nan"))))]
    y_ann = [float(r["annret"]) for r in yearly_rows if np.isfinite(float(r.get("annret", float("nan"))))]
    ir_std = float(np.std(y_ir, ddof=0)) if len(y_ir) >= 2 else 0.0
    ann_std = float(np.std(y_ann, ddof=0)) if len(y_ann) >= 2 else 0.0
    return float(ir + 0.35 * ann - 0.28 * to - 0.35 * abs(mdd) - 0.45 * ir_std - 0.20 * ann_std)


def _pass_stability_gate(overall: Dict[str, float], yearly_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    ir = float(overall["ir"])
    ann = float(overall["annret"])
    mdd = float(overall["max_drawdown"])
    to = float(overall["turnover"])
    if not np.isfinite(ir) or not np.isfinite(ann):
        return False, "overall_nonfinite"
    if ir < 0.70:
        return False, "overall_ir_below_0.70"
    if ann < 0.06:
        return False, "overall_annret_below_0.06"
    if mdd < -0.20:
        return False, "overall_mdd_below_-0.20"
    if to > 0.45:
        return False, "overall_turnover_above_0.45"
    for r in yearly_rows:
        ir_y = float(r.get("ir", float("nan")))
        ann_y = float(r.get("annret", float("nan")))
        mdd_y = float(r.get("max_drawdown", float("nan")))
        if not np.isfinite(ir_y) or not np.isfinite(ann_y):
            return False, f"year_{r.get('year')}_nonfinite"
        if ir_y < 0.0:
            return False, f"year_{r.get('year')}_ir_negative"
        if ann_y < 0.0:
            return False, f"year_{r.get('year')}_annret_negative"
        if mdd_y < -0.25:
            return False, f"year_{r.get('year')}_mdd_below_-0.25"
    return True, ""


def _min_pred_date(run_map: Dict[str, nrs.RunSignal]) -> str:
    dates = []
    for rs in run_map.values():
        idx = rs.raw.index
        if isinstance(idx, pd.MultiIndex):
            d = pd.to_datetime(idx.get_level_values(0)).min()
        else:
            d = pd.to_datetime(idx).min()
        dates.append(d)
    return str(min(dates).date())


def _load_candidate_pool(results_csv: Path, pool_size: int) -> Tuple[List[nrs.CandidateSpec], Dict[str, Any]]:
    df = pd.read_csv(results_csv)
    df = df[(df["metrics_split"] == "test_full") & (df["error"].fillna("") == "")]
    if df.empty:
        raise RuntimeError("no valid candidates in nonlinear results")
    df = df.sort_values(["objective", "ir", "annret"], ascending=False)
    keep = df.head(int(pool_size)).copy()
    out: List[nrs.CandidateSpec] = []
    seen = set()
    for row in keep.to_dict(orient="records"):
        c = _candidate_from_row(row)
        cid = nrs._candidate_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    pool_meta = {
        "rows_total": int(len(df)),
        "rows_used": int(len(keep)),
        "unique_candidates_used": int(len(out)),
        "top_candidate_ids": [str(x) for x in keep["candidate_id"].head(10).tolist()],
    }
    return out, pool_meta


def _periods_with_degrade(min_pred_date: str, test_end: str) -> Tuple[List[Dict[str, str]], List[str]]:
    notes: List[str] = []
    preferred = [
        {
            "select_tag": "2022_2023",
            "select_start": "2022-01-01",
            "select_end": "2023-12-31",
            "apply_tag": "2024",
            "apply_start": "2024-01-01",
            "apply_end": "2024-12-31",
        },
        {
            "select_tag": "2024",
            "select_start": "2024-01-01",
            "select_end": "2024-12-31",
            "apply_tag": "2025",
            "apply_start": "2025-01-01",
            "apply_end": "2025-12-31",
        },
        {
            "select_tag": "up_to_2025",
            "select_start": "2024-01-01",
            "select_end": "2025-12-31",
            "apply_tag": "2026_ytd",
            "apply_start": "2026-01-01",
            "apply_end": test_end,
        },
    ]
    min_date = pd.Timestamp(min_pred_date)
    steps: List[Dict[str, str]] = []
    for st in preferred:
        sel_start = pd.Timestamp(st["select_start"])
        sel_end = pd.Timestamp(st["select_end"])
        app_start = pd.Timestamp(st["apply_start"])
        app_end = pd.Timestamp(st["apply_end"])
        if sel_end < min_date:
            if st["apply_tag"] == "2024":
                notes.append(
                    (
                        f"selection window {st['select_tag']} unavailable (pred starts {min_pred_date}); "
                        "degrade to 2024H1->2024H2."
                    )
                )
                steps.append(
                    {
                        "select_tag": "2024H1_degraded",
                        "select_start": str(max(min_date, pd.Timestamp("2024-01-01")).date()),
                        "select_end": "2024-06-30",
                        "apply_tag": "2024H2_degraded",
                        "apply_start": "2024-07-01",
                        "apply_end": "2024-12-31",
                        "degraded_from": st["select_tag"],
                    }
                )
                continue
            notes.append(f"selection window {st['select_tag']} unavailable (pred starts {min_pred_date}); skipped.")
            continue
        if sel_start < min_date:
            notes.append(
                f"selection window {st['select_tag']} clipped start {st['select_start']} -> {min_pred_date} due to pred coverage."
            )
            sel_start = min_date
        if app_end < app_start:
            continue
        steps.append(
            {
                "select_tag": st["select_tag"],
                "select_start": str(sel_start.date()),
                "select_end": str(sel_end.date()),
                "apply_tag": st["apply_tag"],
                "apply_start": str(app_start.date()),
                "apply_end": str(app_end.date()),
                "degraded_from": "",
            }
        )
    return steps, notes


def _select_candidate_for_window(
    *,
    candidates: Sequence[nrs.CandidateSpec],
    run_map: Dict[str, nrs.RunSignal],
    anchor_key: str,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Any, Any],
    select_start: str,
    select_end: str,
) -> Tuple[Optional[nrs.CandidateSpec], List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rows: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        cid = nrs._candidate_id(cand)
        row: Dict[str, Any] = {
            "candidate_id": cid,
            "select_start": select_start,
            "select_end": select_end,
            "stage": cand.stage,
            "members": "|".join(cand.members),
            "conversion_family": cand.conversion_family,
            "rebalance_mode": cand.rebalance_mode,
        }
        try:
            sig_df, diag = nrs._build_signal_for_candidate(
                candidate=cand,
                run_map=run_map,
                anchor_key=anchor_key,
                start_date=select_start,
                end_date=select_end,
            )
            ev, _ = nrs._eval_candidate_period(
                candidate=cand,
                signal_df=sig_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(open_cost),
                close_cost=float(close_cost),
                start_date=select_start,
                end_date=select_end,
                exchange_cache=exchange_cache,
                metrics_split="lockstep_select",
            )
            overall = {
                "ir": float(ev.ir),
                "annret": float(ev.annret),
                "max_drawdown": float(ev.max_drawdown),
                "turnover": float(ev.turnover),
            }
            yearly_rows: List[Dict[str, Any]] = []
            for ytag, yst, yed in _year_subwindows(select_start, select_end):
                try:
                    sig_y, _ = nrs._build_signal_for_candidate(
                        candidate=cand,
                        run_map=run_map,
                        anchor_key=anchor_key,
                        start_date=yst,
                        end_date=yed,
                    )
                    ev_y, _ = nrs._eval_candidate_period(
                        candidate=cand,
                        signal_df=sig_y,
                        base_port_cfg=base_port_cfg,
                        base_strategy_kwargs=base_strategy_kwargs,
                        open_cost=float(open_cost),
                        close_cost=float(close_cost),
                        start_date=yst,
                        end_date=yed,
                        exchange_cache=exchange_cache,
                        metrics_split=f"lockstep_select_year_{ytag}",
                    )
                    yearly_rows.append(
                        {
                            "year": ytag,
                            "start_date": yst,
                            "end_date": yed,
                            "ir": float(ev_y.ir),
                            "annret": float(ev_y.annret),
                            "max_drawdown": float(ev_y.max_drawdown),
                            "turnover": float(ev_y.turnover),
                            "error": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    yearly_rows.append(
                        {
                            "year": ytag,
                            "start_date": yst,
                            "end_date": yed,
                            "ir": float("nan"),
                            "annret": float("nan"),
                            "max_drawdown": float("nan"),
                            "turnover": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            gate_ok, gate_reason = _pass_stability_gate(overall=overall, yearly_rows=yearly_rows)
            stable_score = _stable_score(overall=overall, yearly_rows=yearly_rows)
            row.update(diag)
            row.update(
                {
                    "select_ir": float(ev.ir),
                    "select_annret": float(ev.annret),
                    "select_max_drawdown": float(ev.max_drawdown),
                    "select_turnover": float(ev.turnover),
                    "select_objective": float(ev.objective),
                    "stable_score": stable_score,
                    "stability_gate_pass": gate_ok,
                    "stability_gate_reason": gate_reason,
                    "yearly_rows_json": json.dumps(yearly_rows, ensure_ascii=False),
                    "error": "",
                }
            )
            scored_rows.append(row.copy())
            eval_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "select_ir": float("nan"),
                    "select_annret": float("nan"),
                    "select_max_drawdown": float("nan"),
                    "select_turnover": float("nan"),
                    "select_objective": float("nan"),
                    "stable_score": -1e9,
                    "stability_gate_pass": False,
                    "stability_gate_reason": "eval_failed",
                    "yearly_rows_json": "[]",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            eval_rows.append(row)

    valid = [r for r in scored_rows if r["error"] == ""]
    passed = [r for r in valid if bool(r["stability_gate_pass"])]
    if passed:
        best = sorted(
            passed,
            key=lambda r: (
                float(r["stable_score"]),
                float(r["select_ir"]),
                float(r["select_annret"]),
            ),
            reverse=True,
        )[0]
    elif valid:
        best = sorted(
            valid,
            key=lambda r: (
                float(r["stable_score"]),
                float(r["select_ir"]),
                float(r["select_annret"]),
            ),
            reverse=True,
        )[0]
    else:
        return None, eval_rows, scored_rows

    chosen = None
    target_cid = str(best["candidate_id"])
    for cand in candidates:
        if nrs._candidate_id(cand) == target_cid:
            chosen = cand
            break
    return chosen, eval_rows, scored_rows


def _eval_apply_period(
    *,
    candidate: nrs.CandidateSpec,
    run_map: Dict[str, nrs.RunSignal],
    anchor_key: str,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Any, Any],
    apply_start: str,
    apply_end: str,
    tag: str,
) -> Tuple[Dict[str, Any], pd.Series]:
    sig_df, diag = nrs._build_signal_for_candidate(
        candidate=candidate,
        run_map=run_map,
        anchor_key=anchor_key,
        start_date=apply_start,
        end_date=apply_end,
    )
    ev, excess = nrs._eval_candidate_period(
        candidate=candidate,
        signal_df=sig_df,
        base_port_cfg=base_port_cfg,
        base_strategy_kwargs=base_strategy_kwargs,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        start_date=apply_start,
        end_date=apply_end,
        exchange_cache=exchange_cache,
        metrics_split=tag,
    )
    row = {
        "candidate_id": nrs._candidate_id(candidate),
        "apply_tag": tag,
        "apply_start": apply_start,
        "apply_end": apply_end,
        "apply_ir": float(ev.ir),
        "apply_annret": float(ev.annret),
        "apply_max_drawdown": float(ev.max_drawdown),
        "apply_turnover": float(ev.turnover),
        "apply_objective": float(ev.objective),
        "error": "",
    }
    row.update(diag)
    return row, excess


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lockstep walk-forward robust regime selection (non-test-period per segment).")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=nrs.DEFAULT_BASE_RUN)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--test-start", default=DEFAULT_TEST_START)
    p.add_argument("--test-end", default=DEFAULT_TEST_END)
    p.add_argument("--candidate-pool", type=int, default=18)
    p.add_argument("--output-prefix", default="robust_regime_lockstep")
    p.add_argument("--nonlinear-results-csv", default="")
    p.add_argument("--nonlinear-summary-json", default="")
    p.add_argument("--nonlinear-candidate-json", default="")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    trans_dir = Path("examples/benchmarks/Transcendence").resolve()
    trans_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()

    summary_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    md_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    period_csv_path = trans_dir / f"{args.output_prefix}_periods_{stamp}.csv"
    candidate_json_path = trans_dir / f"{args.output_prefix}_candidate_{stamp}.json"
    audit_json_path = trans_dir / f"{args.output_prefix}_audit_{stamp}.json"

    nonlinear_results_csv = Path(args.nonlinear_results_csv) if args.nonlinear_results_csv else _find_latest(
        trans_dir, "nonlinear_regime_results_*.csv"
    )
    nonlinear_summary_json = Path(args.nonlinear_summary_json) if args.nonlinear_summary_json else _find_latest(
        trans_dir, "nonlinear_regime_summary_*.json"
    )
    nonlinear_candidate_json = (
        Path(args.nonlinear_candidate_json) if args.nonlinear_candidate_json else _find_latest(trans_dir, "nonlinear_regime_candidate_*.json")
    )
    sota_path = trans_dir / "sota_snapshot.json"

    tracking_dir = nrs._parse_tracking_dir(args.tracking_uri)
    base_run_dir = nrs._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = nrs._load_config(base_run_dir / "artifacts" / "config")
    nrs._init_quant_master(base_cfg)
    base_port_cfg = nrs._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    nonlin_summary = _read_json(nonlinear_summary_json)
    nonlin_candidate = _read_json(nonlinear_candidate_json)
    run_signals, run_audit_rows = nrs._discover_pred_runs(
        tracking_dir=tracking_dir,
        comparable_instruments="csi300",
        require_comparable=True,
        test_start=args.test_start,
        test_end=args.test_end,
        min_coverage_test=0.35,
    )
    if len(run_signals) < 2:
        raise RuntimeError("usable run_signals < 2")
    run_map = {x.key: x for x in run_signals}
    anchor_key = next((x.key for x in run_signals if x.run_id == nrs._resolve_run_token(args.base_run_id)), run_signals[0].key)

    candidates, pool_meta = _load_candidate_pool(nonlinear_results_csv, pool_size=int(args.candidate_pool))
    min_pred_date = _min_pred_date(run_map)
    lockstep_plan, degrade_notes = _periods_with_degrade(min_pred_date=min_pred_date, test_end=args.test_end)

    exchange_cache: Dict[Any, Any] = {}
    period_rows: List[Dict[str, Any]] = []
    selection_trace: List[Dict[str, Any]] = []
    stitched_excess_parts: List[pd.Series] = []
    selected_candidates: List[Dict[str, Any]] = []

    for step in lockstep_plan:
        chosen, eval_rows, scored_rows = _select_candidate_for_window(
            candidates=candidates,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
            select_start=step["select_start"],
            select_end=step["select_end"],
        )
        selection_trace.extend(eval_rows)
        if chosen is None:
            period_rows.append(
                {
                    "select_tag": step["select_tag"],
                    "select_start": step["select_start"],
                    "select_end": step["select_end"],
                    "apply_tag": step["apply_tag"],
                    "apply_start": step["apply_start"],
                    "apply_end": step["apply_end"],
                    "candidate_id": "",
                    "apply_ir": float("nan"),
                    "apply_annret": float("nan"),
                    "apply_max_drawdown": float("nan"),
                    "apply_turnover": float("nan"),
                    "apply_objective": float("nan"),
                    "stability_gate_pass": False,
                    "stability_gate_reason": "no_valid_candidate",
                    "error": "no valid candidate on selection window",
                }
            )
            continue

        chosen_cid = nrs._candidate_id(chosen)
        chosen_row = None
        for r in sorted(scored_rows, key=lambda x: float(x.get("stable_score", -1e9)), reverse=True):
            if str(r.get("candidate_id")) == chosen_cid:
                chosen_row = r
                break
        if chosen_row is None:
            chosen_row = {"stability_gate_pass": False, "stability_gate_reason": "missing_chosen_row"}

        selected_candidates.append(
            {
                "candidate_id": chosen_cid,
                "select_tag": step["select_tag"],
                "select_start": step["select_start"],
                "select_end": step["select_end"],
                "selected_by_score": float(chosen_row.get("stable_score", float("nan"))),
                "stability_gate_pass": bool(chosen_row.get("stability_gate_pass", False)),
                "stability_gate_reason": str(chosen_row.get("stability_gate_reason", "")),
                "candidate": asdict(chosen),
            }
        )

        try:
            apply_row, ex = _eval_apply_period(
                candidate=chosen,
                run_map=run_map,
                anchor_key=anchor_key,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=exchange_cache,
                apply_start=step["apply_start"],
                apply_end=step["apply_end"],
                tag=step["apply_tag"],
            )
            stitched_excess_parts.append(ex)
            period_rows.append(
                {
                    "select_tag": step["select_tag"],
                    "select_start": step["select_start"],
                    "select_end": step["select_end"],
                    "apply_tag": step["apply_tag"],
                    "apply_start": step["apply_start"],
                    "apply_end": step["apply_end"],
                    "candidate_id": chosen_cid,
                    "apply_ir": apply_row["apply_ir"],
                    "apply_annret": apply_row["apply_annret"],
                    "apply_max_drawdown": apply_row["apply_max_drawdown"],
                    "apply_turnover": apply_row["apply_turnover"],
                    "apply_objective": apply_row["apply_objective"],
                    "stability_gate_pass": bool(chosen_row.get("stability_gate_pass", False)),
                    "stability_gate_reason": str(chosen_row.get("stability_gate_reason", "")),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            period_rows.append(
                {
                    "select_tag": step["select_tag"],
                    "select_start": step["select_start"],
                    "select_end": step["select_end"],
                    "apply_tag": step["apply_tag"],
                    "apply_start": step["apply_start"],
                    "apply_end": step["apply_end"],
                    "candidate_id": chosen_cid,
                    "apply_ir": float("nan"),
                    "apply_annret": float("nan"),
                    "apply_max_drawdown": float("nan"),
                    "apply_turnover": float("nan"),
                    "apply_objective": float("nan"),
                    "stability_gate_pass": bool(chosen_row.get("stability_gate_pass", False)),
                    "stability_gate_reason": str(chosen_row.get("stability_gate_reason", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    _write_csv(period_csv_path, period_rows)

    stitched_metrics = None
    if stitched_excess_parts:
        stitched_excess = pd.concat(stitched_excess_parts).sort_index()
        stitched_metrics = nrs._metrics_from_excess(stitched_excess)

    yearly_metrics: List[Dict[str, Any]] = []
    if stitched_excess_parts:
        stitched_excess = pd.concat(stitched_excess_parts).sort_index()
        idx_dt = pd.to_datetime(stitched_excess.index)
        for y in sorted(set(idx_dt.year.tolist())):
            m = idx_dt.year == y
            ex_y = stitched_excess.loc[m]
            if len(ex_y) < 20:
                continue
            ym = nrs._metrics_from_excess(ex_y)
            yearly_metrics.append(
                {
                    "year": int(y),
                    "ir": float(ym["ir"]),
                    "annret": float(ym["annret"]),
                    "max_drawdown": float(ym["max_drawdown"]),
                }
            )

    best_period_row = None
    valid_period_rows = [r for r in period_rows if str(r.get("error", "")) == "" and np.isfinite(_safe_float(r.get("apply_ir")))]
    if valid_period_rows:
        best_period_row = sorted(
            valid_period_rows,
            key=lambda r: (_safe_float(r["apply_ir"]), _safe_float(r["apply_annret"])),
            reverse=True,
        )[0]

    hard_gate_pass = bool(
        stitched_metrics is not None
        and float(stitched_metrics.get("ir", float("nan"))) > TARGET_IR
        and float(stitched_metrics.get("annret", float("nan"))) > TARGET_ANNRET
    )

    summary = {
        "timestamp_utc": _now_utc(),
        "protocol": {
            "name": "lockstep_walk_forward_non_test_period_selection",
            "test_period": {"start": args.test_start, "end": args.test_end},
            "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
            "selection_rule": "choose candidate on prior window only; apply to next window",
            "stability_gate": {
                "overall_ir_min": 0.70,
                "overall_annret_min": 0.06,
                "overall_mdd_min": -0.20,
                "overall_turnover_max": 0.45,
                "yearly_ir_min": 0.0,
                "yearly_annret_min": 0.0,
                "yearly_mdd_min": -0.25,
            },
            "degrade_notes": degrade_notes,
            "lockstep_plan": lockstep_plan,
            "min_pred_date": min_pred_date,
        },
        "source_artifacts": {
            "nonlinear_results_csv": str(nonlinear_results_csv).replace("\\", "/"),
            "nonlinear_summary_json": str(nonlinear_summary_json).replace("\\", "/"),
            "nonlinear_candidate_json": str(nonlinear_candidate_json).replace("\\", "/"),
            "sota_snapshot_json": str(sota_path).replace("\\", "/"),
        },
        "structure_extract": {
            "worker_c_best": nonlin_candidate.get("candidate", {}),
            "worker_c_best_members": nonlin_candidate.get("candidate", {}).get("members", []),
            "worker_c_best_signs": nonlin_candidate.get("candidate", {}).get("signs", []),
            "worker_c_best_topk_dropout_threshold": {
                "topk_normal": nonlin_candidate.get("candidate", {}).get("topk_normal"),
                "topk_stress": nonlin_candidate.get("candidate", {}).get("topk_stress"),
                "n_drop": nonlin_candidate.get("candidate", {}).get("n_drop"),
                "threshold_normal": nonlin_candidate.get("candidate", {}).get("threshold_normal"),
                "threshold_stress": nonlin_candidate.get("candidate", {}).get("threshold_stress"),
            },
            "worker_c_best_rank_transform": {
                "rank_power": nonlin_candidate.get("candidate", {}).get("rank_power"),
                "winsor_q": nonlin_candidate.get("candidate", {}).get("winsor_q"),
                "neutralize": nonlin_candidate.get("candidate", {}).get("neutralize"),
                "interaction_strength": nonlin_candidate.get("candidate", {}).get("interaction_strength"),
                "tanh_temp": nonlin_candidate.get("candidate", {}).get("tanh_temp"),
            },
            "worker_c_best_regime_logic": {
                "regime_lookback": nonlin_candidate.get("candidate", {}).get("regime_lookback"),
                "regime_z": nonlin_candidate.get("candidate", {}).get("regime_z"),
                "w_normal": nonlin_candidate.get("candidate", {}).get("w_normal"),
                "w_stress": nonlin_candidate.get("candidate", {}).get("w_stress"),
            },
            "pool_meta": pool_meta,
        },
        "results": {
            "stitched_metrics": stitched_metrics,
            "yearly_metrics": yearly_metrics,
            "hard_gate": {"ir_gt": TARGET_IR, "annret_gt": TARGET_ANNRET, "pass": hard_gate_pass},
            "best_apply_period": best_period_row,
        },
        "selected_candidates": selected_candidates,
        "artifacts": {
            "summary_json": str(summary_path).replace("\\", "/"),
            "summary_md": str(md_path).replace("\\", "/"),
            "period_csv": str(period_csv_path).replace("\\", "/"),
            "candidate_json": str(candidate_json_path).replace("\\", "/"),
            "audit_json": str(audit_json_path).replace("\\", "/"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_obj = {
        "timestamp_utc": _now_utc(),
        "protocol": "lockstep_walk_forward_non_test_period_selection",
        "test_period": {"start": args.test_start, "end": args.test_end},
        "selected_candidates": selected_candidates,
        "best_apply_period": best_period_row,
        "stitched_metrics": stitched_metrics,
    }
    candidate_json_path.write_text(json.dumps(candidate_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_obj = {
        "timestamp_utc": _now_utc(),
        "run_audit_rows": run_audit_rows,
        "selection_trace_rows": selection_trace,
        "period_rows": period_rows,
        "degrade_notes": degrade_notes,
        "lockstep_plan": lockstep_plan,
        "min_pred_date": min_pred_date,
    }
    audit_json_path.write_text(json.dumps(audit_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# Robust Regime Lockstep Summary ({stamp})

- Protocol: lockstep / walk-forward, select on prior window only.
- Test period: `{args.test_start}..{args.test_end}`.
- Costs: `open={args.open_cost}`, `close={args.close_cost}`.
- Hard gate: `IR>{TARGET_IR}` and `AnnRet>{TARGET_ANNRET}`.
- Hard gate pass: `{hard_gate_pass}`.
- Stitched metrics: `{json.dumps(stitched_metrics, ensure_ascii=False)}`.
- Degrade notes: `{json.dumps(degrade_notes, ensure_ascii=False)}`.
"""
    md_path.write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
