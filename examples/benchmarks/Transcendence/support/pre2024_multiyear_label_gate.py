#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
for _path in (REPO_ROOT, BENCHMARK_ROOT, BENCHMARK_ROOT / "model", BENCHMARK_ROOT / "strategy", THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import drawdown_conditional_label_pre2024 as dd
import liquidity_survival_label_pre2024 as liq
import market_state_target_pre2024 as ms
from quant_master.config import resolve_provider_uri


EXPERIMENT_NAME = "pre-2024 multi-year stable label gate"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
GATE_END = "2023-12-31"
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")

GATE_MIN_COMBINED_IR = 1.8
GATE_MAX_COMBINED_TURNOVER = 0.16
GATE_MIN_IR_POSITIVE_YEARS = 3
GATE_MIN_YEAR_IR = 0.0
DEFAULT_WORKFLOW_CONFIG = "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(_json_sanitize(list(rows)))


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in _parse_csv(text)]


def _parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in _parse_csv(text)]


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "candidates_csv": THIS_DIR / f"{output_prefix}_candidates_{stamp}.csv",
        "year_metrics_csv": THIS_DIR / f"{output_prefix}_year_metrics_{stamp}.csv",
    }


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[float, float]:
    valid_ic = _safe_float(row.get("valid_rank_ic"))
    train_ic = _safe_float(row.get("train_rank_ic"))
    return (
        valid_ic if np.isfinite(valid_ic) else -1e9,
        train_ic if np.isfinite(train_ic) else -1e9,
    )


def _signal_frame(signal: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"score": signal.astype(float).replace([np.inf, -np.inf], np.nan).dropna()})


def _year_window(year: int) -> Tuple[str, str]:
    return f"{int(year)}-01-01", f"{int(year)}-12-31"


def _daily_rank_ic_mean(pred: pd.Series, label: pd.Series, start: str, end: str) -> Tuple[float, int]:
    daily = dd._daily_rank_ic_series(pred, label)
    sliced = daily.loc[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]
    clean = sliced.replace([np.inf, -np.inf], np.nan).dropna()
    return (float(clean.mean()) if len(clean) else float("nan"), int(len(clean)))


def _combined_signal(pred_map: Dict[str, pd.Series]) -> pd.Series:
    parts = [pred_map[k] for k in ("train", "valid") if k in pred_map and pred_map[k] is not None]
    if not parts:
        return pd.Series(dtype=float, name="score")
    return pd.concat(parts).sort_index()


def _train_and_predict(
    family: str,
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    label_meta: Dict[str, Any],
    alpha_grid: Sequence[float],
    weight_grid: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    valid_mask = _mask(dt_idx, "2023-01-01", GATE_END)
    target_modes = list(label_meta)
    train_mask_by_target = {
        target: dd._active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, int(label_meta[target].get("horizon_days", label_meta[target].get("max_horizon_days", 10))))
        for target in target_modes
    }

    if family == "drawdown":
        rows, preds = dd._make_predictions(
            dataset=dataset,
            train_mask_by_target=train_mask_by_target,
            valid_mask=valid_mask,
            test_mask=None,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            target_meta=label_meta,
        )
    elif family == "liquidity":
        rows, preds = liq._make_predictions(
            dataset=dataset,
            train_mask_by_target=train_mask_by_target,
            valid_mask=valid_mask,
            test_mask=None,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            target_meta=label_meta,
        )
    elif family == "market_state":
        rows, preds = ms._make_predictions(
            dataset=dataset,
            train_mask_by_target=train_mask_by_target,
            valid_mask=valid_mask,
            test_mask=None,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            weight_grid=weight_grid,
            target_meta=label_meta,
        )
    else:
        raise ValueError(f"unsupported family: {family}")

    for row in rows:
        row["family"] = family
    return rows, preds


def _build_family_labels(
    family: str,
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]:
    if family == "drawdown":
        labels, meta = dd._build_drawdown_conditional_labels(
            panel_raw,
            feature_index,
            horizons=_parse_int_csv(args.horizon_grid),
            lambdas=_parse_float_csv(args.lambda_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        return labels, meta, None
    if family == "liquidity":
        labels, meta = liq._build_liquidity_survival_labels(
            panel_raw,
            feature_index,
            horizons=_parse_int_csv(args.horizon_grid),
            liquidity_weights=_parse_float_csv(args.liquidity_weight_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        return labels, meta, None
    if family == "market_state":
        state_df, state_meta = ms._build_market_state_table(panel_raw, feature_index)
        labels, meta = ms._build_market_state_targets(
            panel_raw,
            feature_index,
            state_df=state_df,
            policies=_parse_csv(args.policy_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        for target in meta:
            meta[target]["market_state_meta"] = state_meta
        return labels, meta, state_df
    raise ValueError(f"unsupported family: {family}")


def _expected_report_rows(provider_uri: Path, year: int) -> int:
    return int(dd._count_calendar_rows(provider_uri, *_year_window(year)))


def _gate_reasons(
    years: Sequence[int],
    per_year: Dict[str, Dict[str, Any]],
    combined: Dict[str, Any],
    expected_rows: Dict[str, int],
) -> List[str]:
    reasons: List[str] = []
    rank_ics = [_safe_float(per_year[str(y)].get("rank_ic")) for y in years]
    if not all(np.isfinite(v) and v > 0.0 for v in rank_ics):
        bad = [str(y) for y, v in zip(years, rank_ics) if not (np.isfinite(v) and v > 0.0)]
        reasons.append(f"rank_ic_not_positive_years={','.join(bad)}")

    irs = [_safe_float(per_year[str(y)].get("portfolio_ir")) for y in years]
    ir_gt_one = sum(1 for v in irs if np.isfinite(v) and v > 1.0)
    if ir_gt_one < GATE_MIN_IR_POSITIVE_YEARS:
        reasons.append(f"portfolio_ir_gt_1_years={ir_gt_one}<3")
    if not all(np.isfinite(v) and v > GATE_MIN_YEAR_IR for v in irs):
        bad = [str(y) for y, v in zip(years, irs) if not (np.isfinite(v) and v > GATE_MIN_YEAR_IR)]
        reasons.append(f"worst_year_ir_not_positive_years={','.join(bad)}")

    combined_ir = _safe_float(combined.get("ir"))
    if not (np.isfinite(combined_ir) and combined_ir >= GATE_MIN_COMBINED_IR):
        reasons.append(f"combined_ir={combined_ir:.6g}<1.8" if np.isfinite(combined_ir) else "combined_ir_nonfinite")
    combined_turnover = _safe_float(combined.get("turnover"))
    if not (np.isfinite(combined_turnover) and combined_turnover <= GATE_MAX_COMBINED_TURNOVER):
        reasons.append(
            f"combined_turnover={combined_turnover:.6g}>0.16" if np.isfinite(combined_turnover) else "combined_turnover_nonfinite"
        )

    for y in years:
        key = str(y)
        row = per_year[key]
        expected = int(expected_rows[key])
        finite_rows = int(row.get("finite_rows", 0) or 0)
        nonfinite = int(row.get("nonfinite_rows", 0) or 0)
        row_count = int(row.get("row_count", 0) or 0)
        if row_count != expected or finite_rows != expected or nonfinite != 0:
            reasons.append(f"finite_rows_incomplete_{key}=rows:{row_count}/finite:{finite_rows}/expected:{expected}/nonfinite:{nonfinite}")
    return reasons


def _evaluate_candidate(
    candidate: Dict[str, Any],
    pred: pd.Series,
    label: pd.Series,
    years: Sequence[int],
    provider_uri: Path,
    port_cfg: Dict[str, Any],
    benchmark: str,
    args: argparse.Namespace,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_id = str(candidate["candidate_id"])
    signal_df = _signal_frame(pred)
    expected_rows = {str(y): _expected_report_rows(provider_uri, int(y)) for y in years}
    per_year: Dict[str, Dict[str, Any]] = {}
    flat_rows: List[Dict[str, Any]] = []

    for year in years:
        start, end = _year_window(year)
        rank_ic, rank_ic_days = _daily_rank_ic_mean(pred, label, start, end)
        metric, _report = dd._run_backtest_with_report(
            signal_df=signal_df,
            split_name=str(year),
            candidate_id=candidate_id,
            start_time=start,
            end_time=end,
            topk=int(args.topk),
            n_drop=int(args.n_drop),
            port_cfg_template=port_cfg,
            benchmark=benchmark,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        metric_d = dict(metric.__dict__)
        year_row = {
            "candidate_id": candidate_id,
            "family": candidate.get("family"),
            "year": str(year),
            "rank_ic": rank_ic,
            "rank_ic_days": rank_ic_days,
            "portfolio_ir": metric_d["ir"],
            "portfolio_annret": metric_d["annret"],
            "portfolio_mdd": metric_d["max_drawdown"],
            "portfolio_turnover": metric_d["turnover"],
            "row_count": metric_d["row_count"],
            "finite_rows": metric_d["finite_rows"],
            "nonfinite_rows": metric_d["nonfinite_rows"],
            "expected_rows": expected_rows[str(year)],
            "backtest_error": metric_d["error"],
        }
        per_year[str(year)] = year_row
        flat_rows.append(year_row)

    combined_metric, _combined_report = dd._run_backtest_with_report(
        signal_df=signal_df,
        split_name=f"{years[0]}_{years[-1]}",
        candidate_id=candidate_id,
        start_time=f"{int(years[0])}-01-01",
        end_time=f"{int(years[-1])}-12-31",
        topk=int(args.topk),
        n_drop=int(args.n_drop),
        port_cfg_template=port_cfg,
        benchmark=benchmark,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        exchange_cache=exchange_cache,
    )
    combined = {
        "annret": combined_metric.annret,
        "ir": combined_metric.ir,
        "max_drawdown": combined_metric.max_drawdown,
        "turnover": combined_metric.turnover,
        "row_count": combined_metric.row_count,
        "finite_rows": combined_metric.finite_rows,
        "nonfinite_rows": combined_metric.nonfinite_rows,
        "error": combined_metric.error,
    }
    reasons = _gate_reasons(years, per_year, combined, expected_rows)

    out = dict(candidate)
    out.update(
        {
            "per_year": per_year,
            "combined_2020_2023": combined,
            "combined_ir": combined["ir"],
            "combined_annret": combined["annret"],
            "combined_mdd": combined["max_drawdown"],
            "combined_turnover": combined["turnover"],
            "gate_pass": not reasons,
            "fail_reasons": reasons,
        }
    )
    return out, flat_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic pre-2024 multi-year label gate; never loads/evaluates 2024+ data.")
    p.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument("--benchmark", default="SH000300")
    p.add_argument("--years", default="2020,2021,2022,2023")
    p.add_argument("--families", default="drawdown,liquidity,market_state")
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument("--output-prefix", default="pre2024_multiyear_label_gate")
    p.add_argument("--workflow-config", default=DEFAULT_WORKFLOW_CONFIG)
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--horizon-grid", default="5,10,20")
    p.add_argument("--lambda-grid", default="0.25,0.5,1.0")
    p.add_argument("--liquidity-weight-grid", default="0.25,0.5,1.0")
    p.add_argument(
        "--policy-grid",
        default="bull20_bear5_vol_def,bull10_bear_def_vol5,bull20_bear_dd10_vol_dd10,bull10_bear_dd10_vol_def",
    )
    p.add_argument("--weight-grid", default="flat,defensive,trend")
    p.add_argument("--alpha-grid", default="10")
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--n-drop", type=int, default=3)
    p.add_argument("--min-names-per-day", type=int, default=40)
    return p


def _markdown_summary(summary: Dict[str, Any]) -> str:
    rows = summary.get("top_candidates", [])
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        f"- status: {summary.get('status')}",
        f"- verdict: {summary.get('verdict')}",
        f"- candidates_evaluated: {summary.get('candidate_count')}",
        f"- gate_pass_count: {summary.get('gate_pass_count')}",
        f"- data_end: {summary.get('leakage_guardrails', {}).get('load_end')}",
        "",
        "## Top candidates",
    ]
    if not rows:
        lines.append("- none")
    for row in rows:
        reasons = "; ".join(row.get("fail_reasons") or ["PASS"])
        lines.append(
            f"- {row.get('candidate_id')} | family={row.get('family')} | combined_ir={_safe_float(row.get('combined_ir')):.4f} | "
            f"turnover={_safe_float(row.get('combined_turnover')):.4f} | gate_pass={row.get('gate_pass')} | reasons={reasons}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
    years = _parse_int_csv(args.years)
    families = _parse_csv(args.families)

    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "experiment_name": EXPERIMENT_NAME,
        "status": "started",
        "verdict": "NO_GO",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
        "config": {
            "provider_uri": str(provider_uri),
            "market": str(args.market),
            "benchmark": str(args.benchmark),
            "years": years,
            "families": families,
            "max_candidates_per_family": int(args.max_candidates),
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "topk": int(args.topk),
            "n_drop": int(args.n_drop),
        },
        "gate_thresholds": {
            "rank_ic_each_year": "> 0",
            "portfolio_ir_gt_1_years": f">= {GATE_MIN_IR_POSITIVE_YEARS}/{len(years)}",
            "worst_year_portfolio_ir": "> 0",
            "combined_portfolio_ir": f">= {GATE_MIN_COMBINED_IR}",
            "combined_turnover": f"<= {GATE_MAX_COMBINED_TURNOVER}",
            "finite_rows": "each report row finite and complete; nonfinite=0",
        },
        "leakage_guardrails": {
            "raw_lookback_start": RAW_START,
            "train_window": [TRAIN_START, TRAIN_END],
            "selection_window": [f"{min(years)}-01-01", GATE_END],
            "load_end": GATE_END,
            "loads_or_evaluates_2024_2026": False,
            "guard": "script rejects years after 2023 and always builds the raw panel with end_date=2023-12-31",
        },
    }

    candidate_pool_rows: List[Dict[str, Any]] = []
    candidate_rows_out: List[Dict[str, Any]] = []
    year_rows_out: List[Dict[str, Any]] = []
    evaluated: List[Dict[str, Any]] = []

    try:
        if not years or max(years) > 2023:
            raise ValueError(f"years must be <= 2023; got {years}")
        if min(years) < 2020:
            raise ValueError(f"years must start no earlier than 2020 for this gate; got {years}")
        unsupported = sorted(set(families) - {"drawdown", "liquidity", "market_state"})
        if unsupported:
            raise ValueError(f"unsupported families: {unsupported}")

        dd.quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = dd.base._load_config(dd._resolve_workflow_config(str(args.workflow_config)))
        port_cfg = dd.base._extract_port_config(wf_cfg)
        benchmark = str(args.benchmark or wf_cfg.get("benchmark", "SH000300"))

        panel_raw, coverage_df = dd.base._build_panel(provider_uri, str(args.market), RAW_START, GATE_END, BASE_FIELDS)
        feature_df, all_feature_cols = dd.base._build_features_and_targets(panel_raw)
        feature_cols = dd._select_feature_cols(all_feature_cols)
        day_counts = feature_df.groupby(level=0)[feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index

        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
        alpha_grid = _parse_float_csv(args.alpha_grid)
        weight_grid = _parse_csv(args.weight_grid)

        for family in families:
            labels_df, label_meta, state_df = _build_family_labels(family, panel_raw, feature_df.index, args)
            parts = [feature_df[feature_cols], labels_df]
            if state_df is not None:
                parts.append(state_df)
            dataset = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
            dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()
            rows, preds = _train_and_predict(
                family=family,
                dataset=dataset,
                feature_cols=feature_cols,
                label_meta=label_meta,
                alpha_grid=alpha_grid,
                weight_grid=weight_grid,
            )
            rows_sorted = sorted(rows, key=_candidate_sort_key, reverse=True)
            selected_rows = rows_sorted[: max(0, int(args.max_candidates))]
            candidate_pool_rows.extend(rows_sorted)

            for row in selected_rows:
                cid = str(row["candidate_id"])
                target = str(row.get("target_mode") or row.get("label_name"))
                if cid not in preds:
                    row_eval = dict(row)
                    row_eval.update({"gate_pass": False, "fail_reasons": ["prediction_missing_or_family_training_veto"]})
                    evaluated.append(row_eval)
                    continue
                result, year_rows = _evaluate_candidate(
                    candidate=row,
                    pred=_combined_signal(preds[cid]),
                    label=dataset[target],
                    years=years,
                    provider_uri=provider_uri,
                    port_cfg=port_cfg,
                    benchmark=benchmark,
                    args=args,
                    exchange_cache=exchange_cache,
                )
                evaluated.append(result)
                year_rows_out.extend(year_rows)

        evaluated_sorted = sorted(evaluated, key=lambda r: (_safe_float(r.get("combined_ir")), _safe_float(r.get("combined_annret"))), reverse=True)
        gate_pass = [r for r in evaluated_sorted if bool(r.get("gate_pass"))]
        for row in evaluated_sorted:
            flat = {k: v for k, v in row.items() if k not in {"per_year", "combined_2020_2023"}}
            flat["fail_reasons"] = ";".join(row.get("fail_reasons") or [])
            candidate_rows_out.append(flat)

        summary.update(
            {
                "status": "completed",
                "verdict": "PASS_CANDIDATES_REVIEW_ONLY" if gate_pass else "NO_GO",
                "candidate_count": int(len(evaluated_sorted)),
                "candidate_pool_count": int(len(candidate_pool_rows)),
                "gate_pass_count": int(len(gate_pass)),
                "coverage_instruments": int(len(coverage_df)),
                "feature_count": int(len(feature_cols)),
                "elapsed_sec": float(time.perf_counter() - t0),
                "top_candidates": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "family": row.get("family"),
                        "combined_ir": row.get("combined_ir"),
                        "combined_annret": row.get("combined_annret"),
                        "combined_mdd": row.get("combined_mdd"),
                        "combined_turnover": row.get("combined_turnover"),
                        "gate_pass": row.get("gate_pass"),
                        "fail_reasons": row.get("fail_reasons"),
                    }
                    for row in evaluated_sorted[:10]
                ],
                "gate_pass_candidates": [row.get("candidate_id") for row in gate_pass],
            }
        )

    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "verdict": "NO_GO",
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": float(time.perf_counter() - t0),
            }
        )

    _write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(_markdown_summary(summary), encoding="utf-8")
    _write_csv(paths["candidates_csv"], candidate_rows_out)
    _write_csv(paths["year_metrics_csv"], year_rows_out)

    print(json.dumps(_json_sanitize(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

