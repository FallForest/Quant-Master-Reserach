#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

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


EXPERIMENT_NAME = "pre-2024 fixed execution rule stability audit"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
GATE_END = "2023-12-31"
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")
DEFAULT_WORKFLOW_CONFIG = "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"

GATE_MIN_COMBINED_IR = 1.8
GATE_MIN_YEAR_IR = 0.0
GATE_MIN_IR_GT_ONE_YEARS = 3
GATE_MIN_COMBINED_MDD = -0.12
GATE_MAX_COMBINED_TURNOVER = 0.16


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


class _StreamingCsvWriter:
    def __init__(self, path: Path, fieldnames: Optional[Sequence[str]] = None):
        self.path = path
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames else None
        self.row_count = 0

    def write(self, row: Dict[str, Any]) -> None:
        sanitized = _json_sanitize(row)
        if self._writer is None:
            if self._fieldnames is None:
                self._fieldnames = list(sanitized.keys())
            self._file = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        assert self._writer is not None
        assert self._fieldnames is not None
        self._writer.writerow({k: sanitized.get(k) for k in self._fieldnames})
        self.row_count += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._writer is None:
            self.path.write_text("", encoding="utf-8")


def _write_stream_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    writer = _StreamingCsvWriter(path)
    try:
        for row in rows:
            writer.write(row)
    finally:
        writer.close()


def _row_for_fields(row: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field) for field in fields}


def _add_top_rule(top_rules: List[Dict[str, Any]], row: Dict[str, Any], limit: int = 10) -> None:
    top_rules.append(
        {
            "family": row.get("family"),
            "candidate_id": row.get("candidate_id"),
            "topk": row.get("topk"),
            "n_drop": row.get("n_drop"),
            "combined_ir": row.get("combined_ir"),
            "combined_annret": row.get("combined_annret"),
            "combined_mdd": row.get("combined_mdd"),
            "combined_turnover": row.get("combined_turnover"),
            "year_ir_gt_1_count": row.get("year_ir_gt_1_count"),
            "gate_pass": row.get("gate_pass"),
            "fail_reasons": row.get("fail_reasons"),
        }
    )
    top_rules.sort(key=lambda r: (_safe_float(r.get("combined_ir")), _safe_float(r.get("combined_annret"))), reverse=True)
    del top_rules[limit:]


def _limit_rule_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {k: v for k, v in row.items() if k not in {"per_year", "combined_2020_2023"}}
    flat["fail_reasons"] = ";".join(row.get("fail_reasons") or [])
    return flat


def _limit_gate_pass_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "topk": row.get("topk"),
        "n_drop": row.get("n_drop"),
        "combined_ir": row.get("combined_ir"),
        "combined_annret": row.get("combined_annret"),
        "combined_mdd": row.get("combined_mdd"),
        "combined_turnover": row.get("combined_turnover"),
    }


CANDIDATE_POOL_FIELDS = [
    "family",
    "candidate_id",
    "model_family",
    "target_mode",
    "state_policy",
    "horizon_days",
    "lambda_penalty",
    "liquidity_penalty_weight",
    "alpha",
    "bull_weight",
    "bear_weight",
    "volatile_weight",
    "feature_count",
    "train_sample_count",
    "fit_sec",
    "train_rank_ic",
    "train_rank_ic_ir",
    "valid_rank_ic",
    "valid_rank_ic_ir",
    "candidate_selection_basis",
]

RULE_FIELDS = CANDIDATE_POOL_FIELDS + [
    "topk",
    "n_drop",
    "combined_annret",
    "combined_ir",
    "combined_mdd",
    "combined_turnover",
    "combined_row_count",
    "combined_finite_rows",
    "combined_nonfinite_rows",
    "combined_error",
    "year_ir_gt_1_count",
    "gate_pass",
    "fail_reasons",
]

YEAR_METRIC_FIELDS = [
    "family",
    "candidate_id",
    "topk",
    "n_drop",
    "year",
    "ir",
    "annret",
    "max_drawdown",
    "turnover",
    "row_count",
    "finite_rows",
    "nonfinite_rows",
    "expected_rows",
    "elapsed_sec",
    "error",
]


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in _parse_csv(text)]


def _parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in _parse_csv(text)]


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _year_window(year: int) -> Tuple[str, str]:
    return f"{int(year)}-01-01", f"{int(year)}-12-31"


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "rules_csv": THIS_DIR / f"{output_prefix}_rules_{stamp}.csv",
        "year_metrics_csv": THIS_DIR / f"{output_prefix}_year_metrics_{stamp}.csv",
        "candidate_pool_csv": THIS_DIR / f"{output_prefix}_candidate_pool_{stamp}.csv",
    }


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    train_ic_ir = _safe_float(row.get("train_rank_ic_ir"))
    train_ic = _safe_float(row.get("train_rank_ic"))
    valid_ic_ir = _safe_float(row.get("valid_rank_ic_ir"))
    valid_ic = _safe_float(row.get("valid_rank_ic"))
    return (
        train_ic_ir if np.isfinite(train_ic_ir) else -1e9,
        train_ic if np.isfinite(train_ic) else -1e9,
        valid_ic_ir if np.isfinite(valid_ic_ir) else -1e9,
        valid_ic if np.isfinite(valid_ic) else -1e9,
    )


def _expected_report_rows(provider_uri: Path, year: int) -> int:
    return int(dd._count_calendar_rows(provider_uri, *_year_window(year)))


def _signal_frame(signal: pd.Series) -> pd.DataFrame:
    score = signal.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return pd.DataFrame({"score": score}).sort_index()


def _combined_signal(pred_map: Dict[str, pd.Series]) -> pd.Series:
    parts = [pred_map[k] for k in ("train", "valid") if k in pred_map and pred_map[k] is not None]
    if not parts:
        return pd.Series(dtype=float, name="score")
    return pd.concat(parts).sort_index()


def _build_family_labels(
    family: str,
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]:
    if family == "drawdown":
        labels, meta = dd._build_drawdown_conditional_labels(
            panel_raw=panel_raw,
            feature_index=feature_index,
            horizons=_parse_int_csv(args.horizon_grid),
            lambdas=_parse_float_csv(args.lambda_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        return labels, meta, None
    if family == "liquidity":
        labels, meta = liq._build_liquidity_survival_labels(
            panel_raw=panel_raw,
            feature_index=feature_index,
            horizons=_parse_int_csv(args.horizon_grid),
            liquidity_weights=_parse_float_csv(args.liquidity_weight_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        return labels, meta, None
    if family == "market_state":
        state_df, state_meta = ms._build_market_state_table(panel_raw, feature_index)
        labels, meta = ms._build_market_state_targets(
            panel_raw=panel_raw,
            feature_index=feature_index,
            state_df=state_df,
            policies=_parse_csv(args.policy_grid),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        for target in meta:
            meta[target]["market_state_meta"] = state_meta
        return labels, meta, state_df
    raise ValueError(f"unsupported family: {family}")


def _train_and_predict(
    family: str,
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    label_meta: Dict[str, Any],
    alpha_grid: Sequence[float],
    weight_grid: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    valid_mask = _mask(dt_idx, VALID_START, GATE_END)
    target_modes = list(label_meta)
    train_mask_by_target = {
        target: dd._active_sample_mask(
            dataset.index,
            TRAIN_START,
            TRAIN_END,
            int(label_meta[target].get("horizon_days", label_meta[target].get("max_horizon_days", 10))),
        )
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
        row["candidate_selection_basis"] = "train_2020_2022_rank_ic_ir_then_rank_ic"
    return rows, preds


def _gate_reasons(
    years: Sequence[int],
    per_year: Dict[str, Dict[str, Any]],
    combined: Dict[str, Any],
    expected_rows: Dict[str, int],
) -> List[str]:
    reasons: List[str] = []
    irs = [_safe_float(per_year[str(y)].get("ir")) for y in years]
    if not all(np.isfinite(v) and v > GATE_MIN_YEAR_IR for v in irs):
        bad = [str(y) for y, v in zip(years, irs) if not (np.isfinite(v) and v > GATE_MIN_YEAR_IR)]
        reasons.append(f"year_ir_not_positive={','.join(bad)}")

    ir_gt_one = sum(1 for v in irs if np.isfinite(v) and v > 1.0)
    if ir_gt_one < GATE_MIN_IR_GT_ONE_YEARS:
        reasons.append(f"year_ir_gt_1_count={ir_gt_one}<3")

    combined_ir = _safe_float(combined.get("ir"))
    if not (np.isfinite(combined_ir) and combined_ir >= GATE_MIN_COMBINED_IR):
        reasons.append(f"combined_ir={combined_ir:.6g}<1.8" if np.isfinite(combined_ir) else "combined_ir_nonfinite")

    combined_mdd = _safe_float(combined.get("max_drawdown"))
    if not (np.isfinite(combined_mdd) and combined_mdd >= GATE_MIN_COMBINED_MDD):
        reasons.append(f"combined_mdd={combined_mdd:.6g}<-0.12" if np.isfinite(combined_mdd) else "combined_mdd_nonfinite")

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
        if row.get("error"):
            reasons.append(f"backtest_error_{key}={row.get('error')}")

    combined_expected = sum(int(expected_rows[str(y)]) for y in years)
    if (
        int(combined.get("row_count", 0) or 0) != combined_expected
        or int(combined.get("finite_rows", 0) or 0) != combined_expected
        or int(combined.get("nonfinite_rows", 0) or 0) != 0
    ):
        reasons.append(
            "finite_rows_incomplete_combined="
            f"rows:{int(combined.get('row_count', 0) or 0)}/"
            f"finite:{int(combined.get('finite_rows', 0) or 0)}/"
            f"expected:{combined_expected}/"
            f"nonfinite:{int(combined.get('nonfinite_rows', 0) or 0)}"
        )
    if combined.get("error"):
        reasons.append(f"backtest_error_combined={combined.get('error')}")
    return reasons


def _evaluate_rule(
    candidate: Dict[str, Any],
    pred: pd.Series,
    years: Sequence[int],
    provider_uri: Path,
    port_cfg: Dict[str, Any],
    benchmark: str,
    topk: int,
    n_drop: int,
    args: argparse.Namespace,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_id = str(candidate["candidate_id"])
    family = str(candidate.get("family"))
    signal_df = _signal_frame(pred)
    expected_rows = {str(y): _expected_report_rows(provider_uri, int(y)) for y in years}
    per_year: Dict[str, Dict[str, Any]] = {}
    flat_year_rows: List[Dict[str, Any]] = []

    for year in years:
        start, end = _year_window(year)
        metric, report = dd._run_backtest_with_report(
            signal_df=signal_df,
            split_name=str(year),
            candidate_id=candidate_id,
            start_time=start,
            end_time=end,
            topk=int(topk),
            n_drop=int(n_drop),
            port_cfg_template=port_cfg,
            benchmark=benchmark,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        del report
        row = asdict(metric)
        year_row = {
            "family": family,
            "candidate_id": candidate_id,
            "topk": int(topk),
            "n_drop": int(n_drop),
            "year": str(year),
            "ir": row["ir"],
            "annret": row["annret"],
            "max_drawdown": row["max_drawdown"],
            "turnover": row["turnover"],
            "row_count": row["row_count"],
            "finite_rows": row["finite_rows"],
            "nonfinite_rows": row["nonfinite_rows"],
            "expected_rows": expected_rows[str(year)],
            "elapsed_sec": row["elapsed_sec"],
            "error": row["error"],
        }
        per_year[str(year)] = year_row
        flat_year_rows.append(year_row)

    combined_metric, combined_report = dd._run_backtest_with_report(
        signal_df=signal_df,
        split_name=f"{int(years[0])}_{int(years[-1])}",
        candidate_id=candidate_id,
        start_time=f"{int(years[0])}-01-01",
        end_time=f"{int(years[-1])}-12-31",
        topk=int(topk),
        n_drop=int(n_drop),
        port_cfg_template=port_cfg,
        benchmark=benchmark,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        exchange_cache=exchange_cache,
    )
    del combined_report
    combined = {
        "annret": combined_metric.annret,
        "ir": combined_metric.ir,
        "max_drawdown": combined_metric.max_drawdown,
        "turnover": combined_metric.turnover,
        "row_count": combined_metric.row_count,
        "finite_rows": combined_metric.finite_rows,
        "nonfinite_rows": combined_metric.nonfinite_rows,
        "elapsed_sec": combined_metric.elapsed_sec,
        "error": combined_metric.error,
    }
    reasons = _gate_reasons(years, per_year, combined, expected_rows)

    out = {k: v for k, v in candidate.items() if k not in {"per_year", "combined_2020_2023"}}
    out.update(
        {
            "topk": int(topk),
            "n_drop": int(n_drop),
            "combined_annret": combined["annret"],
            "combined_ir": combined["ir"],
            "combined_mdd": combined["max_drawdown"],
            "combined_turnover": combined["turnover"],
            "combined_row_count": combined["row_count"],
            "combined_finite_rows": combined["finite_rows"],
            "combined_nonfinite_rows": combined["nonfinite_rows"],
            "combined_error": combined["error"],
            "per_year": per_year,
            "combined_2020_2023": combined,
            "year_ir_gt_1_count": int(sum(1 for y in years if _safe_float(per_year[str(y)].get("ir")) > 1.0)),
            "gate_pass": not reasons,
            "fail_reasons": reasons,
        }
    )
    return out, flat_year_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-2024 fixed topk/n_drop rule stability audit; never loads or evaluates 2024+ data.")
    p.add_argument("--provider-uri", default=".qmData/cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument("--benchmark", default="SH000300")
    p.add_argument("--families", default="drawdown,liquidity,market_state")
    p.add_argument("--years", default="2020,2021,2022,2023")
    p.add_argument("--topk-grid", default="30,35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--output-prefix", default="pre2024_fixed_rule_stability_eval")
    p.add_argument("--workflow-config", default=DEFAULT_WORKFLOW_CONFIG)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--horizon-grid", default="5,10,20")
    p.add_argument("--lambda-grid", default="0.25,0.5,1.0")
    p.add_argument("--liquidity-weight-grid", default="0.25,0.5,1.0")
    p.add_argument(
        "--policy-grid",
        default="bull20_bear5_vol_def,bull10_bear_def_vol5,bull20_bear_dd10_vol_dd10,bull10_bear_dd10_vol_def",
    )
    p.add_argument("--weight-grid", default="flat,defensive,trend")
    p.add_argument("--alpha-grid", default="10")
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--workers", type=int, default=1, help="Cap QuantMaster/joblib workers used by local data loaders.")
    p.add_argument("--low-memory", action="store_true", help="Stream artifacts and aggressively release reports/signals after each rule.")
    return p


def _markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        f"- status: {summary.get('status')}",
        f"- verdict: {summary.get('verdict')}",
        f"- rules_evaluated: {summary.get('rule_count')}",
        f"- gate_pass_count: {summary.get('gate_pass_count')}",
        f"- data_end_loaded: {summary.get('leakage_guardrails', {}).get('load_end')}",
        f"- summary_json: {summary.get('artifacts', {}).get('summary_json')}",
        "",
        "## Top rules",
    ]
    top_rules = summary.get("top_rules", [])
    if not top_rules:
        lines.append("- none")
    for row in top_rules:
        reasons = "; ".join(row.get("fail_reasons") or ["PASS"])
        lines.append(
            f"- {row.get('family')} | {row.get('candidate_id')} | topk={row.get('topk')} n_drop={row.get('n_drop')} | "
            f"IR={_safe_float(row.get('combined_ir')):.4f} AnnRet={_safe_float(row.get('combined_annret')):.4f} "
            f"MDD={_safe_float(row.get('combined_mdd')):.4f} TO={_safe_float(row.get('combined_turnover')):.4f} | "
            f"gate_pass={row.get('gate_pass')} | reasons={reasons}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    years = _parse_int_csv(args.years)
    families = _parse_csv(args.families)
    topk_grid = _parse_int_csv(args.topk_grid)
    ndrop_grid = _parse_int_csv(args.ndrop_grid)
    combos = [(topk, ndrop) for topk in topk_grid for ndrop in ndrop_grid if int(ndrop) < int(topk)]
    workers = max(1, int(args.workers))

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
            "families": families,
            "years": years,
            "topk_grid": topk_grid,
            "ndrop_grid": ndrop_grid,
            "max_candidates_per_family": int(args.max_candidates),
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "workers": int(workers),
            "low_memory": bool(args.low_memory),
        },
        "gate_thresholds": {
            "combined_2020_2023_ir": f">= {GATE_MIN_COMBINED_IR}",
            "each_year_ir": f"> {GATE_MIN_YEAR_IR}",
            "year_ir_gt_1_count": f">= {GATE_MIN_IR_GT_ONE_YEARS}/{len(years)}",
            "combined_mdd": f">= {GATE_MIN_COMBINED_MDD}",
            "combined_turnover": f"<= {GATE_MAX_COMBINED_TURNOVER}",
            "finite_rows": "each annual and combined report row finite and complete; nonfinite=0",
        },
        "leakage_guardrails": {
            "raw_lookback_start": RAW_START,
            "train_window": [TRAIN_START, TRAIN_END],
            "audit_window": [f"{min(years) if years else 2020}-01-01", GATE_END],
            "load_end": GATE_END,
            "loads_or_evaluates_2024_2026": False,
            "guard": "years > 2023 are rejected and raw panel end_date is hard-coded to 2023-12-31",
        },
        "low_memory_controls": {
            "thread_env_defaults": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "quant_master_kernels": int(workers),
            "joblib_backend": "threading",
            "artifact_mode": "streaming_csv_rows; summary keeps top/pass rows only",
        },
    }

    candidate_pool_count = 0
    selected_candidate_count = 0
    rule_count = 0
    gate_pass_count = 0
    top_rules: List[Dict[str, Any]] = []
    gate_pass_rules: List[Dict[str, Any]] = []
    family_candidate_counts: Dict[str, int] = {}
    family_selected_counts: Dict[str, int] = {}

    candidate_pool_writer = _StreamingCsvWriter(paths["candidate_pool_csv"], CANDIDATE_POOL_FIELDS)
    rules_writer = _StreamingCsvWriter(paths["rules_csv"], RULE_FIELDS)
    year_writer = _StreamingCsvWriter(paths["year_metrics_csv"], YEAR_METRIC_FIELDS)

    try:
        if not years:
            raise ValueError("years cannot be empty")
        if min(years) < 2020 or max(years) > 2023:
            raise ValueError(f"years must be within 2020..2023; got {years}")
        if sorted(years) != list(years):
            raise ValueError(f"years must be sorted ascending; got {years}")
        unsupported = sorted(set(families) - {"drawdown", "liquidity", "market_state"})
        if unsupported:
            raise ValueError(f"unsupported families: {unsupported}")
        if not combos:
            raise ValueError("no valid topk/n_drop combinations")

        dd.quant_master.init(provider_uri=str(provider_uri), region="cn", kernels=workers, joblib_backend="threading")
        wf_cfg = dd.base._load_config(dd._resolve_workflow_config(str(args.workflow_config)))
        port_cfg = dd.base._extract_port_config(wf_cfg)
        benchmark = str(args.benchmark or wf_cfg.get("benchmark", "SH000300"))

        panel_raw, coverage_df = dd.base._build_panel(provider_uri, str(args.market), RAW_START, GATE_END, BASE_FIELDS)
        feature_df, all_feature_cols = dd.base._build_features_and_targets(panel_raw)
        feature_cols = dd._select_feature_cols(all_feature_cols)
        day_counts = feature_df.groupby(level=0)[feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        alpha_grid = _parse_float_csv(args.alpha_grid)
        weight_grid = _parse_csv(args.weight_grid)
        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

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
            family_candidate_counts[family] = int(len(rows_sorted))
            candidate_pool_count += int(len(rows_sorted))
            for candidate_row in rows_sorted:
                candidate_pool_writer.write(_row_for_fields(candidate_row, CANDIDATE_POOL_FIELDS))
            selected_rows = rows_sorted[: max(0, int(args.max_candidates))]
            family_selected_counts[family] = int(len(selected_rows))
            selected_candidate_count += int(len(selected_rows))

            for selected in selected_rows:
                cid = str(selected["candidate_id"])
                if cid not in preds:
                    missing_row = dict(selected)
                    missing_row.update({"gate_pass": False, "fail_reasons": ["prediction_missing_or_family_training_veto"]})
                    rule_count += 1
                    _add_top_rule(top_rules, missing_row)
                    rules_writer.write(_row_for_fields(_limit_rule_row(missing_row), RULE_FIELDS))
                    continue
                pred = _combined_signal(preds[cid])
                for topk, ndrop in combos:
                    rule, yrows = _evaluate_rule(
                        candidate=selected,
                        pred=pred,
                        years=years,
                        provider_uri=provider_uri,
                        port_cfg=port_cfg,
                        benchmark=benchmark,
                        topk=int(topk),
                        n_drop=int(ndrop),
                        args=args,
                        exchange_cache=exchange_cache,
                    )
                    rule_count += 1
                    if bool(rule.get("gate_pass")):
                        gate_pass_count += 1
                        gate_pass_rules.append(_limit_gate_pass_row(rule))
                    _add_top_rule(top_rules, rule)
                    rules_writer.write(_row_for_fields(_limit_rule_row(rule), RULE_FIELDS))
                    for yrow in yrows:
                        year_writer.write(_row_for_fields(yrow, YEAR_METRIC_FIELDS))
                    if bool(args.low_memory):
                        exchange_cache.clear()
                        gc.collect()
                del pred
            del rows, rows_sorted, selected_rows, preds, dataset, parts, labels_df, label_meta, state_df
            if bool(args.low_memory):
                gc.collect()

        summary.update(
            {
                "status": "completed",
                "verdict": "PASS_CANDIDATES_REVIEW_ONLY" if gate_pass_count else "NO_GO",
                "rule_count": int(rule_count),
                "candidate_pool_count": int(candidate_pool_count),
                "selected_candidate_count": int(selected_candidate_count),
                "gate_pass_count": int(gate_pass_count),
                "family_candidate_counts": family_candidate_counts,
                "family_selected_counts": family_selected_counts,
                "coverage_instruments": int(len(coverage_df)),
                "feature_count": int(len(feature_cols)),
                "elapsed_sec": float(time.perf_counter() - t0),
                "top_rules": top_rules,
                "gate_pass_rules": gate_pass_rules,
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

    finally:
        candidate_pool_writer.close()
        rules_writer.close()
        year_writer.close()

    _write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(_markdown_summary(summary), encoding="utf-8")

    print(json.dumps(_json_sanitize(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
