#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import drawdown_conditional_label_pre2024 as dd


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "execution survival liquidity label"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")
PROMOTE_IR = 2.30
PROMOTE_ANNRET = 0.16
PROMOTE_TURNOVER = 0.16
FULL_HARD_GATE_IR = 2.90
FULL_HARD_GATE_ANNRET = 0.27


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    model_family: str
    target_mode: str
    horizon_days: int
    liquidity_penalty_weight: float
    alpha: float
    feature_count: int
    fit_sec: float
    train_rank_ic: float
    train_rank_ic_ir: float
    valid_rank_ic: float
    valid_rank_ic_ir: float


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _robust_scale(s: pd.Series) -> Tuple[float, float]:
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    q25, q60, q75 = vals.quantile([0.25, 0.60, 0.75]).astype(float).tolist()
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(vals.std(ddof=1))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return float(q60), scale


def _train_q75_positive(s: pd.Series, train_mask: pd.Series) -> float:
    vals = s.loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return 1.0
    q75 = float(vals.quantile(0.75))
    return q75 if np.isfinite(q75) and q75 > 1e-8 else 1.0


def _future_mean_by_instrument(s: pd.Series, horizon: int) -> pd.Series:
    steps = [s.groupby(level=1, sort=False).shift(-step) for step in range(1, int(horizon) + 1)]
    return pd.concat(steps, axis=1).mean(axis=1, skipna=False)


def _rolling_median_by_instrument(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level=1, sort=False).transform(lambda x: x.rolling(window, min_periods=max(3, window // 4)).median())


def _rolling_mean_by_instrument(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level=1, sort=False).transform(lambda x: x.rolling(window, min_periods=max(3, window // 4)).mean())


def _build_liquidity_survival_labels(
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
    horizons: Sequence[int],
    liquidity_weights: Sequence[float],
    open_cost: float,
    close_cost: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = panel_raw.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)

    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = (p["close"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    high = (p["high"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    low = (p["low"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    vwap = (p["vwap"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    amount = p["amount"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0).sort_index()

    labels = pd.DataFrame(index=feature_index)
    meta: Dict[str, Any] = {}
    round_trip_cost = float(open_cost) + float(close_cost)
    entry = close.groupby(level=1, sort=False).shift(-1)
    amount_base = _rolling_median_by_instrument(amount, 20).replace(0.0, np.nan)
    vwap_close_gap = ((vwap - close).abs() / (close.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
    intraday_range = ((high - low).abs() / (close.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
    range_base = _rolling_mean_by_instrument(intraday_range, 20).replace(0.0, np.nan)

    for horizon in horizons:
        h = int(horizon)
        exit_ = close.groupby(level=1, sort=False).shift(-h)
        raw_ret = exit_ / (entry + 1e-12) - 1.0
        mkt_ret = raw_ret.groupby(level=0, sort=False).transform("mean")
        excess = (raw_ret - mkt_ret).reindex(feature_index)
        train_mask = dd._active_sample_mask(feature_index, TRAIN_START, TRAIN_END, h)

        future_amount = _future_mean_by_instrument(amount, h)
        amount_contraction = np.log((amount_base + 1.0) / (future_amount + 1.0)).clip(lower=0.0).reindex(feature_index)
        future_vwap_gap = _future_mean_by_instrument(vwap_close_gap, h).clip(lower=0.0).reindex(feature_index)
        future_range = _future_mean_by_instrument(intraday_range, h)
        range_worsening = ((future_range / (range_base + 1e-12)) - 1.0).clip(lower=0.0).reindex(feature_index)

        amount_unit = _train_q75_positive(amount_contraction, train_mask)
        vwap_unit = _train_q75_positive(future_vwap_gap, train_mask)
        range_unit = _train_q75_positive(range_worsening, train_mask)
        liquidity_badness = (
            0.45 * (amount_contraction / amount_unit).clip(0.0, 3.0)
            + 0.30 * (future_vwap_gap / vwap_unit).clip(0.0, 3.0)
            + 0.25 * (range_worsening / range_unit).clip(0.0, 3.0)
        )

        excess_train = excess.loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
        return_scale = float((excess_train.quantile(0.75) - excess_train.quantile(0.25)) / 1.349) if not excess_train.empty else 1.0
        if not np.isfinite(return_scale) or return_scale < 1e-6:
            return_scale = float(excess_train.std(ddof=1)) if len(excess_train) > 1 else 1.0
        if not np.isfinite(return_scale) or return_scale < 1e-6:
            return_scale = 1.0

        for weight in liquidity_weights:
            w = float(weight)
            liquidity_penalty = w * return_scale * liquidity_badness
            net_quality = (excess - round_trip_cost - liquidity_penalty).replace([np.inf, -np.inf], np.nan)
            train_vals = net_quality.loc[train_mask].dropna()
            if train_vals.empty:
                raise RuntimeError(f"no train values for liquidity horizon {h} weight {w:g}")
            q60, scale = _robust_scale(train_vals)

            w_tag = f"{w:g}".replace(".", "p")
            normalized = np.tanh((net_quality - q60) / scale)
            raw_col = f"liqsurv_{h}d_w{w_tag}_net_quality"
            bad_col = f"liqsurv_{h}d_w{w_tag}_badness"
            target_col = f"liqsurv_{h}d_w{w_tag}_trainfit_rank"
            labels[raw_col] = net_quality
            labels[bad_col] = liquidity_badness
            labels[f"liqsurv_{h}d_amount_contraction"] = amount_contraction
            labels[f"liqsurv_{h}d_vwap_close_gap"] = future_vwap_gap
            labels[f"liqsurv_{h}d_range_worsening"] = range_worsening
            labels[target_col] = (dd._cs_rank_pct(pd.Series(normalized, index=feature_index)) - 0.5) * 2.0
            labels.loc[~dd._active_sample_mask(feature_index, TRAIN_START, VALID_END, h), target_col] = np.nan
            meta[target_col] = {
                "horizon_days": h,
                "liquidity_penalty_weight": w,
                "round_trip_cost": round_trip_cost,
                "return_scale": float(return_scale),
                "train_q60_threshold": float(q60),
                "train_iqr_scale": float(scale),
                "train_sample_count": int(len(train_vals)),
                "amount_contraction_q75_unit": float(amount_unit),
                "vwap_close_gap_q75_unit": float(vwap_unit),
                "range_worsening_q75_unit": float(range_unit),
                "label_formula": "future_excess_return - round_trip_cost - weight * train_return_scale * liquidity_badness; tanh train-normalized and cross-section ranked",
                "liquidity_badness": "0.45*future amount contraction + 0.30*future vwap-close gap + 0.25*future range/liquidity worsening, each train-q75 normalized and clipped",
                "train_exit_guard": f"training samples require horizon exit <= {TRAIN_END}",
                "smoke_future_guard": f"smoke panel ends at {VALID_END}; late-2023 labels without in-window exits are NaN",
            }
    return labels.replace([np.inf, -np.inf], np.nan), meta


def _make_predictions(
    dataset: pd.DataFrame,
    train_mask_by_target: Dict[str, pd.Series],
    valid_mask: np.ndarray,
    test_mask: Optional[np.ndarray],
    feature_cols: Sequence[str],
    target_modes: Sequence[str],
    alpha_grid: Sequence[float],
    target_meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    valid_df = dataset.loc[valid_mask]
    x_valid = valid_df[list(feature_cols)].astype(np.float64).values
    test_df = dataset.loc[test_mask] if test_mask is not None else None
    x_test = test_df[list(feature_cols)].astype(np.float64).values if test_df is not None else None
    candidate_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}

    for target in target_modes:
        train_mask = train_mask_by_target[target]
        train_df = dataset.loc[train_mask].dropna(subset=list(feature_cols) + [target])
        if train_df.empty:
            raise RuntimeError(f"empty train split for target={target}")
        x_train = train_df[list(feature_cols)].astype(np.float64).values
        y_train = train_df[target].astype(np.float64).values
        for alpha in alpha_grid:
            candidate_id = f"ridge_liqsurv_{target}_a{float(alpha):g}"
            coef, mu, sd, fit_sec = dd._fit_ridge(x_train, y_train, float(alpha))
            pred_train = dd._cs_z(pd.Series(dd._predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
            pred_valid = dd._cs_z(pd.Series(dd._predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
            pred_test = (
                dd._cs_z(pd.Series(dd._predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"))
                if x_test is not None and test_df is not None
                else None
            )
            train_ic, train_ic_ir = dd._mean_and_ir(dd._daily_rank_ic_series(pred_train, train_df[target]))
            valid_ic, valid_ic_ir = dd._mean_and_ir(dd._daily_rank_ic_series(pred_valid, valid_df[target]))
            candidate_rows.append(
                asdict(
                    CandidateMetric(
                        candidate_id=candidate_id,
                        model_family="closed_form_ridge",
                        target_mode=target,
                        horizon_days=int(target_meta[target]["horizon_days"]),
                        liquidity_penalty_weight=float(target_meta[target]["liquidity_penalty_weight"]),
                        alpha=float(alpha),
                        feature_count=int(len(feature_cols)),
                        fit_sec=fit_sec,
                        train_rank_ic=train_ic,
                        train_rank_ic_ir=train_ic_ir,
                        valid_rank_ic=valid_ic,
                        valid_rank_ic_ir=valid_ic_ir,
                    )
                )
            )
            pred_map: Dict[str, pd.Series] = {"train": pred_train, "valid": pred_valid}
            if pred_test is not None:
                pred_map["test"] = pred_test
            predictions[candidate_id] = pred_map
    return candidate_rows, predictions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Liquidity survival label 2023-smoke / locked full eval.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--provider-uri", default=".qmData/cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default="workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml",
    )
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--horizon-grid", default="5,10,20")
    p.add_argument("--liquidity-weight-grid", default="0.25,0.5,1.0")
    p.add_argument("--alpha-grid", default="10")
    p.add_argument("--topk-grid", default="35,40")
    p.add_argument("--ndrop-grid", default="2,3")
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="liquidity_survival_label_pre2024")
    return p


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "candidates_csv": THIS_DIR / f"{output_prefix}_candidates_{stamp}.csv",
        "validation_backtests_csv": THIS_DIR / f"{output_prefix}_validation_backtests_{stamp}.csv",
        "selected_valid_report_csv": THIS_DIR / f"{output_prefix}_selected_valid_report_{stamp}.csv",
        "selected_valid_signal_csv": THIS_DIR / f"{output_prefix}_selected_valid_signal_{stamp}.csv",
        "selected_test_report_csv": THIS_DIR / f"{output_prefix}_selected_test_report_{stamp}.csv",
        "selected_test_signal_csv": THIS_DIR / f"{output_prefix}_selected_test_signal_{stamp}.csv",
        "test_slices_csv": THIS_DIR / f"{output_prefix}_test_slices_{stamp}.csv",
    }


def _selection_status(ir: float, annret: float, turnover: float, finite_gate_pass: bool) -> Tuple[str, str]:
    if not finite_gate_pass:
        return "failed_finite_gate", "NO_GO"
    if np.isfinite(ir) and np.isfinite(annret) and np.isfinite(turnover) and ir >= PROMOTE_IR and annret >= PROMOTE_ANNRET and turnover <= PROMOTE_TURNOVER:
        return "promotion_passed", "PROMOTE"
    return "gate_failed", "NO_GO"


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    mode = str(args.mode)
    data_end = TEST_END if mode == "full" else VALID_END
    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "experiment_name": EXPERIMENT_NAME,
        "mode": mode,
        "status": "started",
        "verdict": "NO_GO",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
        "leakage_guardrails": {
            "raw_lookback_start": RAW_START,
            "train_window": [TRAIN_START, TRAIN_END],
            "selection_window": [VALID_START, VALID_END],
            "smoke_load_end": VALID_END,
            "smoke_evaluates_2024_2026": False,
            "full_test_window": [TEST_START, TEST_END],
            "full_selection_uses_test_metrics": False,
            "full_mode_guard": "candidate and portfolio selection are locked from 2023 validation only; 2024-2026 is evaluated once after selection",
        },
    }

    try:
        dd.quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = dd.base._load_config(dd._resolve_workflow_config(str(args.workflow_config)))
        port_cfg = dd.base._extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))

        panel_raw, _coverage_df = dd.base._build_panel(provider_uri, str(args.market), RAW_START, data_end, BASE_FIELDS)
        feature_df, all_feature_cols = dd.base._build_features_and_targets(panel_raw)
        feature_cols = dd._select_feature_cols(all_feature_cols)
        horizon_grid = [int(x) for x in str(args.horizon_grid).split(",") if x.strip()]
        liquidity_weight_grid = [float(x) for x in str(args.liquidity_weight_grid).split(",") if x.strip()]
        labels_df, label_meta = _build_liquidity_survival_labels(
            panel_raw,
            feature_df.index,
            horizons=horizon_grid,
            liquidity_weights=liquidity_weight_grid,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        dataset = pd.concat([feature_df[feature_cols], labels_df], axis=1).replace([np.inf, -np.inf], np.nan)
        day_counts = dataset.groupby(level=0)[feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

        dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
        valid_mask = dd._mask(dt_idx, VALID_START, VALID_END)
        test_mask = dd._mask(dt_idx, TEST_START, TEST_END) if mode == "full" else None
        if not valid_mask.any():
            raise RuntimeError("empty 2023 validation split")
        if mode == "full" and (test_mask is None or not test_mask.any()):
            raise RuntimeError("empty 2024-2026 test split")

        target_modes = [c for c in label_meta]
        train_mask_by_target = {
            target: dd._active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, int(label_meta[target]["horizon_days"]))
            for target in target_modes
        }
        alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
        topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
        ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
        combos = [(topk, ndrop) for topk in topk_grid for ndrop in ndrop_grid if ndrop < topk]
        if not combos:
            raise RuntimeError("no valid topk/n_drop combinations")

        candidate_rows, predictions = _make_predictions(
            dataset=dataset,
            train_mask_by_target=train_mask_by_target,
            valid_mask=valid_mask,
            test_mask=test_mask,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            target_meta=label_meta,
        )
        expected_valid_rows = dd._count_calendar_rows(provider_uri, VALID_START, VALID_END)
        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
        valid_bt_rows: List[Dict[str, Any]] = []
        valid_reports: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        finite_gate_errors: List[str] = []

        for cand in candidate_rows:
            cid = str(cand["candidate_id"])
            sig = predictions[cid]["valid"].rename("score").to_frame("score")
            for topk, ndrop in combos:
                metric, report = dd._run_backtest_with_report(
                    sig,
                    "valid_2023_selection_only",
                    cid,
                    VALID_START,
                    VALID_END,
                    topk,
                    ndrop,
                    port_cfg,
                    benchmark,
                    float(args.open_cost),
                    float(args.close_cost),
                    exchange_cache,
                )
                row = asdict(metric)
                valid_bt_rows.append(row)
                if report is not None and not report.empty:
                    valid_reports[(cid, int(topk), int(ndrop))] = report
                if metric.error or metric.row_count != expected_valid_rows or metric.finite_rows != expected_valid_rows:
                    finite_gate_errors.append(
                        f"{cid} topk={topk} n_drop={ndrop}: error={metric.error!r} rows={metric.row_count} finite={metric.finite_rows} expected={expected_valid_rows}"
                    )

        selectable = [
            r
            for r in valid_bt_rows
            if not r["error"]
            and int(r["row_count"]) == expected_valid_rows
            and int(r["finite_rows"]) == expected_valid_rows
            and np.isfinite(_safe_float(r["ir"]))
            and np.isfinite(_safe_float(r["annret"]))
        ]
        if not selectable:
            raise RuntimeError("no validation portfolio passed finite-row and finite-metric gates")
        selected = sorted(selectable, key=dd._sort_metric_key, reverse=True)[0]
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])
        selected_key = (str(selected["candidate_id"]), int(selected["topk"]), int(selected["n_drop"]))
        selected_report = valid_reports[selected_key]
        selected_signal = predictions[str(selected["candidate_id"])]["valid"].rename("score").to_frame("score").sort_index()

        _write_csv(paths["candidates_csv"], candidate_rows)
        _write_csv(paths["validation_backtests_csv"], valid_bt_rows)
        selected_report.to_csv(paths["selected_valid_report_csv"])
        selected_signal.reset_index().to_csv(paths["selected_valid_signal_csv"], index=False)

        expected_test_rows = 0
        test_metric: Optional[Dict[str, Any]] = None
        test_slice_rows: List[Dict[str, Any]] = []
        test_finite_gate_errors: List[str] = []
        if mode == "full":
            expected_test_rows = dd._count_calendar_rows(provider_uri, TEST_START, TEST_END)
            selected_test_signal = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
            test_bt_metric, test_report = dd._run_backtest_with_report(
                selected_test_signal,
                "test_2024_2026_one_shot",
                str(selected["candidate_id"]),
                TEST_START,
                TEST_END,
                int(selected["topk"]),
                int(selected["n_drop"]),
                port_cfg,
                benchmark,
                float(args.open_cost),
                float(args.close_cost),
                exchange_cache,
            )
            test_metric = asdict(test_bt_metric)
            if test_report.empty:
                test_finite_gate_errors.append(f"empty test report: {test_metric['error']}")
            else:
                test_report.to_csv(paths["selected_test_report_csv"])
                selected_test_signal.reset_index().to_csv(paths["selected_test_signal_csv"], index=False)
                test_slice_rows = dd._slice_report_metrics(test_report)
                _write_csv(paths["test_slices_csv"], test_slice_rows)
            if (
                test_bt_metric.error
                or test_bt_metric.row_count != expected_test_rows
                or test_bt_metric.finite_rows != expected_test_rows
                or test_bt_metric.nonfinite_rows != 0
            ):
                test_finite_gate_errors.append(
                    f"{selected['candidate_id']} test: error={test_bt_metric.error!r} rows={test_bt_metric.row_count} finite={test_bt_metric.finite_rows} expected={expected_test_rows}"
                )

        finite_gate_pass = bool(not finite_gate_errors)
        status, verdict = _selection_status(
            _safe_float(selected["ir"]),
            _safe_float(selected["annret"]),
            _safe_float(selected["turnover"]),
            finite_gate_pass,
        )
        full_hard_gate = None
        if mode == "full":
            if test_metric is None:
                raise RuntimeError("full mode did not produce test metrics")
            test_ir = _safe_float(test_metric["ir"])
            test_annret = _safe_float(test_metric["annret"])
            full_hard_gate_errors = list(test_finite_gate_errors)
            if not (np.isfinite(test_ir) and test_ir > FULL_HARD_GATE_IR):
                full_hard_gate_errors.append(f"test IR {test_ir:.6f} must be > {FULL_HARD_GATE_IR:.2f}")
            if not (np.isfinite(test_annret) and test_annret > FULL_HARD_GATE_ANNRET):
                full_hard_gate_errors.append(f"test AnnRet {test_annret:.6f} must be > {FULL_HARD_GATE_ANNRET:.2f}")
            full_hard_gate_pass = bool(
                not full_hard_gate_errors
                and int(test_metric["row_count"]) == int(expected_test_rows)
                and int(test_metric["finite_rows"]) == int(expected_test_rows)
                and int(test_metric["nonfinite_rows"]) == 0
                and np.isfinite(test_ir)
                and test_ir > FULL_HARD_GATE_IR
                and np.isfinite(test_annret)
                and test_annret > FULL_HARD_GATE_ANNRET
            )
            full_hard_gate = {
                "passed": full_hard_gate_pass,
                "rule": "no test finite errors; row_count == finite_rows == expected_test_rows; nonfinite_rows == 0; test IR > 2.90; test AnnRet > 0.27",
                "row_count_required": expected_test_rows,
                "row_count_actual": test_metric["row_count"],
                "finite_rows_required": expected_test_rows,
                "finite_rows_actual": test_metric["finite_rows"],
                "nonfinite_rows_required": 0,
                "nonfinite_rows_actual": test_metric["nonfinite_rows"],
                "test_ir_required_gt": FULL_HARD_GATE_IR,
                "test_ir_actual": test_metric["ir"],
                "test_annret_required_gt": FULL_HARD_GATE_ANNRET,
                "test_annret_actual": test_metric["annret"],
                "selection_locked_from_validation_only": True,
                "selection_candidate_id": str(selected["candidate_id"]),
                "selection_topk": int(selected["topk"]),
                "selection_n_drop": int(selected["n_drop"]),
                "test_finite_gate_errors": test_finite_gate_errors,
                "fail_closed_errors": full_hard_gate_errors,
            }
            status = "full_hard_gate_passed" if full_hard_gate_pass else "full_hard_gate_failed"
            verdict = "FULL_HARD_GATE_PASS" if full_hard_gate_pass else "NO_GO"

        summary.update(
            {
                "status": status,
                "verdict": verdict,
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "benchmark": benchmark,
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "protocol": {
                    "raw_start": RAW_START,
                    "data_end_loaded": data_end,
                    "train": [TRAIN_START, TRAIN_END],
                    "validation_selection_only": [VALID_START, VALID_END],
                    "smoke_no_2024_2026_load_or_eval": mode == "smoke" and data_end == VALID_END,
                    "full_mode": "locked one-shot final test only" if mode == "full" else "available via --mode full",
                    "test_window_if_full": [TEST_START, TEST_END],
                    "full_does_not_tune_using_2024_2026": True,
                    "model": "closed-form ridge only",
                    "candidate_labels": label_meta,
                    "feature_count": int(len(feature_cols)),
                    "features": list(feature_cols),
                    "horizon_grid": horizon_grid,
                    "liquidity_weight_grid": liquidity_weight_grid,
                    "alpha_grid": alpha_grid,
                    "portfolio_grid": [{"topk": topk, "n_drop": ndrop} for topk, ndrop in combos],
                    "selection_rule": "2023 net-cost information ratio, tie by annualized return",
                    "promotion_gate": {
                        "costed_ir_min": PROMOTE_IR,
                        "costed_annret_min": PROMOTE_ANNRET,
                        "turnover_max": PROMOTE_TURNOVER,
                    },
                },
                "selected_candidate": {**selected_candidate, "topk": int(selected["topk"]), "n_drop": int(selected["n_drop"])},
                "validation_metrics": {
                    "costed_ir": selected["ir"],
                    "costed_annret": selected["annret"],
                    "max_drawdown": selected["max_drawdown"],
                    "turnover": selected["turnover"],
                    "row_count": selected["row_count"],
                    "finite_rows": selected["finite_rows"],
                    "nonfinite_rows": selected["nonfinite_rows"],
                    "expected_validation_trading_rows": expected_valid_rows,
                },
                "test_metrics": None
                if mode != "full"
                else {
                    "costed_ir": test_metric["ir"] if test_metric else None,
                    "costed_annret": test_metric["annret"] if test_metric else None,
                    "max_drawdown": test_metric["max_drawdown"] if test_metric else None,
                    "turnover": test_metric["turnover"] if test_metric else None,
                    "row_count": test_metric["row_count"] if test_metric else None,
                    "finite_rows": test_metric["finite_rows"] if test_metric else None,
                    "nonfinite_rows": test_metric["nonfinite_rows"] if test_metric else None,
                    "expected_test_trading_rows": expected_test_rows,
                    "yearly_slices": test_slice_rows,
                },
                "yearly_slices": test_slice_rows if mode == "full" else [],
                "full_hard_gate": full_hard_gate,
                "finite_row_check": {
                    "passed": finite_gate_pass,
                    "expected_validation_trading_rows": expected_valid_rows,
                    "fail_closed_errors": finite_gate_errors,
                    "rule": "every validation report must have row_count == finite_rows == validation trading rows",
                },
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        _write_json(paths["summary_json"], summary)
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Liquidity Survival Label ({stamp})",
                    "",
                    f"- status: `{status}`",
                    f"- verdict: `{verdict}`",
                    f"- mode: `{mode}`",
                    f"- smoke_no_2024_2026_load_or_eval: `{mode == 'smoke' and data_end == VALID_END}`",
                    f"- selected_candidate: `{selected['candidate_id']}`",
                    f"- selected_rule: `topk={int(selected['topk'])}, n_drop={int(selected['n_drop'])}`",
                    f"- 2023 costed IR / AnnRet / turnover: `{_safe_float(selected['ir']):.6f}` / `{_safe_float(selected['annret']):.6f}` / `{_safe_float(selected['turnover']):.6f}`",
                    *(
                        [
                            f"- 2024-2026 costed IR / AnnRet / turnover: `{_safe_float(test_metric['ir']):.6f}` / `{_safe_float(test_metric['annret']):.6f}` / `{_safe_float(test_metric['turnover']):.6f}`",
                            f"- 2024-2026 max_drawdown: `{_safe_float(test_metric['max_drawdown']):.6f}`",
                            f"- 2024-2026 finite_rows: `{int(test_metric['finite_rows'])}` / `{expected_test_rows}`",
                            f"- 2024-2026 nonfinite_rows: `{int(test_metric['nonfinite_rows'])}` / `0`",
                            f"- full_hard_gate_rule: `row_count == finite_rows == {expected_test_rows}; nonfinite_rows == 0; IR > {FULL_HARD_GATE_IR:.2f}; AnnRet > {FULL_HARD_GATE_ANNRET:.2f}`",
                            f"- full_hard_gate_actual: `row_count={int(test_metric['row_count'])}, finite_rows={int(test_metric['finite_rows'])}, nonfinite_rows={int(test_metric['nonfinite_rows'])}, IR={_safe_float(test_metric['ir']):.6f}, AnnRet={_safe_float(test_metric['annret']):.6f}`",
                            f"- full_hard_gate_passed: `{bool(full_hard_gate and full_hard_gate['passed'])}`",
                        ]
                        if mode == "full" and test_metric is not None
                        else []
                    ),
                    f"- validation_finite_rows: `{int(selected['finite_rows'])}` / `{expected_valid_rows}`",
                    f"- validation_nonfinite_rows: `{int(selected['nonfinite_rows'])}`",
                    f"- fail_closed_finite_gate_passed: `{finite_gate_pass}`",
                    f"- promotion_gate: `IR >= {PROMOTE_IR}, AnnRet >= {PROMOTE_ANNRET}, turnover <= {PROMOTE_TURNOVER}`",
                    f"- runtime_sec: `{summary['runtime_sec_total']:.3f}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        if mode == "full":
            return 0 if verdict == "FULL_HARD_GATE_PASS" else 2
        return 0 if verdict == "PROMOTE" else 2
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "verdict": "NO_GO",
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        _write_json(paths["summary_json"], summary)
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Liquidity Survival Label ({stamp})",
                    "",
                    "- status: `failed`",
                    "- verdict: `NO_GO`",
                    f"- blocker: `{type(exc).__name__}: {exc}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
