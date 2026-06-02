#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import factor_augmented_meta_ensemble as fame
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"
DEFAULT_META_PRED_GLOB = "factor_augmented_meta_candidate_pred_*.pkl"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    residual_beta: float = 0.0
    exposure_mode: str = "full"
    exposure_scale: float = 1.0


class ExposureBufferedTopkStrategy(fame.BufferedTopkWeightStrategy):
    def __init__(self, *, exposure: Optional[pd.Series] = None, **kwargs):
        self.exposure = None
        if exposure is not None:
            ex = exposure.copy()
            ex.index = pd.to_datetime(ex.index)
            self.exposure = ex.sort_index().astype(float)
        super().__init__(**kwargs)

    def _exposure_at(self, trade_start_time: pd.Timestamp) -> float:
        if self.exposure is None or self.exposure.empty:
            return 1.0
        ts = pd.Timestamp(trade_start_time).normalize()
        if ts in self.exposure.index:
            return float(self.exposure.loc[ts])
        prior = self.exposure.loc[self.exposure.index <= ts]
        if prior.empty:
            return 1.0
        return float(prior.iloc[-1])

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        target = super().generate_target_weight_position(score, current, trade_start_time, trade_end_time)
        if target is None:
            return None
        scale = max(0.0, min(1.0, self._exposure_at(pd.Timestamp(trade_start_time))))
        return {code: float(weight) * scale for code, weight in target.items()}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        writer.writerows(rows)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _find_latest(path: Path, pattern: str) -> Path:
    hits = sorted(path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no files match {pattern} under {path}")
    return hits[-1]


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, float]:
    if report.empty:
        return {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    excess = report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].astype(float).mean()),
    }


def _daily_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")


def _slice_report(report: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].copy()


def _rank_centered(series: pd.Series) -> pd.Series:
    return 2.0 * series.groupby(level=0).rank(method="average", pct=True) - 1.0


def _candidate_signal(meta: pd.Series, anchor: pd.Series, spec: CandidateSpec) -> pd.DataFrame:
    meta_rank = _rank_centered(meta)
    if spec.residual_beta == 0.0:
        out = meta_rank
    else:
        anchor_rank = _rank_centered(anchor).reindex(meta_rank.index)
        resid_rank = _rank_centered((meta_rank - anchor_rank).dropna()).reindex(meta_rank.index)
        out = meta_rank + float(spec.residual_beta) * resid_rank.fillna(0.0)
    return out.rename("score").to_frame("score").dropna()


def _risk_exposures(identity_report: pd.DataFrame, start: str, end: str, scale: float, mode: str) -> pd.Series:
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    exposure = pd.Series(1.0, index=idx, dtype=float)
    excess = _daily_excess(identity_report).sort_index()
    if mode == "full":
        return exposure
    if mode == "vol75":
        trailing_vol = excess.rolling(20, min_periods=10).std().shift(1)
        vol_cut = trailing_vol.expanding(min_periods=60).quantile(0.75).shift(1)
        risk_off = (trailing_vol > vol_cut).reindex(exposure.index).fillna(False)
        exposure.loc[risk_off] = float(scale)
        return exposure
    if mode == "dd2":
        equity = (1.0 + excess.fillna(0.0)).cumprod()
        drawdown = (equity / equity.cummax() - 1.0).shift(1)
        risk_off = (drawdown < -0.02).reindex(exposure.index).fillna(False)
        exposure.loc[risk_off] = float(scale)
        return exposure
    raise ValueError(f"unsupported exposure mode: {mode}")


def _get_day_report(pm: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in pm:
        return pm["1day"][0].copy()
    if "day" in pm:
        return pm["day"][0].copy()
    return pm[next(iter(pm.keys()))][0].copy()


def _run_backtest_report(
    *,
    signal_df: pd.DataFrame,
    exposure: Optional[pd.Series],
    base_port_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    open_cost: float,
    close_cost: float,
    topk: int,
    hold_topk: int,
) -> Tuple[pd.DataFrame, float]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = start_date
    backtest_cfg["end_time"] = end_date
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exch = dict(backtest_cfg.get("exchange_kwargs", {}))
    exch["open_cost"] = float(open_cost)
    exch["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exch["exchange"] = get_exchange(
        freq=freq,
        start_time=start_date,
        end_time=end_date,
        deal_price=str(exch.get("deal_price", "close")),
        limit_threshold=float(exch.get("limit_threshold", 0.095)),
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=float(exch.get("min_cost", 5)),
    )

    strategy = ExposureBufferedTopkStrategy(
        signal=signal_df,
        exposure=exposure,
        topk=int(topk),
        hold_topk=int(hold_topk),
        weight_mode="equal",
        rebalance_mode="weekly",
        rebalance_interval=1,
    )
    t0 = time.perf_counter()
    pm, _ = run_backtest(
        start_time=start_date,
        end_time=end_date,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exch,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    return _get_day_report(pm), float(time.perf_counter() - t0)


def _candidate_specs() -> List[CandidateSpec]:
    return [
        CandidateSpec("identity"),
        CandidateSpec("resid_plus_005", residual_beta=0.05),
        CandidateSpec("resid_plus_010", residual_beta=0.10),
        CandidateSpec("resid_minus_005", residual_beta=-0.05),
        CandidateSpec("vol75_scale85", exposure_mode="vol75", exposure_scale=0.85),
        CandidateSpec("dd2_scale85", exposure_mode="dd2", exposure_scale=0.85),
    ]


def _quarter_windows(start: str, end: str) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    periods = pd.period_range(st, ed, freq="Q")
    out: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for p in periods:
        qs = max(st, p.start_time.normalize())
        qe = min(ed, p.end_time.normalize())
        out.append((str(p), qs, qe))
    return out


def _year_slices(start: str, end: str) -> List[Tuple[str, str, str]]:
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    out = []
    for y in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{y}-01-01"), st)
        ye = min(pd.Timestamp(f"{y}-12-31"), ed)
        if ys <= ye:
            tag = f"{y}_ytd" if ye < pd.Timestamp(f"{y}-12-31") else str(y)
            out.append((tag, str(ys.date()), str(ye.date())))
    return out


def _report_coverage(report: pd.DataFrame) -> Dict[str, Any]:
    if report.empty:
        return {"rows": 0, "valid_excess_rows": 0, "start": "", "end": ""}
    excess = _daily_excess(report)
    valid = excess.dropna()
    return {
        "rows": int(report.shape[0]),
        "valid_excess_rows": int(valid.shape[0]),
        "start": str(pd.Timestamp(report.index.min()).date()),
        "end": str(pd.Timestamp(report.index.max()).date()),
    }


def _coverage_is_admissible(report: pd.DataFrame) -> bool:
    cov = _report_coverage(report)
    rows = int(cov["rows"])
    valid = int(cov["valid_excess_rows"])
    return rows > 0 and valid >= int(np.floor(0.95 * rows))


def _run_backtest_report_by_year(
    *,
    signal_df: pd.DataFrame,
    exposure: Optional[pd.Series],
    base_port_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    open_cost: float,
    close_cost: float,
    topk: int,
    hold_topk: int,
) -> Tuple[pd.DataFrame, float]:
    parts: List[pd.DataFrame] = []
    elapsed_total = 0.0
    for split, st, ed in _year_slices(start_date, end_date):
        report, elapsed = _run_backtest_report(
            signal_df=signal_df,
            exposure=exposure,
            base_port_cfg=base_port_cfg,
            start_date=st,
            end_date=ed,
            open_cost=open_cost,
            close_cost=close_cost,
            topk=topk,
            hold_topk=hold_topk,
        )
        coverage = _report_coverage(report)
        if coverage["valid_excess_rows"] <= 0:
            raise RuntimeError(f"{split} produced no valid excess rows")
        parts.append(report)
        elapsed_total += float(elapsed)
    stitched = pd.concat(parts, axis=0).sort_index()
    stitched = stitched.loc[~stitched.index.duplicated(keep="first")]
    return stitched, elapsed_total


def _selection_score(metrics: Dict[str, float]) -> float:
    ir = float(metrics.get("ir", float("nan")))
    ann = float(metrics.get("annret", float("nan")))
    turnover = float(metrics.get("turnover", float("nan")))
    max_dd = float(metrics.get("max_drawdown", float("nan")))
    if not np.isfinite(ir):
        return -1e9
    ann_term = ann if np.isfinite(ann) else 0.0
    turnover_term = turnover if np.isfinite(turnover) else 0.0
    dd_pen = max(0.0, abs(max_dd) - 0.05) if np.isfinite(max_dd) else 0.0
    return float(ir + 0.35 * ann_term - 0.40 * turnover_term - 2.0 * dd_pen)


def _select_and_stitch(
    reports: Dict[str, pd.DataFrame],
    start: str,
    end: str,
    min_train_days: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    selections: List[Dict[str, Any]] = []
    stitched_parts: List[pd.DataFrame] = []
    identity = "identity"
    for seq, (period, apply_start, apply_end) in enumerate(_quarter_windows(start, end), start=1):
        train_end = apply_start - pd.Timedelta(days=1)
        train_rows: List[Dict[str, Any]] = []
        best_name = identity
        best_score = -1e9
        train_days = 0
        if train_end >= pd.Timestamp(start):
            for name, report in reports.items():
                train_report = _slice_report(report, start, train_end)
                train_days = max(train_days, int(train_report.shape[0]))
                if train_report.shape[0] < int(min_train_days):
                    continue
                metrics = _metrics_from_report(train_report)
                score = _selection_score(metrics)
                train_rows.append({"candidate": name, "score": score, **metrics})
                if score > best_score:
                    best_name = name
                    best_score = score
        reason = "prior_objective" if train_rows else "warmup_identity"
        chosen_report = _slice_report(reports[best_name], apply_start, apply_end)
        stitched_parts.append(chosen_report.assign(selected_candidate=best_name, lockstep_period=period))
        chosen_train = next((r for r in train_rows if r["candidate"] == best_name), {})
        selections.append(
            {
                "period_seq": seq,
                "period": period,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "train_start": start if train_rows else "",
                "train_end": str(train_end.date()) if train_rows else "",
                "train_days": int(train_days),
                "selected_candidate": best_name,
                "selection_reason": reason,
                "selection_score": float(best_score) if train_rows else float("nan"),
                "train_ir": chosen_train.get("ir"),
                "train_annret": chosen_train.get("annret"),
                "train_max_drawdown": chosen_train.get("max_drawdown"),
                "train_turnover": chosen_train.get("turnover"),
                "candidate_count_scored": len(train_rows),
            }
        )
    stitched = pd.concat(stitched_parts, axis=0).sort_index()
    return stitched, selections


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict prior-only residual/risk overlay lockstep on factor meta prediction.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--topk", type=int, default=55)
    p.add_argument("--hold-topk", type=int, default=85)
    p.add_argument("--min-train-days", type=int, default=126)
    p.add_argument("--meta-pred-path", default="")
    p.add_argument("--output-prefix", default="residual_overlay_lockstep")
    return p


def main() -> int:
    args = build_parser().parse_args()
    trans_dir = Path(__file__).resolve().parent
    tracking_dir = fame._parse_tracking_dir(args.tracking_uri)
    stamp = _stamp()

    base_run_dir = fame._find_run_dir(tracking_dir, args.base_run_id)
    cfg = fame._load_config(base_run_dir / "artifacts" / "config")
    fame._init_quant_master(cfg)
    base_port_cfg = fame._extract_port_config(cfg)

    meta_pred_path = Path(args.meta_pred_path).expanduser().resolve() if args.meta_pred_path else _find_latest(trans_dir, DEFAULT_META_PRED_GLOB)
    meta = fame._slice_period(fame._as_score_series(_load_pickle(meta_pred_path)), args.start_date, args.end_date)
    anchor = fame._slice_period(fame._as_score_series(_load_pickle(base_run_dir / "artifacts" / "pred.pkl")), args.start_date, args.end_date)

    specs = _candidate_specs()
    reports: Dict[str, pd.DataFrame] = {}
    candidate_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    identity_signal = _candidate_signal(meta, anchor, specs[0])
    identity_report, identity_elapsed = _run_backtest_report_by_year(
        signal_df=identity_signal,
        exposure=None,
        base_port_cfg=base_port_cfg,
        start_date=args.start_date,
        end_date=args.end_date,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.topk),
        hold_topk=int(args.hold_topk),
    )
    reports["identity"] = identity_report
    identity_metrics = _metrics_from_report(identity_report)
    candidate_rows.append(
        {
            "candidate": "identity",
            "residual_beta": 0.0,
            "exposure_mode": "full",
            "exposure_scale": 1.0,
            "elapsed_sec": identity_elapsed,
            **{f"coverage_{k}": v for k, v in _report_coverage(identity_report).items()},
            **identity_metrics,
        }
    )

    for spec in specs[1:]:
        signal = _candidate_signal(meta, anchor, spec)
        exposure = None
        if spec.exposure_mode != "full":
            exposure = _risk_exposures(identity_report, args.start_date, args.end_date, spec.exposure_scale, spec.exposure_mode)
        try:
            report, elapsed = _run_backtest_report_by_year(
                signal_df=signal,
                exposure=exposure,
                base_port_cfg=base_port_cfg,
                start_date=args.start_date,
                end_date=args.end_date,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                topk=int(args.topk),
                hold_topk=int(args.hold_topk),
            )
        except Exception as exc:  # noqa: BLE001
            error_rows.append(
                {
                    "candidate": spec.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
                }
            )
            continue
        reports[spec.name] = report
        candidate_rows.append(
            {
                "candidate": spec.name,
                "residual_beta": spec.residual_beta,
                "exposure_mode": spec.exposure_mode,
                "exposure_scale": spec.exposure_scale,
                "elapsed_sec": elapsed,
                **{f"coverage_{k}": v for k, v in _report_coverage(report).items()},
                **_metrics_from_report(report),
            }
        )

    stitched, selection_rows = _select_and_stitch(reports, args.start_date, args.end_date, int(args.min_train_days))
    stitched_metrics = _metrics_from_report(stitched)
    split_rows: List[Dict[str, Any]] = []
    for split, st, ed in _year_slices(args.start_date, args.end_date):
        split_rows.append({"split": split, "start": st, "end": ed, **_metrics_from_report(_slice_report(stitched, st, ed))})

    stitched_coverage = _report_coverage(stitched)
    admissible = _coverage_is_admissible(stitched)
    hard_gate_pass = bool(
        admissible and stitched_metrics["ir"] > HARD_GATE_IR and stitched_metrics["annret"] > HARD_GATE_ANNRET
    )
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    candidates_csv = trans_dir / f"{args.output_prefix}_candidates_{stamp}.csv"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    daily_csv = trans_dir / f"{args.output_prefix}_stitched_daily_{stamp}.csv"
    errors_csv = trans_dir / f"{args.output_prefix}_errors_{stamp}.csv"

    _write_csv(candidates_csv, candidate_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(splits_csv, split_rows)
    _write_csv(errors_csv, error_rows)
    daily_out = stitched.copy()
    daily_out["excess"] = _daily_excess(stitched)
    daily_out.reset_index().to_csv(daily_csv, index=False)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "WorkerE residual_overlay_lockstep",
        "test_period": {"start": args.start_date, "end": args.end_date},
        "costs": {"open": float(args.open_cost), "close": float(args.close_cost)},
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "hard_gate_pass": hard_gate_pass,
        "admissible": admissible,
        "admissibility": {
            "daily_report_coverage": stitched_coverage,
            "required_valid_excess_fraction": 0.95,
            "not_admissible_reason": "" if admissible else "incomplete reconstructed daily excess coverage",
        },
        "protocol": {
            "candidate_set": "predeclared bounded residual/risk overlays",
            "selection": "quarterly lockstep; score candidates only on prior daily backtest reports; apply selected candidate to next quarter",
            "no_test_window_parameter_selection": True,
            "min_train_days": int(args.min_train_days),
            "selection_objective": "IR + 0.35*AnnRet - 0.40*Turnover - 2.0*max(0, abs(MaxDD)-0.05)",
            "base_execution": {
                "strategy": "BufferedTopkWeightStrategy with optional exposure scale",
                "rebalance_mode": "weekly",
                "topk": int(args.topk),
                "hold_topk": int(args.hold_topk),
                "weight_mode": "equal",
            },
        },
        "source_artifacts": {
            "meta_pred_path": str(meta_pred_path),
            "anchor_pred_path": str(base_run_dir / "artifacts" / "pred.pkl"),
            "base_config_path": str(base_run_dir / "artifacts" / "config"),
        },
        "metrics": {
            "stitched_full": stitched_metrics,
            "identity_full": identity_metrics,
            "delta_vs_identity": {
                "ir": float(stitched_metrics["ir"] - identity_metrics["ir"]),
                "annret": float(stitched_metrics["annret"] - identity_metrics["annret"]),
                "max_drawdown": float(stitched_metrics["max_drawdown"] - identity_metrics["max_drawdown"]),
                "turnover": float(stitched_metrics["turnover"] - identity_metrics["turnover"]),
            },
            "splits": split_rows,
            "candidates": candidate_rows,
            "candidate_errors": error_rows,
        },
        "risk_notes": [
            "Daily reports are reconstructed by real QuantMaster backtests from saved prediction artifacts, not proxy metrics.",
            "Risk exposure schedules use only prior identity daily excess via shifted rolling/expanding statistics.",
            "All overlay candidates are fixed in code before evaluation; quarterly selection never scores the apply quarter.",
        ],
        "artifacts": {
            "summary_json": str(summary_json),
            "candidates_csv": str(candidates_csv),
            "selections_csv": str(selections_csv),
            "splits_csv": str(splits_csv),
            "stitched_daily_csv": str(daily_csv),
            "errors_csv": str(errors_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"summary_json": str(summary_json), "hard_gate_pass": hard_gate_pass, "metrics": stitched_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

