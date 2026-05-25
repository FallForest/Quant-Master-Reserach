#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class OverlayCandidate:
    candidate_id: str
    vol_lookback: int = 0
    target_ann_vol: float = 0.0
    dd_lookback: int = 0
    dd_threshold: float = 0.0
    stress_scale: float = 1.0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_score_df(run_dir: Path) -> pd.DataFrame:
    return conv._as_score_df(conv._load_pickle(run_dir / "artifacts" / "pred.pkl")).sort_index()


def _date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    dates = pd.to_datetime(df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index)
    unique_dates = pd.Index(dates.normalize().unique()).sort_values()
    return pd.Timestamp(unique_dates.min()), pd.Timestamp(unique_dates.max()), int(len(unique_dates))


def _combo() -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": 40,
        "n_drop": 2,
        "hold_topk": 40,
        "weight_mode": "equal",
        "score_power": 1.0,
    }


def _reconstruct_base40_report(
    *,
    pred_df: pd.DataFrame,
    port_cfg: Dict[str, Any],
    strategy_kwargs: Dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    open_cost: float,
    close_cost: float,
) -> Tuple[pd.DataFrame, float]:
    cfg = copy.deepcopy(port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = str(start.date())
    backtest_cfg["end_time"] = str(end.date())
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange = conv.get_exchange(
        freq=freq,
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        deal_price=deal_price,
        limit_threshold=limit_threshold,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=min_cost,
    )
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange

    pred_slice = conv._slice_pred(pred_df, start, end)
    if pred_slice.empty:
        raise ValueError(f"empty base40 prediction slice in {start.date()} ~ {end.date()}")
    strategy = conv._build_strategy_object(combo=_combo(), pred_df=pred_slice, base_strategy_kwargs=strategy_kwargs)

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = conv.run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = conv._get_report_for_day_freq(portfolio_metric_dict).copy().sort_index()
    report.index = pd.to_datetime(report.index)
    return report, float(time.perf_counter() - t0)


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _base_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"] - report["bench"] - report["cost"]).astype(float).sort_index()


def _candidate_grid() -> List[OverlayCandidate]:
    return [
        OverlayCandidate("identity_scale_1"),
        OverlayCandidate("vol63_target08", vol_lookback=63, target_ann_vol=0.08),
        OverlayCandidate("vol63_target09", vol_lookback=63, target_ann_vol=0.09),
        OverlayCandidate("vol126_target09", vol_lookback=126, target_ann_vol=0.09),
        OverlayCandidate("vol126_target10", vol_lookback=126, target_ann_vol=0.10),
        OverlayCandidate("dd63_thr025_scale070", dd_lookback=63, dd_threshold=-0.025, stress_scale=0.70),
        OverlayCandidate("dd126_thr040_scale080", dd_lookback=126, dd_threshold=-0.040, stress_scale=0.80),
        OverlayCandidate(
            "vol63_target09_dd63_thr025_scale075",
            vol_lookback=63,
            target_ann_vol=0.09,
            dd_lookback=63,
            dd_threshold=-0.025,
            stress_scale=0.75,
        ),
        OverlayCandidate(
            "vol126_target10_dd126_thr040_scale080",
            vol_lookback=126,
            target_ann_vol=0.10,
            dd_lookback=126,
            dd_threshold=-0.040,
            stress_scale=0.80,
        ),
    ]


def _trailing_drawdown(past: pd.Series) -> float:
    if past.empty:
        return 0.0
    equity = (1.0 + past.astype(float)).cumprod()
    high = equity.cummax()
    dd = equity / high - 1.0
    return float(dd.iloc[-1])


def _scale_series(excess: pd.Series, candidate: OverlayCandidate) -> pd.Series:
    values = excess.astype(float).sort_index()
    scales: List[float] = []
    for i in range(len(values)):
        scale = 1.0
        if candidate.vol_lookback > 0 and i >= candidate.vol_lookback:
            trailing = values.iloc[i - candidate.vol_lookback : i]
            ann_vol = float(trailing.std(ddof=0) * np.sqrt(TRADING_DAYS))
            if ann_vol > 0.0:
                scale = min(scale, float(candidate.target_ann_vol) / ann_vol)
        if candidate.dd_lookback > 0 and i >= candidate.dd_lookback:
            trailing = values.iloc[i - candidate.dd_lookback : i]
            if _trailing_drawdown(trailing) <= float(candidate.dd_threshold):
                scale = min(scale, float(candidate.stress_scale))
        scales.append(float(min(1.0, max(0.0, scale))))
    return pd.Series(scales, index=values.index, name="scale")


def _apply_candidate(excess: pd.Series, candidate: OverlayCandidate) -> Tuple[pd.Series, pd.Series]:
    scale = _scale_series(excess, candidate)
    return excess.astype(float).mul(scale, fill_value=0.0).rename("scaled_excess"), scale


def _selection_score(metrics: Dict[str, float]) -> float:
    annret_penalty = max(0.0, HARD_GATE_ANNRET - float(metrics["annret"])) * 2.0
    return float(metrics["ir"]) + 0.35 * float(metrics["annret"]) - 0.25 * abs(float(metrics["max_drawdown"])) - annret_penalty


def _selection_windows() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2024H1", TEST_START, pd.Timestamp("2024-06-30"), "2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2024", TEST_START, pd.Timestamp("2024-12-31"), "2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2024_2025", TEST_START, pd.Timestamp("2025-12-31"), "2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]


def _apply_slices() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    return [
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_scale_1_no_selection"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]


def _slice_series(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.loc[(series.index >= start) & (series.index <= end)].copy()


def _split_rows(excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, start, end, selection_rule in _apply_slices():
        part = _slice_series(excess, start, end)
        if part.empty:
            continue
        rows.append(
            {
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "days": int(len(part)),
                "selection_rule": selection_rule,
                **_metrics_from_excess(part),
            }
        )
    full = _slice_series(excess, TEST_START, TEST_END)
    if not full.empty:
        rows.insert(
            0,
            {
                "split": "test_full",
                "start": str(TEST_START.date()),
                "end": str(TEST_END.date()),
                "days": int(len(full)),
                "selection_rule": "stitched_past_only_overlay",
                **_metrics_from_excess(full),
            },
        )
    return rows


def _daily_rows(base_report: pd.DataFrame, base_excess: pd.Series, scaled_excess: pd.Series, scale: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dt in base_excess.index:
        rows.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "base_return": float(base_report.loc[dt, "return"]),
                "base_bench": float(base_report.loc[dt, "bench"]),
                "base_cost": float(base_report.loc[dt, "cost"]),
                "base_turnover": float(base_report.loc[dt, "turnover"]) if "turnover" in base_report else float("nan"),
                "base_excess": float(base_excess.loc[dt]),
                "selected_scale": float(scale.loc[dt]),
                "report_level_scaled_excess": float(scaled_excess.loc[dt]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report-level volatility/drawdown exposure audit for base40.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--output-prefix", default="volatility_scaled_base40_audit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent
    train_csv = out_dir / f"{args.output_prefix}_candidate_train_metrics_{stamp}.csv"
    selections_csv = out_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    daily_csv = out_dir / f"{args.output_prefix}_daily_report_overlay_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    wf_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(wf_cfg)
    port_cfg = conv._extract_port_config(wf_cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)
    base_pred = _load_score_df(base_dir)
    coverage_start, coverage_end, coverage_days = _date_range(base_pred)

    base_report, reconstruct_sec = _reconstruct_base40_report(
        pred_df=base_pred,
        port_cfg=port_cfg,
        strategy_kwargs=strategy_kwargs,
        start=TEST_START,
        end=TEST_END,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
    )
    base_excess = _base_excess(base_report)
    base_metrics = _metrics_from_excess(base_excess)

    candidates = _candidate_grid()
    by_id = {c.candidate_id: c for c in candidates}
    selected_by_apply: Dict[str, OverlayCandidate] = {"2024H1_default": by_id["identity_scale_1"]}
    train_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = [
        {
            "apply_tag": "2024H1_default",
            "apply_start": str(TEST_START.date()),
            "apply_end": str(pd.Timestamp("2024-06-30").date()),
            "selected": True,
            "selection_train_tag": "predeclared_default_no_2024H1_selection",
            "candidate_id": "identity_scale_1",
            "selection_score": "",
            "train_ir": "",
            "train_annret": "",
            "train_mdd": "",
            "strict_non_test_selected": True,
            "candidate_family_predeclared": True,
        }
    ]

    for train_tag, train_start, train_end, apply_tag, apply_start, apply_end in _selection_windows():
        train_excess = _slice_series(base_excess, train_start, train_end)
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            scaled, scale = _apply_candidate(train_excess, candidate)
            metrics = _metrics_from_excess(scaled)
            row = {
                "train_tag": train_tag,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "apply_tag": apply_tag,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "candidate_id": candidate.candidate_id,
                **asdict(candidate),
                "scale_min": float(scale.min()),
                "scale_mean": float(scale.mean()),
                "scale_max": float(scale.max()),
                "scaled_days": int((scale < 0.999999).sum()),
                "annret": float(metrics["annret"]),
                "ir": float(metrics["ir"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "selection_score": float(_selection_score(metrics)),
                "strict_non_test_selected": True,
                "candidate_family_predeclared": True,
            }
            scored.append(row)
            train_rows.append(row.copy())

        ranked = sorted(scored, key=lambda x: float(x["selection_score"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["train_rank"] = rank
        best = ranked[0]
        selected = by_id[str(best["candidate_id"])]
        selected_by_apply[apply_tag] = selected
        selection_rows.append(
            {
                "apply_tag": apply_tag,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "selected": True,
                "selection_train_tag": train_tag,
                "candidate_id": selected.candidate_id,
                "selection_score": float(best["selection_score"]),
                "train_ir": float(best["ir"]),
                "train_annret": float(best["annret"]),
                "train_mdd": float(best["max_drawdown"]),
                "train_scale_mean": float(best["scale_mean"]),
                "train_scaled_days": int(best["scaled_days"]),
                "strict_non_test_selected": True,
                "candidate_family_predeclared": True,
            }
        )

    scaled_parts: List[pd.Series] = []
    scale_parts: List[pd.Series] = []
    apply_rows: List[Dict[str, Any]] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        candidate = selected_by_apply[apply_tag]
        apply_excess = _slice_series(base_excess, apply_start, apply_end)
        scaled, scale = _apply_candidate(apply_excess, candidate)
        metrics = _metrics_from_excess(scaled)
        apply_rows.append(
            {
                "apply_tag": apply_tag,
                "start": str(apply_start.date()),
                "end": str(apply_end.date()),
                "selection_rule": selection_rule,
                "candidate_id": candidate.candidate_id,
                **asdict(candidate),
                "days": int(len(scaled)),
                "scale_min": float(scale.min()),
                "scale_mean": float(scale.mean()),
                "scale_max": float(scale.max()),
                "scaled_days": int((scale < 0.999999).sum()),
                **metrics,
            }
        )
        scaled_parts.append(scaled)
        scale_parts.append(scale)

    stitched_scaled = pd.concat(scaled_parts).sort_index()
    stitched_scale = pd.concat(scale_parts).sort_index()
    full_metrics = _metrics_from_excess(stitched_scaled)
    split_rows = _split_rows(stitched_scaled)
    base_split_rows = [{"candidate_id": "base40_control", **row} for row in _split_rows(base_excess)]
    overlay_split_rows = [{"candidate_id": "report_level_overlay", **row} for row in split_rows]
    all_split_rows = base_split_rows + overlay_split_rows

    report_level_metric_gate_pass = bool(
        full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET
    )
    admissible = False
    hard_gate_pass = bool(admissible and report_level_metric_gate_pass)
    strict_non_test_selected = bool(
        set(selected_by_apply) == {"2024H1_default", "2024H2", "2025", "2026_ytd"}
        and all(bool(row.get("strict_non_test_selected")) for row in selection_rows)
    )

    _write_csv(train_csv, train_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(splits_csv, all_split_rows)
    _write_csv(daily_csv, _daily_rows(base_report, base_excess, stitched_scaled, stitched_scale))

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "volatility_scaled_base40_audit",
        "verdict": "NO_GO_GOVERNANCE_AUDIT_ONLY" if report_level_metric_gate_pass else "NO_GO",
        "hard_gate_pass": hard_gate_pass,
        "report_level_metric_gate_pass": report_level_metric_gate_pass,
        "admissible": admissible,
        "admissibility_reason": (
            "Audit scales already-costed daily excess returns from a reconstructed report. "
            "It is not a variable-exposure trade replay with path-dependent positions and costs."
        ),
        "strict_non_test_selected": strict_non_test_selected,
        "protocol": {
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "base_signal": "base40 run pred.pkl replayed with topk=40,n_drop=2,daily",
            "overlay_family": "predeclared no-leverage report-level volatility/drawdown scales",
            "selection_rule": "2024H1 fixed scale=1; 2024H2 selected by 2024H1; 2025 by 2024; 2026_ytd by 2024-2025",
            "max_leverage": 1.0,
        },
        "metadata": {
            "base_run_id": str(base_dir.name),
            "base_pred_coverage_start": str(coverage_start.date()),
            "base_pred_coverage_end": str(coverage_end.date()),
            "base_pred_coverage_days": coverage_days,
            "reconstruct_base40_report_sec": reconstruct_sec,
        },
        "base40_metrics": base_metrics,
        "overlay_metrics": full_metrics,
        "base40_split_metrics": base_split_rows,
        "overlay_split_metrics": overlay_split_rows,
        "selections": selection_rows,
        "apply_rows": apply_rows,
        "candidate_count": len(candidates),
        "counts": {
            "train_rows": len(train_rows),
            "selection_rows": len(selection_rows),
            "apply_rows": len(apply_rows),
            "daily_rows": int(len(stitched_scaled)),
        },
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "candidate_train_metrics_csv": str(train_csv),
            "selections_csv": str(selections_csv),
            "split_metrics_csv": str(splits_csv),
            "daily_report_overlay_csv": str(daily_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Volatility Scaled Base40 Audit {stamp}",
                "",
                f"- hard_gate_pass: `{hard_gate_pass}`",
                f"- report_level_metric_gate_pass: `{report_level_metric_gate_pass}`",
                f"- admissible: `{admissible}`",
                f"- base40_metrics: `{json.dumps(base_metrics, ensure_ascii=True)}`",
                f"- overlay_metrics: `{json.dumps(full_metrics, ensure_ascii=True)}`",
                "- governance: report-level return scaling audit only; not a variable-exposure trade replay.",
                f"- summary_json: `{summary_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "hard_gate_pass": hard_gate_pass,
                "report_level_metric_gate_pass": report_level_metric_gate_pass,
                "admissible": admissible,
                "strict_non_test_selected": strict_non_test_selected,
                "base40_metrics": base_metrics,
                "overlay_metrics": full_metrics,
                "summary_json": str(summary_json),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
