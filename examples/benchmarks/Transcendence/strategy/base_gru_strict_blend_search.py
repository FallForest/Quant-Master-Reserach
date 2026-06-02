#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
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
GRU45_RUN_IDS = (
    "7406e47063e9479cb34d300b9ed03bad",
    "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d8413b4bb0b388a63a6b5b6a86",
)
GRU45_RUN_WEIGHTS = (0.4, 0.2, 0.4)
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
VALID_END = "2023-12-31"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _slice_df(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index)
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask].copy()


def _rank_ensemble(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        run_dir = conv._find_run_dir(tracking_dir, run_id)
        pred = _slice_df(_load_score_df(run_dir), start, end)
        ranked = conv._cross_section_rank(pred["score"].astype(float))
        ranked.name = run_id
        cols.append(ranked)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    score = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _blend_base_gru(base40: pd.DataFrame, gru45: pd.DataFrame, gru_weight: float) -> pd.DataFrame:
    return conv._blend_rank_scores(base40, gru45, 1.0 - float(gru_weight), float(gru_weight)).sort_index()


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _combo(topk: int, n_drop: int) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(topk),
        "n_drop": int(n_drop),
        "hold_topk": int(topk),
        "weight_mode": "equal",
        "score_power": 1.0,
    }


def _eval_with_report(
    *,
    pred_df: pd.DataFrame,
    port_cfg: Dict[str, Any],
    strategy_kwargs: Dict[str, Any],
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    out = conv._eval_combo_period(
        combo=_combo(topk, n_drop),
        pred_df=pred_df,
        base_port_cfg=port_cfg,
        base_strategy_kwargs=strategy_kwargs,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        start_time=pd.Timestamp(start),
        end_time=pd.Timestamp(end),
        exchange_cache=exchange_cache,
    )
    # Re-run via local copy only when the caller needs series; the main path keeps
    # the conversion utility as the source of admissible net-cost metrics.
    return out


def _eval_with_series(
    *,
    pred_df: pd.DataFrame,
    port_cfg: Dict[str, Any],
    strategy_kwargs: Dict[str, Any],
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = start
    backtest_cfg["end_time"] = end
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
    cache_key = (start, end, float(open_cost), float(close_cost), limit_threshold, deal_price)
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = conv.get_exchange(
            freq=freq,
            start_time=start,
            end_time=end,
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    strategy = conv._build_strategy_object(
        combo=_combo(topk, n_drop),
        pred_df=_slice_df(pred_df, start, end),
        base_strategy_kwargs=strategy_kwargs,
    )
    t0 = time.perf_counter()
    portfolio_metric_dict, _ = conv.run_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = conv._get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = conv._calc_costed_metrics(report)
    excess = report["return"] - report["bench"] - report["cost"]
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "elapsed_sec": float(time.perf_counter() - t0),
        "excess": excess,
        "report": report,
    }


def _split_rows(excess: pd.Series) -> List[Dict[str, Any]]:
    rows = []
    for split, start, end in [
        ("test_full", TEST_START, TEST_END),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", TEST_END),
    ]:
        s = excess.loc[(excess.index >= pd.Timestamp(start)) & (excess.index <= pd.Timestamp(end))]
        if s.empty:
            continue
        rows.append({"split": split, "start": start, "end": end, "days": int(len(s)), **_metrics_from_excess(s)})
    return rows


def _parse_weights(text: str) -> List[float]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            val = float(part)
            if val < 0.0 or val > 1.0:
                raise ValueError(f"weight outside [0, 1]: {val}")
            out.append(val)
    if not out:
        raise ValueError("empty weight grid")
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict pre-2024 base40/gru45 rank blend search.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--valid-start", default="2021-01-01")
    p.add_argument("--valid-end", default=VALID_END)
    p.add_argument("--test-start", default=TEST_START)
    p.add_argument("--test-end", default=TEST_END)
    p.add_argument("--weight-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--min-valid-days", type=int, default=20)
    p.add_argument("--output-prefix", default="base_gru_strict_blend_search")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    selection_csv = out_dir / f"{args.output_prefix}_selection_{stamp}.csv"
    eval_csv = out_dir / f"{args.output_prefix}_eval_{stamp}.csv"
    split_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    pred_pkl = out_dir / f"{args.output_prefix}_selected_pred_{stamp}.pkl"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    wf_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(wf_cfg)
    port_cfg = conv._extract_port_config(wf_cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    base40 = _load_score_df(base_dir)
    coverage_start, coverage_end, coverage_days = _date_range(base40)
    blend_start = min(coverage_start, pd.Timestamp(args.valid_start))
    blend_end = max(coverage_end, pd.Timestamp(args.test_end))
    gru45 = _rank_ensemble(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, blend_start, blend_end)

    metadata = {
        "base40": {
            "run_id": args.base_run_id,
            "signal_definition": "base run pred.pkl evaluated with topk=40,n_drop=2",
            "coverage_start": str(coverage_start.date()),
            "coverage_end": str(coverage_end.date()),
            "coverage_days": coverage_days,
        },
        "gru45": {
            "run_ids": list(GRU45_RUN_IDS),
            "weights": list(GRU45_RUN_WEIGHTS),
            "signal_definition": "rank ensemble evaluated with topk=45,n_drop=4, matching existing gru45 evidence",
            "coverage_start": str(_date_range(gru45)[0].date()),
            "coverage_end": str(_date_range(gru45)[1].date()),
            "coverage_days": int(_date_range(gru45)[2]),
        },
    }

    valid_base = _slice_df(base40, args.valid_start, args.valid_end)
    valid_gru = _slice_df(gru45, args.valid_start, args.valid_end)
    valid_dates = pd.Index([])
    if not valid_base.empty and not valid_gru.empty:
        valid_dates = pd.Index(pd.to_datetime(valid_base.index.get_level_values(0)).normalize().unique()).intersection(
            pd.Index(pd.to_datetime(valid_gru.index.get_level_values(0)).normalize().unique())
        )

    selection_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    selected_weight: float | None = None
    selected_reason = ""
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    admissible_selection = int(len(valid_dates)) >= int(args.min_valid_days)
    if admissible_selection:
        for weight in _parse_weights(args.weight_grid):
            candidate = _blend_base_gru(base40, gru45, weight)
            ev = _eval_with_report(
                pred_df=candidate,
                port_cfg=port_cfg,
                strategy_kwargs=strategy_kwargs,
                start=args.valid_start,
                end=args.valid_end,
                topk=40,
                n_drop=2,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=exchange_cache,
            )
            score = float(ev["costed_ir"]) + 0.35 * float(ev["costed_annret"])
            selection_rows.append(
                {
                    "gru_weight": float(weight),
                    "base_weight": float(1.0 - weight),
                    "valid_start": args.valid_start,
                    "valid_end": args.valid_end,
                    "valid_days": int(len(valid_dates)),
                    "selection_metric": "valid_ir_plus_0.35_annret",
                    "selection_score": score,
                    **ev,
                }
            )
        selection_rows.sort(key=lambda x: float(x["selection_score"]), reverse=True)
        for rank, row in enumerate(selection_rows, start=1):
            row["selection_rank"] = rank
        selected_weight = float(selection_rows[0]["gru_weight"])
        selected_reason = "selected_by_pre_2024_validation"
    else:
        selected_reason = "no_admissible_selection_data"
        selection_rows.append(
            {
                "selected": False,
                "reason": selected_reason,
                "valid_start": args.valid_start,
                "valid_end": args.valid_end,
                "valid_overlap_days": int(len(valid_dates)),
                "min_valid_days": int(args.min_valid_days),
                "note": "pred/label/report artifacts for base40 and gru45 do not provide enough pre-2024 overlap; no test-period weight selection performed",
            }
        )

    control_defs = [
        ("base40_control", base40, 40, 2, 0.0),
        ("gru45_control", gru45, 45, 4, 1.0),
    ]
    if selected_weight is not None:
        selected_signal = _blend_base_gru(base40, gru45, selected_weight)
        control_defs.append(("selected_blend", selected_signal, 40, 2, selected_weight))
    else:
        selected_signal = None

    best_selected: Dict[str, Any] | None = None
    for name, signal, topk, n_drop, weight in control_defs:
        ev = _eval_with_series(
            pred_df=signal,
            port_cfg=port_cfg,
            strategy_kwargs=strategy_kwargs,
            start=args.test_start,
            end=args.test_end,
            topk=topk,
            n_drop=n_drop,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        row = {
            "candidate_id": name,
            "phase": "test_evaluation_control" if name != "selected_blend" else "test_evaluation_selected",
            "test_start": args.test_start,
            "test_end": args.test_end,
            "topk": int(topk),
            "n_drop": int(n_drop),
            "gru_weight": float(weight),
            "base_weight": float(1.0 - weight),
            "costed_annret": float(ev["costed_annret"]),
            "costed_ir": float(ev["costed_ir"]),
            "max_drawdown": float(ev["max_drawdown"]),
            "turnover": float(ev["turnover"]),
            "elapsed_sec": float(ev["elapsed_sec"]),
        }
        eval_rows.append(row)
        for split_row in _split_rows(ev["excess"]):
            split_rows.append({"candidate_id": name, **split_row})
        if name == "selected_blend":
            best_selected = row

    hard_gate_pass = bool(
        admissible_selection
        and best_selected is not None
        and float(best_selected["costed_ir"]) > HARD_GATE_IR
        and float(best_selected["costed_annret"]) > HARD_GATE_ANNRET
    )
    if selected_signal is not None:
        with pred_pkl.open("wb") as f:
            pickle.dump(selected_signal["score"], f)

    _write_csv(selection_csv, selection_rows)
    _write_csv(eval_csv, eval_rows)
    _write_csv(split_csv, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "base40_gru45_strict_blend_search",
        "protocol": {
            "selection_rule": "candidate linear/rank blend weight must be selected only on data <= 2023-12-31",
            "test_rule": "only selected weight may be judged on 2024-01-01..2026-04-30",
            "net_cost": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "non_admissible_proxy": False,
        },
        "admissible_selection": bool(admissible_selection),
        "selection_reason": selected_reason,
        "selected_weight": selected_weight,
        "hard_gate_pass": hard_gate_pass,
        "metadata": metadata,
        "selection_rows": selection_rows,
        "test_eval_rows": eval_rows,
        "split_rows": split_rows,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "selection_csv": str(selection_csv),
            "eval_csv": str(eval_csv),
            "split_csv": str(split_csv),
            "selected_pred_pkl": str(pred_pkl) if selected_signal is not None else None,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(summary_json, summary)
    md_lines = [
        f"# Base/GRU Strict Blend Search {stamp}",
        "",
        f"- hard_gate_pass: `{hard_gate_pass}`",
        f"- admissible_selection: `{admissible_selection}`",
        f"- selection_reason: `{selected_reason}`",
        f"- selected_weight: `{selected_weight}`",
        f"- controls_or_selected: `{json.dumps(eval_rows, ensure_ascii=False)}`",
        f"- artifacts: `{json.dumps(summary['artifacts'], ensure_ascii=False)}`",
    ]
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "hard_gate_pass": hard_gate_pass,
                "admissible_selection": bool(admissible_selection),
                "selection_reason": selected_reason,
                "selected_weight": selected_weight,
                "eval_rows": eval_rows,
                "summary_json": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

