#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.data import D

import signal_portfolio_conversion_scan as conv


BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
GRU45_RUN_IDS = (
    "7406e47063e9479cb34d300b9ed03bad",
    "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d8413b4bb0b388a63a6b5b6a86",
)
GRU45_RUN_WEIGHTS = (0.4, 0.2, 0.4)

MARKET = "csi300"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
OPEN_COST = 0.0001
CLOSE_COST = 0.0006
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

AUDIT_START = "2021-01-01"
AUDIT_END = "2023-12-31"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    signal_kind: str
    topk: int
    n_drop: int
    signal: pd.DataFrame
    source_paths: Tuple[str, ...]
    notes: str


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    candidates = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not candidates:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id}")
    if len(candidates) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _load_pickle(path)


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port_analysis_config")


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", "~/.quant_master/quant_master_data/tdx_cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


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


def _datetime_values(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        names = list(index.names)
        if "datetime" in names:
            return pd.to_datetime(index.get_level_values("datetime"))
        lv0 = pd.to_datetime(pd.Index(index.get_level_values(0)[:64]), errors="coerce")
        lv1 = pd.to_datetime(pd.Index(index.get_level_values(1)[:64]), errors="coerce")
        level = 0 if lv0.notna().mean() >= lv1.notna().mean() else 1
        return pd.to_datetime(index.get_level_values(level))
    return pd.to_datetime(index)


def _instrument_values(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("expected MultiIndex with an instrument level")
    names = list(index.names)
    if "instrument" in names:
        return index.get_level_values("instrument")
    dt0 = pd.to_datetime(pd.Index(index.get_level_values(0)[:64]), errors="coerce")
    dt1 = pd.to_datetime(pd.Index(index.get_level_values(1)[:64]), errors="coerce")
    level = 1 if dt0.notna().mean() >= dt1.notna().mean() else 0
    return index.get_level_values(level)


def _rank_ensemble(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        run_dir = _find_run_dir(tracking_dir, run_id)
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


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _calc_costed_metrics(report: pd.DataFrame) -> Dict[str, float]:
    excess = (report["return"] - report["bench"] - report["cost"]).astype(float)
    metrics = _metrics_from_excess(excess)
    metrics["turnover"] = float(report["turnover"].mean())
    metrics["rows"] = int(len(report))
    return metrics


def _build_strategy_object(combo: Dict[str, Any], pred_df: pd.DataFrame, base_strategy_kwargs: Dict[str, Any]):
    common_kwargs: Dict[str, Any] = {"signal": pred_df}
    if "risk_degree" in base_strategy_kwargs:
        common_kwargs["risk_degree"] = float(base_strategy_kwargs["risk_degree"])
    kwargs = {
        "topk": int(combo["topk"]),
        "n_drop": int(combo["n_drop"]),
        "method_sell": base_strategy_kwargs.get("method_sell", "bottom"),
        "method_buy": base_strategy_kwargs.get("method_buy", "top"),
        "hold_thresh": int(base_strategy_kwargs.get("hold_thresh", 1)),
        "only_tradable": bool(base_strategy_kwargs.get("only_tradable", False)),
        "forbid_all_trade_at_limit": bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
    }
    kwargs.update(common_kwargs)
    return conv.TopkDropoutStrategy(**kwargs) if hasattr(conv, "TopkDropoutStrategy") else None


def _eval_candidate(
    *,
    candidate: CandidateSpec,
    port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start: str,
    end: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str, Tuple[str, ...]], Any],
    universe: Sequence[str],
    out_dir: Path,
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

    signal = _slice_df(candidate.signal, start, end)
    signal = signal.loc[_instrument_values(signal.index).isin(universe)].copy()
    if signal.empty:
        return {
            "candidate_id": candidate.candidate_id,
            "status": "error",
            "error": "empty signal after universe filtering",
        }

    strategy = conv._build_strategy_object(combo=_combo(candidate.topk, candidate.n_drop), pred_df=signal, base_strategy_kwargs=base_strategy_kwargs)

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = (
        str(backtest_cfg["start_time"]),
        str(backtest_cfg["end_time"]),
        float(open_cost),
        float(close_cost),
        limit_threshold,
        deal_price,
        tuple(universe),
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            codes=list(universe),
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    try:
        portfolio_metric_dict, indicator_dict = run_backtest(
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            strategy=strategy,
            executor=executor_cfg,
            benchmark=backtest_cfg.get("benchmark", "SH000300"),
            account=backtest_cfg.get("account", 100000000),
            exchange_kwargs=exchange_kwargs,
            pos_type=backtest_cfg.get("pos_type", "Position"),
        )
        elapsed = time.perf_counter() - t0
        report = conv._get_report_for_day_freq(portfolio_metric_dict).copy()
        report.index = pd.to_datetime(report.index)
        report = report.loc[(report.index >= pd.Timestamp(start)) & (report.index <= pd.Timestamp(end))].sort_index()
        metrics = _calc_costed_metrics(report)
        report_pkl = out_dir / f"{candidate.candidate_id}_report.pkl"
        report_csv = out_dir / f"{candidate.candidate_id}_report.csv"
        positions_pkl = out_dir / f"{candidate.candidate_id}_positions.pkl"
        indicators_pkl = out_dir / f"{candidate.candidate_id}_indicators.pkl"
        with report_pkl.open("wb") as f:
            pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)
        report.to_csv(report_csv)
        _, positions = next(iter(portfolio_metric_dict.values()))
        with positions_pkl.open("wb") as f:
            pickle.dump(positions, f, protocol=pickle.HIGHEST_PROTOCOL)
        with indicators_pkl.open("wb") as f:
            pickle.dump(indicator_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        split_rows: List[Dict[str, Any]] = []
        for split, split_start, split_end in [
            ("test_full", TEST_START, TEST_END),
            ("2024", "2024-01-01", "2024-12-31"),
            ("2025", "2025-01-01", "2025-12-31"),
            ("2026_ytd", "2026-01-01", TEST_END),
        ]:
            part = report.loc[(report.index >= pd.Timestamp(split_start)) & (report.index <= pd.Timestamp(split_end))]
            if part.empty:
                continue
            split_metrics = _calc_costed_metrics(part)
            split_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "split": split,
                    "start": split_start,
                    "end": split_end,
                    "days": int(len(part)),
                    **{k: split_metrics[k] for k in ("annret", "ir", "max_drawdown")},
                }
            )
        return {
            "candidate_id": candidate.candidate_id,
            "status": "ok",
            "signal_kind": candidate.signal_kind,
            "topk": int(candidate.topk),
            "n_drop": int(candidate.n_drop),
            "metrics": {
                **metrics,
                "elapsed_sec": float(elapsed),
            },
            "hard_gate_pass": bool(metrics["ir"] > HARD_GATE_IR and metrics["annret"] > HARD_GATE_ANNRET),
            "split_metrics": split_rows,
            "source_paths": list(candidate.source_paths),
            "notes": candidate.notes,
            "artifacts": {
                "report_pkl": str(report_pkl),
                "report_csv": str(report_csv),
                "positions_pkl": str(positions_pkl),
                "indicators_pkl": str(indicators_pkl),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": candidate.candidate_id,
            "status": "error",
            "signal_kind": candidate.signal_kind,
            "topk": int(candidate.topk),
            "n_drop": int(candidate.n_drop),
            "elapsed_sec": float(time.perf_counter() - t0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_paths": list(candidate.source_paths),
            "notes": candidate.notes,
        }


def _calendar_union(spans: Sequence[Tuple[pd.Timestamp, pd.Timestamp]], calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex([])
    for start, end in spans:
        mask = (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
        out = out.union(calendar[mask])
    return out


def _build_universe_audit(
    *,
    market: str,
    audit_start: str,
    audit_end: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    market_cfg = D.instruments(market)
    inst_spans = D.list_instruments(market_cfg, start_time=audit_start, end_time=audit_end, freq="day", as_list=False)
    universe = sorted(inst_spans.keys())
    raw_close = D.features(universe, ["$close"], start_time=audit_start, end_time=audit_end, freq="day", disk_cache=0)
    close_df = conv._as_score_df(raw_close)
    close_series = close_df.iloc[:, 0].astype(float)
    calendar = pd.DatetimeIndex(D.calendar(start_time=audit_start, end_time=audit_end, freq="day")).normalize()

    coverage_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    admissible_universe: List[str] = []

    for inst in universe:
        spans = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in inst_spans.get(inst, [])]
        expected_dates = _calendar_union(spans, calendar)
        inst_values = _instrument_values(close_series.index)
        inst_series = close_series.loc[inst_values == inst]
        if not inst_series.empty:
            inst_series = inst_series.dropna()
            observed_raw = pd.DatetimeIndex(_datetime_values(inst_series.index).normalize().unique()).sort_values()
            observed_dates = observed_raw.intersection(expected_dates)
        else:
            observed_dates = pd.DatetimeIndex([])
        expected_count = int(len(expected_dates))
        observed_count = int(len(observed_dates))
        missing_dates = expected_dates.difference(observed_dates)
        missing_count = int(len(missing_dates))
        coverage_ratio = float(observed_count / expected_count) if expected_count > 0 else 0.0
        row = {
            "instrument": inst,
            "active_spans": json.dumps([(str(s.date()), str(e.date())) for s, e in spans], ensure_ascii=False),
            "expected_days": expected_count,
            "observed_days": observed_count,
            "missing_days": missing_count,
            "coverage_ratio": coverage_ratio,
            "admissible": bool(missing_count == 0 and expected_count > 0),
            "first_observed": str(observed_dates.min().date()) if observed_count else "",
            "last_observed": str(observed_dates.max().date()) if observed_count else "",
            "missing_dates_sample": "|".join(str(x.date()) for x in missing_dates[:10]),
        }
        coverage_rows.append(row)
        if row["admissible"]:
            admissible_universe.append(inst)
        else:
            excluded_rows.append(
                {
                    "instrument": inst,
                    "reason": "pre_2024_close_coverage_incomplete",
                    "expected_days": expected_count,
                    "observed_days": observed_count,
                    "missing_days": missing_count,
                    "coverage_ratio": coverage_ratio,
                    "missing_dates_sample": row["missing_dates_sample"],
                }
            )

    audit_meta = {
        "market": market,
        "audit_window": {"start": audit_start, "end": audit_end},
        "rule": "keep instruments whose $close is non-missing for every trading day in each active pre-2024 span",
        "admissible": True,
        "future_price_audit_used": False,
        "future_price_audit_note": "No 2024-2026 price availability was used for filtering; that would be audit-only and non-admissible.",
        "universe_size": int(len(admissible_universe)),
        "excluded_size": int(len(excluded_rows)),
    }
    return admissible_universe, coverage_rows, excluded_rows, audit_meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay base40/gru45 under a pre-2024 finite-price universe.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--output-prefix", default="finite_price_universe_replay")
    p.add_argument("--market", default=MARKET)
    p.add_argument("--audit-start", default=AUDIT_START)
    p.add_argument("--audit-end", default=AUDIT_END)
    p.add_argument("--test-start", default=TEST_START)
    p.add_argument("--test-end", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=OPEN_COST)
    p.add_argument("--close-cost", type=float, default=CLOSE_COST)
    return p


def main() -> int:
    args = build_parser().parse_args()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent / f"{args.output_prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    summary_path = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    coverage_csv = out_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    excluded_csv = out_dir / f"{args.output_prefix}_excluded_{stamp}.csv"
    candidate_csv = out_dir / f"{args.output_prefix}_candidates_{stamp}.csv"
    split_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"

    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    base_run_dir = _find_run_dir(tracking_dir, BASE_RUN_ID)
    base_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(base_cfg)
    port_cfg = _extract_port_config(base_cfg)
    base_strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base40_run_dir = base_run_dir
    gru45_base_pred = _load_score_df(base40_run_dir)
    coverage_start, coverage_end, coverage_days = _date_range(gru45_base_pred)
    blend_start = min(coverage_start, pd.Timestamp(args.test_start))
    blend_end = max(coverage_end, pd.Timestamp(args.test_end))
    gru45_signal = _rank_ensemble(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, blend_start, blend_end)

    universe, coverage_rows, excluded_rows, audit_meta = _build_universe_audit(
        market=args.market,
        audit_start=args.audit_start,
        audit_end=args.audit_end,
    )

    base40_signal = _slice_df(gru45_base_pred, args.test_start, args.test_end)
    gru45_signal = _slice_df(gru45_signal, args.test_start, args.test_end)

    candidate_specs = [
        CandidateSpec(
            candidate_id="base40_control",
            signal_kind="topk_dropout",
            topk=40,
            n_drop=2,
            signal=base40_signal,
            source_paths=(str(base_run_dir / "artifacts" / "pred.pkl"),),
            notes="base run 7406e470 TopkDropoutStrategy topk=40 n_drop=2",
        ),
        CandidateSpec(
            candidate_id="gru45_control",
            signal_kind="topk_dropout",
            topk=45,
            n_drop=4,
            signal=gru45_signal,
            source_paths=tuple(str(_find_run_dir(tracking_dir, rid) / "artifacts" / "pred.pkl") for rid in GRU45_RUN_IDS),
            notes="rank ensemble 7406e470/1a085ff9/773bd6d weights=0.4/0.2/0.4 topk=45 n_drop=4",
        ),
    ]

    exchange_cache: Dict[Tuple[str, str, float, float, float, str, Tuple[str, ...]], Any] = {}
    candidate_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    candidate_results: List[Dict[str, Any]] = []
    passed_candidates: List[str] = []

    for candidate in candidate_specs:
        result = _eval_candidate(
            candidate=candidate,
            port_cfg=port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start=args.test_start,
            end=args.test_end,
            exchange_cache=exchange_cache,
            universe=universe,
            out_dir=out_dir,
        )
        candidate_results.append(result)
        if result.get("status") == "ok":
            metrics = result["metrics"]
            candidate_rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "signal_kind": result["signal_kind"],
                    "topk": result["topk"],
                    "n_drop": result["n_drop"],
                    "annret": metrics["annret"],
                    "ir": metrics["ir"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "rows": metrics["rows"],
                    "elapsed_sec": metrics["elapsed_sec"],
                    "hard_gate_pass": result["hard_gate_pass"],
                    "status": "ok",
                }
            )
            split_rows.extend(result["split_metrics"])
            if result["hard_gate_pass"]:
                passed_candidates.append(result["candidate_id"])
        else:
            candidate_rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "signal_kind": result.get("signal_kind", ""),
                    "topk": result.get("topk", ""),
                    "n_drop": result.get("n_drop", ""),
                    "status": "error",
                    "error_type": result.get("error_type", ""),
                    "error": result.get("error", ""),
                    "elapsed_sec": result.get("elapsed_sec", ""),
                }
            )

    hard_gate_pass = bool(passed_candidates)

    _write_csv(coverage_csv, coverage_rows)
    _write_csv(excluded_csv, excluded_rows)
    _write_csv(candidate_csv, candidate_rows)
    _write_csv(split_csv, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "replay base40/gru45 under a fixed pre-2024 finite-price universe",
        "admissible": bool(audit_meta["admissible"]),
        "filter_definition": audit_meta,
        "universe": {
            "market": args.market,
            "size": int(len(universe)),
            "codes": universe,
        },
        "excluded_stocks": excluded_rows,
        "candidate_results": candidate_results,
        "full_metrics": candidate_rows,
        "split_metrics": split_rows,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "hard_gate_pass": hard_gate_pass,
        "hard_gate_pass_candidates": passed_candidates,
        "artifacts": {
            "summary_json": str(summary_path),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "excluded_csv": str(excluded_csv),
            "candidate_csv": str(candidate_csv),
            "split_csv": str(split_csv),
            "out_dir": str(out_dir),
        },
        "source_meta": {
            "base_run_id": BASE_RUN_ID,
            "gru45_run_ids": list(GRU45_RUN_IDS),
            "gru45_run_weights": list(GRU45_RUN_WEIGHTS),
            "coverage_anchor_start": str(coverage_start.date()),
            "coverage_anchor_end": str(coverage_end.date()),
            "coverage_anchor_days": coverage_days,
            "test_window": {"start": args.test_start, "end": args.test_end},
            "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        },
    }
    _write_json(summary_path, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Finite Price Universe Replay {stamp}",
                "",
                f"- admissible: `{summary['admissible']}`",
                f"- hard_gate_pass: `{hard_gate_pass}`",
                f"- passed_candidates: `{passed_candidates}`",
                f"- universe_size: `{len(universe)}`",
                f"- excluded_size: `{len(excluded_rows)}`",
                f"- artifacts: `{json.dumps(summary['artifacts'], ensure_ascii=False)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "hard_gate_pass": hard_gate_pass,
                "passed_candidates": passed_candidates,
                "candidate_results": candidate_results,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

