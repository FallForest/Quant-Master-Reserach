#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
VALID_END = "2023-12-31"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

RUN_ALIAS = {
    "7406": "7406e47063e9479cb34d300b9ed03bad",
    "7406e470": "7406e47063e9479cb34d300b9ed03bad",
    "773": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "e2300230": "e2300230e0994a1a9ccbbd3bc4606d97",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "1a085": "1a085ff9b5a34f408a44ad74055fc5da",
    "05ef8bd1": "05ef8bd12e0e407f9fdf0cad3ef72652",
    "0ed35c": "0ed35c572e104ddab555a8af6a7fe981",
    "2ac6": "2ac6ebc249bf42e5a9f83c6ca0725941",
}


@dataclass(frozen=True)
class DiscoveredRun:
    run_id: str
    exp_id: str
    model_class: str
    dataset_class: str
    instruments: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    pred_path: str
    full_test_coverage: bool
    coverage_days: int
    valid_l2: float | None
    train_l2: float | None
    icir: float | None
    rank_icir: float | None
    source_test_ir: float | None
    source_test_annret: float | None
    source_topk: int | None
    source_n_drop: int | None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cluster_key: str
    member_run_ids: Tuple[str, ...]
    weights: Tuple[float, ...]
    exec_topk: int
    exec_n_drop: int
    family: str
    selection_score: float
    metadata_score_breakdown: Dict[str, float]


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    cands = [p for p in tracking_dir.glob(f"*/{run_id}") if p.is_dir() and (p / "artifacts").exists()]
    if not cands:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id}")
    if len(cands) > 1:
        exact = [p for p in cands if p.name == run_id]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in cands]}")
    return cands[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        with path.open("rb") as f:
            return pickle.load(f)


def _parse_metric_file(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    parts = metric_path.read_text(encoding="utf-8", errors="ignore").strip().split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port_analysis_config or task.record[PortAnaRecord].kwargs.config")


def _extract_dataset_meta(config: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    task = config.get("task", {})
    model_class = ""
    dataset_class = ""
    instruments = ""
    valid_start = ""
    valid_end = ""
    test_end = ""
    if isinstance(task, dict):
        model_cfg = task.get("model", {})
        if isinstance(model_cfg, dict):
            model_class = str(model_cfg.get("class", ""))
        dataset_cfg = task.get("dataset", {})
        if isinstance(dataset_cfg, dict):
            kwargs = dataset_cfg.get("kwargs", {})
            if isinstance(kwargs, dict):
                segments = kwargs.get("segments", {})
                if isinstance(segments, dict):
                    valid = segments.get("valid", [])
                    test = segments.get("test", [])
                    if len(valid) == 2:
                        valid_start = str(valid[0])
                        valid_end = str(valid[1])
                    if len(test) == 2:
                        test_end = str(test[1])
                handler = kwargs.get("handler", {})
                if isinstance(handler, dict):
                    dataset_class = str(handler.get("class", ""))
                    hkwargs = handler.get("kwargs", {})
                    if isinstance(hkwargs, dict):
                        instruments = str(hkwargs.get("instruments", ""))
    return model_class, dataset_class, instruments, valid_start, valid_end, test_end


def _as_score_df(pred_obj: Any) -> pd.DataFrame:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float).to_frame("score")
    if isinstance(pred_obj, pd.DataFrame):
        if pred_obj.empty:
            return pd.DataFrame(columns=["score"], index=pred_obj.index)
        if "score" in pred_obj.columns:
            return pred_obj[["score"]].astype(float)
        if pred_obj.shape[1] == 1:
            col = pred_obj.columns[0]
            return pred_obj[[col]].rename(columns={col: "score"}).astype(float)
        return pred_obj.iloc[:, [0]].rename(columns={pred_obj.columns[0]: "score"}).astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _slice_pred(pred_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.to_datetime(pred_df.index.get_level_values(0))
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return pred_df.loc[mask].copy()


def _cross_section_rank(score: pd.Series) -> pd.Series:
    if isinstance(score.index, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(score.index).rank(method="average", pct=True)


def _extract_exec_params(port_cfg: Dict[str, Any]) -> Tuple[int | None, int | None]:
    strat = port_cfg.get("strategy", {})
    kwargs = strat.get("kwargs", {}) if isinstance(strat, dict) else {}
    topk = kwargs.get("topk")
    n_drop = kwargs.get("n_drop")
    return (int(topk) if topk is not None else None, int(n_drop) if n_drop is not None else None)


def _discover_runs(tracking_dir: Path) -> List[DiscoveredRun]:
    rows: List[DiscoveredRun] = []
    for run_dir in sorted(tracking_dir.glob("*/*")):
        if not run_dir.is_dir():
            continue
        if len(run_dir.name) != 32:
            continue
        pred_path = run_dir / "artifacts" / "pred.pkl"
        cfg_path = run_dir / "artifacts" / "config"
        if not pred_path.exists() or not cfg_path.exists():
            continue
        config = _load_config(cfg_path)
        model_class, dataset_class, instruments, valid_start, valid_end, test_end = _extract_dataset_meta(config)
        port_cfg = _extract_port_config(config)
        topk, n_drop = _extract_exec_params(port_cfg)

        pred_df = _as_score_df(_load_pickle(pred_path))
        dates = pd.to_datetime(pred_df.index.get_level_values(0))
        unique_dates = pd.Index(dates.normalize().unique()).sort_values()
        min_date = unique_dates.min() if len(unique_dates) else pd.NaT
        max_date = unique_dates.max() if len(unique_dates) else pd.NaT
        first_admissible_date = pd.Timestamp(TEST_START) + pd.Timedelta(days=1)
        full_coverage = bool(
            pd.notna(min_date)
            and pd.notna(max_date)
            and min_date <= first_admissible_date
            and max_date >= pd.Timestamp(TEST_END)
        )
        coverage_days = int(len(unique_dates))

        metric_dir = run_dir / "metrics"
        valid_l2 = _parse_metric_file(metric_dir / "l2.valid")
        train_l2 = _parse_metric_file(metric_dir / "l2.train")
        icir = _parse_metric_file(metric_dir / "ICIR")
        rank_icir = _parse_metric_file(metric_dir / "Rank ICIR")
        source_test_ir = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.information_ratio")
        source_test_annret = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.annualized_return")

        rows.append(
            DiscoveredRun(
                run_id=run_dir.name,
                exp_id=run_dir.parent.name,
                model_class=model_class,
                dataset_class=dataset_class,
                instruments=instruments,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=TEST_START,
                test_end=test_end or TEST_END,
                pred_path=str(pred_path).replace("\\", "/"),
                full_test_coverage=full_coverage,
                coverage_days=coverage_days,
                valid_l2=valid_l2,
                train_l2=train_l2,
                icir=icir,
                rank_icir=rank_icir,
                source_test_ir=source_test_ir,
                source_test_annret=source_test_annret,
                source_topk=topk,
                source_n_drop=n_drop,
            )
        )
    return rows


def _cluster_key(run: DiscoveredRun) -> str:
    return "|".join([run.model_class, run.dataset_class, run.instruments])


def _normalize_positive(weights: Sequence[float]) -> Tuple[float, ...]:
    vals = [float(x) for x in weights]
    if not vals:
        raise ValueError("empty weights")
    mn = min(vals)
    if mn <= 0:
        vals = [x - mn + 1e-12 for x in vals]
    s = float(sum(vals))
    if s <= 0:
        raise ValueError("weights sum <= 0")
    return tuple(float(x) / s for x in vals)


def _harmonic_weights(n: int) -> Tuple[float, ...]:
    return _normalize_positive([1.0 / (i + 1) for i in range(n)])


def _rank_equal_weights(n: int) -> Tuple[float, ...]:
    return tuple([1.0 / n] * n)


def _most_common_exec(member_runs: Sequence[DiscoveredRun]) -> Tuple[int, int]:
    counts = Counter((r.source_topk, r.source_n_drop) for r in member_runs if r.source_topk is not None and r.source_n_drop is not None)
    if counts:
        return counts.most_common(1)[0][0]
    return (45, 4)


def _build_candidates(cluster_key: str, runs: Sequence[DiscoveredRun], max_members: int = 4) -> List[Candidate]:
    ranked = sorted(
        [r for r in runs if r.full_test_coverage and r.valid_l2 is not None],
        key=lambda r: (
            float(r.valid_l2),
            int(r.coverage_days) * -1,
            int(r.source_topk or 0),
            int(r.source_n_drop or 0),
            r.run_id,
        ),
    )
    if not ranked:
        return []
    exec_templates: List[Tuple[int, int]] = []
    for r in ranked[: max(3, max_members)]:
        tpl = (r.source_topk or 45, r.source_n_drop or 4)
        if tpl not in exec_templates:
            exec_templates.append(tpl)
    if not exec_templates:
        exec_templates = [(45, 4)]

    candidates: List[Candidate] = []
    for depth in range(1, min(max_members, len(ranked)) + 1):
        members = tuple(r.run_id for r in ranked[:depth])
        member_runs = ranked[:depth]
        families: List[Tuple[str, Tuple[float, ...]]] = [
            ("equal", _rank_equal_weights(depth)),
        ]
        if depth >= 2:
            families.append(("harmonic", _harmonic_weights(depth)))
        for family, weights in families:
            exec_topk, exec_n_drop = _most_common_exec(member_runs)
            aligned = sum(
                1 for r in member_runs if (r.source_topk or 0, r.source_n_drop or 0) == (exec_topk, exec_n_drop)
            )
            valid_l2_values = [float(r.valid_l2) for r in member_runs if r.valid_l2 is not None]
            valid_l2_weighted = float(np.average(valid_l2_values, weights=list(weights)))
            valid_l2_worst = float(np.max(valid_l2_values))
            coverage_mean = float(np.mean([r.coverage_days for r in member_runs]))
            family_bonus = 1e-6 if family == "equal" else 0.0
            selection_score = float(-valid_l2_weighted - valid_l2_worst * 1e-3 + len(member_runs) * 1e-5 + aligned * 1e-7 + family_bonus)
            for tpl_topk, tpl_n_drop in exec_templates:
                candidate_id = f"{cluster_key.replace('|', '_')}__{family}__top{depth}__tk{tpl_topk}_nd{tpl_n_drop}"
                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        cluster_key=cluster_key,
                        member_run_ids=members,
                        weights=weights,
                        exec_topk=int(tpl_topk),
                        exec_n_drop=int(tpl_n_drop),
                        family=family,
                        selection_score=selection_score,
                        metadata_score_breakdown={
                            "member_count": float(len(member_runs)),
                            "exec_alignment": float(aligned),
                            "coverage_mean_days": coverage_mean,
                            "family_bonus": family_bonus,
                            "valid_l2_weighted": valid_l2_weighted,
                            "valid_l2_worst": valid_l2_worst,
                        },
                    )
                )
    candidates.sort(key=lambda c: (c.selection_score, len(c.member_run_ids), c.family), reverse=True)
    return candidates


def _blend_rank_signal(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: str,
    end: str,
) -> pd.DataFrame:
    cols: List[pd.Series] = []
    for run_id in run_ids:
        pred = _as_score_df(_load_pickle(_find_run_dir(tracking_dir, run_id) / "artifacts" / "pred.pkl"))
        pred = _slice_pred(pred, start, end)
        if pred.empty:
            raise ValueError(f"empty pred slice for run_id={run_id} in {start}..{end}")
        rank_s = _cross_section_rank(pred["score"].astype(float))
        rank_s.name = run_id
        cols.append(rank_s)
    panel = pd.concat(cols, axis=1)
    w = pd.Series([float(x) for x in weights], index=panel.columns, dtype=float)
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


def _get_day_report(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    return portfolio_metric_dict[next(iter(portfolio_metric_dict.keys()))][0]


def _calc_costed_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    excess = report_df["return"].astype(float) - report_df["bench"].astype(float) - report_df["cost"].astype(float)
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report_df["turnover"].mean()),
    }


def _run_backtest_eval(
    *,
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_time: str,
    end_time: str,
    topk: int,
    n_drop: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_time).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_time).date())
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    exchange_kwargs["exchange"] = get_exchange(
        freq=freq,
        start_time=str(backtest_cfg["start_time"]),
        end_time=str(backtest_cfg["end_time"]),
        deal_price=deal_price,
        limit_threshold=limit_threshold,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=min_cost,
    )
    base_strategy_kwargs = dict(backtest_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)
    strategy = TopkDropoutStrategy(
        signal=pred_df,
        topk=int(topk),
        n_drop=int(n_drop),
        method_sell=base_strategy_kwargs.get("method_sell", "bottom"),
        method_buy=base_strategy_kwargs.get("method_buy", "top"),
        hold_thresh=int(base_strategy_kwargs.get("hold_thresh", 1)),
        only_tradable=bool(base_strategy_kwargs.get("only_tradable", False)),
        forbid_all_trade_at_limit=bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
    )
    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=str(backtest_cfg["start_time"]),
        end_time=str(backtest_cfg["end_time"]),
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report_df = _get_day_report(portfolio_metric_dict)
    metrics = _calc_costed_metrics(report_df)
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    return {"metrics": metrics, "report": report_df}


def _slice_metrics(report_df: pd.DataFrame, start: str, end: str) -> Dict[str, float]:
    idx = pd.to_datetime(report_df.index)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    sliced = report_df.loc[mask].copy()
    if sliced.empty:
        return {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    return _calc_costed_metrics(sliced)


def _parse_run_list(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.split(","):
        tok = raw.strip()
        if not tok:
            continue
        out.append(RUN_ALIAS.get(tok, tok))
    return list(dict.fromkeys(out))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search pre-2024 validated rank/blend candidates and test once on 2024-2026.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--output-prefix", default="historical_validation_ensemble_search")
    p.add_argument("--preferred-cluster", default="", help="Optional cluster key override for candidate generation.")
    p.add_argument("--max-members", type=int, default=4, help="Max runs per candidate.")
    p.add_argument("--max-candidates-per-cluster", type=int, default=36)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    out_dir = Path(__file__).resolve().parent
    stamp = _stamp()

    discovered = _discover_runs(tracking_dir)
    if not discovered:
        raise RuntimeError("no usable pred.pkl runs discovered")

    discovered_rows = [
        {
            "run_id": r.run_id,
            "exp_id": r.exp_id,
            "cluster_key": _cluster_key(r),
            "model_class": r.model_class,
            "dataset_class": r.dataset_class,
            "instruments": r.instruments,
            "valid_start": r.valid_start,
            "valid_end": r.valid_end,
            "test_start": r.test_start,
            "test_end": r.test_end,
            "pred_path": r.pred_path,
            "full_test_coverage": r.full_test_coverage,
            "coverage_days": r.coverage_days,
            "valid_l2": r.valid_l2,
            "train_l2": r.train_l2,
            "icir": r.icir,
            "rank_icir": r.rank_icir,
            "source_test_ir": r.source_test_ir,
            "source_test_annret": r.source_test_annret,
            "source_topk": r.source_topk,
            "source_n_drop": r.source_n_drop,
        }
        for r in discovered
    ]

    discovery_path = out_dir / f"{args.output_prefix}_discovered_runs_{stamp}.csv"
    _write_csv(discovery_path, discovered_rows)

    usable = [r for r in discovered if r.full_test_coverage]
    validation_usable = [r for r in usable if r.valid_l2 is not None]
    if not validation_usable:
        raise RuntimeError("no full-coverage run has pre-2024 validation metric l2.valid; refusing test-selected search")
    clusters: Dict[str, List[DiscoveredRun]] = {}
    for r in validation_usable:
        clusters.setdefault(_cluster_key(r), []).append(r)
    if args.preferred_cluster:
        candidate_clusters = {_cluster_key(r): clusters.get(_cluster_key(r), []) for r in usable if _cluster_key(r) == args.preferred_cluster}
        if not candidate_clusters:
            raise RuntimeError(f"preferred cluster not found: {args.preferred_cluster}")
    else:
        candidate_clusters = clusters
    candidate_clusters = {k: v for k, v in candidate_clusters.items() if len(v) >= 2}
    if not candidate_clusters:
        raise RuntimeError("no compatible cluster with at least two full-coverage runs")

    validation_rows: List[Dict[str, Any]] = []
    candidate_pool: List[Candidate] = []
    for cluster_key, runs in candidate_clusters.items():
        cands = _build_candidates(cluster_key, runs, max_members=args.max_members)
        candidate_pool.extend(cands[: args.max_candidates_per_cluster])
        for c in cands[: args.max_candidates_per_cluster]:
            validation_rows.append(
                {
                    "candidate_id": c.candidate_id,
                    "cluster_key": c.cluster_key,
                    "family": c.family,
                    "member_run_ids": "|".join(c.member_run_ids),
                    "weights": "|".join(f"{w:.8f}" for w in c.weights),
                    "exec_topk": c.exec_topk,
                    "exec_n_drop": c.exec_n_drop,
                    "selection_score": c.selection_score,
                    "metadata_member_count": c.metadata_score_breakdown["member_count"],
                    "metadata_exec_alignment": c.metadata_score_breakdown["exec_alignment"],
                    "metadata_coverage_mean_days": c.metadata_score_breakdown["coverage_mean_days"],
                    "metadata_family_bonus": c.metadata_score_breakdown["family_bonus"],
                    "validation_l2_weighted": c.metadata_score_breakdown["valid_l2_weighted"],
                    "validation_l2_worst": c.metadata_score_breakdown["valid_l2_worst"],
                    "member_count": len(c.member_run_ids),
                }
            )

    validation_rows.sort(key=lambda r: (float(r["selection_score"]), int(r["member_count"])), reverse=True)
    leaderboard_path = out_dir / f"{args.output_prefix}_validation_leaderboard_{stamp}.csv"
    _write_csv(leaderboard_path, validation_rows)

    if not candidate_pool:
        raise RuntimeError("no candidates generated")
    selected = max(candidate_pool, key=lambda c: (c.selection_score, len(c.member_run_ids), c.family))
    selected_members = [next(r for r in discovered if r.run_id == rid) for rid in selected.member_run_ids]
    selected_cluster = selected.cluster_key
    selected_base_run = selected_members[0]
    run_dir = _find_run_dir(tracking_dir, selected_base_run.run_id)
    workflow_cfg = _load_config(run_dir / "artifacts" / "config")
    quant_master.init(provider_uri=".qmData/cn_data", region="cn")
    base_port_cfg = _extract_port_config(workflow_cfg)
    base_pred = _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
    base_index = base_pred.index

    selected_pred = _blend_rank_signal(
        tracking_dir=tracking_dir,
        run_ids=selected.member_run_ids,
        weights=selected.weights,
        start=TEST_START,
        end=TEST_END,
    ).reindex(base_index).dropna()
    if selected_pred.empty:
        raise RuntimeError("selected candidate has empty test signal")

    full_eval = _run_backtest_eval(
        pred_df=selected_pred,
        base_port_cfg=base_port_cfg,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        start_time=TEST_START,
        end_time=TEST_END,
        topk=selected.exec_topk,
        n_drop=selected.exec_n_drop,
    )
    full_metrics = full_eval["metrics"]
    full_report = full_eval["report"]
    split_rows: List[Dict[str, Any]] = []
    for slice_name, start, end in [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", TEST_END),
        ("2024_2026_full", TEST_START, TEST_END),
    ]:
        m = _slice_metrics(full_report, start, end)
        split_rows.append(
            {
                "candidate_id": selected.candidate_id,
                "slice": slice_name,
                "start": start,
                "end": end,
                "annret": m["annret"],
                "ir": m["ir"],
                "max_drawdown": m["max_drawdown"],
                "turnover": m["turnover"],
            }
        )

    hard_gate_pass = bool(full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)
    verification: Dict[str, Any] | None = None
    if hard_gate_pass:
        verify_eval = _run_backtest_eval(
            pred_df=selected_pred,
            base_port_cfg=base_port_cfg,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_time=TEST_START,
            end_time=TEST_END,
            topk=selected.exec_topk,
            n_drop=selected.exec_n_drop,
        )
        verify_metrics = verify_eval["metrics"]
        verification = {
            "rerun_metrics": verify_metrics,
            "match": {
                k: bool(math.isclose(float(full_metrics[k]), float(verify_metrics[k]), rel_tol=0.0, abs_tol=1e-9))
                for k in ("annret", "ir", "max_drawdown", "turnover")
            },
        }

    selected_json = {
        "timestamp_utc": _now_utc(),
        "candidate_id": selected.candidate_id,
        "cluster_key": selected.cluster_key,
        "member_run_ids": list(selected.member_run_ids),
        "weights": [float(x) for x in selected.weights],
        "exec_topk": selected.exec_topk,
        "exec_n_drop": selected.exec_n_drop,
        "family": selected.family,
        "selection_score": selected.selection_score,
        "metadata_score_breakdown": selected.metadata_score_breakdown,
        "selection_inputs": {
            "validation_only": True,
            "used_metrics": ["metrics/l2.valid", "metadata/source_topk", "metadata/source_n_drop", "fixed_rules"],
            "excluded_metrics": [
                "metrics/ICIR",
                "metrics/Rank ICIR",
                "metrics/1day.excess_return_with_cost.annualized_return",
                "metrics/1day.excess_return_with_cost.information_ratio",
                "metrics/1day.excess_return_with_cost.max_drawdown",
            ],
            "selection_scope": "pre-2024 validation loss only",
            "test_window": {"start": TEST_START, "end": TEST_END},
            "selection_score_definition": "-weighted_mean(l2.valid) - 0.001*worst(l2.valid) + tiny deterministic tie breakers",
            "candidate_rules": ["equal_rank", "harmonic_rank"],
        },
    }
    selected_path = out_dir / f"{args.output_prefix}_selected_candidate_{stamp}.json"
    _write_json(selected_path, selected_json)

    test_metrics_path = out_dir / f"{args.output_prefix}_test_metrics_{stamp}.json"
    _write_json(
        test_metrics_path,
        {
            "candidate_id": selected.candidate_id,
            "metrics": full_metrics,
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "hard_gate_pass": hard_gate_pass,
            "verification": verification,
        },
    )
    split_path = out_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    _write_csv(split_path, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "tracking_uri": args.tracking_uri,
        "selection_scope": {
            "validation_only": True,
            "test_window": {"start": TEST_START, "end": TEST_END},
            "selection_score_definition": "-weighted_mean(l2.valid) - 0.001*worst(l2.valid) + tiny deterministic tie breakers",
            "test_metrics_used_in_selection": False,
            "pre_2024_validation_metric": "metrics/l2.valid",
            "pre_2024_validation_run_count": len(validation_usable),
        },
        "discovery_artifact": str(discovery_path),
        "validation_leaderboard_artifact": str(leaderboard_path),
        "selected_candidate_artifact": str(selected_path),
        "test_metrics_artifact": str(test_metrics_path),
        "split_metrics_artifact": str(split_path),
        "discovered_run_count": len(discovered_rows),
        "full_coverage_run_count": len(usable),
        "pre_2024_validation_run_count": len(validation_usable),
        "candidate_cluster": selected_cluster,
        "selected_candidate": selected_json,
        "test_metrics": full_metrics,
        "hard_gate_pass": hard_gate_pass,
        "verification": verification,
        "residual_risks": [
            "Selection uses pre-2024 l2.valid model-validation loss, not pre-2024 portfolio IR/AnnRet, because validation-period portfolio backtest artifacts were not discoverable in the current mlruns tree.",
            "The candidate search is intentionally bounded to a small set of fixed rank/blend families; a more exhaustive blend search could still uncover additional candidates.",
        ],
        "artifacts": {
            "discovered_runs_csv": str(discovery_path),
            "validation_leaderboard_csv": str(leaderboard_path),
            "selected_candidate_json": str(selected_path),
            "test_metrics_json": str(test_metrics_path),
            "split_metrics_csv": str(split_path),
        },
    }
    summary_path = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
