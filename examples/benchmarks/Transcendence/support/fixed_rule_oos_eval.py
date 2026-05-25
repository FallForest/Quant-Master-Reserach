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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TARGET_COSTED_IR_7406 = 2.799983676714277
TARGET_COSTED_ANNRET_7406 = 0.24466463608994535
TARGET_COSTED_IR_SOTA_STRAT = 3.0230019401859436

RUN_ALIAS = {
    "e2300230": "e2300230e0994a1a9ccbbd3bc4606d97",
    "7406e470": "7406e47063e9479cb34d300b9ed03bad",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "0ed35c": "0ed35c572e104ddab555a8af6a7fe981",
    "2ac6": "2ac6ebc249bf42e5a9f83c6ca0725941",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
}


class RebalanceMixin:
    def __init__(self, rebalance_mode: str = "daily", rebalance_interval: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.rebalance_mode = str(rebalance_mode).lower()
        self.rebalance_interval = int(max(1, rebalance_interval))
        self._last_rebalance_key: Optional[Tuple[int, int]] = None

    def _get_rebalance_key(self, ts: pd.Timestamp) -> Tuple[int, int]:
        if self.rebalance_mode == "daily":
            return int(ts.year), int(ts.dayofyear)
        if self.rebalance_mode == "weekly":
            iso = ts.isocalendar()
            return int(iso.year), int(iso.week)
        if self.rebalance_mode == "monthly":
            return int(ts.year), int(ts.month)
        return int(ts.year), int(ts.dayofyear)

    def should_rebalance(self, trade_step: int, trade_start_time: pd.Timestamp) -> bool:
        if self.rebalance_mode == "interval":
            return trade_step % self.rebalance_interval == 0
        if self.rebalance_mode in {"daily", "weekly", "monthly"}:
            key = self._get_rebalance_key(pd.Timestamp(trade_start_time))
            if self._last_rebalance_key != key:
                self._last_rebalance_key = key
                return True
            return False
        raise ValueError(f"unsupported rebalance_mode={self.rebalance_mode}")


class ScheduledTopkDropoutStrategy(RebalanceMixin, TopkDropoutStrategy):
    def __init__(
        self,
        *,
        n_drop_schedule: Optional[Sequence[int]] = None,
        rebalance_mode: str = "daily",
        rebalance_interval: int = 5,
        **kwargs,
    ):
        self.n_drop_schedule = [int(x) for x in (n_drop_schedule or [])]
        self._rebalance_count = 0
        super().__init__(rebalance_mode=rebalance_mode, rebalance_interval=rebalance_interval, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return TradeDecisionWO([], self)

        orig_n_drop = int(self.n_drop)
        if self.n_drop_schedule:
            self.n_drop = int(self.n_drop_schedule[self._rebalance_count % len(self.n_drop_schedule)])
        self._rebalance_count += 1
        try:
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.n_drop = orig_n_drop


class BufferedTopkWeightStrategy(RebalanceMixin, WeightStrategyBase):
    def __init__(
        self,
        *,
        topk: int,
        hold_topk: Optional[int] = None,
        weight_mode: str = "equal",
        score_power: float = 1.0,
        rebalance_mode: str = "daily",
        rebalance_interval: int = 5,
        **kwargs,
    ):
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.weight_mode = str(weight_mode).lower()
        self.score_power = float(score_power)
        super().__init__(
            rebalance_mode=rebalance_mode,
            rebalance_interval=rebalance_interval,
            order_generator_cls_or_obj=OrderGenWInteract,
            **kwargs,
        )

    def _to_series(self, score: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        return score.dropna()

    def _calc_weights(self, ranked_score: pd.Series, target: List[str]) -> Dict[str, float]:
        if not target:
            return {}
        if self.weight_mode == "equal":
            w = 1.0 / len(target)
            return {code: w for code in target}
        if self.weight_mode == "score":
            s = ranked_score.reindex(target).astype(float)
            shifted = s - s.min()
            if float(shifted.sum()) <= 0:
                raw = pd.Series(np.arange(len(target), 0, -1), index=target, dtype=float)
            else:
                raw = (shifted + 1e-12) ** max(1e-6, self.score_power)
            norm = float(raw.sum())
            if norm <= 0:
                w = 1.0 / len(target)
                return {code: w for code in target}
            return {code: float(raw.loc[code] / norm) for code in target}
        raise ValueError(f"unsupported weight_mode={self.weight_mode}")

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_step = self.trade_calendar.get_trade_step()
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return None

        score_s = self._to_series(score)
        if score_s.empty:
            return {}
        ranked = score_s.sort_values(ascending=False)

        tradable_ranked_idx = []
        for code in ranked.index:
            try:
                ok = self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                tradable_ranked_idx.append(code)
        if not tradable_ranked_idx:
            return None
        ranked = ranked.reindex(tradable_ranked_idx).dropna()
        if ranked.empty:
            return None

        rank_pos = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        current_stocks = [s for s in current.get_stock_list() if s in ranked.index]
        keep = [s for s in current_stocks if int(rank_pos.get(s, 10**9)) <= self.hold_topk]
        keep = sorted(keep, key=lambda x: float(ranked.loc[x]), reverse=True)
        if len(keep) > self.topk:
            keep = keep[: self.topk]

        need = max(0, self.topk - len(keep))
        add_list = [s for s in ranked.index if s not in keep][:need]
        target = keep + add_list
        if not target:
            return {}
        return self._calc_weights(ranked_score=ranked, target=target)


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
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        with path.open("rb") as f:
            return pickle.load(f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


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


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _get_report_for_day_freq(portfolio_metric_dict):
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    first_key = next(iter(portfolio_metric_dict.keys()))
    return portfolio_metric_dict[first_key][0]


def _calc_costed_metrics(report_df) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, max_drawdown, turnover


def _pred_date_values(pred_df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    return pd.to_datetime(idx)


def _slice_pred(pred_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        lv = pd.to_datetime(idx.get_level_values(0))
        m = (lv >= pd.Timestamp(start)) & (lv <= pd.Timestamp(end))
        return pred_df.loc[m]
    m = (pd.to_datetime(idx) >= pd.Timestamp(start)) & (pd.to_datetime(idx) <= pd.Timestamp(end))
    return pred_df.loc[m]


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


def _cross_section_rank(score: pd.Series) -> pd.Series:
    idx = score.index
    if isinstance(idx, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(idx).rank(method="average", pct=True)


def _build_strategy_object(combo: Dict[str, Any], pred_df, base_strategy_kwargs: Dict[str, Any]):
    common_kwargs: Dict[str, Any] = {"signal": pred_df}
    if "risk_degree" in base_strategy_kwargs:
        common_kwargs["risk_degree"] = float(base_strategy_kwargs["risk_degree"])
    if combo["family"] == "topk_dropout_sched":
        kwargs = {
            "topk": int(combo["topk"]),
            "n_drop": int(combo["n_drop"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
            "n_drop_schedule": combo.get("n_drop_schedule", []),
            "method_sell": base_strategy_kwargs.get("method_sell", "bottom"),
            "method_buy": base_strategy_kwargs.get("method_buy", "top"),
            "hold_thresh": int(base_strategy_kwargs.get("hold_thresh", 1)),
            "only_tradable": bool(base_strategy_kwargs.get("only_tradable", False)),
            "forbid_all_trade_at_limit": bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
        }
        kwargs.update(common_kwargs)
        return ScheduledTopkDropoutStrategy(**kwargs)
    if combo["family"] == "buffered_weight":
        kwargs = {
            "topk": int(combo["topk"]),
            "hold_topk": int(combo["hold_topk"]),
            "weight_mode": str(combo["weight_mode"]),
            "score_power": float(combo["score_power"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
        }
        kwargs.update(common_kwargs)
        return BufferedTopkWeightStrategy(**kwargs)
    raise ValueError(f"unsupported family={combo['family']}")


def _eval_combo_period(
    *,
    combo: Dict[str, Any],
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
):
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_time).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_time).date())
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    pred_slice = _slice_pred(pred_df, start_time, end_time)
    if len(pred_slice) == 0:
        raise ValueError(f"empty signal slice in {start_time} ~ {end_time}")
    strategy_obj = _build_strategy_object(combo=combo, pred_df=pred_slice, base_strategy_kwargs=base_strategy_kwargs)

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["benchmark"] = "SH000300"

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
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark="SH000300",
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    report_df = _get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report_df)
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "elapsed_sec": float(elapsed),
    }


def _resolve_run_ids(tracking_dir: Path, run_tokens: Sequence[str]) -> List[str]:
    all_run_ids = {
        p.name for p in tracking_dir.glob("*/*") if p.is_dir() and (p / "artifacts").exists() and len(p.name) == 32
    }
    resolved: List[str] = []
    for token in run_tokens:
        tok = token.strip()
        if not tok:
            continue
        mapped = RUN_ALIAS.get(tok, tok)
        if mapped in all_run_ids:
            resolved.append(mapped)
            continue
        cands = [rid for rid in all_run_ids if rid.startswith(mapped)]
        if len(cands) == 1:
            resolved.append(cands[0])
            continue
        if len(cands) == 0:
            raise FileNotFoundError(f"run token cannot be resolved: {token}")
        raise RuntimeError(f"run token matches multiple run_ids: token={token}, candidates={cands}")
    return list(dict.fromkeys(resolved))


def _parse_float_list(text: str) -> List[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _build_rank_ensemble_pred(
    *,
    tracking_dir: Path,
    run_tokens: Sequence[str],
    weights: Sequence[float],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    run_ids = _resolve_run_ids(tracking_dir, run_tokens)
    if len(run_ids) != len(weights):
        raise ValueError(f"ensemble run_ids({len(run_ids)}) != weights({len(weights)})")
    if float(np.sum(weights)) <= 0:
        raise ValueError("ensemble weights sum <= 0")

    series_list: List[pd.Series] = []
    for run_id in run_ids:
        run_dir = _find_run_dir(tracking_dir, run_id)
        pred = _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
        pred = _slice_pred(pred, start_time, end_time)
        if pred.empty:
            raise ValueError(f"empty pred slice for run_id={run_id} in {start_time}~{end_time}")
        rank_s = _cross_section_rank(pred["score"].astype(float))
        rank_s.name = run_id
        series_list.append(rank_s)
    panel = pd.concat(series_list, axis=1)
    w = pd.Series([float(x) for x in weights], index=panel.columns, dtype=float)
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0))
    blend = blend.dropna()
    return blend.to_frame("score")


def _build_slices(coverage_start: pd.Timestamp, coverage_end: pd.Timestamp) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    slices: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for year in (2024, 2025, 2026):
        y_start = max(pd.Timestamp(f"{year}-01-01"), coverage_start)
        y_end = min(pd.Timestamp(f"{year}-12-31"), coverage_end)
        if y_start > y_end:
            continue
        tag = f"{year}_ytd" if y_end < pd.Timestamp(f"{year}-12-31") else str(year)
        slices.append((tag, y_start, y_end))
    if coverage_start <= coverage_end:
        slices.append(("2024_2026_full", coverage_start, coverage_end))
    return slices


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fixed portfolio rule family by year slices without per-slice selection.")
    p.add_argument("--run-id", default=TARGET_RUN_ID)
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--config-path", default="")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--ensemble-runs", default="7406e470,773bd6d,bc641")
    p.add_argument("--ensemble-weights", default="0.6,0.2,0.2")
    p.add_argument("--output-prefix", default="portfolio_fixed_oos")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    run_dir = _find_run_dir(tracking_dir, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    pred_path = artifacts_dir / "pred.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing pred artifact: {pred_path}")

    run_cfg_path = Path(args.config_path).expanduser().resolve() if args.config_path else artifacts_dir / "config"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"missing config file: {run_cfg_path}")

    workflow_cfg = _load_config(run_cfg_path)
    _init_quant_master(workflow_cfg)
    base_port_cfg = _extract_port_config(workflow_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base_pred_df = _as_score_df(_load_pickle(pred_path))
    coverage_dates = _pred_date_values(base_pred_df)
    coverage_start = pd.Timestamp(coverage_dates.min())
    coverage_end = pd.Timestamp(coverage_dates.max())
    slices = _build_slices(coverage_start=coverage_start, coverage_end=coverage_end)

    ensemble_weights = _parse_float_list(args.ensemble_weights)
    ensemble_pred_df = _build_rank_ensemble_pred(
        tracking_dir=tracking_dir,
        run_tokens=[x.strip() for x in args.ensemble_runs.split(",") if x.strip()],
        weights=ensemble_weights,
        start_time=coverage_start,
        end_time=coverage_end,
    )

    strategies = [
        {
            "strategy_id": "baseline_7406_default_tk45_nd4_daily",
            "family": "topk_dropout_sched",
            "signal_source": "base",
            "combo": {
                "family": "topk_dropout_sched",
                "tag": "baseline_tk45_nd4_daily",
                "topk": 45,
                "n_drop": 4,
                "rebalance_mode": "daily",
                "rebalance_interval": 1,
                "n_drop_schedule": [],
            },
            "is_candidate_family": False,
        },
        {
            "strategy_id": "fixed_topk50_nd5_daily",
            "family": "topk_dropout_sched",
            "signal_source": "base",
            "combo": {
                "family": "topk_dropout_sched",
                "tag": "fixed_tk50_nd5_daily",
                "topk": 50,
                "n_drop": 5,
                "rebalance_mode": "daily",
                "rebalance_interval": 1,
                "n_drop_schedule": [],
            },
            "is_candidate_family": True,
        },
        {
            "strategy_id": "fixed_topk40_nd2_daily",
            "family": "topk_dropout_sched",
            "signal_source": "base",
            "combo": {
                "family": "topk_dropout_sched",
                "tag": "fixed_tk40_nd2_daily",
                "topk": 40,
                "n_drop": 2,
                "rebalance_mode": "daily",
                "rebalance_interval": 1,
                "n_drop_schedule": [],
            },
            "is_candidate_family": True,
        },
        {
            "strategy_id": "fixed_buffered_tk55_hk85_equal_weekly",
            "family": "buffered_weight",
            "signal_source": "base",
            "combo": {
                "family": "buffered_weight",
                "tag": "buffered_tk55_hk85_equal_weekly",
                "topk": 55,
                "hold_topk": 85,
                "weight_mode": "equal",
                "score_power": 1.0,
                "rebalance_mode": "weekly",
                "rebalance_interval": 10,
            },
            "is_candidate_family": True,
        },
        {
            "strategy_id": "fixed_rank_ensemble_tk45_nd4_daily",
            "family": "topk_dropout_sched",
            "signal_source": "rank_ensemble",
            "combo": {
                "family": "topk_dropout_sched",
                "tag": "rank_ensemble_tk45_nd4_daily",
                "topk": 45,
                "n_drop": 4,
                "rebalance_mode": "daily",
                "rebalance_interval": 1,
                "n_drop_schedule": [],
            },
            "is_candidate_family": True,
        },
    ]

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    eval_rows: List[Dict[str, Any]] = []
    strategy_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for st in strategies:
        sid = str(st["strategy_id"])
        pred_df = base_pred_df if st["signal_source"] == "base" else ensemble_pred_df
        strategy_metrics[sid] = {}
        for slice_name, s_start, s_end in slices:
            ev = _eval_combo_period(
                combo=st["combo"],
                pred_df=pred_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_time=s_start,
                end_time=s_end,
                exchange_cache=exchange_cache,
            )
            rec = {
                "strategy_id": sid,
                "family": st["family"],
                "signal_source": st["signal_source"],
                "slice": slice_name,
                "start_time": str(pd.Timestamp(s_start).date()),
                "end_time": str(pd.Timestamp(s_end).date()),
                "costed_annret": float(ev["costed_annret"]),
                "costed_ir": float(ev["costed_ir"]),
                "max_drawdown": float(ev["max_drawdown"]),
                "turnover": float(ev["turnover"]),
                "elapsed_sec": float(ev["elapsed_sec"]),
                "is_candidate_family": bool(st["is_candidate_family"]),
            }
            eval_rows.append(rec)
            strategy_metrics[sid][slice_name] = rec
            print(
                f"[{sid}][{slice_name}] IR={ev['costed_ir']:.6f} AnnRet={ev['costed_annret']:.6f} "
                f"MDD={ev['max_drawdown']:.6f} TO={ev['turnover']:.6f}",
                flush=True,
            )

    candidate_ids = [s["strategy_id"] for s in strategies if s["is_candidate_family"]]
    robust_rows: List[Dict[str, Any]] = []
    year_slices = ["2024", "2025", "2026_ytd"]
    for sid in candidate_ids:
        m = strategy_metrics[sid]
        yearly_irs = [m[x]["costed_ir"] for x in year_slices if x in m]
        yearly_ann = [m[x]["costed_annret"] for x in year_slices if x in m]
        full = m["2024_2026_full"]
        robust_rows.append(
            {
                "strategy_id": sid,
                "full_ir": float(full["costed_ir"]),
                "full_annret": float(full["costed_annret"]),
                "full_max_drawdown": float(full["max_drawdown"]),
                "min_year_ir": float(min(yearly_irs)) if yearly_irs else float("nan"),
                "min_year_annret": float(min(yearly_ann)) if yearly_ann else float("nan"),
                "year_count": int(len(yearly_irs)),
                "all_year_ir_positive": bool(all(x > 0 for x in yearly_irs)),
                "all_year_annret_positive": bool(all(x > 0 for x in yearly_ann)),
                "beats_7406_threshold": bool(
                    float(full["costed_ir"]) > TARGET_COSTED_IR_7406 and float(full["costed_annret"]) >= TARGET_COSTED_ANNRET_7406
                ),
            }
        )
    robust_rows.sort(
        key=lambda x: (x["all_year_ir_positive"], x["min_year_ir"], x["full_ir"], x["full_annret"]),
        reverse=True,
    )
    best_robust = robust_rows[0] if robust_rows else None
    breakthroughs = [
        x
        for x in robust_rows
        if x["all_year_ir_positive"]
        and x["all_year_annret_positive"]
        and x["beats_7406_threshold"]
    ]
    breakthrough = breakthroughs[0] if breakthroughs else None

    run_short = args.run_id[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / f"{args.output_prefix}_metrics_{run_short}_{stamp}.csv"
    robust_csv = out_dir / f"{args.output_prefix}_robust_{run_short}_{stamp}.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"
    candidate_path = out_dir / f"{args.output_prefix}_candidate_{run_short}_{stamp}.json"

    _write_csv(metrics_csv, eval_rows)
    _write_csv(robust_csv, robust_rows)

    baseline_full = strategy_metrics["baseline_7406_default_tk45_nd4_daily"]["2024_2026_full"]
    summary = {
        "run_id": args.run_id,
        "tracking_uri": args.tracking_uri,
        "artifact_dir": str(artifacts_dir),
        "config_path": str(run_cfg_path),
        "scan_time_utc": _now_utc(),
        "signal_coverage": {
            "start": str(pd.Timestamp(coverage_start).date()),
            "end": str(pd.Timestamp(coverage_end).date()),
            "rows": int(len(base_pred_df)),
            "unique_trade_days": int(pd.Index(coverage_dates).nunique()),
        },
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "slices": [
            {"slice": name, "start": str(pd.Timestamp(st).date()), "end": str(pd.Timestamp(ed).date())}
            for name, st, ed in slices
        ],
        "fixed_family": [x["strategy_id"] for x in strategies if x["is_candidate_family"]],
        "comparator": "baseline_7406_default_tk45_nd4_daily",
        "rank_ensemble_spec": {
            "runs": [x.strip() for x in args.ensemble_runs.split(",") if x.strip()],
            "weights": ensemble_weights,
        },
        "strategy_slice_metrics": eval_rows,
        "robust_ranking": robust_rows,
        "best_robust_rule": best_robust,
        "breakthrough_rule": breakthrough,
        "has_cross_year_breakthrough": bool(breakthrough is not None),
        "thresholds": {
            "legacy_7406_costed_ir": TARGET_COSTED_IR_7406,
            "legacy_7406_costed_annret": TARGET_COSTED_ANNRET_7406,
            "strategy_sota_costed_ir": TARGET_COSTED_IR_SOTA_STRAT,
        },
        "baseline_full_period": baseline_full,
        "artifacts": {
            "metrics_csv": str(metrics_csv),
            "robust_csv": str(robust_csv),
            "summary_json": str(summary_path),
            "candidate_json": str(candidate_path) if breakthrough is not None else None,
        },
    }

    if breakthrough is not None:
        candidate_path.write_text(json.dumps(breakthrough, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
