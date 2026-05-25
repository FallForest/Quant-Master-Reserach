#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRANS_DIR = Path(__file__).resolve().parent
if str(TRANS_DIR) not in sys.path:
    sys.path.insert(0, str(TRANS_DIR))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import Order, TradeDecisionWO
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy
from quant_master.strategy.base import BaseStrategy

import replay_action_reports_cache as action_cache
from factor_augmented_meta_ensemble import BufferedTopkWeightStrategy


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
DEFAULT_SCHEDULE = (
    "base40:2024-01-01:2024-06-30",
    "factor_augmented_meta:2024-07-01:2026-04-30",
)


@dataclass(frozen=True)
class ScheduleWindow:
    action: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class CashWindow:
    start: pd.Timestamp
    end: pd.Timestamp


class RegimeSwitchReplayStrategy(BaseStrategy):
    """Delegate to prebuilt action strategies while sharing one account/executor path."""

    def __init__(
        self,
        *,
        schedule: Sequence[ScheduleWindow],
        action_strategies: Dict[str, BaseStrategy],
        transition_cash_days: int = 0,
        cash_windows: Sequence[CashWindow] = (),
    ):
        super().__init__()
        self.schedule = list(schedule)
        self.action_strategies = dict(action_strategies)
        self._decision_counts = {name: 0 for name in self.action_strategies}
        self.transition_cash_days = int(max(0, transition_cash_days))
        self.cash_windows = list(cash_windows)
        self._active_action: str | None = None
        self._transition_days_left = 0
        self._transition_events: List[Dict[str, Any]] = []
        self._cash_window_events: List[Dict[str, Any]] = []
        self._transition_decision_count = 0
        self._transition_sell_order_count = 0
        self._transition_sell_amount = 0.0

    def reset_common_infra(self, common_infra):
        super().reset_common_infra(common_infra)
        for strategy in self.action_strategies.values():
            strategy.reset_common_infra(common_infra)

    def reset_level_infra(self, level_infra):
        super().reset_level_infra(level_infra)
        for strategy in self.action_strategies.values():
            strategy.reset_level_infra(level_infra)

    def reset(self, level_infra=None, common_infra=None, outer_trade_decision=None, **kwargs):
        super().reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)
        for strategy in self.action_strategies.values():
            strategy.reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)
        self._active_action = None
        self._transition_days_left = 0

    def _action_for_date(self, trade_date: pd.Timestamp) -> str:
        ts = pd.Timestamp(trade_date).normalize()
        for window in self.schedule:
            if window.start <= ts <= window.end:
                return window.action
        raise RuntimeError(f"no scheduled action covers trade date {ts.date()}")

    def _cash_window_for_date(self, trade_date: pd.Timestamp) -> CashWindow | None:
        ts = pd.Timestamp(trade_date).normalize()
        for window in self.cash_windows:
            if window.start <= ts <= window.end:
                return window
        return None

    def _make_liquidation_decision(self, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> TradeDecisionWO:
        current_amount = self.trade_position.get_stock_amount_dict()
        orders = [
            Order(
                stock_id=stock_id,
                amount=float(amount),
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            for stock_id, amount in sorted(current_amount.items())
            if float(amount) > 1e-8
        ]
        self._transition_decision_count += 1
        self._transition_sell_order_count += len(orders)
        self._transition_sell_amount += float(sum(o.amount for o in orders))
        return TradeDecisionWO(orders, self)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        action = self._action_for_date(pd.Timestamp(trade_start_time))
        if self._active_action is None:
            self._active_action = action
        elif action != self._active_action:
            prev_action = self._active_action
            self._active_action = action
            self._transition_days_left = self.transition_cash_days
            if self.transition_cash_days > 0:
                self._transition_events.append(
                    {
                        "date": str(pd.Timestamp(trade_start_time).date()),
                        "from_action": prev_action,
                        "to_action": action,
                        "transition_cash_days": self.transition_cash_days,
                    }
                )

        if self._transition_days_left > 0:
            self._transition_days_left -= 1
            return self._make_liquidation_decision(trade_start_time, trade_end_time)

        cash_window = self._cash_window_for_date(pd.Timestamp(trade_start_time))
        if cash_window is not None:
            self._cash_window_events.append(
                {
                    "date": str(pd.Timestamp(trade_start_time).date()),
                    "start": str(cash_window.start.date()),
                    "end": str(cash_window.end.date()),
                    "scheduled_action": action,
                }
            )
            return self._make_liquidation_decision(trade_start_time, trade_end_time)

        self._decision_counts[action] = self._decision_counts.get(action, 0) + 1
        strategy = self.action_strategies[action]
        decision = strategy.generate_trade_decision(execute_result=execute_result)
        if isinstance(decision, TradeDecisionWO):
            decision.strategy = self
        return decision


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _parse_schedule(values: Sequence[str]) -> List[ScheduleWindow]:
    windows: List[ScheduleWindow] = []
    for raw in values:
        parts = [x.strip() for x in raw.split(":")]
        if len(parts) != 3:
            raise ValueError(f"schedule entry must be action:start:end, got {raw!r}")
        action, start, end = parts
        windows.append(ScheduleWindow(action=action, start=pd.Timestamp(start), end=pd.Timestamp(end)))
    return sorted(windows, key=lambda x: x.start)


def _parse_cash_windows(values: Sequence[str]) -> List[CashWindow]:
    windows: List[CashWindow] = []
    for raw in values:
        parts = [x.strip() for x in raw.split(":")]
        if len(parts) != 2:
            raise ValueError(f"cash-window entry must be start:end, got {raw!r}")
        start, end = parts
        windows.append(CashWindow(start=pd.Timestamp(start), end=pd.Timestamp(end)))
    return sorted(windows, key=lambda x: x.start)


def _extract_day_result(portfolio_metric: Dict[str, Any]) -> Tuple[str, pd.DataFrame, Dict[pd.Timestamp, Any]]:
    if "1day" in portfolio_metric:
        freq = "1day"
    elif "day" in portfolio_metric:
        freq = "day"
    else:
        freq = next(iter(portfolio_metric.keys()))
    report, positions = portfolio_metric[freq]
    return freq, report, positions


def _calc_metrics(report: pd.DataFrame) -> Dict[str, float]:
    risk = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="1day")
    return {
        "annret": float(risk.loc["annualized_return", "risk"]),
        "ir": float(risk.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].mean()),
        "rows": int(len(report)),
    }


def _slice_report(report: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= start) & (idx <= end)].copy()


def _make_action_strategy(action: action_cache.ActionSpec, base_strategy_kwargs: Dict[str, Any]) -> BaseStrategy:
    if action.strategy == "TopkDropoutStrategy":
        kwargs = copy.deepcopy(base_strategy_kwargs)
        kwargs.pop("signal", None)
        kwargs.update(action.strategy_kwargs)
        return TopkDropoutStrategy(signal=action.signal, **kwargs)
    if action.strategy == "BufferedTopkWeightStrategy":
        return BufferedTopkWeightStrategy(signal=action.signal.to_frame("score"), **action.strategy_kwargs)
    raise ValueError(f"unsupported action strategy: {action.strategy}")


def _build_exchange_kwargs(backtest_cfg: Dict[str, Any], executor_cfg: Dict[str, Any], start: str, end: str) -> Dict[str, Any]:
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = OPEN_COST
    exchange_kwargs["close_cost"] = CLOSE_COST
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs["exchange"] = get_exchange(
        freq=freq,
        start_time=start,
        end_time=end,
        deal_price=exchange_kwargs.get("deal_price", "close"),
        limit_threshold=exchange_kwargs.get("limit_threshold", 0.095),
        open_cost=OPEN_COST,
        close_cost=CLOSE_COST,
        min_cost=exchange_kwargs.get("min_cost", 5),
    )
    return exchange_kwargs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continuous trade replay probe for date-switched cached actions.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--schedule", nargs="*", default=list(DEFAULT_SCHEDULE))
    p.add_argument("--output-prefix", default="regime_switch_strategy_replay_probe")
    p.add_argument(
        "--transition-cash-days",
        type=int,
        default=0,
        help=(
            "On each scheduled action change, emit real sell orders for all current holdings for this many "
            "trading days before allowing the new action to trade. Costs/tradability still come from Exchange."
        ),
    )
    p.add_argument(
        "--liquidate-on-switch",
        action="store_true",
        help="Alias for --transition-cash-days 1 unless a larger value is supplied.",
    )
    p.add_argument(
        "--cash-window",
        nargs="*",
        default=[],
        help=(
            "Optional fixed cash windows as start:end. During these dates the probe emits only real SELL orders "
            "for remaining holdings and does not call the scheduled action strategy."
        ),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent / f"{args.output_prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    tracking_dir = action_cache._parse_tracking_dir(args.tracking_uri)
    schedule = _parse_schedule(args.schedule)
    cash_windows = _parse_cash_windows(args.cash_window)

    base_run_dir = action_cache._find_run_dir(tracking_dir, action_cache.RUNS["7406e470"])
    base_cfg = action_cache._load_config(base_run_dir / "artifacts" / "config")
    action_cache._init_quant_master(base_cfg)
    base_port_cfg = action_cache._extract_port_config(base_cfg)
    actions, load_meta = action_cache._load_actions(tracking_dir, args.start_date, args.end_date)

    missing = sorted({w.action for w in schedule}.difference(actions))
    if missing:
        raise KeyError(f"scheduled actions are unavailable: {missing}")

    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    action_strategies = {name: _make_action_strategy(actions[name], base_strategy_kwargs) for name in {w.action for w in schedule}}
    transition_cash_days = int(max(args.transition_cash_days, 1 if args.liquidate_on_switch else 0))
    strategy = RegimeSwitchReplayStrategy(
        schedule=schedule,
        action_strategies=action_strategies,
        transition_cash_days=transition_cash_days,
        cash_windows=cash_windows,
    )

    backtest_cfg = copy.deepcopy(base_port_cfg["backtest"])
    backtest_cfg["start_time"] = args.start_date
    backtest_cfg["end_time"] = args.end_date
    executor_cfg = base_port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exchange_kwargs = _build_exchange_kwargs(backtest_cfg, executor_cfg, args.start_date, args.end_date)

    t0 = time.perf_counter()
    portfolio_metric_dict, indicator_dict = run_backtest(
        start_time=args.start_date,
        end_time=args.end_date,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0

    freq, report, positions = _extract_day_result(portfolio_metric_dict)
    metrics = _calc_metrics(report)
    metrics["elapsed_sec"] = float(elapsed)

    window_metrics: List[Dict[str, Any]] = []
    for window in schedule:
        part = _slice_report(report, window.start, window.end)
        row = {"action": window.action, "start": str(window.start.date()), "end": str(window.end.date())}
        row.update(_calc_metrics(part))
        window_metrics.append(row)

    report_pkl = out_dir / f"{args.output_prefix}_report_{stamp}.pkl"
    report_csv = out_dir / f"{args.output_prefix}_report_{stamp}.csv"
    positions_pkl = out_dir / f"{args.output_prefix}_positions_{stamp}.pkl"
    indicators_pkl = out_dir / f"{args.output_prefix}_indicators_{stamp}.pkl"
    summary_path = out_dir / f"{args.output_prefix}_summary_{stamp}.json"

    _dump_pickle(report_pkl, report)
    report.to_csv(report_csv)
    _dump_pickle(positions_pkl, positions)
    _dump_pickle(indicators_pkl, indicator_dict)

    summary = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "continuous trade-level replay of date-switched cached action signals; no report splicing",
        "tracking_dir": str(tracking_dir),
        "test_period": {"start": args.start_date, "end": args.end_date},
        "costs": {"open": OPEN_COST, "close": CLOSE_COST},
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "hard_gate_pass": bool(metrics["ir"] > HARD_GATE_IR and metrics["annret"] > HARD_GATE_ANNRET),
        "freq": freq,
        "transition_policy": {
            "transition_cash_days": transition_cash_days,
            "liquidate_on_switch": bool(args.liquidate_on_switch),
            "cash_windows": [{"start": str(w.start.date()), "end": str(w.end.date())} for w in cash_windows],
            "note": (
                "Transition uses real SELL orders for inherited holdings; exchange tradability, deal prices, "
                "rounding, and close_cost remain active. It does not mutate positions directly."
            ),
            "events": strategy._transition_events,
            "cash_window_events": strategy._cash_window_events,
            "transition_decision_count": strategy._transition_decision_count,
            "sell_order_count": strategy._transition_sell_order_count,
            "sell_order_amount_requested": strategy._transition_sell_amount,
        },
        "schedule": [
            {"action": w.action, "start": str(w.start.date()), "end": str(w.end.date())} for w in schedule
        ],
        "decision_counts": strategy._decision_counts,
        "action_specs": {
            name: {
                "strategy": actions[name].strategy,
                "strategy_kwargs": actions[name].strategy_kwargs,
                "source_paths": actions[name].source_paths,
                "notes": actions[name].notes,
            }
            for name in sorted({w.action for w in schedule})
        },
        "load_meta": load_meta,
        "metrics": metrics,
        "window_metrics": window_metrics,
        "artifacts": {
            "summary_json": str(summary_path),
            "report_pkl": str(report_pkl),
            "report_csv": str(report_csv),
            "positions_pkl": str(positions_pkl),
            "indicators_pkl": str(indicators_pkl),
            "out_dir": str(out_dir),
        },
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "metrics": metrics, "hard_gate_pass": summary["hard_gate_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
