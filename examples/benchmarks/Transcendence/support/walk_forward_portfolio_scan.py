#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri_in_config
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase
from examples.benchmarks.Transcendence._bootstrap import init_quant_master_from_config, load_config_with_resolved_provider


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TARGET_COSTED_IR_7406 = 2.799983676714277
TARGET_COSTED_IR_SOTA_STRAT = 3.0230019401859436


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


class DrawdownAwareScheduledTopkDropoutStrategy(ScheduledTopkDropoutStrategy):
    def __init__(
        self,
        *,
        dd_trigger: float = 0.05,
        dd_full: float = 0.12,
        min_risk_scale: float = 0.4,
        min_n_drop: int = 0,
        n_drop_scale_mode: str = "floor",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dd_trigger = max(0.0, float(dd_trigger))
        self.dd_full = max(self.dd_trigger + 1e-6, float(dd_full))
        self.min_risk_scale = float(np.clip(min_risk_scale, 0.0, 1.0))
        self.min_n_drop = int(max(0, min_n_drop))
        self.n_drop_scale_mode = str(n_drop_scale_mode).lower()
        self._peak_equity: Optional[float] = None
        self._latest_risk_scale: float = 1.0

    def _update_risk_scale(self) -> None:
        try:
            eq = float(self.trade_position.calculate_value())
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(eq) or eq <= 0:
            return
        if self._peak_equity is None or eq > self._peak_equity:
            self._peak_equity = eq
        if self._peak_equity is None or self._peak_equity <= 0:
            self._latest_risk_scale = 1.0
            return
        drawdown = max(0.0, 1.0 - eq / self._peak_equity)
        if drawdown <= self.dd_trigger:
            scale = 1.0
        elif drawdown >= self.dd_full:
            scale = self.min_risk_scale
        else:
            frac = (drawdown - self.dd_trigger) / (self.dd_full - self.dd_trigger)
            scale = 1.0 - frac * (1.0 - self.min_risk_scale)
        self._latest_risk_scale = float(np.clip(scale, self.min_risk_scale, 1.0))

    def _scale_n_drop(self, n_drop: int) -> int:
        raw = max(0.0, float(n_drop) * self._latest_risk_scale)
        if self.n_drop_scale_mode == "ceil":
            scaled = int(np.ceil(raw))
        elif self.n_drop_scale_mode == "round":
            scaled = int(np.rint(raw))
        else:
            scaled = int(np.floor(raw))
        return int(np.clip(max(self.min_n_drop, scaled), 0, max(0, n_drop)))

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return TradeDecisionWO([], self)

        self._update_risk_scale()
        orig_n_drop = int(self.n_drop)
        orig_risk_degree = float(self.risk_degree)
        scheduled_n_drop = orig_n_drop
        if self.n_drop_schedule:
            scheduled_n_drop = int(self.n_drop_schedule[self._rebalance_count % len(self.n_drop_schedule)])
        self._rebalance_count += 1

        scaled_n_drop = self._scale_n_drop(scheduled_n_drop)
        if scaled_n_drop <= 0:
            return TradeDecisionWO([], self)
        try:
            self.n_drop = scaled_n_drop
            self.risk_degree = float(np.clip(orig_risk_degree * self._latest_risk_scale, 0.0, 1.0))
            return TopkDropoutStrategy.generate_trade_decision(self, execute_result=execute_result)
        finally:
            self.n_drop = orig_n_drop
            self.risk_degree = orig_risk_degree


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


class ConfidenceScaledBufferedTopkStrategy(BufferedTopkWeightStrategy):
    def __init__(
        self,
        *,
        confidence_gap_low: float = 0.0,
        confidence_gap_high: float = 0.5,
        confidence_min_risk: float = 0.4,
        confidence_compare_k: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_gap_low = float(confidence_gap_low)
        self.confidence_gap_high = max(self.confidence_gap_low + 1e-6, float(confidence_gap_high))
        self.confidence_min_risk = float(np.clip(confidence_min_risk, 0.0, 1.0))
        self.confidence_compare_k = int(max(self.topk + 1, confidence_compare_k))
        self._base_risk_degree = float(getattr(self, "risk_degree", 0.95))
        self._latest_risk_scale = 1.0

    def _update_confidence_scale(self, score) -> None:
        score_s = self._to_series(score)
        if score_s.empty or len(score_s) <= self.topk:
            self._latest_risk_scale = self.confidence_min_risk
            return
        ranked = score_s.sort_values(ascending=False)
        compare_end = min(len(ranked), self.confidence_compare_k)
        top = ranked.iloc[: self.topk]
        compare = ranked.iloc[self.topk : compare_end]
        if compare.empty:
            self._latest_risk_scale = self.confidence_min_risk
            return
        denom = float(ranked.iloc[:compare_end].std(ddof=0))
        if not np.isfinite(denom) or denom <= 1e-12:
            self._latest_risk_scale = self.confidence_min_risk
            return
        gap = (float(top.mean()) - float(compare.mean())) / denom
        raw = (gap - self.confidence_gap_low) / (self.confidence_gap_high - self.confidence_gap_low)
        conf = float(np.clip(raw, 0.0, 1.0))
        self._latest_risk_scale = self.confidence_min_risk + conf * (1.0 - self.confidence_min_risk)

    def get_risk_degree(self, trade_step=None):
        return float(np.clip(self._base_risk_degree * self._latest_risk_scale, 0.0, 1.0))

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        self._update_confidence_scale(score)
        return super().generate_target_weight_position(score, current, trade_start_time, trade_end_time)


class DrawdownAwareBufferedTopkWeightStrategy(BufferedTopkWeightStrategy):
    def __init__(
        self,
        *,
        dd_trigger: float = 0.05,
        dd_full: float = 0.12,
        min_risk_scale: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dd_trigger = max(0.0, float(dd_trigger))
        self.dd_full = max(self.dd_trigger + 1e-6, float(dd_full))
        self.min_risk_scale = float(np.clip(min_risk_scale, 0.0, 1.0))
        self._peak_equity: Optional[float] = None
        self._latest_risk_scale: float = 1.0
        self._base_risk_degree = float(getattr(self, "risk_degree", 0.95))

    def _update_risk_scale(self, current) -> None:
        try:
            eq = float(current.calculate_value())
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(eq) or eq <= 0:
            return
        if self._peak_equity is None or eq > self._peak_equity:
            self._peak_equity = eq
        if self._peak_equity is None or self._peak_equity <= 0:
            self._latest_risk_scale = 1.0
            return
        drawdown = max(0.0, 1.0 - eq / self._peak_equity)
        if drawdown <= self.dd_trigger:
            scale = 1.0
        elif drawdown >= self.dd_full:
            scale = self.min_risk_scale
        else:
            frac = (drawdown - self.dd_trigger) / (self.dd_full - self.dd_trigger)
            scale = 1.0 - frac * (1.0 - self.min_risk_scale)
        self._latest_risk_scale = float(np.clip(scale, self.min_risk_scale, 1.0))

    def get_risk_degree(self, trade_step=None):
        return float(np.clip(self._base_risk_degree * self._latest_risk_scale, 0.0, 1.0))

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        self._update_risk_scale(current)
        return super().generate_target_weight_position(score, current, trade_start_time, trade_end_time)


@dataclass
class ScanResult:
    protocol: str
    fold_id: str
    split: str
    family: str
    tag: str
    topk: int
    hold_topk: int
    n_drop: int
    hold_thresh: int
    rebalance_mode: str
    rebalance_interval: int
    n_drop_schedule: str
    weight_mode: str
    score_power: float
    confidence_gap_low: float
    confidence_gap_high: float
    confidence_min_risk: float
    confidence_compare_k: int
    dd_trigger: float
    dd_full: float
    min_risk_scale: float
    min_n_drop: int
    n_drop_scale_mode: str
    start_time: str
    end_time: str
    costed_annret: float
    costed_ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float


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
    return load_config_with_resolved_provider(
        path,
        loader=lambda config_path: yaml.safe_load(config_path.read_text(encoding="utf-8")),
        binary_fallback=_load_pickle,
    )


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
    init_quant_master_from_config(config, base_dir=REPO_ROOT, region="cn")


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


def _parse_int_list(text: str) -> List[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(text: str) -> List[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _parse_mode_list(text: str) -> List[str]:
    out = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty mode list")
    return out


def _parse_schedule_specs(text: str) -> List[List[int]]:
    specs: List[List[int]] = []
    for seg in text.split(";"):
        seg = seg.strip().lower()
        if not seg or seg == "none":
            specs.append([])
            continue
        specs.append([int(x.strip()) for x in seg.split(",") if x.strip()])
    return specs or [[]]


def _dedup_keep_order(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in combos:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _clip_period(start: pd.Timestamp, end: pd.Timestamp, min_dt: pd.Timestamp, max_dt: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    left = max(pd.Timestamp(start), pd.Timestamp(min_dt))
    right = min(pd.Timestamp(end), pd.Timestamp(max_dt))
    if left > right:
        return None
    return left, right


def _pred_date_values(pred_df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    return pd.to_datetime(idx)


def _count_trade_days(pred_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    dates = _pred_date_values(pred_df)
    m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not m.any():
        return 0
    return int(pd.Index(dates[m]).nunique())


def _slice_pred(pred_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        lv = pd.to_datetime(idx.get_level_values(0))
        m = (lv >= pd.Timestamp(start)) & (lv <= pd.Timestamp(end))
        return pred_df.loc[m]
    m = (pd.to_datetime(idx) >= pd.Timestamp(start)) & (pd.to_datetime(idx) <= pd.Timestamp(end))
    return pred_df.loc[m]


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
            "hold_thresh": int(combo.get("hold_thresh", base_strategy_kwargs.get("hold_thresh", 1))),
            "only_tradable": bool(base_strategy_kwargs.get("only_tradable", False)),
            "forbid_all_trade_at_limit": bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
        }
        kwargs.update(common_kwargs)
        return ScheduledTopkDropoutStrategy(**kwargs)
    if combo["family"] == "topk_dropout_derisk":
        kwargs = {
            "topk": int(combo["topk"]),
            "n_drop": int(combo["n_drop"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
            "n_drop_schedule": combo.get("n_drop_schedule", []),
            "method_sell": base_strategy_kwargs.get("method_sell", "bottom"),
            "method_buy": base_strategy_kwargs.get("method_buy", "top"),
            "hold_thresh": int(combo.get("hold_thresh", base_strategy_kwargs.get("hold_thresh", 1))),
            "only_tradable": bool(base_strategy_kwargs.get("only_tradable", False)),
            "forbid_all_trade_at_limit": bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
            "dd_trigger": float(combo["dd_trigger"]),
            "dd_full": float(combo["dd_full"]),
            "min_risk_scale": float(combo["min_risk_scale"]),
            "min_n_drop": int(combo["min_n_drop"]),
            "n_drop_scale_mode": str(combo["n_drop_scale_mode"]),
        }
        kwargs.update(common_kwargs)
        return DrawdownAwareScheduledTopkDropoutStrategy(**kwargs)
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
    if combo["family"] == "confidence_buffered":
        kwargs = {
            "topk": int(combo["topk"]),
            "hold_topk": int(combo["hold_topk"]),
            "weight_mode": str(combo["weight_mode"]),
            "score_power": float(combo["score_power"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
            "confidence_gap_low": float(combo["confidence_gap_low"]),
            "confidence_gap_high": float(combo["confidence_gap_high"]),
            "confidence_min_risk": float(combo["confidence_min_risk"]),
            "confidence_compare_k": int(combo["confidence_compare_k"]),
        }
        kwargs.update(common_kwargs)
        return ConfidenceScaledBufferedTopkStrategy(**kwargs)
    if combo["family"] == "buffered_weight_derisk":
        kwargs = {
            "topk": int(combo["topk"]),
            "hold_topk": int(combo["hold_topk"]),
            "weight_mode": str(combo["weight_mode"]),
            "score_power": float(combo["score_power"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
            "dd_trigger": float(combo["dd_trigger"]),
            "dd_full": float(combo["dd_full"]),
            "min_risk_scale": float(combo["min_risk_scale"]),
        }
        kwargs.update(common_kwargs)
        return DrawdownAwareBufferedTopkWeightStrategy(**kwargs)
    raise ValueError(f"unsupported family={combo['family']}")


def _json_default(x):
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    raise TypeError(f"cannot serialize type={type(x)}")


def _build_rebalance_profiles(modes: List[str], intervals: List[int], fallback_interval: int) -> List[Tuple[str, int]]:
    profiles: List[Tuple[str, int]] = []
    for mode in modes:
        mode = mode.lower()
        if mode == "interval":
            for v in intervals:
                profiles.append(("interval", int(v)))
        else:
            profiles.append((mode, int(fallback_interval)))
    return _dedup_keep_order_profiles(profiles)


def _dedup_keep_order_profiles(items: Iterable[Tuple[str, int]]) -> List[Tuple[str, int]]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_combos(args) -> List[Dict[str, Any]]:
    topks = _parse_int_list(args.topk_grid)
    hold_gaps = _parse_int_list(args.hold_gap_grid)
    weight_modes = _parse_mode_list(args.weight_modes)
    score_powers = _parse_float_list(args.score_power_grid)
    n_drops = _parse_int_list(args.n_drop_grid)
    hold_threshes = _parse_int_list(args.hold_thresh_grid)
    n_drop_schedules = _parse_schedule_specs(args.n_drop_schedules)
    rebalance_profiles = _build_rebalance_profiles(
        _parse_mode_list(args.rebalance_modes),
        _parse_int_list(args.rebalance_interval_grid),
        args.rebalance_interval,
    )

    combos: List[Dict[str, Any]] = []
    for topk in topks:
        for gap in hold_gaps:
            hold_topk = topk + max(0, gap)
            for mode, interval in rebalance_profiles:
                for wm in weight_modes:
                    for sp in score_powers:
                        if wm == "equal":
                            sp = 1.0
                        combos.append(
                            {
                                "family": "buffered_weight",
                                "tag": f"buffered_tk{topk}_hk{hold_topk}_{wm}_{mode}_ri{interval}_sp{sp:.2f}",
                                "topk": int(topk),
                                "hold_topk": int(hold_topk),
                                "weight_mode": wm,
                                "score_power": float(sp),
                                "rebalance_mode": mode,
                                "rebalance_interval": int(interval),
                            }
                        )

    for topk in topks:
        for n_drop in n_drops:
            for hold_thresh in hold_threshes:
                for mode, interval in rebalance_profiles:
                    for sched in n_drop_schedules:
                        combos.append(
                            {
                                "family": "topk_dropout_sched",
                                "tag": (
                                    f"topkdrop_tk{topk}_nd{n_drop}_ht{hold_thresh}_{mode}"
                                    f"_ri{interval}_sch{','.join(map(str, sched)) or 'none'}"
                                ),
                                "topk": int(topk),
                                "n_drop": int(n_drop),
                                "hold_thresh": int(hold_thresh),
                                "rebalance_mode": mode,
                                "rebalance_interval": int(interval),
                                "n_drop_schedule": list(sched),
                            }
                        )

    if args.enable_topk_derisk:
        topk_derisk_topks = _parse_int_list(args.topk_derisk_topk_grid)
        topk_derisk_n_drops = _parse_int_list(args.topk_derisk_n_drop_grid)
        topk_derisk_hold_threshes = _parse_int_list(args.topk_derisk_hold_thresh_grid)
        topk_derisk_schedules = _parse_schedule_specs(args.topk_derisk_n_drop_schedules)
        topk_derisk_profiles = _build_rebalance_profiles(
            _parse_mode_list(args.topk_derisk_rebalance_modes),
            _parse_int_list(args.topk_derisk_rebalance_interval_grid),
            args.rebalance_interval,
        )
        dd_triggers = _parse_float_list(args.topk_derisk_dd_trigger_grid)
        dd_fulls = _parse_float_list(args.topk_derisk_dd_full_grid)
        min_risks = _parse_float_list(args.topk_derisk_min_risk_grid)
        min_n_drops = _parse_int_list(args.topk_derisk_min_n_drop_grid)
        scale_modes = _parse_mode_list(args.topk_derisk_n_drop_scale_modes)
        for topk in topk_derisk_topks:
            for n_drop in topk_derisk_n_drops:
                for hold_thresh in topk_derisk_hold_threshes:
                    for mode, interval in topk_derisk_profiles:
                        for sched in topk_derisk_schedules:
                            for dd_trig in dd_triggers:
                                for dd_full in dd_fulls:
                                    if dd_full <= dd_trig:
                                        continue
                                    for min_risk in min_risks:
                                        for min_n_drop in min_n_drops:
                                            for scale_mode in scale_modes:
                                                combos.append(
                                                    {
                                                        "family": "topk_dropout_derisk",
                                                        "tag": (
                                                            f"topkdrop_derisk_tk{topk}_nd{n_drop}_ht{hold_thresh}_{mode}"
                                                            f"_ri{interval}_sch{','.join(map(str, sched)) or 'none'}"
                                                            f"_dd{dd_trig:.2f}-{dd_full:.2f}_mr{min_risk:.2f}"
                                                            f"_mnd{min_n_drop}_{scale_mode}"
                                                        ),
                                                        "topk": int(topk),
                                                        "n_drop": int(n_drop),
                                                        "hold_thresh": int(hold_thresh),
                                                        "rebalance_mode": mode,
                                                        "rebalance_interval": int(interval),
                                                        "n_drop_schedule": list(sched),
                                                        "dd_trigger": float(dd_trig),
                                                        "dd_full": float(dd_full),
                                                        "min_risk_scale": float(min_risk),
                                                        "min_n_drop": int(min_n_drop),
                                                        "n_drop_scale_mode": scale_mode,
                                                    }
                                                )

    if args.enable_confidence:
        confidence_topks = _parse_int_list(args.confidence_topk_grid)
        confidence_gaps = _parse_int_list(args.confidence_hold_gap_grid)
        confidence_weight_modes = _parse_mode_list(args.confidence_weight_modes)
        confidence_score_powers = _parse_float_list(args.confidence_score_power_grid)
        confidence_profiles = _build_rebalance_profiles(
            _parse_mode_list(args.confidence_rebalance_modes),
            _parse_int_list(args.confidence_rebalance_interval_grid),
            args.rebalance_interval,
        )
        gap_lows = _parse_float_list(args.confidence_gap_low_grid)
        gap_highs = _parse_float_list(args.confidence_gap_high_grid)
        min_risks = _parse_float_list(args.confidence_min_risk_grid)
        compare_ks = _parse_int_list(args.confidence_compare_k_grid)
        for topk in confidence_topks:
            for gap in confidence_gaps:
                hold_topk = topk + max(0, gap)
                for mode, interval in confidence_profiles:
                    for wm in confidence_weight_modes:
                        for sp in confidence_score_powers:
                            if wm == "equal":
                                sp = 1.0
                            for gap_low in gap_lows:
                                for gap_high in gap_highs:
                                    if gap_high <= gap_low:
                                        continue
                                    for min_risk in min_risks:
                                        for compare_k in compare_ks:
                                            combos.append(
                                                {
                                                    "family": "confidence_buffered",
                                                    "tag": (
                                                        f"confbuf_tk{topk}_hk{hold_topk}_{wm}_{mode}_ri{interval}"
                                                        f"_gl{gap_low:.2f}_gh{gap_high:.2f}_mr{min_risk:.2f}"
                                                        f"_ck{compare_k}_sp{sp:.2f}"
                                                    ),
                                                    "topk": int(topk),
                                                    "hold_topk": int(hold_topk),
                                                    "weight_mode": wm,
                                                    "score_power": float(sp),
                                                    "rebalance_mode": mode,
                                                    "rebalance_interval": int(interval),
                                                    "confidence_gap_low": float(gap_low),
                                                    "confidence_gap_high": float(gap_high),
                                                    "confidence_min_risk": float(min_risk),
                                                    "confidence_compare_k": int(compare_k),
                                                }
                                            )

    if args.enable_derisk:
        derisk_topks = _parse_int_list(args.derisk_topk_grid)
        derisk_gaps = _parse_int_list(args.derisk_hold_gap_grid)
        derisk_weight_modes = _parse_mode_list(args.derisk_weight_modes)
        derisk_score_powers = _parse_float_list(args.derisk_score_power_grid)
        derisk_profiles = _build_rebalance_profiles(
            _parse_mode_list(args.derisk_rebalance_modes),
            _parse_int_list(args.derisk_rebalance_interval_grid),
            args.rebalance_interval,
        )
        dd_triggers = _parse_float_list(args.derisk_dd_trigger_grid)
        dd_fulls = _parse_float_list(args.derisk_dd_full_grid)
        min_risks = _parse_float_list(args.derisk_min_risk_grid)
        for topk in derisk_topks:
            for gap in derisk_gaps:
                hold_topk = topk + max(0, gap)
                for mode, interval in derisk_profiles:
                    for wm in derisk_weight_modes:
                        for sp in derisk_score_powers:
                            if wm == "equal":
                                sp = 1.0
                            for dd_trig in dd_triggers:
                                for dd_full in dd_fulls:
                                    if dd_full <= dd_trig:
                                        continue
                                    for min_risk in min_risks:
                                        combos.append(
                                            {
                                                "family": "buffered_weight_derisk",
                                                "tag": (
                                                    f"buffered_derisk_tk{topk}_hk{hold_topk}_{wm}_{mode}_ri{interval}"
                                                    f"_dd{dd_trig:.2f}-{dd_full:.2f}_mr{min_risk:.2f}_sp{sp:.2f}"
                                                ),
                                                "topk": int(topk),
                                                "hold_topk": int(hold_topk),
                                                "weight_mode": wm,
                                                "score_power": float(sp),
                                                "rebalance_mode": mode,
                                                "rebalance_interval": int(interval),
                                                "dd_trigger": float(dd_trig),
                                                "dd_full": float(dd_full),
                                                "min_risk_scale": float(min_risk),
                                            }
                                        )
    return _dedup_keep_order(combos)


def _combo_signature(combo: Dict[str, Any]) -> str:
    return json.dumps(combo, sort_keys=True, ensure_ascii=False)


def _eval_combo_period(
    *,
    combo: Dict[str, Any],
    pred_df,
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
        "report_df": report_df,
    }


def _aggregate_reports(report_list: Sequence[pd.DataFrame]) -> Dict[str, float]:
    if not report_list:
        return {"costed_annret": float("nan"), "costed_ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    merged = pd.concat(report_list, axis=0).sort_index()
    annret, ir, maxdd, turnover = _calc_costed_metrics(merged)
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "trade_days": int(len(merged)),
        "start_time": str(pd.Timestamp(merged.index.min()).date()),
        "end_time": str(pd.Timestamp(merged.index.max()).date()),
    }


def _build_default_protocol_folds(pred_df: pd.DataFrame, min_train_days: int, min_test_days: int):
    dates = _pred_date_values(pred_df)
    min_dt, max_dt = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    min_year, max_year = int(min_dt.year), int(max_dt.year)

    rolling_folds: List[Dict[str, Any]] = []
    for y in range(min_year, max_year):
        train_clip = _clip_period(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"), min_dt, max_dt)
        test_clip = _clip_period(pd.Timestamp(f"{y + 1}-01-01"), pd.Timestamp(f"{y + 1}-12-31"), min_dt, max_dt)
        if train_clip is None or test_clip is None:
            continue
        train_days = _count_trade_days(pred_df, train_clip[0], train_clip[1])
        test_days = _count_trade_days(pred_df, test_clip[0], test_clip[1])
        if train_days < min_train_days or test_days < min_test_days:
            continue
        rolling_folds.append(
            {
                "fold_id": f"{y}_to_{y+1}",
                "train_start": train_clip[0],
                "train_end": train_clip[1],
                "test_start": test_clip[0],
                "test_end": test_clip[1],
                "train_days": train_days,
                "test_days": test_days,
            }
        )

    anchor_train_start = pd.Timestamp(f"{min_year}-01-01")
    anchor_train_end = pd.Timestamp(f"{min_year}-12-31")
    anchor_test_start = pd.Timestamp(f"{min_year + 1}-01-01")
    anchor_test_end = max_dt
    train_clip = _clip_period(anchor_train_start, anchor_train_end, min_dt, max_dt)
    test_clip = _clip_period(anchor_test_start, anchor_test_end, min_dt, max_dt)
    anchored_folds: List[Dict[str, Any]] = []
    if train_clip is not None and test_clip is not None:
        train_days = _count_trade_days(pred_df, train_clip[0], train_clip[1])
        test_days = _count_trade_days(pred_df, test_clip[0], test_clip[1])
        if train_days >= min_train_days and test_days >= min_test_days:
            anchored_folds.append(
                {
                    "fold_id": f"{min_year}_fixed_to_{int(test_clip[1].year)}",
                    "train_start": train_clip[0],
                    "train_end": train_clip[1],
                    "test_start": test_clip[0],
                    "test_end": test_clip[1],
                    "train_days": train_days,
                    "test_days": test_days,
                }
            )
    return rolling_folds, anchored_folds


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward and holdout portfolio scan on existing signal artifacts.")
    p.add_argument("--run-id", default=TARGET_RUN_ID)
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--config-path", default="")
    p.add_argument("--output-prefix", default="portfolio_wf_scan")

    p.add_argument("--topk-grid", default="45,50,55")
    p.add_argument("--hold-gap-grid", default="10,20,30")
    p.add_argument("--weight-modes", default="equal,score")
    p.add_argument("--score-power-grid", default="1.0")
    p.add_argument("--n-drop-grid", default="2,4")
    p.add_argument("--hold-thresh-grid", default="1")
    p.add_argument("--n-drop-schedules", default="none;2,3")
    p.add_argument("--rebalance-modes", default="weekly,interval")
    p.add_argument("--rebalance-interval-grid", default="5,10")
    p.add_argument("--rebalance-interval", type=int, default=5)

    p.add_argument("--enable-topk-derisk", action="store_true")
    p.add_argument("--topk-derisk-topk-grid", default="3")
    p.add_argument("--topk-derisk-n-drop-grid", default="1")
    p.add_argument("--topk-derisk-hold-thresh-grid", default="2")
    p.add_argument("--topk-derisk-n-drop-schedules", default="none")
    p.add_argument("--topk-derisk-rebalance-modes", default="daily")
    p.add_argument("--topk-derisk-rebalance-interval-grid", default="5")
    p.add_argument("--topk-derisk-dd-trigger-grid", default="0.08")
    p.add_argument("--topk-derisk-dd-full-grid", default="0.20")
    p.add_argument("--topk-derisk-min-risk-grid", default="0.5")
    p.add_argument("--topk-derisk-min-n-drop-grid", default="0")
    p.add_argument("--topk-derisk-n-drop-scale-modes", default="floor")

    p.add_argument("--enable-confidence", action="store_true")
    p.add_argument("--confidence-topk-grid", default="3")
    p.add_argument("--confidence-hold-gap-grid", default="0")
    p.add_argument("--confidence-weight-modes", default="equal")
    p.add_argument("--confidence-score-power-grid", default="1.0")
    p.add_argument("--confidence-rebalance-modes", default="daily")
    p.add_argument("--confidence-rebalance-interval-grid", default="5")
    p.add_argument("--confidence-gap-low-grid", default="0.0")
    p.add_argument("--confidence-gap-high-grid", default="0.5")
    p.add_argument("--confidence-min-risk-grid", default="0.4")
    p.add_argument("--confidence-compare-k-grid", default="20")

    p.add_argument("--enable-derisk", action="store_true")
    p.add_argument("--derisk-topk-grid", default="50,55")
    p.add_argument("--derisk-hold-gap-grid", default="20,30")
    p.add_argument("--derisk-weight-modes", default="equal")
    p.add_argument("--derisk-score-power-grid", default="1.0")
    p.add_argument("--derisk-rebalance-modes", default="weekly,interval")
    p.add_argument("--derisk-rebalance-interval-grid", default="10")
    p.add_argument("--derisk-dd-trigger-grid", default="0.04,0.06")
    p.add_argument("--derisk-dd-full-grid", default="0.10,0.14")
    p.add_argument("--derisk-min-risk-grid", default="0.4,0.6")

    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--min-train-days", type=int, default=180)
    p.add_argument("--min-test-days", type=int, default=60)
    p.add_argument("--max-combos", type=int, default=0)
    p.add_argument("--run-global-cheat-ref", action="store_true")
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
    pred_df = _load_pickle(pred_path)
    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError(f"pred artifact is not DataFrame: {type(pred_df)}")
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    combos = _build_combos(args)
    if args.max_combos > 0:
        combos = combos[: int(args.max_combos)]
    if not combos:
        raise RuntimeError("no combos generated")

    rolling_folds, anchored_folds = _build_default_protocol_folds(
        pred_df=pred_df,
        min_train_days=int(args.min_train_days),
        min_test_days=int(args.min_test_days),
    )
    if not rolling_folds and not anchored_folds:
        raise RuntimeError("no valid folds generated from pred coverage")

    baseline_combo = {
        "family": "topk_dropout_sched",
        "tag": "baseline_tk45_nd4_daily",
        "topk": 45,
        "n_drop": 4,
        "hold_thresh": 1,
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "n_drop_schedule": [],
    }
    sota_combo = {
        "family": "buffered_weight",
        "tag": "buffered_tk55_hk85_equal_weekly",
        "topk": 55,
        "hold_topk": 85,
        "weight_mode": "equal",
        "score_power": 1.0,
        "rebalance_mode": "weekly",
        "rebalance_interval": 10,
    }

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    eval_cache: Dict[str, Dict[str, Any]] = {}

    def cached_eval(combo: Dict[str, Any], start_time: pd.Timestamp, end_time: pd.Timestamp):
        key = json.dumps(
            {
                "combo": combo,
                "start": str(pd.Timestamp(start_time).date()),
                "end": str(pd.Timestamp(end_time).date()),
                "open_cost": float(args.open_cost),
                "close_cost": float(args.close_cost),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if key not in eval_cache:
            eval_cache[key] = _eval_combo_period(
                combo=combo,
                pred_df=pred_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_time=start_time,
                end_time=end_time,
                exchange_cache=exchange_cache,
            )
        return eval_cache[key]

    protocol_defs = [("rolling_yearly", rolling_folds), ("anchored_holdout", anchored_folds)]
    train_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    protocol_summary: Dict[str, Any] = {}

    for protocol_name, folds in protocol_defs:
        if not folds:
            continue
        selected_reports = []
        baseline_reports = []
        sota_reports = []
        fold_summaries = []
        print(f"[{protocol_name}] folds={len(folds)} combos={len(combos)}", flush=True)
        for fold in folds:
            fold_id = str(fold["fold_id"])
            train_start = pd.Timestamp(fold["train_start"])
            train_end = pd.Timestamp(fold["train_end"])
            test_start = pd.Timestamp(fold["test_start"])
            test_end = pd.Timestamp(fold["test_end"])
            train_eval_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for idx, combo in enumerate(combos, start=1):
                try:
                    ev = cached_eval(combo, train_start, train_end)
                except Exception as e:  # noqa: BLE001
                    failed_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold_id": fold_id,
                            "split": "train",
                            "tag": combo.get("tag", ""),
                            "start_time": str(train_start.date()),
                            "end_time": str(train_end.date()),
                            "error": str(e),
                        }
                    )
                    continue
                row = ScanResult(
                    protocol=protocol_name,
                    fold_id=fold_id,
                    split="train",
                    family=str(combo["family"]),
                    tag=str(combo["tag"]),
                    topk=int(combo.get("topk", 0)),
                    hold_topk=int(combo.get("hold_topk", combo.get("topk", 0))),
                    n_drop=int(combo.get("n_drop", 0)),
                    hold_thresh=int(combo.get("hold_thresh", 0)),
                    rebalance_mode=str(combo.get("rebalance_mode", "")),
                    rebalance_interval=int(combo.get("rebalance_interval", 0)),
                    n_drop_schedule=",".join(str(x) for x in combo.get("n_drop_schedule", [])),
                    weight_mode=str(combo.get("weight_mode", "")),
                    score_power=float(combo.get("score_power", 1.0)),
                    confidence_gap_low=float(combo.get("confidence_gap_low", 0.0)),
                    confidence_gap_high=float(combo.get("confidence_gap_high", 0.0)),
                    confidence_min_risk=float(combo.get("confidence_min_risk", 1.0)),
                    confidence_compare_k=int(combo.get("confidence_compare_k", 0)),
                    dd_trigger=float(combo.get("dd_trigger", 0.0)),
                    dd_full=float(combo.get("dd_full", 0.0)),
                    min_risk_scale=float(combo.get("min_risk_scale", 1.0)),
                    min_n_drop=int(combo.get("min_n_drop", 0)),
                    n_drop_scale_mode=str(combo.get("n_drop_scale_mode", "")),
                    start_time=str(train_start.date()),
                    end_time=str(train_end.date()),
                    costed_annret=float(ev["costed_annret"]),
                    costed_ir=float(ev["costed_ir"]),
                    max_drawdown=float(ev["max_drawdown"]),
                    turnover=float(ev["turnover"]),
                    elapsed_sec=float(ev["elapsed_sec"]),
                )
                train_rows.append(asdict(row))
                train_eval_rows.append((combo, ev))
                if idx % 30 == 0 or idx == len(combos):
                    print(f"[{protocol_name}:{fold_id}] train {idx}/{len(combos)}", flush=True)

            if not train_eval_rows:
                print(f"[{protocol_name}:{fold_id}] no successful train combos, skip fold", flush=True)
                continue
            ranked = sorted(train_eval_rows, key=lambda x: (x[1]["costed_ir"], x[1]["costed_annret"]), reverse=True)
            best_combo = None
            best_train = None
            best_test = None
            for candidate_combo, candidate_train in ranked:
                try:
                    candidate_test = cached_eval(candidate_combo, test_start, test_end)
                except Exception as e:  # noqa: BLE001
                    failed_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold_id": fold_id,
                            "split": "test_selected",
                            "tag": candidate_combo.get("tag", ""),
                            "start_time": str(test_start.date()),
                            "end_time": str(test_end.date()),
                            "error": str(e),
                        }
                    )
                    continue
                best_combo = candidate_combo
                best_train = candidate_train
                best_test = candidate_test
                break
            if best_combo is None or best_train is None or best_test is None:
                print(f"[{protocol_name}:{fold_id}] no candidate survived on test period, skip fold", flush=True)
                continue
            try:
                baseline_test = cached_eval(baseline_combo, test_start, test_end)
                sota_test = cached_eval(sota_combo, test_start, test_end)
            except Exception as e:  # noqa: BLE001
                failed_rows.append(
                    {
                        "protocol": protocol_name,
                        "fold_id": fold_id,
                        "split": "test_comparator",
                        "tag": "baseline_or_sota_fixed",
                        "start_time": str(test_start.date()),
                        "end_time": str(test_end.date()),
                        "error": str(e),
                    }
                )
                print(f"[{protocol_name}:{fold_id}] comparator failed, skip fold", flush=True)
                continue

            selected_reports.append(best_test["report_df"])
            baseline_reports.append(baseline_test["report_df"])
            sota_reports.append(sota_test["report_df"])
            fold_summary = {
                "fold_id": fold_id,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_days": int(fold["train_days"]),
                "test_days": int(fold["test_days"]),
                "selected_combo": best_combo,
                "selected_train": {k: float(best_train[k]) for k in ("costed_annret", "costed_ir", "max_drawdown", "turnover")},
                "selected_test": {k: float(best_test[k]) for k in ("costed_annret", "costed_ir", "max_drawdown", "turnover")},
                "baseline_test": {k: float(baseline_test[k]) for k in ("costed_annret", "costed_ir", "max_drawdown", "turnover")},
                "sota_fixed_test": {k: float(sota_test[k]) for k in ("costed_annret", "costed_ir", "max_drawdown", "turnover")},
            }
            fold_summaries.append(fold_summary)

            selected_row = ScanResult(
                protocol=protocol_name,
                fold_id=fold_id,
                split="test_selected",
                family=str(best_combo["family"]),
                tag=str(best_combo["tag"]),
                topk=int(best_combo.get("topk", 0)),
                hold_topk=int(best_combo.get("hold_topk", best_combo.get("topk", 0))),
                n_drop=int(best_combo.get("n_drop", 0)),
                hold_thresh=int(best_combo.get("hold_thresh", 0)),
                rebalance_mode=str(best_combo.get("rebalance_mode", "")),
                rebalance_interval=int(best_combo.get("rebalance_interval", 0)),
                n_drop_schedule=",".join(str(x) for x in best_combo.get("n_drop_schedule", [])),
                weight_mode=str(best_combo.get("weight_mode", "")),
                score_power=float(best_combo.get("score_power", 1.0)),
                confidence_gap_low=float(best_combo.get("confidence_gap_low", 0.0)),
                confidence_gap_high=float(best_combo.get("confidence_gap_high", 0.0)),
                confidence_min_risk=float(best_combo.get("confidence_min_risk", 1.0)),
                confidence_compare_k=int(best_combo.get("confidence_compare_k", 0)),
                dd_trigger=float(best_combo.get("dd_trigger", 0.0)),
                dd_full=float(best_combo.get("dd_full", 0.0)),
                min_risk_scale=float(best_combo.get("min_risk_scale", 1.0)),
                min_n_drop=int(best_combo.get("min_n_drop", 0)),
                n_drop_scale_mode=str(best_combo.get("n_drop_scale_mode", "")),
                start_time=str(test_start.date()),
                end_time=str(test_end.date()),
                costed_annret=float(best_test["costed_annret"]),
                costed_ir=float(best_test["costed_ir"]),
                max_drawdown=float(best_test["max_drawdown"]),
                turnover=float(best_test["turnover"]),
                elapsed_sec=float(best_test["elapsed_sec"]),
            )
            selected_rows.append(asdict(selected_row))
            print(
                f"[{protocol_name}:{fold_id}] selected={best_combo['tag']} "
                f"trainIR={best_train['costed_ir']:.4f} testIR={best_test['costed_ir']:.4f} "
                f"baselineIR={baseline_test['costed_ir']:.4f} sotaFixedIR={sota_test['costed_ir']:.4f}",
                flush=True,
            )

        if not selected_reports:
            protocol_summary[protocol_name] = {
                "fold_count": 0,
                "folds": [],
                "note": "all folds failed during train/test evaluation",
            }
            continue
        agg_selected = _aggregate_reports(selected_reports)
        agg_baseline = _aggregate_reports(baseline_reports)
        agg_sota = _aggregate_reports(sota_reports)
        protocol_summary[protocol_name] = {
            "fold_count": len(folds),
            "folds": fold_summaries,
            "aggregate_selected_oos": agg_selected,
            "aggregate_baseline_7406_oos": agg_baseline,
            "aggregate_sota_fixed_oos": agg_sota,
            "beats_7406_oos": bool(agg_selected["costed_ir"] > agg_baseline["costed_ir"] + 1e-9),
            "beats_sota_fixed_oos": bool(agg_selected["costed_ir"] > agg_sota["costed_ir"] + 1e-9),
            "beats_legacy_7406_threshold": bool(agg_selected["costed_ir"] > TARGET_COSTED_IR_7406 + 1e-9),
            "beats_3p023_threshold": bool(agg_selected["costed_ir"] > TARGET_COSTED_IR_SOTA_STRAT + 1e-9),
        }

    global_reference = {}
    if args.run_global_cheat_ref:
        dates = _pred_date_values(pred_df)
        full_start = pd.Timestamp(dates.min())
        full_end = pd.Timestamp(dates.max())
        ranked_full = []
        for combo in combos:
            ev = cached_eval(combo, full_start, full_end)
            ranked_full.append((combo, ev))
        ranked_full = sorted(ranked_full, key=lambda x: (x[1]["costed_ir"], x[1]["costed_annret"]), reverse=True)
        best_combo, best_ev = ranked_full[0]
        global_reference = {
            "full_period_start": str(full_start.date()),
            "full_period_end": str(full_end.date()),
            "best_combo": best_combo,
            "best_metrics": {k: float(best_ev[k]) for k in ("costed_annret", "costed_ir", "max_drawdown", "turnover")},
        }

    run_short = args.run_id[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / f"{args.output_prefix}_train_{run_short}_{stamp}.csv"
    selected_csv = out_dir / f"{args.output_prefix}_selected_{run_short}_{stamp}.csv"
    failed_csv = out_dir / f"{args.output_prefix}_failed_{run_short}_{stamp}.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(selected_csv, selected_rows)
    _write_csv(failed_csv, failed_rows)

    date_values = _pred_date_values(pred_df)
    summary = {
        "run_id": args.run_id,
        "tracking_uri": args.tracking_uri,
        "artifact_dir": str(artifacts_dir),
        "config_path": str(run_cfg_path),
        "scan_time_utc": _now_utc(),
        "signal_coverage": {
            "start": str(pd.Timestamp(date_values.min()).date()),
            "end": str(pd.Timestamp(date_values.max()).date()),
            "rows": int(len(pred_df)),
            "unique_trade_days": int(pd.Index(date_values).nunique()),
        },
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "combo_count": len(combos),
        "failed_eval_count": len(failed_rows),
        "protocols": protocol_summary,
        "global_full_period_reference": global_reference,
        "thresholds": {
            "legacy_7406_costed_ir": TARGET_COSTED_IR_7406,
            "strategy_sota_costed_ir": TARGET_COSTED_IR_SOTA_STRAT,
        },
        "artifacts": {
            "train_csv": str(train_csv),
            "selected_csv": str(selected_csv),
            "failed_csv": str(failed_csv),
        },
        "failed_examples": failed_rows[:30],
    }
    summary_path = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

