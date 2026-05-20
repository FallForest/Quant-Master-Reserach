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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


SOTA_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
SOTA_IR = 2.799983676714277
SOTA_ANNRET = 0.24466463608994535

TARGET_RUNS = {
    "gru_bcbecf55": "bcbecf55a3924357ba93fc55b1140e99",
    "metalabel_top_bottom_29864": "29864d9c5d00463b9fdbc065c10b0093",
    "metalabel_rank_4a98": "4a98f99bdb6848bab789ff6c46d0a1ff",
}

RUN_ALIAS = {
    "7406": SOTA_RUN_ID,
    "7406e470": SOTA_RUN_ID,
    "bcbecf55": "bcbecf55a3924357ba93fc55b1140e99",
    "29864": "29864d9c5d00463b9fdbc065c10b0093",
    "4a98": "4a98f99bdb6848bab789ff6c46d0a1ff",
}


@dataclass
class EvalResult:
    signal_key: str
    run_id: str
    scenario: str
    transform: str
    blend: str
    family: str
    rebalance_mode: str
    topk: int
    n_drop: int
    hold_topk: int
    costed_annret: float
    costed_ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float
    error: str = ""


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
        rebalance_mode: str = "weekly",
        rebalance_interval: int = 1,
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


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _resolve_run_id(token: str) -> str:
    t = token.strip()
    if t in RUN_ALIAS:
        return RUN_ALIAS[t]
    if len(t) in {8, 32}:
        return t
    raise ValueError(f"unknown run token: {token}")


def _find_run_dir(tracking_dir: Path, run_id_or_token: str) -> Path:
    run_id = _resolve_run_id(run_id_or_token)
    candidates = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not candidates:
        if len(run_id) == 8:
            candidates = [p for p in tracking_dir.glob(f"*/{run_id}*") if (p / "artifacts").exists()]
    if not candidates:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id_or_token}")
    if len(candidates) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
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
    raise KeyError("cannot find port_analysis_config or task.record[PortAnaRecord].kwargs.config")


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


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


def _cross_section_rank(score: pd.Series) -> pd.Series:
    idx = score.index
    if isinstance(idx, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(idx).rank(method="average", pct=True)


def _smooth_by_instrument(score: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return score
    if not isinstance(score.index, pd.MultiIndex):
        return score.rolling(window, min_periods=1).mean()
    return score.groupby(level=1, group_keys=False).apply(lambda x: x.rolling(window, min_periods=1).mean())


def _build_transforms(base_pred: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    s = base_pred["score"].astype(float)
    rank = _cross_section_rank(s)
    out: Dict[str, pd.DataFrame] = {
        "raw": s.to_frame("score"),
        "inverted": (-s).to_frame("score"),
        "winsor_rank": rank.clip(0.02, 0.98).to_frame("score"),
        "ls_spread": (2.0 * rank - 1.0).to_frame("score"),
        "exclude_middle60": s.where((rank <= 0.2) | (rank >= 0.8)).to_frame("score"),
        "top_decile_only": s.where(rank >= 0.9).to_frame("score"),
        "smooth3": _smooth_by_instrument(s, 3).to_frame("score"),
        "smooth5": _smooth_by_instrument(s, 5).to_frame("score"),
    }
    return out


def _blend_rank_scores(base_df: pd.DataFrame, cand_df: pd.DataFrame, base_weight: float, cand_weight: float) -> pd.DataFrame:
    base_rank = _cross_section_rank(base_df["score"].astype(float))
    cand_rank = _cross_section_rank(cand_df["score"].astype(float))
    panel = pd.concat([base_rank.rename("base"), cand_rank.rename("cand")], axis=1)
    w = pd.Series({"base": float(base_weight), "cand": float(cand_weight)})
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


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


def _build_strategy_object(combo: Dict[str, Any], pred_df, base_strategy_kwargs: Dict[str, Any]):
    common_kwargs: Dict[str, Any] = {"signal": pred_df}
    if "risk_degree" in base_strategy_kwargs:
        common_kwargs["risk_degree"] = float(base_strategy_kwargs["risk_degree"])
    if combo["family"] == "topk_dropout":
        kwargs = {
            "topk": int(combo["topk"]),
            "n_drop": int(combo["n_drop"]),
            "rebalance_mode": str(combo["rebalance_mode"]),
            "rebalance_interval": int(combo["rebalance_interval"]),
            "n_drop_schedule": [],
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
) -> Dict[str, float]:
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
    if pred_slice.empty:
        raise ValueError(f"empty signal slice in {start_time} ~ {end_time}")

    strategy_obj = _build_strategy_object(combo=combo, pred_df=pred_slice, base_strategy_kwargs=base_strategy_kwargs)

    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
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
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    report_df = _get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report_df)
    return {
        "costed_annret": annret,
        "costed_ir": ir,
        "max_drawdown": maxdd,
        "turnover": turnover,
        "elapsed_sec": elapsed,
    }


def _build_combo_space(topk_grid: Sequence[int], n_drop_grid: Sequence[int], hold_buffers: Sequence[int]) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for topk in topk_grid:
        for n_drop in n_drop_grid:
            if n_drop >= topk:
                continue
            for mode in ("daily", "weekly"):
                combos.append(
                    {
                        "family": "topk_dropout",
                        "rebalance_mode": mode,
                        "rebalance_interval": 1,
                        "topk": int(topk),
                        "n_drop": int(n_drop),
                        "hold_topk": int(topk),
                    }
                )
        for buf in hold_buffers:
            combos.append(
                {
                    "family": "buffered_weight",
                    "rebalance_mode": "weekly",
                    "rebalance_interval": 1,
                    "topk": int(topk),
                    "n_drop": 0,
                    "hold_topk": int(topk + buf),
                    "weight_mode": "equal",
                    "score_power": 1.0,
                }
            )
    return combos


def _split_ints(text: str) -> List[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty int list")
    return out


def _split_floats(text: str) -> List[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_diagnostics(sig_df: pd.DataFrame, base_df: pd.DataFrame) -> Dict[str, float]:
    score = sig_df["score"].astype(float)
    base = base_df["score"].astype(float)
    nonnull_ratio = float(score.notna().mean())

    if isinstance(score.index, pd.MultiIndex):
        g = score.groupby(level=0)
        p90 = g.quantile(0.9)
        p10 = g.quantile(0.1)
        spread = float((p90 - p10).mean())
        by_inst = score.groupby(level=1)
        lag1_s = pd.to_numeric(by_inst.apply(lambda x: x.autocorr(lag=1)), errors="coerce").dropna()
        lag1 = float(lag1_s.mean()) if not lag1_s.empty else float("nan")
    else:
        spread = float(score.quantile(0.9) - score.quantile(0.1))
        lag_raw = score.autocorr(lag=1)
        lag1 = float(lag_raw) if lag_raw is not None and pd.notna(lag_raw) else float("nan")

    panel = pd.concat(
        [
            _cross_section_rank(score).rename("sig"),
            _cross_section_rank(base).rename("base"),
        ],
        axis=1,
    ).dropna()
    if panel.empty:
        corr = float("nan")
    elif isinstance(panel.index, pd.MultiIndex):
        corr_s = pd.to_numeric(panel.groupby(level=0).apply(lambda x: x["sig"].corr(x["base"])), errors="coerce").dropna()
        corr = float(corr_s.mean()) if not corr_s.empty else float("nan")
    else:
        corr_raw = panel["sig"].corr(panel["base"])
        corr = float(corr_raw) if corr_raw is not None and pd.notna(corr_raw) else float("nan")

    return {
        "nonnull_ratio": nonnull_ratio,
        "avg_p90_p10_spread": spread,
        "lag1_autocorr": lag1,
        "avg_cs_rank_corr_vs_7406": corr,
    }


def _year_slices(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    out: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for year in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{year}-01-01"), start)
        ye = min(pd.Timestamp(f"{year}-12-31"), end)
        if ys > ye:
            continue
        tag = f"{year}_ytd" if ye < pd.Timestamp(f"{year}-12-31") else str(year)
        out.append((tag, ys, ye))
    out.append(("2024_2026_full", start, end))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert high-RankIC signals into portfolio rules or prove failure.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=SOTA_RUN_ID)
    p.add_argument(
        "--signal-runs",
        default="gru_bcbecf55:bcbecf55,metalabel_top_bottom_29864:29864,metalabel_rank_4a98:4a98",
        help="comma separated key:run_token",
    )
    p.add_argument("--topk-grid", default="40,45,50,55")
    p.add_argument("--n-drop-grid", default="2,3,4")
    p.add_argument("--hold-buffers", default="20,30")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--blend-candidate-weights", default="0.2,0.3")
    p.add_argument("--blend-transforms", default="raw,inverted,winsor_rank,smooth3")
    p.add_argument("--output-prefix", default="signal_portfolio_conversion")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    combo_space = _build_combo_space(
        topk_grid=_split_ints(args.topk_grid),
        n_drop_grid=_split_ints(args.n_drop_grid),
        hold_buffers=_split_ints(args.hold_buffers),
    )
    blend_weights = _split_floats(args.blend_candidate_weights)
    blend_transform_set = {x.strip() for x in args.blend_transforms.split(",") if x.strip()}

    base_dir = _find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = _load_config(base_dir / "artifacts" / "config")
    _init_quant_master(base_cfg)
    base_port_cfg = _extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base_pred = _as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    coverage_start = pd.Timestamp(_pred_date_values(base_pred).min())
    coverage_end = pd.Timestamp(_pred_date_values(base_pred).max())
    slices = _year_slices(coverage_start, coverage_end)

    signal_specs: List[Tuple[str, str]] = []
    for part in [x.strip() for x in args.signal_runs.split(",") if x.strip()]:
        if ":" not in part:
            raise ValueError(f"invalid signal spec: {part}, expected key:run")
        k, v = part.split(":", 1)
        signal_specs.append((k.strip(), v.strip()))

    scan_rows: List[Dict[str, Any]] = []
    scenario_best: Dict[str, Dict[str, Any]] = {}
    signal_best: Dict[str, Dict[str, Any]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    total = 0
    for sig_key, run_token in signal_specs:
        run_dir = _find_run_dir(tracking_dir, run_token)
        run_id = run_dir.name
        sig_pred_raw = _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
        sig_pred = _slice_pred(sig_pred_raw, coverage_start, coverage_end)
        transforms = _build_transforms(sig_pred)

        signal_candidates: List[Tuple[str, str, pd.DataFrame, str]] = []
        for t_name, t_df in transforms.items():
            signal_candidates.append(("single", t_name, t_df, "single"))
            if t_name in blend_transform_set:
                for cw in blend_weights:
                    bw = 1.0 - cw
                    bname = f"blend_7406_{int(round(bw * 100))}_{int(round(cw * 100))}"
                    bdf = _blend_rank_scores(base_pred, t_df, bw, cw)
                    signal_candidates.append(("blend", t_name, bdf, bname))

        for scenario_type, t_name, scenario_df, blend_name in signal_candidates:
            scenario_id = f"{sig_key}|{scenario_type}|{t_name}|{blend_name}"
            scenario_diag = _score_diagnostics(scenario_df, base_pred)
            diagnostics[scenario_id] = scenario_diag

            best_row: Optional[EvalResult] = None
            for combo in combo_space:
                total += 1
                try:
                    metrics = _eval_combo_period(
                        combo=combo,
                        pred_df=scenario_df,
                        base_port_cfg=base_port_cfg,
                        base_strategy_kwargs=base_strategy_kwargs,
                        open_cost=float(args.open_cost),
                        close_cost=float(args.close_cost),
                        start_time=coverage_start,
                        end_time=coverage_end,
                        exchange_cache=exchange_cache,
                    )
                    row = EvalResult(
                        signal_key=sig_key,
                        run_id=run_id,
                        scenario=scenario_type,
                        transform=t_name,
                        blend=blend_name,
                        family=str(combo["family"]),
                        rebalance_mode=str(combo["rebalance_mode"]),
                        topk=int(combo["topk"]),
                        n_drop=int(combo["n_drop"]),
                        hold_topk=int(combo["hold_topk"]),
                        costed_annret=float(metrics["costed_annret"]),
                        costed_ir=float(metrics["costed_ir"]),
                        max_drawdown=float(metrics["max_drawdown"]),
                        turnover=float(metrics["turnover"]),
                        elapsed_sec=float(metrics["elapsed_sec"]),
                        error="",
                    )
                except Exception as exc:  # noqa: BLE001
                    row = EvalResult(
                        signal_key=sig_key,
                        run_id=run_id,
                        scenario=scenario_type,
                        transform=t_name,
                        blend=blend_name,
                        family=str(combo["family"]),
                        rebalance_mode=str(combo["rebalance_mode"]),
                        topk=int(combo["topk"]),
                        n_drop=int(combo["n_drop"]),
                        hold_topk=int(combo["hold_topk"]),
                        costed_annret=float("nan"),
                        costed_ir=float("nan"),
                        max_drawdown=float("nan"),
                        turnover=float("nan"),
                        elapsed_sec=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                row_dict = row.__dict__.copy()
                row_dict.update(scenario_diag)
                scan_rows.append(row_dict)
                if (
                    row.error == ""
                    and np.isfinite(row.costed_ir)
                    and np.isfinite(row.costed_annret)
                    and (best_row is None or (row.costed_ir, row.costed_annret) > (best_row.costed_ir, best_row.costed_annret))
                ):
                    best_row = row

            if best_row is None:
                continue
            best_dict = best_row.__dict__.copy()
            best_dict.update(scenario_diag)
            best_dict["scenario_id"] = scenario_id
            scenario_best[scenario_id] = best_dict
            print(
                f"[{sig_key}] {scenario_type}/{t_name}/{blend_name} best IR={best_row.costed_ir:.6f} "
                f"AnnRet={best_row.costed_annret:.6f} TO={best_row.turnover:.6f}",
                flush=True,
            )

        sig_scenarios = [x for x in scenario_best.values() if x["signal_key"] == sig_key]
        if not sig_scenarios:
            continue
        sig_best = sorted(sig_scenarios, key=lambda x: (x["costed_ir"], x["costed_annret"]), reverse=True)[0]

        # Re-evaluate yearly slices using the best scenario + combo for this signal.
        best_transform_df = None
        tname = str(sig_best["transform"])
        if sig_best["scenario"] == "single":
            best_transform_df = transforms[tname]
        else:
            cw = int(str(sig_best["blend"]).split("_")[-1]) / 100.0
            bw = 1.0 - cw
            best_transform_df = _blend_rank_scores(base_pred, transforms[tname], bw, cw)

        combo = {
            "family": sig_best["family"],
            "rebalance_mode": sig_best["rebalance_mode"],
            "rebalance_interval": 1,
            "topk": int(sig_best["topk"]),
            "n_drop": int(sig_best["n_drop"]),
            "hold_topk": int(sig_best["hold_topk"]),
            "weight_mode": "equal",
            "score_power": 1.0,
        }
        slice_metrics: List[Dict[str, Any]] = []
        for slice_name, s_start, s_end in slices:
            try:
                m = _eval_combo_period(
                    combo=combo,
                    pred_df=best_transform_df,
                    base_port_cfg=base_port_cfg,
                    base_strategy_kwargs=base_strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start_time=s_start,
                    end_time=s_end,
                    exchange_cache=exchange_cache,
                )
            except Exception as exc:  # noqa: BLE001
                m = {
                    "costed_ir": float("nan"),
                    "costed_annret": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            slice_metrics.append(
                {
                    "slice": slice_name,
                    "start": str(pd.Timestamp(s_start).date()),
                    "end": str(pd.Timestamp(s_end).date()),
                    "costed_ir": float(m["costed_ir"]),
                    "costed_annret": float(m["costed_annret"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "turnover": float(m["turnover"]),
                    "error": str(m.get("error", "")),
                }
            )
        sig_best["year_slices"] = slice_metrics
        signal_best[sig_key] = sig_best

    run_short = _resolve_run_id(args.base_run_id)[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_csv = out_dir / f"{args.output_prefix}_scan_{run_short}_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.md"
    _write_csv(scan_csv, scan_rows)

    breakthroughs = {}
    for sig_key, best in signal_best.items():
        breakthroughs[sig_key] = bool(best["costed_ir"] > SOTA_IR and best["costed_annret"] >= SOTA_ANNRET)

    summary = {
        "scan_time_utc": _now_utc(),
        "tracking_uri": args.tracking_uri,
        "base_run_id": _resolve_run_id(args.base_run_id),
        "base_thresholds": {"costed_ir": SOTA_IR, "costed_annret": SOTA_ANNRET},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "search_space": {
            "topk_grid": _split_ints(args.topk_grid),
            "n_drop_grid": _split_ints(args.n_drop_grid),
            "hold_buffers": _split_ints(args.hold_buffers),
            "blend_candidate_weights": blend_weights,
            "blend_transforms": sorted(list(blend_transform_set)),
            "total_combo_evals": int(total),
        },
        "coverage": {
            "start": str(pd.Timestamp(coverage_start).date()),
            "end": str(pd.Timestamp(coverage_end).date()),
            "rows_per_signal": int(len(base_pred)),
        },
        "best_by_signal": signal_best,
        "breakthrough_by_signal": breakthroughs,
        "any_breakthrough": bool(any(breakthroughs.values())),
        "artifacts": {
            "scan_csv": str(scan_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Signal Portfolio Conversion Summary ({stamp})")
    lines.append("")
    lines.append(f"- base_run_id: `{summary['base_run_id']}`")
    lines.append(f"- threshold: IR>{SOTA_IR:.6f} and AnnRet>={SOTA_ANNRET:.6f}")
    lines.append(f"- combo_evals: `{total}`")
    lines.append("")
    lines.append("## Best Conversion Per Signal")
    lines.append("")
    lines.append("| signal | scenario | transform | blend | family | rebalance | topk | n_drop | hold_topk | IR | AnnRet | MDD | Turnover | breakthrough |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for sig_key, best in signal_best.items():
        lines.append(
            "| {sig} | {scenario} | {transform} | {blend} | {family} | {mode} | {topk} | {nd} | {hk} | {ir:.6f} | {ann:.6f} | {mdd:.6f} | {to:.6f} | {br} |".format(
                sig=sig_key,
                scenario=best["scenario"],
                transform=best["transform"],
                blend=best["blend"],
                family=best["family"],
                mode=best["rebalance_mode"],
                topk=int(best["topk"]),
                nd=int(best["n_drop"]),
                hk=int(best["hold_topk"]),
                ir=float(best["costed_ir"]),
                ann=float(best["costed_annret"]),
                mdd=float(best["max_drawdown"]),
                to=float(best["turnover"]),
                br="yes" if breakthroughs[sig_key] else "no",
            )
        )
    lines.append("")
    lines.append("## Year Slices")
    lines.append("")
    for sig_key, best in signal_best.items():
        lines.append(f"### {sig_key}")
        lines.append("")
        lines.append("| slice | IR | AnnRet | MDD | Turnover |")
        lines.append("|---|---:|---:|---:|---:|")
        for rec in best.get("year_slices", []):
            lines.append(
                f"| {rec['slice']} | {float(rec['costed_ir']):.6f} | {float(rec['costed_annret']):.6f} | "
                f"{float(rec['max_drawdown']):.6f} | {float(rec['turnover']):.6f} |"
            )
        lines.append("")
        lines.append(
            "- diagnostics: nonnull={nn:.3f}, spread(p90-p10)={sp:.6f}, lag1={lag:.4f}, rank_corr_vs_7406={corr:.4f}".format(
                nn=float(best.get("nonnull_ratio", float("nan"))),
                sp=float(best.get("avg_p90_p10_spread", float("nan"))),
                lag=float(best.get("lag1_autocorr", float("nan"))),
                corr=float(best.get("avg_cs_rank_corr_vs_7406", float("nan"))),
            )
        )
        lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
