#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import nonlinear_regime_portfolio_search as nrs

from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import Order
from quant_master.backtest.position import Position
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase


DEFAULT_TEST_START = "2024-01-01"
DEFAULT_TEST_END = "2026-04-30"
DEFAULT_APPLY_START = "2025-01-01"
DEFAULT_APPLY_END = "2025-12-31"

ERR_RE = re.compile(
    r"only have\s+([0-9eE\.\+\-]+)\s+([A-Z0-9]+)\s*,\s*require\s+([0-9eE\.\+\-]+)"
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def _find_latest(trans_dir: Path, pattern: str) -> Path:
    candidates = sorted(trans_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing artifact: {pattern}")
    return candidates[-1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pipe_str_list(x: Any) -> Tuple[str, ...]:
    s = str(x or "").strip()
    return tuple([t for t in s.split("|") if t])


def _parse_pipe_int_list(x: Any) -> Tuple[int, ...]:
    vals = []
    for t in _parse_pipe_str_list(x):
        vals.append(int(float(t)))
    return tuple(vals)


def _parse_pipe_float_list(x: Any) -> Tuple[float, ...]:
    vals = []
    for t in _parse_pipe_str_list(x):
        vals.append(float(t))
    return tuple(vals)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _candidate_from_row(row: Dict[str, Any]) -> nrs.CandidateSpec:
    return nrs.CandidateSpec(
        stage=str(row.get("stage", "stage2")),
        members=_parse_pipe_str_list(row.get("members", "")),
        signs=_parse_pipe_int_list(row.get("signs", "")),
        w_normal=_parse_pipe_float_list(row.get("w_normal", "")),
        w_stress=_parse_pipe_float_list(row.get("w_stress", "")),
        rank_power=float(row.get("rank_power", 1.0)),
        winsor_q=float(row.get("winsor_q", 0.01)),
        neutralize=_as_bool(row.get("neutralize", True)),
        interaction_strength=float(row.get("interaction_strength", 0.0)),
        tanh_temp=float(row.get("tanh_temp", 1.0)),
        regime_lookback=int(float(row.get("regime_lookback", 40))),
        regime_z=float(row.get("regime_z", 0.8)),
        threshold_normal=float(row.get("threshold_normal", 0.0)),
        threshold_stress=float(row.get("threshold_stress", 0.0)),
        topk_normal=int(float(row.get("topk_normal", 40))),
        topk_stress=int(float(row.get("topk_stress", 35))),
        n_drop=int(float(row.get("n_drop", 2))),
        conversion_family=str(row.get("conversion_family", "convex_softmax")),
        rebalance_mode=str(row.get("rebalance_mode", "weekly")),
        hold_buffer=int(float(row.get("hold_buffer", 10))),
        softmax_temp=float(row.get("softmax_temp", 8.0)),
        softmax_power=float(row.get("softmax_power", 1.0)),
        max_weight=float(row.get("max_weight", 0.07)),
        vol_target=float(row.get("vol_target", 0.0)),
        lambda_turnover=float(row.get("lambda_turnover", 0.25)),
        lambda_dd=float(row.get("lambda_dd", 0.35)),
        parent_id=str(row.get("parent_id", "")),
    )


def _candidate_from_obj(obj: Dict[str, Any]) -> nrs.CandidateSpec:
    return nrs.CandidateSpec(
        stage=str(obj.get("stage", "stage2")),
        members=tuple(str(x) for x in obj.get("members", [])),
        signs=tuple(int(x) for x in obj.get("signs", [])),
        w_normal=tuple(float(x) for x in obj.get("w_normal", [])),
        w_stress=tuple(float(x) for x in obj.get("w_stress", [])),
        rank_power=float(obj.get("rank_power", 1.0)),
        winsor_q=float(obj.get("winsor_q", 0.01)),
        neutralize=_as_bool(obj.get("neutralize", True)),
        interaction_strength=float(obj.get("interaction_strength", 0.0)),
        tanh_temp=float(obj.get("tanh_temp", 1.0)),
        regime_lookback=int(float(obj.get("regime_lookback", 40))),
        regime_z=float(obj.get("regime_z", 0.8)),
        threshold_normal=float(obj.get("threshold_normal", 0.0)),
        threshold_stress=float(obj.get("threshold_stress", 0.0)),
        topk_normal=int(float(obj.get("topk_normal", 40))),
        topk_stress=int(float(obj.get("topk_stress", 35))),
        n_drop=int(float(obj.get("n_drop", 2))),
        conversion_family=str(obj.get("conversion_family", "convex_softmax")),
        rebalance_mode=str(obj.get("rebalance_mode", "weekly")),
        hold_buffer=int(float(obj.get("hold_buffer", 10))),
        softmax_temp=float(obj.get("softmax_temp", 8.0)),
        softmax_power=float(obj.get("softmax_power", 1.0)),
        max_weight=float(obj.get("max_weight", 0.07)),
        vol_target=float(obj.get("vol_target", 0.0)),
        lambda_turnover=float(obj.get("lambda_turnover", 0.25)),
        lambda_dd=float(obj.get("lambda_dd", 0.35)),
        parent_id=str(obj.get("parent_id", "")),
    )


def _load_candidate_map(results_csv: Path) -> Dict[str, nrs.CandidateSpec]:
    df = pd.read_csv(results_csv)
    out: Dict[str, nrs.CandidateSpec] = {}
    for row in df.to_dict(orient="records"):
        cid = str(row.get("candidate_id", "")).strip()
        if not cid or cid in out:
            continue
        out[cid] = _candidate_from_row(row)
    return out


def _extract_failed_events(lockstep_audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    period_rows = lockstep_audit.get("period_rows", [])
    for r in period_rows:
        err = str(r.get("error", ""))
        m = ERR_RE.search(err)
        if not m:
            continue
        out.append(
            {
                "source": "period_rows",
                "candidate_id": str(r.get("candidate_id", "")),
                "select_tag": str(r.get("select_tag", "")),
                "apply_tag": str(r.get("apply_tag", "")),
                "apply_start": str(r.get("apply_start", "")),
                "apply_end": str(r.get("apply_end", "")),
                "symbol": str(m.group(2)),
                "only_have": float(m.group(1)),
                "require": float(m.group(3)),
                "diff": float(m.group(3)) - float(m.group(1)),
                "error": err,
            }
        )
    for r in lockstep_audit.get("selection_trace_rows", []):
        yj = str(r.get("yearly_rows_json", ""))
        if "only have" not in yj:
            continue
        m = ERR_RE.search(yj)
        if not m:
            continue
        out.append(
            {
                "source": "selection_trace_rows",
                "candidate_id": str(r.get("candidate_id", "")),
                "select_tag": "",
                "apply_tag": "yearly_trace",
                "apply_start": "",
                "apply_end": "",
                "symbol": str(m.group(2)),
                "only_have": float(m.group(1)),
                "require": float(m.group(3)),
                "diff": float(m.group(3)) - float(m.group(1)),
                "error": yj,
            }
        )
    seen = set()
    uniq = []
    for e in out:
        key = (e["candidate_id"], e["symbol"], e["only_have"], e["require"], e["source"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def _build_signal_debug(
    candidate: nrs.CandidateSpec,
    run_map: Dict[str, nrs.RunSignal],
    anchor_key: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    member_cols: List[pd.Series] = []
    member_names: List[str] = []
    for key, sign in zip(candidate.members, candidate.signs):
        rank_pct = nrs._slice_series(run_map[key].rank_pct, start_date, end_date)
        rank_pct = nrs._cap_winsor_rank(rank_pct, candidate.winsor_q)
        transformed = nrs._rank_power_transform(rank_pct, candidate.rank_power) * float(sign)
        transformed.name = key
        member_cols.append(transformed)
        member_names.append(key)

    panel = pd.concat(member_cols, axis=1)
    panel.columns = member_names
    panel = panel.dropna(how="all")
    if panel.empty:
        raise ValueError("empty panel after member merge")

    anchor_rank = nrs._slice_series(run_map[anchor_key].rank_pct, start_date, end_date)
    regime_flag_by_date = nrs._build_regime_flag(
        anchor_rank=anchor_rank,
        lookback=candidate.regime_lookback,
        z_thr=candidate.regime_z,
    )
    row_dates = pd.to_datetime(panel.index.get_level_values(0))
    is_stress = pd.Series(regime_flag_by_date).reindex(row_dates).fillna(False).astype(bool).to_numpy()

    w_norm = np.asarray(candidate.w_normal, dtype=float)
    w_stress = np.asarray(candidate.w_stress, dtype=float)
    w_mat = np.where(is_stress[:, None], w_stress[None, :], w_norm[None, :])
    x = panel.to_numpy(dtype=float)
    valid = np.isfinite(x)

    wx = np.where(valid, x * w_mat, 0.0)
    wsum = np.where(valid, w_mat, 0.0).sum(axis=1)
    linear = wx.sum(axis=1) / np.where(wsum > 1e-12, wsum, np.nan)

    pair_values = np.zeros(len(panel), dtype=float)
    pair_weights = np.zeros(len(panel), dtype=float)
    k = x.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            wij = np.sqrt(np.clip(w_mat[:, i] * w_mat[:, j], 0.0, None))
            both = valid[:, i] & valid[:, j]
            pair_values += np.where(both, wij * x[:, i] * x[:, j], 0.0)
            pair_weights += np.where(both, wij, 0.0)
    pair_term = pair_values / np.where(pair_weights > 1e-12, pair_weights, np.nan)

    blended = linear + float(candidate.interaction_strength) * np.nan_to_num(pair_term, nan=0.0)
    score = pd.Series(np.tanh(float(candidate.tanh_temp) * blended), index=panel.index, name="score", dtype=float)
    if candidate.neutralize:
        score = nrs._date_neutralize(score)
    score = nrs._apply_vol_targeting(score, target_vol=candidate.vol_target)

    rank_pos = score.groupby(level=0).rank(method="first", ascending=False)
    rank_pct = score.groupby(level=0).rank(method="average", pct=True)
    threshold = np.where(is_stress, float(candidate.threshold_stress), float(candidate.threshold_normal))
    topk_req = np.where(is_stress, int(candidate.topk_stress), int(candidate.topk_normal))

    eligible = rank_pct.to_numpy() >= threshold
    selected = eligible & (rank_pos.to_numpy() <= topk_req)

    ddf = pd.DataFrame(
        {
            "score": score.to_numpy(dtype=float),
            "rank_pos": rank_pos.to_numpy(dtype=float),
            "rank_pct": rank_pct.to_numpy(dtype=float),
            "is_stress": is_stress.astype(int),
            "threshold": threshold.astype(float),
            "topk_req": topk_req.astype(int),
            "eligible": eligible.astype(int),
            "selected": selected.astype(int),
        },
        index=score.index,
    )
    ddf = ddf.reset_index()
    ddf.columns = ["datetime", "instrument", "score", "rank_pos", "rank_pct", "is_stress", "threshold", "topk_req", "eligible", "selected"]
    ddf["date"] = pd.to_datetime(ddf["datetime"]).dt.date.astype(str)

    diag = {
        "rows": int(len(ddf)),
        "date_count": int(ddf["date"].nunique()),
        "selected_ratio": float(ddf["selected"].mean()) if len(ddf) else float("nan"),
        "nonnull_ratio": float(np.isfinite(ddf["score"]).mean()) if len(ddf) else float("nan"),
    }
    return ddf, diag


def _build_daily_diag_rows(
    *,
    candidate_id: str,
    candidate: nrs.CandidateSpec,
    event_symbols: Sequence[str],
    ddf: pd.DataFrame,
    exchange: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    event_set = set(event_symbols)
    for date, g in ddf.groupby("date", sort=True):
        g = g.copy()
        topk_req = int(g["topk_req"].max()) if len(g) else 0
        threshold = float(g["threshold"].max()) if len(g) else float("nan")
        merged_names = int(g["instrument"].nunique())
        nonnan_names = int(g.loc[np.isfinite(g["score"]), "instrument"].nunique())
        eligible_names = int(g.loc[g["eligible"] > 0, "instrument"].nunique())
        selected_names = int(g.loc[g["selected"] > 0, "instrument"].nunique())
        selected_codes = g.loc[g["selected"] > 0, "instrument"].astype(str).tolist()
        eligible_codes = g.loc[g["eligible"] > 0, "instrument"].astype(str).tolist()

        dt = pd.Timestamp(date)
        tradable_selected = 0
        for code in selected_codes:
            if exchange.is_stock_tradable(code, start_time=dt, end_time=dt):
                tradable_selected += 1
        tradable_eligible = 0
        for code in eligible_codes:
            if exchange.is_stock_tradable(code, start_time=dt, end_time=dt):
                tradable_eligible += 1

        event_flags = []
        for sym in sorted(event_set):
            if sym in set(selected_codes):
                event_flags.append(sym)

        out.append(
            {
                "candidate_id": candidate_id,
                "date": date,
                "conversion_family": candidate.conversion_family,
                "rebalance_mode": candidate.rebalance_mode,
                "topk_normal": int(candidate.topk_normal),
                "topk_stress": int(candidate.topk_stress),
                "hold_buffer": int(candidate.hold_buffer),
                "threshold_normal": float(candidate.threshold_normal),
                "threshold_stress": float(candidate.threshold_stress),
                "n_drop": int(candidate.n_drop),
                "is_stress_day": int(g["is_stress"].max()) if len(g) else 0,
                "topk_req": topk_req,
                "threshold_req": threshold,
                "merged_names": merged_names,
                "nonnull_names": nonnan_names,
                "eligible_names": eligible_names,
                "selected_names": selected_names,
                "tradable_eligible_names": int(tradable_eligible),
                "tradable_selected_names": int(tradable_selected),
                "topk_satisfied_by_selected": bool(selected_names >= topk_req),
                "topk_satisfied_by_tradable_selected": bool(tradable_selected >= topk_req),
                "event_symbols_in_selected": "|".join(event_flags),
            }
        )
    return out


class DiagSafeOrderGenWInteract(OrderGenWInteract):
    def __init__(self, *, eps_ratio: float = 1e-9):
        super().__init__()
        self.eps_ratio = float(max(0.0, eps_ratio))

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Any,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        orders = super().generate_order_list_from_target_weight_position(
            current=current,
            trade_exchange=trade_exchange,
            target_weight_position=target_weight_position,
            risk_degree=risk_degree,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        for od in orders:
            if od.direction != Order.SELL:
                continue
            cur_amt = float(current.get_stock_amount(od.stock_id))
            if od.amount > cur_amt:
                od.amount = max(0.0, cur_amt * (1.0 - self.eps_ratio))
            else:
                tol = self.eps_ratio * max(1.0, cur_amt)
                if (cur_amt - od.amount) >= 0 and (cur_amt - od.amount) <= tol:
                    od.amount = max(0.0, cur_amt * (1.0 - self.eps_ratio))
        return orders


class DiagSafeConvexTopKWeightStrategy(nrs.RebalanceMixin, WeightStrategyBase):
    def __init__(
        self,
        *,
        signal,
        topk: int,
        hold_topk: Optional[int] = None,
        softmax_temp: float = 8.0,
        softmax_power: float = 1.0,
        max_weight: float = 0.07,
        rebalance_mode: str = "weekly",
        rebalance_interval: int = 1,
        risk_degree: float = 0.95,
        min_names_guard: int = 3,
        dynamic_topk_clamp: bool = True,
        fallback_mode: str = "cash",
    ):
        self.signal = signal
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.softmax_temp = float(softmax_temp)
        self.softmax_power = float(softmax_power)
        self.max_weight = float(max_weight)
        self.min_names_guard = int(max(1, min_names_guard))
        self.dynamic_topk_clamp = bool(dynamic_topk_clamp)
        self.fallback_mode = str(fallback_mode)
        super().__init__(
            rebalance_mode=rebalance_mode,
            rebalance_interval=rebalance_interval,
            order_generator_cls_or_obj=DiagSafeOrderGenWInteract(eps_ratio=1e-9),
            signal=signal,
            risk_degree=float(risk_degree),
        )

    def _to_series(self, score: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        return score.dropna()

    @staticmethod
    def _apply_cap(raw: pd.Series, cap: float) -> pd.Series:
        if raw.empty:
            return raw
        w = raw.copy()
        cap = float(max(1e-6, cap))
        for _ in range(5):
            over = w > cap
            if not over.any():
                break
            locked = w.where(over, 0.0)
            free = w.where(~over, 0.0)
            locked_sum = float(locked.sum())
            if locked_sum >= 0.999999:
                return locked / max(1e-12, locked_sum)
            target_free = 1.0 - float((over * cap).sum())
            free_sum = float(free.sum())
            if free_sum <= 0:
                w.loc[over] = cap
                rem = 1.0 - float(w.loc[over].sum())
                w.loc[~over] = rem / max(1, (~over).sum())
                return w.clip(lower=0.0)
            w.loc[over] = cap
            w.loc[~over] = free.loc[~over] / free_sum * target_free
        norm = float(w.sum())
        if norm <= 0:
            return pd.Series(1.0 / len(w), index=w.index)
        return w / norm

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_step = self.trade_calendar.get_trade_step()
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return None
        score_s = self._to_series(score)
        if score_s.empty:
            return {} if self.fallback_mode == "cash" else None

        ranked = score_s.sort_values(ascending=False)
        tradable = []
        for code in ranked.index:
            try:
                ok = self.trade_exchange.is_stock_tradable(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                )
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                tradable.append(code)
        if len(tradable) < self.min_names_guard:
            return {} if self.fallback_mode == "cash" else None

        ranked = ranked.reindex(tradable).dropna()
        if ranked.empty:
            return {} if self.fallback_mode == "cash" else None

        topk_use = int(self.topk)
        hold_topk_use = int(self.hold_topk)
        if self.dynamic_topk_clamp:
            topk_use = int(min(topk_use, len(ranked)))
            hold_topk_use = int(min(max(topk_use, hold_topk_use), len(ranked)))

        rank_pos = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        current_stocks = [s for s in current.get_stock_list() if s in ranked.index]
        keep = [s for s in current_stocks if int(rank_pos.get(s, 10**9)) <= hold_topk_use]
        keep = sorted(keep, key=lambda x: float(ranked.loc[x]), reverse=True)
        if len(keep) > topk_use:
            keep = keep[:topk_use]
        need = max(0, topk_use - len(keep))
        add_list = [s for s in ranked.index if s not in keep][:need]
        target = keep + add_list
        if not target:
            return {} if self.fallback_mode == "cash" else None

        s = ranked.reindex(target).astype(float)
        z = (s - float(s.mean())) * self.softmax_temp
        z = z.clip(lower=-50, upper=50)
        raw = np.exp(z)
        raw = pd.Series(raw, index=s.index, dtype=float)
        if self.softmax_power != 1.0:
            raw = raw.pow(max(1e-6, self.softmax_power))
        norm = float(raw.sum())
        if norm <= 0:
            w = pd.Series(1.0 / len(raw), index=raw.index, dtype=float)
        else:
            w = raw / norm
        w = self._apply_cap(w, cap=self.max_weight)
        return {k: float(v) for k, v in w.items()}


def _build_exchange(
    *,
    base_port_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    open_cost: float,
    close_cost: float,
) -> Any:
    backtest_cfg = base_port_cfg["backtest"]
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    executor_cfg = base_port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    return get_exchange(
        freq=freq,
        start_time=start_date,
        end_time=end_date,
        deal_price=deal_price,
        limit_threshold=limit_threshold,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=min_cost,
    )


def _extract_failed_runtime_context(
    exc: Exception,
) -> Dict[str, Any]:
    msg = str(exc)
    out: Dict[str, Any] = {"error": msg}
    m = ERR_RE.search(msg)
    if m:
        out["only_have"] = float(m.group(1))
        out["symbol"] = str(m.group(2))
        out["require"] = float(m.group(3))
        out["diff"] = float(m.group(3)) - float(m.group(1))
    return out


def _eval_candidate_with_runtime_capture(
    *,
    candidate: nrs.CandidateSpec,
    run_map: Dict[str, nrs.RunSignal],
    anchor_key: str,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_date: str,
    end_date: str,
    safe_mode: bool,
) -> Dict[str, Any]:
    signal_df, sig_diag = nrs._build_signal_for_candidate(
        candidate=candidate,
        run_map=run_map,
        anchor_key=anchor_key,
        start_date=start_date,
        end_date=end_date,
    )
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_date).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_date).date())
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    sliced = signal_df.copy()
    if isinstance(sliced.index, pd.MultiIndex):
        d = pd.to_datetime(sliced.index.get_level_values(0))
        sliced = sliced.loc[(d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))]
    if sliced.empty:
        raise ValueError(f"empty signal after slice: {start_date}..{end_date}")

    topk_effective = int(max(candidate.topk_normal, candidate.topk_stress))
    if safe_mode and candidate.conversion_family == "convex_softmax":
        strategy_obj = DiagSafeConvexTopKWeightStrategy(
            signal=sliced,
            topk=topk_effective,
            hold_topk=topk_effective + int(candidate.hold_buffer),
            softmax_temp=float(candidate.softmax_temp),
            softmax_power=float(candidate.softmax_power),
            max_weight=float(candidate.max_weight),
            rebalance_mode=candidate.rebalance_mode,
            rebalance_interval=1,
            risk_degree=float(base_strategy_kwargs.get("risk_degree", 0.95)),
            min_names_guard=3,
            dynamic_topk_clamp=True,
            fallback_mode="cash",
        )
    elif candidate.conversion_family == "convex_softmax":
        strategy_obj = nrs.ConvexTopKWeightStrategy(
            signal=sliced,
            topk=topk_effective,
            hold_topk=topk_effective + int(candidate.hold_buffer),
            softmax_temp=float(candidate.softmax_temp),
            softmax_power=float(candidate.softmax_power),
            max_weight=float(candidate.max_weight),
            rebalance_mode=candidate.rebalance_mode,
            rebalance_interval=1,
            risk_degree=float(base_strategy_kwargs.get("risk_degree", 0.95)),
        )
    else:
        strategy_obj = TopkDropoutStrategy(
            signal=sliced,
            topk=topk_effective,
            n_drop=int(candidate.n_drop),
            method_sell=base_strategy_kwargs.get("method_sell", "bottom"),
            method_buy=base_strategy_kwargs.get("method_buy", "top"),
            hold_thresh=int(base_strategy_kwargs.get("hold_thresh", 1)),
            only_tradable=bool(base_strategy_kwargs.get("only_tradable", False)),
            forbid_all_trade_at_limit=bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
            risk_degree=float(base_strategy_kwargs.get("risk_degree", 0.95)),
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
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        deal_price=deal_price,
        limit_threshold=limit_threshold,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=min_cost,
    )

    runtime_ctx: Dict[str, Any] = {}
    orig_update_order = Position.update_order

    def _patched_update_order(self, order, trade_val, cost, trade_price):
        before_amount = float(self.get_stock_amount(order.stock_id))
        try:
            return orig_update_order(self, order, trade_val, cost, trade_price)
        except Exception as exc:  # noqa: BLE001
            runtime_ctx["order_stock_id"] = str(order.stock_id)
            runtime_ctx["order_direction"] = int(order.direction)
            runtime_ctx["order_amount"] = float(order.amount)
            runtime_ctx["trade_val"] = float(trade_val)
            runtime_ctx["trade_price"] = float(trade_price)
            runtime_ctx["trade_start_time"] = str(order.start_time)
            runtime_ctx["trade_end_time"] = str(order.end_time)
            runtime_ctx["before_amount"] = float(before_amount)
            raise

    Position.update_order = _patched_update_order
    try:
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
    except Exception as exc:  # noqa: BLE001
        err = _extract_failed_runtime_context(exc)
        err.update(runtime_ctx)
        err.update(sig_diag)
        err["safe_mode"] = bool(safe_mode)
        return {"ok": False, "error": err, "metrics": None, "excess": None}
    finally:
        Position.update_order = orig_update_order

    report_df = nrs._get_report_for_day_freq(portfolio_metric_dict)
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    maxdd = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    objective = float(
        ir
        + 0.45 * annret
        - float(candidate.lambda_turnover) * turnover
        - float(candidate.lambda_dd) * abs(maxdd)
    )
    excess = (report_df["return"] - report_df["bench"] - report_df["cost"]).astype(float)
    return {
        "ok": True,
        "error": None,
        "metrics": {
            "annret": annret,
            "ir": ir,
            "max_drawdown": maxdd,
            "turnover": turnover,
            "objective": objective,
            "rows": int(len(report_df)),
            "safe_mode": bool(safe_mode),
            "signal_nonnull_ratio": float(sig_diag.get("nonnull_ratio", float("nan"))),
        },
        "excess": excess,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _load_context(args: argparse.Namespace) -> Dict[str, Any]:
    trans_dir = Path("examples/benchmarks/Transcendence").resolve()
    tracking_dir = nrs._parse_tracking_dir(args.tracking_uri)
    base_run_dir = nrs._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = nrs._load_config(base_run_dir / "artifacts" / "config")
    nrs._init_quant_master(base_cfg)
    base_port_cfg = nrs._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)
    run_signals, run_audit_rows = nrs._discover_pred_runs(
        tracking_dir=tracking_dir,
        comparable_instruments="csi300",
        require_comparable=True,
        test_start=args.test_start,
        test_end=args.test_end,
        min_coverage_test=0.35,
    )
    if len(run_signals) < 2:
        raise RuntimeError("usable run_signals < 2")
    run_map = {x.key: x for x in run_signals}
    anchor_key = next(
        (x.key for x in run_signals if x.run_id == nrs._resolve_run_token(args.base_run_id)),
        run_signals[0].key,
    )
    return {
        "trans_dir": trans_dir,
        "tracking_dir": tracking_dir,
        "run_audit_rows": run_audit_rows,
        "run_map": run_map,
        "anchor_key": anchor_key,
        "base_port_cfg": base_port_cfg,
        "base_strategy_kwargs": base_strategy_kwargs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose lockstep backtest execution constraint failures and safe fallback.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=nrs.DEFAULT_BASE_RUN)
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--test-start", default=DEFAULT_TEST_START)
    p.add_argument("--test-end", default=DEFAULT_TEST_END)
    p.add_argument("--apply-start", default=DEFAULT_APPLY_START)
    p.add_argument("--apply-end", default=DEFAULT_APPLY_END)
    p.add_argument("--lockstep-audit-json", default="")
    p.add_argument("--lockstep-summary-json", default="")
    p.add_argument("--nonlinear-results-csv", default="")
    p.add_argument("--output-prefix", default="robust_backtest_diag")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    ctx = _load_context(args)
    trans_dir: Path = ctx["trans_dir"]
    stamp = _timestamp()

    lockstep_audit_json = (
        Path(args.lockstep_audit_json)
        if args.lockstep_audit_json
        else _find_latest(trans_dir, "robust_regime_lockstep_audit_*.json")
    )
    lockstep_summary_json = (
        Path(args.lockstep_summary_json)
        if args.lockstep_summary_json
        else _find_latest(trans_dir, "robust_regime_lockstep_summary_*.json")
    )
    nonlinear_results_csv = (
        Path(args.nonlinear_results_csv)
        if args.nonlinear_results_csv
        else _find_latest(trans_dir, "nonlinear_regime_results_*.csv")
    )

    summary_json_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    daily_csv_path = trans_dir / f"{args.output_prefix}_daily_{stamp}.csv"
    safe_period_csv_path = trans_dir / f"{args.output_prefix}_safe_periods_{stamp}.csv"
    parse_smoke_path = trans_dir / f"{args.output_prefix}_parse_smoke_{stamp}.json"

    lockstep_audit = _read_json(lockstep_audit_json)
    lockstep_summary = _read_json(lockstep_summary_json)
    candidate_map = _load_candidate_map(nonlinear_results_csv)
    lockstep_selected_by_id: Dict[str, nrs.CandidateSpec] = {}
    for r in lockstep_summary.get("selected_candidates", []):
        cid = str(r.get("candidate_id", "")).strip()
        cobj = r.get("candidate", {})
        if cid and isinstance(cobj, dict) and cobj:
            lockstep_selected_by_id[cid] = _candidate_from_obj(cobj)
    failed_events = _extract_failed_events(lockstep_audit)

    failed_candidate_ids = sorted(set([str(e["candidate_id"]) for e in failed_events if str(e["candidate_id"])]))
    if not failed_candidate_ids:
        # fallback to period row candidate_id for apply_tag=2025 when no explicit parse
        for r in lockstep_audit.get("period_rows", []):
            if str(r.get("apply_tag")) == "2025" and str(r.get("candidate_id")):
                failed_candidate_ids.append(str(r["candidate_id"]))
        failed_candidate_ids = sorted(set(failed_candidate_ids))

    missing_candidates = [
        cid
        for cid in failed_candidate_ids
        if (cid not in candidate_map and cid not in lockstep_selected_by_id)
    ]

    base_port_cfg = ctx["base_port_cfg"]
    base_strategy_kwargs = ctx["base_strategy_kwargs"]
    run_map = ctx["run_map"]
    anchor_key = ctx["anchor_key"]

    daily_rows: List[Dict[str, Any]] = []
    replay_rows: List[Dict[str, Any]] = []
    safe_period_rows: List[Dict[str, Any]] = []
    root_cause_rows: List[Dict[str, Any]] = []
    unresolved_failed_candidates: List[str] = []

    for cid in failed_candidate_ids:
        cand = lockstep_selected_by_id.get(cid, candidate_map.get(cid))
        if cand is None:
            unresolved_failed_candidates.append(cid)
            continue
        symbols = [e["symbol"] for e in failed_events if str(e.get("candidate_id")) == cid]
        symbols = sorted(set([s for s in symbols if s]))

        exchange = _build_exchange(
            base_port_cfg=base_port_cfg,
            start_date=args.apply_start,
            end_date=args.apply_end,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        ddf, sig_diag = _build_signal_debug(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            start_date=args.apply_start,
            end_date=args.apply_end,
        )
        rows = _build_daily_diag_rows(
            candidate_id=cid,
            candidate=cand,
            event_symbols=symbols,
            ddf=ddf,
            exchange=exchange,
        )
        daily_rows.extend(rows)

        replay_base = _eval_candidate_with_runtime_capture(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_date=args.apply_start,
            end_date=args.apply_end,
            safe_mode=False,
        )
        replay_safe = _eval_candidate_with_runtime_capture(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_date=args.apply_start,
            end_date=args.apply_end,
            safe_mode=True,
        )
        replay_rows.append(
            {
                "candidate_id": cid,
                "members": "|".join(cand.members),
                "topk_normal": int(cand.topk_normal),
                "topk_stress": int(cand.topk_stress),
                "hold_buffer": int(cand.hold_buffer),
                "threshold_normal": float(cand.threshold_normal),
                "threshold_stress": float(cand.threshold_stress),
                "baseline_ok": bool(replay_base["ok"]),
                "baseline_error": replay_base["error"],
                "safe_ok": bool(replay_safe["ok"]),
                "safe_error": replay_safe["error"],
                "baseline_metrics": replay_base["metrics"],
                "safe_metrics": replay_safe["metrics"],
                "signal_diag": sig_diag,
            }
        )

        # root cause evidence summary row
        day_df = pd.DataFrame(rows)
        root_cause_rows.append(
            {
                "candidate_id": cid,
                "members": "|".join(cand.members),
                "conversion_family": cand.conversion_family,
                "rebalance_mode": cand.rebalance_mode,
                "min_selected_names": int(day_df["selected_names"].min()) if len(day_df) else -1,
                "p05_selected_names": float(day_df["selected_names"].quantile(0.05)) if len(day_df) else float("nan"),
                "min_tradable_selected_names": int(day_df["tradable_selected_names"].min()) if len(day_df) else -1,
                "p05_tradable_selected_names": float(day_df["tradable_selected_names"].quantile(0.05))
                if len(day_df)
                else float("nan"),
                "days_selected_le_1": int((day_df["selected_names"] <= 1).sum()) if len(day_df) else 0,
                "days_tradable_selected_lt_topk": int((~day_df["topk_satisfied_by_tradable_selected"]).sum())
                if len(day_df)
                else 0,
                "days_selected_lt_topk": int((~day_df["topk_satisfied_by_selected"]).sum()) if len(day_df) else 0,
                "baseline_error": replay_base["error"],
            }
        )

    # lockstep-safe replay using selected_candidates from lockstep summary
    safe_excess_parts: List[pd.Series] = []
    selected_candidates = lockstep_summary.get("selected_candidates", [])
    for r in selected_candidates:
        cid = str(r.get("candidate_id", ""))
        if not cid:
            continue
        cand = lockstep_selected_by_id.get(cid, candidate_map.get(cid))
        if cand is None:
            continue
        apply_tag = str(r.get("select_tag", ""))
        if apply_tag == "2024H1_degraded":
            st, ed = "2024-07-01", "2024-12-31"
        elif apply_tag == "2024":
            st, ed = "2025-01-01", "2025-12-31"
        elif apply_tag == "up_to_2025":
            st, ed = "2026-01-01", args.test_end
        else:
            continue
        rs = _eval_candidate_with_runtime_capture(
            candidate=cand,
            run_map=run_map,
            anchor_key=anchor_key,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_date=st,
            end_date=ed,
            safe_mode=True,
        )
        row = {
            "select_tag": apply_tag,
            "apply_start": st,
            "apply_end": ed,
            "candidate_id": cid,
            "safe_ok": bool(rs["ok"]),
            "error": rs["error"],
        }
        if rs["metrics"]:
            row.update(rs["metrics"])
        safe_period_rows.append(row)
        if rs["ok"] and rs["excess"] is not None:
            safe_excess_parts.append(rs["excess"])

    stitched_safe_metrics = None
    if safe_excess_parts:
        stitched_excess = pd.concat(safe_excess_parts).sort_index()
        stitched_safe_metrics = _metrics_from_excess(stitched_excess)

    _write_csv(daily_csv_path, daily_rows)
    _write_csv(safe_period_csv_path, safe_period_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "source_artifacts": {
            "lockstep_audit_json": str(lockstep_audit_json).replace("\\", "/"),
            "lockstep_summary_json": str(lockstep_summary_json).replace("\\", "/"),
            "nonlinear_results_csv": str(nonlinear_results_csv).replace("\\", "/"),
        },
        "diagnostic_window": {"start": args.apply_start, "end": args.apply_end},
        "failed_events": failed_events,
        "missing_candidate_ids_cross_file": missing_candidates,
        "unresolved_failed_candidate_ids": unresolved_failed_candidates,
        "root_cause_rows": root_cause_rows,
        "candidate_replays": replay_rows,
        "safe_lockstep_period_rows": safe_period_rows,
        "safe_lockstep_stitched_metrics": stitched_safe_metrics,
        "baseline_lockstep_stitched_metrics": lockstep_summary.get("results", {}).get("stitched_metrics"),
        "artifacts": {
            "daily_csv": str(daily_csv_path).replace("\\", "/"),
            "safe_period_csv": str(safe_period_csv_path).replace("\\", "/"),
            "summary_json": str(summary_json_path).replace("\\", "/"),
            "summary_md": str(summary_md_path).replace("\\", "/"),
            "parse_smoke_json": str(parse_smoke_path).replace("\\", "/"),
        },
        "safe_fallback_proposal": {
            "wrapper_only": True,
            "strategy": "ConvexTopKWeightStrategy + OrderGenWInteract safe wrapper",
            "guards": {
                "min_names_guard": 3,
                "dynamic_topk_clamp": True,
                "fallback_mode": "cash",
                "sell_amount_epsilon_clip": "1e-9 ratio",
            },
            "note": "diagnostic-only fallback; no core library changes in this worker scope",
        },
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Robust Backtest Constraint Diagnostics ({stamp})",
        "",
        f"- Window: `{args.apply_start}..{args.apply_end}`",
        f"- Failed candidate count: `{len(failed_candidate_ids)}`",
        f"- Failed events parsed: `{len(failed_events)}`",
        f"- Baseline stitched metrics (from lockstep summary): `{json.dumps(lockstep_summary.get('results', {}).get('stitched_metrics'), ensure_ascii=False)}`",
        f"- Safe stitched metrics (diagnostic wrapper): `{json.dumps(stitched_safe_metrics, ensure_ascii=False)}`",
        "",
        "## Root Cause Snapshot",
    ]
    for r in root_cause_rows:
        md_lines.append(
            "- candidate `{cid}`: min selected `{mn}`, min tradable selected `{mtn}`, "
            "days selected<=1 `{d1}`, days tradable selected<topk `{dtopk}`; baseline error `{err}`".format(
                cid=r["candidate_id"],
                mn=r["min_selected_names"],
                mtn=r["min_tradable_selected_names"],
                d1=r["days_selected_le_1"],
                dtopk=r["days_tradable_selected_lt_topk"],
                err=r["baseline_error"],
            )
        )
    md_lines.extend(
        [
            "",
            "## Safe Fallback (Diagnostic Only)",
            "- min_names guard + dynamic topk clamp + cash fallback + sell epsilon clip.",
            "- Implemented only in this script wrapper, not in production strategy library.",
        ]
    )
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # parse smoke
    smoke = {
        "timestamp_utc": _now_utc(),
        "summary_json_exists": summary_json_path.exists(),
        "summary_md_exists": summary_md_path.exists(),
        "daily_csv_exists": daily_csv_path.exists(),
        "safe_period_csv_exists": safe_period_csv_path.exists(),
        "summary_json_parse_ok": False,
        "daily_csv_rows": 0,
    }
    try:
        _ = json.loads(summary_json_path.read_text(encoding="utf-8"))
        smoke["summary_json_parse_ok"] = True
    except Exception as exc:  # noqa: BLE001
        smoke["summary_json_parse_ok"] = False
        smoke["summary_json_parse_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if daily_csv_path.exists():
            with daily_csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                smoke["daily_csv_rows"] = max(0, len(rows) - 1)
    except Exception as exc:  # noqa: BLE001
        smoke["daily_csv_parse_error"] = f"{type(exc).__name__}: {exc}"

    parse_smoke_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

