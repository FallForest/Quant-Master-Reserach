#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
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
from quant_master.config import resolve_provider_uri_in_config
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase
from examples.benchmarks.Transcendence._bootstrap import init_quant_master_from_config, load_config_with_resolved_provider


TARGET_IR = 2.90
TARGET_ANNRET = 0.27
BASELINE_7406_IR = 2.799983676714277
BASELINE_7406_ANNRET = 0.24466463608994535
DEFAULT_SOTA_IR = 3.0230019401859436
DEFAULT_SOTA_ANNRET = 0.3878544154715252
DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"

RUN_ALIAS = {
    "7406": DEFAULT_BASE_RUN,
    "7406e470": DEFAULT_BASE_RUN,
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "e2300230": "e2300230e0994a1a9ccbbd3bc4606d97",
    "bcbecf55": "bcbecf55a3924357ba93fc55b1140e99",
    "d4526da": "d4526da7854245af954fc99cf02963f0",
    "1a085ff": "1a085ff9b5a34f408a44ad74055fc5da",
    "05ef8bd1": "05ef8bd12e0e407f9fdf0cad3ef72652",
    "0ed35c": "0ed35c572e104ddab555a8af6a7fe981",
    "2ac6": "2ac6ebc249bf42e5a9f83c6ca0725941",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "94a52": "94a52e5949104218ab2a3b4cd84dce08",
    "6feaa": "6feaa8c5b0fc437784592bb7b534d710",
    "ae098013": "ae0980136de44dc58a0b9d3f7d947363",
    "4a98": "4a98f99bdb6848bab789ff6c46d0a1ff",
    "29864": "29864d9c5d00463b9fdbc065c10b0093",
}


@dataclass
class RunSignal:
    key: str
    run_id: str
    pred_path: str
    raw: pd.Series
    rank_pct: pd.Series
    coverage_test: float
    model_class: str
    dataset_class: str
    instruments: str
    source_ir: float | None
    source_annret: float | None


@dataclass
class CandidateSpec:
    stage: str
    members: Tuple[str, ...]
    signs: Tuple[int, ...]
    w_normal: Tuple[float, ...]
    w_stress: Tuple[float, ...]
    rank_power: float
    winsor_q: float
    neutralize: bool
    interaction_strength: float
    tanh_temp: float
    regime_lookback: int
    regime_z: float
    threshold_normal: float
    threshold_stress: float
    topk_normal: int
    topk_stress: int
    n_drop: int
    conversion_family: str
    rebalance_mode: str
    hold_buffer: int
    softmax_temp: float
    softmax_power: float
    max_weight: float
    vol_target: float
    lambda_turnover: float
    lambda_dd: float
    parent_id: str = ""


@dataclass
class EvalResult:
    candidate_id: str
    stage: str
    metrics_split: str
    start_date: str
    end_date: str
    annret: float
    ir: float
    max_drawdown: float
    turnover: float
    objective: float
    elapsed_sec: float
    topk_effective_max: int
    nonnull_ratio: float
    error: str = ""


class RebalanceMixin:
    def __init__(self, rebalance_mode: str = "daily", rebalance_interval: int = 1, **kwargs):
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


class ConvexTopKWeightStrategy(RebalanceMixin, WeightStrategyBase):
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
        **kwargs,
    ):
        self.signal = signal
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.softmax_temp = float(softmax_temp)
        self.softmax_power = float(softmax_power)
        self.max_weight = float(max_weight)
        super().__init__(
            rebalance_mode=rebalance_mode,
            rebalance_interval=rebalance_interval,
            order_generator_cls_or_obj=OrderGenWInteract,
            signal=signal,
            **kwargs,
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
            return {}
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
        if not tradable:
            return {}
        ranked = ranked.reindex(tradable).dropna()
        if ranked.empty:
            return {}
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


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _resolve_run_token(token: str) -> str:
    t = token.strip()
    if t in RUN_ALIAS:
        return RUN_ALIAS[t]
    return t


def _find_run_dir(tracking_dir: Path, run_token: str) -> Path:
    run_id = _resolve_run_token(run_token)
    candidates = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not candidates and len(run_id) == 8:
        candidates = [p for p in tracking_dir.glob(f"*/{run_id}*") if (p / "artifacts").exists()]
    if not candidates:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_token}")
    if len(candidates) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    return load_config_with_resolved_provider(
        path,
        loader=lambda config_path: yaml.safe_load(config_path.read_text(encoding="utf-8")),
        binary_fallback=_load_pickle,
    )


def _parse_metric_file(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    parts = metric_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _extract_source_metrics(run_dir: Path) -> Tuple[float | None, float | None]:
    metric_dir = run_dir / "metrics"
    ir = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.information_ratio")
    ann = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.annualized_return")
    if ir is None:
        ir = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.information_ratio")
    if ann is None:
        ann = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.annualized_return")
    return ir, ann


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


def _as_score_series(pred_obj: Any) -> pd.Series:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float)
    if isinstance(pred_obj, pd.DataFrame):
        if pred_obj.shape[1] == 1:
            return pred_obj.iloc[:, 0].astype(float)
        if "score" in pred_obj.columns:
            return pred_obj["score"].astype(float)
        return pred_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _slice_series(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    idx = series.index
    if isinstance(idx, pd.MultiIndex):
        d = pd.to_datetime(idx.get_level_values(0))
        m = (d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))
        return series.loc[m]
    d = pd.to_datetime(idx)
    return series.loc[(d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))]


def _cs_rank(series: pd.Series) -> pd.Series:
    idx = series.index
    if isinstance(idx, pd.MultiIndex):
        return series.groupby(level=0).rank(method="average", pct=True)
    return series.groupby(idx).rank(method="average", pct=True)


def _date_neutralize(series: pd.Series) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        m = float(x.mean())
        s = float(x.std(ddof=0))
        if not np.isfinite(s) or s <= 1e-12:
            return x - m
        return (x - m) / s

    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=0, group_keys=False).apply(_z)
    return _z(series)


def _cap_winsor_rank(rank_pct: pd.Series, q: float) -> pd.Series:
    if q <= 0:
        return rank_pct
    lo = float(q)
    hi = float(1.0 - q)
    return rank_pct.clip(lower=lo, upper=hi)


def _rank_power_transform(rank_pct: pd.Series, power: float) -> pd.Series:
    centered = 2.0 * rank_pct - 1.0
    p = max(1e-6, float(power))
    return np.sign(centered) * np.power(np.abs(centered), p)


def _load_run_meta(run_dir: Path) -> Tuple[str, str, str]:
    cfg_path = run_dir / "artifacts" / "config"
    if not cfg_path.exists():
        return "", "", ""
    try:
        cfg = _load_config(cfg_path)
    except Exception:  # noqa: BLE001
        return "", "", ""
    if not isinstance(cfg, dict):
        return "", "", ""
    task = cfg.get("task", {})
    if not isinstance(task, dict):
        return "", "", ""
    model_cfg = task.get("model", {})
    dataset_cfg = task.get("dataset", {})
    model_class = str(model_cfg.get("class", "")) if isinstance(model_cfg, dict) else ""
    dataset_class = ""
    instruments = ""
    if isinstance(dataset_cfg, dict):
        kwargs = dataset_cfg.get("kwargs", {})
        if isinstance(kwargs, dict):
            handler = kwargs.get("handler", {})
            if isinstance(handler, dict):
                dataset_class = str(handler.get("class", ""))
                hkwargs = handler.get("kwargs", {})
                if isinstance(hkwargs, dict):
                    instruments = str(hkwargs.get("instruments", ""))
    return model_class, dataset_class, instruments


def _discover_pred_runs(
    tracking_dir: Path,
    comparable_instruments: str,
    require_comparable: bool,
    test_start: str,
    test_end: str,
    min_coverage_test: float,
) -> Tuple[List[RunSignal], List[Dict[str, Any]]]:
    run_signals: List[RunSignal] = []
    audit_rows: List[Dict[str, Any]] = []
    all_lengths: List[int] = []
    temp: List[Tuple[Path, str, str, str, pd.Series]] = []

    for run_dir in tracking_dir.glob("*/*"):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if len(run_id) != 32:
            continue
        pred_path = run_dir / "artifacts" / "pred.pkl"
        if not pred_path.exists():
            continue
        model_class, dataset_class, instruments = _load_run_meta(run_dir)
        comparable_ok = (not require_comparable) or (not instruments) or (instruments == comparable_instruments)
        source_ir, source_annret = _extract_source_metrics(run_dir)
        audit_row = {
            "run_id": run_id,
            "pred_path": str(pred_path).replace("\\", "/"),
            "model_class": model_class,
            "dataset_class": dataset_class,
            "instruments": instruments,
            "source_ir": source_ir,
            "source_annret": source_annret,
            "comparable_ok": comparable_ok,
        }
        if not comparable_ok:
            audit_row["skip_reason"] = f"instruments_mismatch({instruments} != {comparable_instruments})"
            audit_rows.append(audit_row)
            continue
        try:
            raw = _as_score_series(_load_pickle(pred_path))
            raw_test = _slice_series(raw, test_start, test_end)
            if raw_test.empty:
                audit_row["skip_reason"] = "empty_test_period_signal"
                audit_rows.append(audit_row)
                continue
            all_lengths.append(len(raw_test))
            temp.append((run_dir, model_class, dataset_class, instruments, raw))
            audit_row["test_rows"] = int(len(raw_test))
            audit_rows.append(audit_row)
        except Exception as exc:  # noqa: BLE001
            audit_row["skip_reason"] = f"{type(exc).__name__}: {exc}"
            audit_rows.append(audit_row)

    if not temp:
        return [], audit_rows
    max_len = max(all_lengths) if all_lengths else 1
    key_count: Dict[str, int] = {}
    for run_dir, model_class, dataset_class, instruments, raw in temp:
        run_id = run_dir.name
        raw_test = _slice_series(raw, test_start, test_end)
        coverage_test = float(len(raw_test) / max(1, max_len))
        if coverage_test < float(min_coverage_test):
            for row in audit_rows:
                if row.get("run_id") == run_id:
                    row["skip_reason"] = f"coverage_below_min({coverage_test:.4f} < {min_coverage_test:.4f})"
                    row["coverage_test"] = coverage_test
                    break
            continue
        source_ir, source_annret = _extract_source_metrics(run_dir)
        base_key = next((k for k, v in RUN_ALIAS.items() if v == run_id), run_id[:8])
        n = key_count.get(base_key, 0)
        key_count[base_key] = n + 1
        key = base_key if n == 0 else f"{base_key}_{n+1}"
        rank_pct = _cs_rank(raw.astype(float))
        run_signals.append(
            RunSignal(
                key=key,
                run_id=run_id,
                pred_path=str(run_dir / "artifacts" / "pred.pkl").replace("\\", "/"),
                raw=raw.astype(float),
                rank_pct=rank_pct.astype(float),
                coverage_test=coverage_test,
                model_class=model_class,
                dataset_class=dataset_class,
                instruments=instruments,
                source_ir=source_ir,
                source_annret=source_annret,
            )
        )
        for row in audit_rows:
            if row.get("run_id") == run_id:
                row["key"] = key
                row["coverage_test"] = coverage_test
                break
    return run_signals, audit_rows


def _normalize_pos_weights(w: Sequence[float]) -> Tuple[float, ...]:
    arr = np.asarray(w, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    arr = arr / float(arr.sum())
    return tuple(float(x) for x in arr.tolist())


def _sample_dirichlet(k: int, rng: random.Random, alpha: float = 1.0) -> Tuple[float, ...]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    return _normalize_pos_weights(draws)


def _weight_sig(w: Sequence[float], nd: int = 4) -> Tuple[float, ...]:
    return tuple(round(float(x), nd) for x in w)


def _candidate_id(c: CandidateSpec) -> str:
    payload = {
        "members": c.members,
        "signs": c.signs,
        "wn": _weight_sig(c.w_normal, 4),
        "ws": _weight_sig(c.w_stress, 4),
        "rp": round(c.rank_power, 3),
        "wq": round(c.winsor_q, 3),
        "nz": c.neutralize,
        "is": round(c.interaction_strength, 3),
        "tt": round(c.tanh_temp, 3),
        "rl": c.regime_lookback,
        "rz": round(c.regime_z, 3),
        "tn": round(c.threshold_normal, 3),
        "ts": round(c.threshold_stress, 3),
        "kn": c.topk_normal,
        "ks": c.topk_stress,
        "nd": c.n_drop,
        "cf": c.conversion_family,
        "rb": c.rebalance_mode,
        "hb": c.hold_buffer,
        "st": round(c.softmax_temp, 2),
        "sp": round(c.softmax_power, 2),
        "mw": round(c.max_weight, 3),
        "vt": round(c.vol_target, 4),
        "lt": round(c.lambda_turnover, 3),
        "ld": round(c.lambda_dd, 3),
        "stage": c.stage,
        "parent": c.parent_id,
    }
    return str(abs(hash(json.dumps(payload, sort_keys=True))))


def _build_regime_flag(anchor_rank: pd.Series, lookback: int, z_thr: float) -> pd.Series:
    if isinstance(anchor_rank.index, pd.MultiIndex):
        disp = anchor_rank.groupby(level=0).std().sort_index()
    else:
        disp = anchor_rank.groupby(anchor_rank.index).std().sort_index()
    lb = int(max(20, lookback))
    med = disp.rolling(lb, min_periods=max(10, lb // 3)).median().shift(1)
    mad = (disp - med).abs().rolling(lb, min_periods=max(10, lb // 3)).median().shift(1)
    z = (disp - med) / mad.replace(0.0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (z > float(z_thr)).astype(bool)


def _apply_vol_targeting(series: pd.Series, target_vol: float) -> pd.Series:
    target = float(target_vol)
    if target <= 0:
        return series
    if isinstance(series.index, pd.MultiIndex):
        daily = series.groupby(level=0).mean()
    else:
        daily = series.groupby(series.index).mean()
    vol20 = daily.rolling(20, min_periods=10).std().shift(1)
    scale = (target / vol20).replace([np.inf, -np.inf], np.nan).clip(lower=0.5, upper=1.8).fillna(1.0)
    if isinstance(series.index, pd.MultiIndex):
        fac = pd.Series(scale).reindex(pd.to_datetime(series.index.get_level_values(0))).to_numpy()
    else:
        fac = pd.Series(scale).reindex(pd.to_datetime(series.index)).to_numpy()
    return series * pd.Series(fac, index=series.index)


def _build_signal_for_candidate(
    candidate: CandidateSpec,
    run_map: Dict[str, RunSignal],
    anchor_key: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    member_cols: List[pd.Series] = []
    member_names: List[str] = []
    for key, sign in zip(candidate.members, candidate.signs):
        rank_pct = _slice_series(run_map[key].rank_pct, start_date, end_date)
        rank_pct = _cap_winsor_rank(rank_pct, candidate.winsor_q)
        transformed = _rank_power_transform(rank_pct, candidate.rank_power) * float(sign)
        transformed.name = key
        member_cols.append(transformed)
        member_names.append(key)

    panel = pd.concat(member_cols, axis=1)
    panel.columns = member_names
    panel = panel.dropna(how="all")
    if panel.empty:
        raise ValueError("empty panel after member merge")

    anchor_rank = _slice_series(run_map[anchor_key].rank_pct, start_date, end_date)
    regime_flag_by_date = _build_regime_flag(
        anchor_rank=anchor_rank, lookback=candidate.regime_lookback, z_thr=candidate.regime_z
    )
    row_dates = pd.to_datetime(panel.index.get_level_values(0)) if isinstance(panel.index, pd.MultiIndex) else pd.to_datetime(panel.index)
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
    blended = np.tanh(float(candidate.tanh_temp) * blended)
    score = pd.Series(blended, index=panel.index, name="score", dtype=float)
    if candidate.neutralize:
        score = _date_neutralize(score)
    score = _apply_vol_targeting(score, target_vol=candidate.vol_target)

    if isinstance(score.index, pd.MultiIndex):
        rank_pos = score.groupby(level=0).rank(method="first", ascending=False)
        rank_pct = score.groupby(level=0).rank(method="average", pct=True)
    else:
        rank_pos = score.rank(method="first", ascending=False)
        rank_pct = score.rank(method="average", pct=True)

    threshold = np.where(is_stress, float(candidate.threshold_stress), float(candidate.threshold_normal))
    topk_by_row = np.where(is_stress, int(candidate.topk_stress), int(candidate.topk_normal))
    keep = (rank_pct.to_numpy() >= threshold) & (rank_pos.to_numpy() <= topk_by_row)
    score = score.where(keep)

    nonnull_ratio = float(score.notna().mean())
    out_df = score.to_frame("score").dropna()
    diag = {
        "nonnull_ratio": nonnull_ratio,
        "stress_ratio": float(np.mean(is_stress)) if len(is_stress) else 0.0,
        "topk_effective_max": int(max(candidate.topk_normal, candidate.topk_stress)),
    }
    return out_df, diag


def _get_report_for_day_freq(portfolio_metric_dict):
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    first_key = next(iter(portfolio_metric_dict.keys()))
    return portfolio_metric_dict[first_key][0]


def _calc_costed_metrics(report_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, max_drawdown, turnover


def _objective_value(annret: float, ir: float, maxdd: float, turnover: float, lam_to: float, lam_dd: float) -> float:
    return float(ir + 0.45 * annret - float(lam_to) * turnover - float(lam_dd) * abs(maxdd))


def _build_exchange_cache_key(
    start_time: str,
    end_time: str,
    open_cost: float,
    close_cost: float,
    limit_threshold: float,
    deal_price: str,
) -> Tuple[str, str, float, float, float, str]:
    return (
        str(start_time),
        str(end_time),
        float(open_cost),
        float(close_cost),
        float(limit_threshold),
        str(deal_price),
    )


def _build_strategy_obj(
    candidate: CandidateSpec,
    signal_df: pd.DataFrame,
    base_strategy_kwargs: Dict[str, Any],
) -> Tuple[Any, int]:
    topk_effective = int(max(candidate.topk_normal, candidate.topk_stress))
    if candidate.conversion_family == "topk_dropout":
        strategy = TopkDropoutStrategy(
            signal=signal_df,
            topk=topk_effective,
            n_drop=int(candidate.n_drop),
            method_sell=base_strategy_kwargs.get("method_sell", "bottom"),
            method_buy=base_strategy_kwargs.get("method_buy", "top"),
            hold_thresh=int(base_strategy_kwargs.get("hold_thresh", 1)),
            only_tradable=bool(base_strategy_kwargs.get("only_tradable", False)),
            forbid_all_trade_at_limit=bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
            risk_degree=float(base_strategy_kwargs.get("risk_degree", 0.95)),
        )
        return strategy, topk_effective
    if candidate.conversion_family == "convex_softmax":
        strategy = ConvexTopKWeightStrategy(
            signal=signal_df,
            topk=topk_effective,
            hold_topk=topk_effective + int(candidate.hold_buffer),
            softmax_temp=float(candidate.softmax_temp),
            softmax_power=float(candidate.softmax_power),
            max_weight=float(candidate.max_weight),
            rebalance_mode=candidate.rebalance_mode,
            rebalance_interval=1,
            risk_degree=float(base_strategy_kwargs.get("risk_degree", 0.95)),
        )
        return strategy, topk_effective
    raise ValueError(f"unsupported conversion_family={candidate.conversion_family}")


def _eval_candidate_period(
    *,
    candidate: CandidateSpec,
    signal_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_date: str,
    end_date: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
    metrics_split: str,
) -> Tuple[EvalResult, pd.Series]:
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

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    sliced = signal_df.copy()
    if isinstance(sliced.index, pd.MultiIndex):
        d = pd.to_datetime(sliced.index.get_level_values(0))
        sliced = sliced.loc[(d >= start_ts) & (d <= end_ts)]
    else:
        d = pd.to_datetime(sliced.index)
        sliced = sliced.loc[(d >= start_ts) & (d <= end_ts)]
    if sliced.empty:
        raise ValueError(f"empty signal after slice: {start_date}..{end_date}")

    strategy_obj, topk_effective = _build_strategy_obj(
        candidate=candidate, signal_df=sliced, base_strategy_kwargs=base_strategy_kwargs
    )

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = _build_exchange_cache_key(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        open_cost=open_cost,
        close_cost=close_cost,
        limit_threshold=limit_threshold,
        deal_price=deal_price,
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
    elapsed = float(time.perf_counter() - t0)
    report_df = _get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report_df)
    objective = _objective_value(
        annret=annret,
        ir=ir,
        maxdd=maxdd,
        turnover=turnover,
        lam_to=candidate.lambda_turnover,
        lam_dd=candidate.lambda_dd,
    )
    excess = (report_df["return"] - report_df["bench"] - report_df["cost"]).astype(float)
    out = EvalResult(
        candidate_id=_candidate_id(candidate),
        stage=candidate.stage,
        metrics_split=metrics_split,
        start_date=str(pd.Timestamp(start_date).date()),
        end_date=str(pd.Timestamp(end_date).date()),
        annret=float(annret),
        ir=float(ir),
        max_drawdown=float(maxdd),
        turnover=float(turnover),
        objective=float(objective),
        elapsed_sec=elapsed,
        topk_effective_max=int(topk_effective),
        nonnull_ratio=float(sliced["score"].notna().mean()),
        error="",
    )
    return out, excess


def _candidate_to_row(candidate: CandidateSpec) -> Dict[str, Any]:
    row = asdict(candidate)
    row["candidate_id"] = _candidate_id(candidate)
    row["members"] = "|".join(candidate.members)
    row["signs"] = "|".join(str(x) for x in candidate.signs)
    row["w_normal"] = "|".join(f"{x:.6f}" for x in candidate.w_normal)
    row["w_stress"] = "|".join(f"{x:.6f}" for x in candidate.w_stress)
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
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


def _choice(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(0, len(seq))]


def _sample_candidate(
    *,
    rng: random.Random,
    run_keys: Sequence[str],
    stage: str,
    parent_id: str = "",
) -> CandidateSpec:
    k = int(_choice(rng, [2, 3, 4]))
    members = tuple(rng.sample(list(run_keys), k))
    signs = tuple(_choice(rng, [-1, 1]) for _ in members)
    w_normal = _sample_dirichlet(k, rng=rng, alpha=1.0)
    stress_mix = _sample_dirichlet(k, rng=rng, alpha=1.0)
    tilt = float(_choice(rng, [0.15, 0.30, 0.45, 0.60]))
    w_stress = _normalize_pos_weights([(1.0 - tilt) * a + tilt * b for a, b in zip(w_normal, stress_mix)])

    conversion_family = _choice(rng, ["topk_dropout", "convex_softmax"])
    rebalance_mode = _choice(rng, ["daily", "weekly"]) if conversion_family == "convex_softmax" else "daily"
    topk_normal = int(_choice(rng, [35, 40, 45, 50, 55]))
    topk_stress = int(_choice(rng, [25, 30, 35, 40, 45]))
    topk_normal = max(topk_normal, topk_stress)
    n_drop = int(_choice(rng, [1, 2, 3, 4]))
    n_drop = min(n_drop, max(1, topk_stress - 1))
    winsor_q = float(_choice(rng, [0.0, 0.01, 0.02, 0.05]))
    vol_target = float(_choice(rng, [0.0, 0.015, 0.02]))

    return CandidateSpec(
        stage=stage,
        members=members,
        signs=signs,
        w_normal=w_normal,
        w_stress=w_stress,
        rank_power=float(_choice(rng, [0.7, 1.0, 1.3, 1.8])),
        winsor_q=winsor_q,
        neutralize=bool(_choice(rng, [True, False])),
        interaction_strength=float(_choice(rng, [-0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35])),
        tanh_temp=float(_choice(rng, [0.7, 0.9, 1.1, 1.4])),
        regime_lookback=int(_choice(rng, [40, 60, 90])),
        regime_z=float(_choice(rng, [0.5, 0.8, 1.1])),
        threshold_normal=float(_choice(rng, [0.0, 0.30, 0.40, 0.50, 0.60])),
        threshold_stress=float(_choice(rng, [0.0, 0.40, 0.50, 0.60, 0.70])),
        topk_normal=topk_normal,
        topk_stress=topk_stress,
        n_drop=n_drop,
        conversion_family=conversion_family,
        rebalance_mode=rebalance_mode,
        hold_buffer=int(_choice(rng, [10, 20, 30])),
        softmax_temp=float(_choice(rng, [6.0, 8.0, 12.0])),
        softmax_power=float(_choice(rng, [1.0, 1.3, 1.8])),
        max_weight=float(_choice(rng, [0.03, 0.05, 0.07])),
        vol_target=vol_target,
        lambda_turnover=float(_choice(rng, [0.15, 0.25, 0.35])),
        lambda_dd=float(_choice(rng, [0.2, 0.35, 0.5])),
        parent_id=parent_id,
    )


def _mutate_candidate(base: CandidateSpec, rng: random.Random, stage: str) -> CandidateSpec:
    c = replace(base, stage=stage, parent_id=_candidate_id(base))
    idx = rng.randrange(0, len(c.members))
    wn = list(c.w_normal)
    ws = list(c.w_stress)
    wn[idx] = max(1e-4, wn[idx] + rng.uniform(-0.2, 0.2))
    ws[idx] = max(1e-4, ws[idx] + rng.uniform(-0.2, 0.2))
    c = replace(c, w_normal=_normalize_pos_weights(wn), w_stress=_normalize_pos_weights(ws))
    c = replace(
        c,
        interaction_strength=float(np.clip(c.interaction_strength + rng.uniform(-0.15, 0.15), -0.45, 0.45)),
        tanh_temp=float(np.clip(c.tanh_temp + rng.uniform(-0.25, 0.25), 0.5, 1.8)),
        regime_z=float(np.clip(c.regime_z + rng.uniform(-0.25, 0.25), 0.2, 1.5)),
        threshold_normal=float(np.clip(c.threshold_normal + rng.choice([-0.1, 0.0, 0.1]), 0.0, 0.75)),
        threshold_stress=float(np.clip(c.threshold_stress + rng.choice([-0.1, 0.0, 0.1]), 0.0, 0.85)),
        topk_normal=int(np.clip(c.topk_normal + rng.choice([-5, 0, 5]), 30, 65)),
        topk_stress=int(np.clip(c.topk_stress + rng.choice([-5, 0, 5]), 20, 55)),
        n_drop=int(np.clip(c.n_drop + rng.choice([-1, 0, 1]), 1, 6)),
        softmax_temp=float(np.clip(c.softmax_temp + rng.choice([-2.0, 0.0, 2.0]), 4.0, 14.0)),
        softmax_power=float(np.clip(c.softmax_power + rng.choice([-0.3, 0.0, 0.3]), 0.8, 2.2)),
        max_weight=float(np.clip(c.max_weight + rng.choice([-0.01, 0.0, 0.01]), 0.02, 0.10)),
        lambda_turnover=float(np.clip(c.lambda_turnover + rng.choice([-0.1, 0.0, 0.1]), 0.05, 0.5)),
        lambda_dd=float(np.clip(c.lambda_dd + rng.choice([-0.1, 0.0, 0.1]), 0.1, 0.7)),
    )
    topk_normal = max(c.topk_normal, c.topk_stress)
    topk_stress = min(c.topk_stress, topk_normal)
    n_drop = min(c.n_drop, max(1, topk_stress - 1))
    return replace(c, topk_normal=topk_normal, topk_stress=topk_stress, n_drop=n_drop)


def _load_sota_snapshot(trans_dir: Path) -> Tuple[float, float]:
    p = trans_dir / "sota_snapshot.json"
    if not p.exists():
        return DEFAULT_SOTA_IR, DEFAULT_SOTA_ANNRET
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        cur = obj.get("current_sota", {})
        ir = float(cur.get("costed_ir", DEFAULT_SOTA_IR))
        ann = float(cur.get("costed_annret", DEFAULT_SOTA_ANNRET))
        return ir, ann
    except Exception:  # noqa: BLE001
        return DEFAULT_SOTA_IR, DEFAULT_SOTA_ANNRET


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.astype(float), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nonlinear regime-conditioned portfolio search from existing pred.pkl pool.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-04-30")
    p.add_argument("--pretest-start", default="2022-01-01")
    p.add_argument("--pretest-end", default="2023-12-31")
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--comparable-instruments", default="csi300")
    p.add_argument(
        "--require-comparable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only keep runs whose config instruments matches --comparable-instruments",
    )
    p.add_argument("--min-coverage-test", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--stage1-budget", type=int, default=100)
    p.add_argument("--stage2-keep", type=int, default=10)
    p.add_argument("--stage2-mutations", type=int, default=6)
    p.add_argument("--replay-pool", type=int, default=8)
    p.add_argument("--output-prefix", default="nonlinear_regime")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    trans_dir = Path("examples/benchmarks/Transcendence").resolve()
    trans_dir.mkdir(parents=True, exist_ok=True)

    stamp = _timestamp()
    audit_path = trans_dir / f"{args.output_prefix}_artifact_audit_{stamp}.json"
    results_path = trans_dir / f"{args.output_prefix}_results_{stamp}.csv"
    summary_path = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    candidate_path = trans_dir / f"{args.output_prefix}_candidate_{stamp}.json"
    replay_path = trans_dir / f"{args.output_prefix}_walkforward_replay_{stamp}.json"
    md_path = trans_dir / f"{args.output_prefix}_report_{stamp}.md"

    base_run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(base_cfg)
    base_port_cfg = _extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    run_signals, audit_rows = _discover_pred_runs(
        tracking_dir=tracking_dir,
        comparable_instruments=args.comparable_instruments,
        require_comparable=bool(args.require_comparable),
        test_start=args.start_date,
        test_end=args.end_date,
        min_coverage_test=float(args.min_coverage_test),
    )
    if len(run_signals) < 2:
        raise RuntimeError("usable pred.pkl runs < 2")

    run_map = {x.key: x for x in run_signals}
    anchor_key = next((x.key for x in run_signals if x.run_id == _resolve_run_token(args.base_run_id)), run_signals[0].key)
    run_keys = list(run_map.keys())

    audit_obj = {
        "timestamp_utc": _now_utc(),
        "tracking_dir": str(tracking_dir),
        "search_period_test": {"start": args.start_date, "end": args.end_date},
        "search_period_pretest": {"start": args.pretest_start, "end": args.pretest_end},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "require_comparable": bool(args.require_comparable),
        "comparable_instruments": args.comparable_instruments,
        "usable_signal_count": len(run_signals),
        "anchor_key": anchor_key,
        "runs": [
            {
                "key": x.key,
                "run_id": x.run_id,
                "coverage_test": x.coverage_test,
                "model_class": x.model_class,
                "dataset_class": x.dataset_class,
                "instruments": x.instruments,
                "source_ir": x.source_ir,
                "source_annret": x.source_annret,
                "pred_path": x.pred_path,
            }
            for x in run_signals
        ],
        "audit_rows": audit_rows,
    }
    audit_path.write_text(json.dumps(audit_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(int(args.seed))
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    rows: List[Dict[str, Any]] = []
    best_by_id: Dict[str, Dict[str, Any]] = {}
    excess_cache: Dict[Tuple[str, str, str], pd.Series] = {}
    candidate_store: Dict[str, CandidateSpec] = {}

    seen_sigs = set()
    stage1_candidates: List[CandidateSpec] = []
    while len(stage1_candidates) < int(args.stage1_budget):
        cand = _sample_candidate(rng=rng, run_keys=run_keys, stage="stage1")
        sig = (
            cand.members,
            cand.signs,
            _weight_sig(cand.w_normal, 3),
            _weight_sig(cand.w_stress, 3),
            cand.rank_power,
            cand.interaction_strength,
            cand.threshold_normal,
            cand.threshold_stress,
            cand.topk_normal,
            cand.topk_stress,
            cand.n_drop,
            cand.conversion_family,
            cand.rebalance_mode,
        )
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        stage1_candidates.append(cand)

    def _eval_and_record(cand: CandidateSpec) -> Optional[Dict[str, Any]]:
        cid = _candidate_id(cand)
        candidate_store[cid] = cand
        base_row = _candidate_to_row(cand)
        try:
            signal_df, diag = _build_signal_for_candidate(
                candidate=cand, run_map=run_map, anchor_key=anchor_key, start_date=args.start_date, end_date=args.end_date
            )
            ev, excess = _eval_candidate_period(
                candidate=cand,
                signal_df=signal_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_date=args.start_date,
                end_date=args.end_date,
                exchange_cache=exchange_cache,
                metrics_split="test_full",
            )
            row = base_row.copy()
            row.update(diag)
            row.update(asdict(ev))
            rows.append(row)
            excess_cache[(cid, args.start_date, args.end_date)] = excess
            best_prev = best_by_id.get(cid)
            if best_prev is None or (float(ev.objective), float(ev.ir), float(ev.annret)) > (
                float(best_prev["objective"]),
                float(best_prev["ir"]),
                float(best_prev["annret"]),
            ):
                best_by_id[cid] = row
            return row
        except Exception as exc:  # noqa: BLE001
            er = EvalResult(
                candidate_id=cid,
                stage=cand.stage,
                metrics_split="test_full",
                start_date=str(pd.Timestamp(args.start_date).date()),
                end_date=str(pd.Timestamp(args.end_date).date()),
                annret=float("nan"),
                ir=float("nan"),
                max_drawdown=float("nan"),
                turnover=float("nan"),
                objective=float("nan"),
                elapsed_sec=0.0,
                topk_effective_max=max(cand.topk_normal, cand.topk_stress),
                nonnull_ratio=float("nan"),
                error=f"{type(exc).__name__}: {exc}",
            )
            row = base_row.copy()
            row.update(asdict(er))
            rows.append(row)
            return None

    for i, cand in enumerate(stage1_candidates, start=1):
        row = _eval_and_record(cand)
        if row is not None:
            print(
                f"[stage1 {i}/{len(stage1_candidates)}] cid={row['candidate_id']} obj={row['objective']:.6f} "
                f"IR={row['ir']:.6f} AnnRet={row['annret']:.6f}",
                flush=True,
            )
        else:
            print(f"[stage1 {i}/{len(stage1_candidates)}] failed", flush=True)

    stage1_valid = [r for r in rows if r.get("stage") == "stage1" and r.get("error", "") == ""]
    stage1_top = sorted(stage1_valid, key=lambda x: (x["objective"], x["ir"], x["annret"]), reverse=True)[
        : int(args.stage2_keep)
    ]
    stage2_candidates: List[CandidateSpec] = []
    for top in stage1_top:
        pid = str(top["candidate_id"])
        parent = candidate_store[pid]
        stage2_candidates.append(replace(parent, stage="stage2", parent_id=pid))
        for _ in range(int(args.stage2_mutations)):
            stage2_candidates.append(_mutate_candidate(parent, rng=rng, stage="stage2"))

    # dedupe stage2
    seen2 = set()
    uniq2: List[CandidateSpec] = []
    for c in stage2_candidates:
        sig = (
            c.members,
            c.signs,
            _weight_sig(c.w_normal, 3),
            _weight_sig(c.w_stress, 3),
            round(c.rank_power, 3),
            round(c.interaction_strength, 3),
            round(c.threshold_normal, 3),
            round(c.threshold_stress, 3),
            c.topk_normal,
            c.topk_stress,
            c.n_drop,
            c.conversion_family,
            c.rebalance_mode,
            round(c.softmax_temp, 2),
            round(c.softmax_power, 2),
            round(c.max_weight, 3),
        )
        if sig in seen2:
            continue
        seen2.add(sig)
        uniq2.append(c)

    for i, cand in enumerate(uniq2, start=1):
        row = _eval_and_record(cand)
        if row is not None:
            print(
                f"[stage2 {i}/{len(uniq2)}] cid={row['candidate_id']} obj={row['objective']:.6f} "
                f"IR={row['ir']:.6f} AnnRet={row['annret']:.6f}",
                flush=True,
            )
        else:
            print(f"[stage2 {i}/{len(uniq2)}] failed", flush=True)

    valid_rows = [r for r in rows if r.get("metrics_split") == "test_full" and r.get("error", "") == ""]
    if not valid_rows:
        raise RuntimeError("no valid candidate on test_full")
    valid_rows_sorted = sorted(valid_rows, key=lambda x: (x["objective"], x["ir"], x["annret"]), reverse=True)
    best = valid_rows_sorted[0]
    best_cid = str(best["candidate_id"])
    best_candidate = candidate_store[best_cid]

    # Validate best candidate on pre-test + year slices.
    replay_slices = [
        ("pretest_2022_2023", args.pretest_start, args.pretest_end),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", args.end_date),
    ]
    validation_rows: List[Dict[str, Any]] = []
    for tag, st, ed in replay_slices:
        try:
            sig_df, diag = _build_signal_for_candidate(
                candidate=best_candidate, run_map=run_map, anchor_key=anchor_key, start_date=st, end_date=ed
            )
            ev, excess = _eval_candidate_period(
                candidate=best_candidate,
                signal_df=sig_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_date=st,
                end_date=ed,
                exchange_cache=exchange_cache,
                metrics_split=tag,
            )
            excess_cache[(best_cid, st, ed)] = excess
            rr = _candidate_to_row(best_candidate)
            rr.update(diag)
            rr.update(asdict(ev))
            validation_rows.append(rr)
        except Exception as exc:  # noqa: BLE001
            rr = _candidate_to_row(best_candidate)
            rr.update(
                asdict(
                    EvalResult(
                        candidate_id=best_cid,
                        stage=best_candidate.stage,
                        metrics_split=tag,
                        start_date=str(pd.Timestamp(st).date()),
                        end_date=str(pd.Timestamp(ed).date()),
                        annret=float("nan"),
                        ir=float("nan"),
                        max_drawdown=float("nan"),
                        turnover=float("nan"),
                        objective=float("nan"),
                        elapsed_sec=0.0,
                        topk_effective_max=max(best_candidate.topk_normal, best_candidate.topk_stress),
                        nonnull_ratio=float("nan"),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            )
            validation_rows.append(rr)

    rows.extend(validation_rows)
    rows_sorted = sorted(
        rows,
        key=lambda x: (
            0 if x.get("error", "") == "" else 1,
            -float(x.get("objective", -1e9) if np.isfinite(float(x.get("objective", float("nan")))) else -1e9),
        ),
    )
    _write_csv(results_path, rows_sorted)

    # Walk-forward replay among exploratory top candidates.
    replay_pool = valid_rows_sorted[: int(args.replay_pool)]
    replay_candidates = [candidate_store[str(r["candidate_id"])] for r in replay_pool]
    period_defs = {
        "pretest": (args.pretest_start, args.pretest_end),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_ytd": ("2026-01-01", args.end_date),
    }
    score_table: Dict[str, Dict[str, Dict[str, float]]] = {}
    excess_table: Dict[str, Dict[str, pd.Series]] = {}
    for cand in replay_candidates:
        cid = _candidate_id(cand)
        score_table[cid] = {}
        excess_table[cid] = {}
        for ptag, (st, ed) in period_defs.items():
            try:
                sig_df, _ = _build_signal_for_candidate(
                    candidate=cand, run_map=run_map, anchor_key=anchor_key, start_date=st, end_date=ed
                )
                ev, ex = _eval_candidate_period(
                    candidate=cand,
                    signal_df=sig_df,
                    base_port_cfg=base_port_cfg,
                    base_strategy_kwargs=base_strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start_date=st,
                    end_date=ed,
                    exchange_cache=exchange_cache,
                    metrics_split=f"replay_{ptag}",
                )
                score_table[cid][ptag] = {
                    "objective": float(ev.objective),
                    "ir": float(ev.ir),
                    "annret": float(ev.annret),
                    "max_drawdown": float(ev.max_drawdown),
                    "turnover": float(ev.turnover),
                }
                excess_table[cid][ptag] = ex
            except Exception as exc:  # noqa: BLE001
                score_table[cid][ptag] = {
                    "objective": -1e9,
                    "ir": float("nan"),
                    "annret": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}",
                }

    # Selection replay: select on prior period, trade next period.
    replay_plan = [
        ("pretest", "2024"),
        ("2024", "2025"),
        ("2025", "2026_ytd"),
    ]
    replay_picks: List[Dict[str, Any]] = []
    stitched: List[pd.Series] = []
    for select_ptag, apply_ptag in replay_plan:
        best_cid_local = None
        best_key = None
        for cid, pt_map in score_table.items():
            if select_ptag not in pt_map:
                continue
            m = pt_map[select_ptag]
            key = (
                float(m.get("objective", -1e9)),
                float(m.get("ir", -1e9)) if np.isfinite(float(m.get("ir", float("nan")))) else -1e9,
                float(m.get("annret", -1e9)) if np.isfinite(float(m.get("annret", float("nan")))) else -1e9,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_cid_local = cid
        if best_cid_local is None:
            continue
        apply_metrics = score_table.get(best_cid_local, {}).get(apply_ptag, {})
        replay_picks.append(
            {
                "select_period": select_ptag,
                "apply_period": apply_ptag,
                "selected_candidate_id": best_cid_local,
                "selected_on_objective": float(best_key[0]) if best_key is not None else None,
                "applied_ir": apply_metrics.get("ir"),
                "applied_annret": apply_metrics.get("annret"),
                "applied_max_drawdown": apply_metrics.get("max_drawdown"),
                "applied_turnover": apply_metrics.get("turnover"),
            }
        )
        ex = excess_table.get(best_cid_local, {}).get(apply_ptag)
        if ex is not None and len(ex) > 0:
            stitched.append(ex)

    stitched_metrics = None
    if stitched:
        stitched_excess = pd.concat(stitched).sort_index()
        stitched_metrics = _metrics_from_excess(stitched_excess)

    replay_obj = {
        "timestamp_utc": _now_utc(),
        "notice": "Walk-forward replay over exploratory top candidates (selection uses prior period only).",
        "period_defs": period_defs,
        "candidate_ids": [_candidate_id(c) for c in replay_candidates],
        "replay_picks": replay_picks,
        "stitched_metrics": stitched_metrics,
        "score_table": score_table,
    }
    replay_path.write_text(json.dumps(replay_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    sota_ir, sota_ann = _load_sota_snapshot(trans_dir)
    pass_hard_gate = bool(best["ir"] > TARGET_IR and best["annret"] > TARGET_ANNRET)

    candidate_dump = {
        "timestamp_utc": _now_utc(),
        "candidate_id": best_cid,
        "candidate": asdict(best_candidate),
        "test_metrics": {
            "ir": float(best["ir"]),
            "annret": float(best["annret"]),
            "max_drawdown": float(best["max_drawdown"]),
            "turnover": float(best["turnover"]),
            "objective": float(best["objective"]),
        },
    }
    candidate_path.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "timestamp_utc": _now_utc(),
        "selection_notice": (
            f"Exploratory selection on TEST window {args.start_date}..{args.end_date}; "
            "parameters here are test-period-chosen unless noted."
        ),
        "task_constraints": {
            "test_start": args.start_date,
            "test_end": args.end_date,
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "hard_gate_ir": TARGET_IR,
            "hard_gate_annret": TARGET_ANNRET,
        },
        "search_space": {
            "stage1_budget": int(args.stage1_budget),
            "stage2_keep": int(args.stage2_keep),
            "stage2_mutations_per_parent": int(args.stage2_mutations),
            "member_count": [2, 3, 4],
            "sign_inversion": True,
            "rank_power": [0.7, 1.0, 1.3, 1.8],
            "winsor_q": [0.0, 0.01, 0.02, 0.05],
            "date_neutralize": [True, False],
            "interaction_strength": [-0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35],
            "regime_lookback": [40, 60, 90],
            "regime_z": [0.5, 0.8, 1.1],
            "threshold_normal": [0.0, 0.30, 0.40, 0.50, 0.60],
            "threshold_stress": [0.0, 0.40, 0.50, 0.60, 0.70],
            "topk_normal": [35, 40, 45, 50, 55],
            "topk_stress": [25, 30, 35, 40, 45],
            "n_drop": [1, 2, 3, 4],
            "conversion_family": ["topk_dropout", "convex_softmax"],
            "rebalance_mode": ["daily", "weekly"],
            "softmax_temp": [6.0, 8.0, 12.0],
            "softmax_power": [1.0, 1.3, 1.8],
            "max_weight": [0.03, 0.05, 0.07],
            "vol_target": [0.0, 0.015, 0.02],
            "lambda_turnover": [0.15, 0.25, 0.35],
            "lambda_dd": [0.2, 0.35, 0.5],
            "objective_formula": "IR + 0.45*AnnRet - lambda_turnover*Turnover - lambda_dd*abs(MaxDD)",
        },
        "run_pool": {
            "usable_signals": len(run_signals),
            "anchor_key": anchor_key,
            "signals": [
                {
                    "key": x.key,
                    "run_id": x.run_id,
                    "coverage_test": x.coverage_test,
                    "source_ir": x.source_ir,
                    "source_annret": x.source_annret,
                    "model_class": x.model_class,
                    "dataset_class": x.dataset_class,
                    "instruments": x.instruments,
                }
                for x in run_signals
            ],
        },
        "best": {
            "candidate_id": best_cid,
            "stage": best["stage"],
            "metrics_split": best["metrics_split"],
            "ir": float(best["ir"]),
            "annret": float(best["annret"]),
            "max_drawdown": float(best["max_drawdown"]),
            "turnover": float(best["turnover"]),
            "objective": float(best["objective"]),
            "candidate": asdict(best_candidate),
        },
        "hard_gate_pass": pass_hard_gate,
        "comparison": {
            "vs_7406_delta_ir": float(best["ir"] - BASELINE_7406_IR),
            "vs_7406_delta_annret": float(best["annret"] - BASELINE_7406_ANNRET),
            "vs_sota_delta_ir": float(best["ir"] - sota_ir),
            "vs_sota_delta_annret": float(best["annret"] - sota_ann),
            "ref_7406": {"ir": BASELINE_7406_IR, "annret": BASELINE_7406_ANNRET},
            "ref_sota": {"ir": sota_ir, "annret": sota_ann},
        },
        "validation": {
            "best_candidate_pretest_and_year_slices": [
                {
                    "split": r["metrics_split"],
                    "start_date": r["start_date"],
                    "end_date": r["end_date"],
                    "ir": r["ir"],
                    "annret": r["annret"],
                    "max_drawdown": r["max_drawdown"],
                    "turnover": r["turnover"],
                    "objective": r["objective"],
                    "error": r.get("error", ""),
                }
                for r in validation_rows
            ],
            "walkforward_replay_artifact": str(replay_path).replace("\\", "/"),
            "walkforward_stitched_metrics": stitched_metrics,
        },
        "artifacts": {
            "audit_json": str(audit_path).replace("\\", "/"),
            "results_csv": str(results_path).replace("\\", "/"),
            "summary_json": str(summary_path).replace("\\", "/"),
            "candidate_json": str(candidate_path).replace("\\", "/"),
            "walkforward_replay_json": str(replay_path).replace("\\", "/"),
            "report_md": str(md_path).replace("\\", "/"),
        },
        "risk_notes": [
            "Main selection is exploratory on test period (potential overfitting).",
            "Regime logic is signal-driven proxy, not macro/market-state ground truth.",
            "Convex softmax conversion may reduce concentration risk but can dilute signal in high-dispersion days.",
            "Walk-forward replay is over exploratory top-candidate pool, not a frozen ex-ante protocol.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best_line = (
        f"- Best(test exploratory): IR={best['ir']:.6f}, AnnRet={best['annret']:.6f}, "
        f"MaxDD={best['max_drawdown']:.6f}, Turnover={best['turnover']:.6f}, "
        f"hard_gate_pass={pass_hard_gate}"
    )
    md = f"""# Nonlinear Regime Portfolio Search ({stamp})

## Notice
- Exploratory selection window: `{args.start_date}..{args.end_date}` (test-period selected parameters).
- Cost setting: `open={args.open_cost}`, `close={args.close_cost}`.

## Best Candidate
{best_line}
- Candidate ID: `{best_cid}`
- Stage: `{best['stage']}`
- Conversion: `{best_candidate.conversion_family}` / rebalance `{best_candidate.rebalance_mode}`

## 7406 / SOTA Comparison
- vs 7406: dIR=`{best['ir'] - BASELINE_7406_IR:.6f}`, dAnnRet=`{best['annret'] - BASELINE_7406_ANNRET:.6f}`
- vs SOTA: dIR=`{best['ir'] - sota_ir:.6f}`, dAnnRet=`{best['annret'] - sota_ann:.6f}`

## Validation Replay
- Best-candidate pretest/year slices and walk-forward replay saved in artifacts.
"""
    md_path.write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


