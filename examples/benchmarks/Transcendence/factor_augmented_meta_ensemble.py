#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repo root import when running directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import WeightStrategyBase


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"


@dataclass
class MemberSpec:
    key: str
    source: str
    path: str
    series: pd.Series
    is_anchor: bool = False
    is_expanded_factor: bool = False


class RebalanceMixin:
    def __init__(self, rebalance_mode: str = "weekly", rebalance_interval: int = 1, **kwargs):
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


class BufferedTopkWeightStrategy(RebalanceMixin, WeightStrategyBase):
    def __init__(
        self,
        *,
        signal,
        topk: int,
        hold_topk: Optional[int] = None,
        weight_mode: str = "equal",
        score_power: float = 1.0,
        rebalance_mode: str = "weekly",
        rebalance_interval: int = 1,
        **kwargs,
    ):
        self.signal = signal
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.weight_mode = str(weight_mode).lower()
        self.score_power = float(score_power)
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id_or_prefix: str) -> Path:
    token = str(run_id_or_prefix).strip()
    cands = [p for p in tracking_dir.glob(f"*/{token}") if (p / "artifacts").exists()]
    if not cands:
        cands = [p for p in tracking_dir.glob(f"*/{token}*") if (p / "artifacts").exists()]
    if not cands:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id_or_prefix}")
    if len(cands) > 1:
        exact = [x for x in cands if x.name == token]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"run token matched multiple runs: {run_id_or_prefix}")
    return cands[0]


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
    raise KeyError("cannot find port_analysis_config")


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _as_score_series(pred_obj: Any, preferred_col: str = "score") -> pd.Series:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float)
    if isinstance(pred_obj, pd.DataFrame):
        if preferred_col in pred_obj.columns:
            return pred_obj[preferred_col].astype(float)
        if pred_obj.shape[1] == 1:
            return pred_obj.iloc[:, 0].astype(float)
        return pred_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _as_label_series(label_obj: Any) -> pd.Series:
    if isinstance(label_obj, pd.Series):
        return label_obj.astype(float)
    if isinstance(label_obj, pd.DataFrame):
        if "label" in label_obj.columns:
            return label_obj["label"].astype(float)
        if label_obj.shape[1] >= 1:
            return label_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported label type: {type(label_obj)}")


def _normalize_mi_dt_inst(obj: pd.Series) -> pd.Series:
    idx = obj.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise TypeError("expected MultiIndex (datetime, instrument)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dt0 = pd.to_datetime(pd.Index(idx.get_level_values(0)[:32]), errors="coerce")
        dt1 = pd.to_datetime(pd.Index(idx.get_level_values(1)[:32]), errors="coerce")
    if dt0.notna().mean() < dt1.notna().mean():
        obj = obj.swaplevel(0, 1)
    obj = obj.sort_index()
    if isinstance(obj.index, pd.MultiIndex):
        obj.index = obj.index.set_names(["datetime", "instrument"] + list(obj.index.names[2:]))
    return obj


def _slice_period(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    s = _normalize_mi_dt_inst(series.astype(float))
    dt = pd.to_datetime(s.index.get_level_values(0))
    m = (dt >= pd.Timestamp(start_date)) & (dt <= pd.Timestamp(end_date))
    return s.loc[m]


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(method="average", pct=True)


def _center_rank(series: pd.Series) -> pd.Series:
    return 2.0 * _cs_rank(series) - 1.0


def _parse_metric_file(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        parts = path.read_text(encoding="utf-8").strip().split()
        if len(parts) < 2:
            return None
        return float(parts[1])
    except Exception:  # noqa: BLE001
        return None


def _read_source_metrics(run_dir: Path) -> Tuple[float | None, float | None]:
    metric_dir = run_dir / "metrics"
    ir = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.information_ratio")
    ann = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.annualized_return")
    if ir is None:
        ir = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.information_ratio")
    if ann is None:
        ann = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.annualized_return")
    return ir, ann


def _discover_top_mlruns(
    tracking_dir: Path,
    include_run_ids: Sequence[str],
    start_date: str,
    end_date: str,
    top_n: int,
) -> List[Tuple[str, str, float]]:
    forced = {str(x).strip() for x in include_run_ids if str(x).strip()}
    rows: List[Tuple[str, str, float]] = []
    for p in tracking_dir.glob("*/*"):
        if not p.is_dir() or len(p.name) != 32:
            continue
        pred_path = p / "artifacts" / "pred.pkl"
        if not pred_path.exists():
            continue
        try:
            obj = _load_pickle(pred_path)
            s = _slice_period(_as_score_series(obj), start_date, end_date)
            if s.empty:
                continue
            coverage = float(s.index.get_level_values(0).nunique())
            ir, _ = _read_source_metrics(p)
            score = float(ir) if ir is not None and np.isfinite(ir) else -999.0
            rows.append((p.name, str(pred_path), score + coverage / 10000.0))
        except Exception:  # noqa: BLE001
            continue
    rows_sorted = sorted(rows, key=lambda x: x[2], reverse=True)
    picked: List[Tuple[str, str, float]] = []
    used = set()
    for run_id in forced:
        try:
            rd = _find_run_dir(tracking_dir, run_id)
            pred_path = rd / "artifacts" / "pred.pkl"
            if pred_path.exists():
                picked.append((rd.name, str(pred_path), 999.0))
                used.add(rd.name)
        except Exception:  # noqa: BLE001
            continue
    for rid, path, sc in rows_sorted:
        if rid in used:
            continue
        picked.append((rid, path, sc))
        used.add(rid)
        if len(picked) >= top_n + len(forced):
            break
    return picked


def _load_sota_refs(trans_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    snapshot_path = trans_dir / "sota_snapshot.json"
    if snapshot_path.exists():
        try:
            out["sota_snapshot"] = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            out["sota_snapshot_error"] = "parse_error"
    conv_paths = sorted(trans_dir.glob("signal_portfolio_conversion_transcendence_summary_*.json"))
    if conv_paths:
        p = conv_paths[-1]
        try:
            out["signal_conversion_summary_path"] = str(p)
            out["signal_conversion_summary"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            out["signal_conversion_summary_error"] = "parse_error"
    nonlinear_paths = sorted(trans_dir.glob("nonlinear_regime_summary_*.json"))
    if nonlinear_paths:
        p = nonlinear_paths[-1]
        try:
            out["nonlinear_summary_path"] = str(p)
            out["nonlinear_summary"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            out["nonlinear_summary_error"] = "parse_error"
    return out


def _extract_referenced_run_ids(refs: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    snapshot = refs.get("sota_snapshot", {})
    if isinstance(snapshot, dict):
        cur = snapshot.get("current_sota", {})
        if isinstance(cur, dict) and cur.get("run_id"):
            ids.append(str(cur["run_id"]))
        latest = snapshot.get("latest_verified_strategy_candidate", {})
        if isinstance(latest, dict) and latest.get("run_id"):
            ids.append(str(latest["run_id"]))
    conv = refs.get("signal_conversion_summary", {})
    if isinstance(conv, dict):
        best_by_signal = conv.get("best_by_signal", {})
        if isinstance(best_by_signal, dict):
            for v in best_by_signal.values():
                if isinstance(v, dict) and v.get("run_id"):
                    ids.append(str(v["run_id"]))
    nonlinear = refs.get("nonlinear_summary", {})
    if isinstance(nonlinear, dict):
        run_pool = nonlinear.get("run_pool", {})
        if isinstance(run_pool, dict):
            for item in run_pool.get("signals", []):
                if isinstance(item, dict) and item.get("run_id"):
                    ids.append(str(item["run_id"]))
    return list(dict.fromkeys(ids))


def _build_prior(member_keys: Sequence[str], anchor_key: str, expanded_key: str | None) -> np.ndarray:
    n = len(member_keys)
    w = np.zeros(n, dtype=float)
    key_to_i = {k: i for i, k in enumerate(member_keys)}
    if anchor_key in key_to_i:
        w[key_to_i[anchor_key]] = 0.74
    if expanded_key is not None and expanded_key in key_to_i:
        w[key_to_i[expanded_key]] = 0.08
    rest_idx = [i for i in range(n) if w[i] == 0.0]
    rem = max(0.0, 1.0 - float(w.sum()))
    if rest_idx:
        fill = rem / len(rest_idx)
        for i in rest_idx:
            w[i] = fill
    if not rest_idx and w.sum() != 1.0:
        w = w / max(1e-12, w.sum())
    return w


def _make_regime_quarters(dates: pd.DatetimeIndex) -> List[pd.Period]:
    return sorted(pd.PeriodIndex(dates, freq="Q").unique())


def _fit_sparse_ridge(
    x: np.ndarray,
    y: np.ndarray,
    w_prior: np.ndarray,
    w_prev: np.ndarray,
    alpha: float,
    corr_pen: float,
    prior_pen: float,
    turnover_pen: float,
    max_members: int,
) -> np.ndarray:
    n, d = x.shape
    if n <= d + 5:
        return w_prior.copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xtx = (x.T @ x) / float(n)
    xty = (x.T @ y) / float(n)
    corr = np.corrcoef(x, rowvar=False)
    if np.isscalar(corr):
        corr = np.eye(d, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    reg = float(alpha + prior_pen + turnover_pen)
    a = xtx + reg * np.eye(d) + float(corr_pen) * corr
    b = xty + float(prior_pen) * w_prior + float(turnover_pen) * w_prev
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b

    k = max(1, min(int(max_members), d))
    if k < d:
        idx = np.argsort(np.abs(w))[::-1]
        keep = idx[:k]
        mask = np.zeros(d, dtype=bool)
        mask[keep] = True
        w = np.where(mask, w, 0.0)

    l1 = float(np.sum(np.abs(w)))
    if l1 <= 1e-12:
        return w_prior.copy()
    return w / l1


def _daily_ic(pred: np.ndarray, y: np.ndarray, date_codes: np.ndarray) -> Tuple[float, float]:
    df = pd.DataFrame({"d": date_codes, "p": pred, "y": y})
    vals = []
    for _, g in df.groupby("d"):
        if len(g) < 30:
            continue
        c = g["p"].corr(g["y"], method="spearman")
        if pd.notna(c):
            vals.append(float(c))
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    ir = float(np.sqrt(252.0) * m / s) if s > 1e-12 else float("nan")
    return m, ir


def _ann_proxy_topk(pred: np.ndarray, y: np.ndarray, date_codes: np.ndarray, topk: int = 55) -> float:
    df = pd.DataFrame({"d": date_codes, "p": pred, "y": y})
    vals = []
    for _, g in df.groupby("d"):
        if len(g) < topk + 5:
            continue
        top = g.nlargest(topk, "p")["y"].mean()
        btm = g.nsmallest(topk, "p")["y"].mean()
        vals.append(float(top - btm))
    if not vals:
        return float("nan")
    return float(np.mean(vals) * 252.0)


def _turnover_proxy(pred: np.ndarray, inst_codes: np.ndarray, date_codes: np.ndarray, topk: int = 55) -> float:
    df = pd.DataFrame({"d": date_codes, "inst": inst_codes, "p": pred})
    prev: Optional[set] = None
    turns = []
    for _, g in df.groupby("d"):
        if len(g) < topk + 5:
            continue
        cur = set(g.nlargest(topk, "p")["inst"].tolist())
        if prev is not None:
            inter = len(cur.intersection(prev))
            turns.append(1.0 - inter / max(1, topk))
        prev = cur
    if not turns:
        return float("nan")
    return float(np.mean(turns))


def _build_cv_folds(train_dates: pd.DatetimeIndex, min_train_days: int, valid_days: int, max_folds: int) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    uniq = pd.Index(sorted(pd.Index(train_dates).unique()))
    if len(uniq) < min_train_days + valid_days:
        return []
    folds: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    end = len(uniq)
    for _ in range(max_folds):
        v_start = end - valid_days
        if v_start < min_train_days:
            break
        tr = pd.DatetimeIndex(uniq[:v_start])
        va = pd.DatetimeIndex(uniq[v_start:end])
        folds.append((tr, va))
        end = v_start
    return list(reversed(folds))


def _cv_select_params(
    x_df: pd.DataFrame,
    y_s: pd.Series,
    w_prior: np.ndarray,
    w_prev: np.ndarray,
    param_grid: List[Dict[str, Any]],
    min_train_days: int,
    valid_days: int,
    max_folds: int,
) -> Dict[str, Any]:
    train_dates = pd.to_datetime(x_df.index.get_level_values(0))
    folds = _build_cv_folds(train_dates, min_train_days=min_train_days, valid_days=valid_days, max_folds=max_folds)
    if not folds:
        out = dict(param_grid[0])
        out["cv_score"] = float("nan")
        out["cv_folds"] = 0
        return out

    best = None
    best_score = -1e18
    x_cols = list(x_df.columns)
    for p in param_grid:
        fold_scores = []
        w_local_prev = w_prev.copy()
        for tr_dates, va_dates in folds:
            tr_m = pd.to_datetime(x_df.index.get_level_values(0)).isin(tr_dates)
            va_m = pd.to_datetime(x_df.index.get_level_values(0)).isin(va_dates)
            x_tr = x_df.loc[tr_m, x_cols].to_numpy(dtype=float)
            y_tr = y_s.loc[tr_m].to_numpy(dtype=float)
            x_va = x_df.loc[va_m, x_cols].to_numpy(dtype=float)
            y_va = y_s.loc[va_m].to_numpy(dtype=float)
            if len(x_tr) < 5000 or len(x_va) < 2000:
                continue
            w = _fit_sparse_ridge(
                x=x_tr,
                y=y_tr,
                w_prior=w_prior,
                w_prev=w_local_prev,
                alpha=float(p["alpha"]),
                corr_pen=float(p["corr_pen"]),
                prior_pen=float(p["prior_pen"]),
                turnover_pen=float(p["turnover_pen"]),
                max_members=int(p["max_members"]),
            )
            pred = x_va @ w
            d = pd.to_datetime(x_df.loc[va_m].index.get_level_values(0)).strftime("%Y-%m-%d").to_numpy()
            inst = x_df.loc[va_m].index.get_level_values(1).astype(str).to_numpy()
            _, ic_ir = _daily_ic(pred, y_va, d)
            ann = _ann_proxy_topk(pred, y_va, d, topk=55)
            to = _turnover_proxy(pred, inst, d, topk=55)
            if not np.isfinite(ic_ir):
                continue
            ann_term = float(ann) if np.isfinite(ann) else 0.0
            to_term = float(to) if np.isfinite(to) else 0.0
            score = float(ic_ir + 0.35 * ann_term - 0.60 * to_term)
            fold_scores.append(score)
            w_local_prev = w
        if not fold_scores:
            continue
        mscore = float(np.mean(fold_scores))
        if mscore > best_score:
            best_score = mscore
            best = dict(p)
            best["cv_score"] = mscore
            best["cv_folds"] = len(fold_scores)
    if best is None:
        out = dict(param_grid[0])
        out["cv_score"] = float("nan")
        out["cv_folds"] = 0
        return out
    return best


def _eval_portfolio_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report_df["turnover"].mean()),
    }


def _get_day_report(pm: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in pm:
        return pm["1day"][0]
    if "day" in pm:
        return pm["day"][0]
    k = next(iter(pm.keys()))
    return pm[k][0]


def _run_backtest_eval(
    signal_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    open_cost: float,
    close_cost: float,
    topk: int,
    hold_topk: int,
) -> Dict[str, float]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = start_date
    backtest_cfg["end_time"] = end_date
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exch = dict(backtest_cfg.get("exchange_kwargs", {}))
    exch["open_cost"] = float(open_cost)
    exch["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_obj = get_exchange(
        freq=freq,
        start_time=start_date,
        end_time=end_date,
        deal_price=str(exch.get("deal_price", "close")),
        limit_threshold=float(exch.get("limit_threshold", 0.095)),
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=float(exch.get("min_cost", 5)),
    )
    exch["exchange"] = exchange_obj

    strategy = BufferedTopkWeightStrategy(
        signal=signal_df,
        topk=int(topk),
        hold_topk=int(hold_topk),
        weight_mode="equal",
        rebalance_mode="weekly",
        rebalance_interval=1,
    )
    t0 = time.perf_counter()
    pm, _ = run_backtest(
        start_time=start_date,
        end_time=end_date,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exch,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = float(time.perf_counter() - t0)
    met = _eval_portfolio_metrics(_get_day_report(pm))
    met["elapsed_sec"] = elapsed
    return met


def _year_slices(start: str, end: str) -> List[Tuple[str, str, str]]:
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    out = []
    for y in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{y}-01-01"), st)
        ye = min(pd.Timestamp(f"{y}-12-31"), ed)
        if ys > ye:
            continue
        tag = f"{y}_ytd" if ye < pd.Timestamp(f"{y}-12-31") else str(y)
        out.append((tag, str(ys.date()), str(ye.date())))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Worker-F factor augmented locked-forward meta ensemble.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--mlruns-topn", type=int, default=10)
    p.add_argument("--min-train-days", type=int, default=126)
    p.add_argument("--cv-valid-days", type=int, default=42)
    p.add_argument("--cv-max-folds", type=int, default=3)
    p.add_argument("--final-topk", type=int, default=55)
    p.add_argument("--final-hold-topk", type=int, default=85)
    p.add_argument("--output-prefix", default="factor_augmented_meta")
    return p


def main() -> int:
    args = build_parser().parse_args()
    trans_dir = Path(__file__).resolve().parent
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    stamp = _timestamp()

    refs = _load_sota_refs(trans_dir)
    ref_run_ids = _extract_referenced_run_ids(refs)
    base_run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    base_run_id = base_run_dir.name
    if base_run_id not in ref_run_ids:
        ref_run_ids.append(base_run_id)

    wf_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(wf_cfg)
    base_port_cfg = _extract_port_config(wf_cfg)

    base_pred = _slice_period(
        _as_score_series(_load_pickle(base_run_dir / "artifacts" / "pred.pkl")),
        args.start_date,
        args.end_date,
    )
    base_label = _slice_period(
        _as_label_series(_load_pickle(base_run_dir / "artifacts" / "label.pkl")),
        args.start_date,
        args.end_date,
    )

    # Expanded factor signal (Worker B)
    expanded_paths = sorted(trans_dir.glob("expanded_factor_best_pred_*.pkl"))
    if not expanded_paths:
        raise FileNotFoundError("no expanded_factor_best_pred_*.pkl found")
    expanded_path = expanded_paths[-1]
    expanded_pred = _slice_period(_as_score_series(_load_pickle(expanded_path)), args.start_date, args.end_date)

    # Worker A deep stack feature member (deep_rank)
    deep_paths = sorted(trans_dir.glob("gpu_deep_stack_workerA_full_pred_*.pkl"))
    deep_member: Optional[pd.Series] = None
    deep_path_used = None
    if deep_paths:
        deep_path_used = deep_paths[-1]
        deep_obj = _load_pickle(deep_path_used)
        if isinstance(deep_obj, pd.DataFrame) and "deep_rank" in deep_obj.columns:
            deep_member = _slice_period(deep_obj["deep_rank"].astype(float), args.start_date, args.end_date)

    ml_candidates = _discover_top_mlruns(
        tracking_dir=tracking_dir,
        include_run_ids=ref_run_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=int(args.mlruns_topn),
    )

    member_specs: List[MemberSpec] = []
    member_specs.append(
        MemberSpec(
            key=f"ml_{base_run_id[:8]}",
            source="mlruns_anchor",
            path=str(base_run_dir / "artifacts" / "pred.pkl"),
            series=base_pred,
            is_anchor=True,
            is_expanded_factor=False,
        )
    )
    member_specs.append(
        MemberSpec(
            key="expanded_factor",
            source="expanded_factor_best",
            path=str(expanded_path),
            series=expanded_pred,
            is_anchor=False,
            is_expanded_factor=True,
        )
    )
    if deep_member is not None:
        member_specs.append(
            MemberSpec(
                key="workerA_deep_rank",
                source="gpu_deep_stack",
                path=str(deep_path_used),
                series=deep_member,
            )
        )

    used_ids = {base_run_id}
    for rid, pred_path, _ in ml_candidates:
        if rid in used_ids:
            continue
        used_ids.add(rid)
        s = _slice_period(_as_score_series(_load_pickle(Path(pred_path))), args.start_date, args.end_date)
        member_specs.append(MemberSpec(key=f"ml_{rid[:8]}", source="mlruns", path=pred_path, series=s))

    # Keep unique keys and reasonable width.
    dedup: Dict[str, MemberSpec] = {}
    for m in member_specs:
        if m.key not in dedup:
            dedup[m.key] = m
    member_specs = list(dedup.values())
    max_members_total = 12
    if len(member_specs) > max_members_total:
        fixed = [m for m in member_specs if m.is_anchor or m.is_expanded_factor or m.source == "gpu_deep_stack"]
        rest = [m for m in member_specs if m not in fixed]
        member_specs = fixed + rest[: max(0, max_members_total - len(fixed))]

    # Build aligned panel with per-date median fill.
    panel_raw = {}
    for m in member_specs:
        panel_raw[m.key] = _center_rank(m.series)
    x_df = pd.concat(panel_raw, axis=1).sort_index()
    x_df.columns = [str(c) for c in x_df.columns]
    y_s = _center_rank(base_label).reindex(x_df.index)
    x_df["__y__"] = y_s
    x_df["__anchor__"] = _center_rank(base_pred).reindex(x_df.index)

    # Fill member holes cross-sectionally to preserve broad panel.
    member_cols = [m.key for m in member_specs]
    for c in member_cols:
        x_df[c] = x_df.groupby(level=0)[c].transform(lambda s: s.fillna(float(s.median()) if s.notna().any() else 0.0))
    x_df = x_df.dropna(subset=["__y__", "__anchor__"])
    y_s = x_df["__y__"].astype(float)
    anchor_s = x_df["__anchor__"].astype(float)
    x_df = x_df[member_cols].astype(float)

    if x_df.empty:
        raise RuntimeError("empty aligned panel after merge/fill")

    # Locked-forward quarterly training and inference.
    dates = pd.to_datetime(x_df.index.get_level_values(0))
    quarters = _make_regime_quarters(pd.DatetimeIndex(dates))
    anchor_key = next(m.key for m in member_specs if m.is_anchor)
    expanded_key = next((m.key for m in member_specs if m.is_expanded_factor), None)
    prior = _build_prior(member_cols, anchor_key=anchor_key, expanded_key=expanded_key)
    w_prev = prior.copy()

    param_grid: List[Dict[str, Any]] = []
    for alpha in (0.5, 2.0, 8.0):
        for corr_pen in (0.0, 0.15):
            for prior_pen in (0.2, 0.6):
                for turn_pen in (0.0, 0.08):
                    for max_m in (4, 6, 8):
                        param_grid.append(
                            {
                                "alpha": float(alpha),
                                "corr_pen": float(corr_pen),
                                "prior_pen": float(prior_pen),
                                "turnover_pen": float(turn_pen),
                                "max_members": int(max_m),
                            }
                        )

    final_pred = pd.Series(np.nan, index=x_df.index, dtype=float, name="score")
    weight_rows: List[Dict[str, Any]] = []
    period_rows: List[Dict[str, Any]] = []
    cv_diag_rows: List[Dict[str, Any]] = []

    for i, q in enumerate(quarters):
        q_start = q.start_time
        q_end = q.end_time
        q_mask = (dates >= q_start) & (dates <= q_end)
        if not q_mask.any():
            continue
        tr_mask = dates < q_start
        x_tr_df = x_df.loc[tr_mask, member_cols]
        y_tr = y_s.loc[tr_mask]
        x_ap = x_df.loc[q_mask, member_cols]
        y_ap = y_s.loc[q_mask]

        if x_tr_df.index.get_level_values(0).nunique() < int(args.min_train_days):
            chosen = {"alpha": None, "corr_pen": None, "prior_pen": None, "turnover_pen": None, "max_members": None, "cv_score": None, "cv_folds": 0}
            w = w_prev.copy()
        else:
            chosen = _cv_select_params(
                x_df=x_tr_df,
                y_s=y_tr,
                w_prior=prior,
                w_prev=w_prev,
                param_grid=param_grid,
                min_train_days=int(args.min_train_days),
                valid_days=int(args.cv_valid_days),
                max_folds=int(args.cv_max_folds),
            )
            w = _fit_sparse_ridge(
                x=x_tr_df.to_numpy(dtype=float),
                y=y_tr.to_numpy(dtype=float),
                w_prior=prior,
                w_prev=w_prev,
                alpha=float(chosen["alpha"]) if chosen["alpha"] is not None else 1.0,
                corr_pen=float(chosen["corr_pen"]) if chosen["corr_pen"] is not None else 0.0,
                prior_pen=float(chosen["prior_pen"]) if chosen["prior_pen"] is not None else 0.3,
                turnover_pen=float(chosen["turnover_pen"]) if chosen["turnover_pen"] is not None else 0.0,
                max_members=int(chosen["max_members"]) if chosen["max_members"] is not None else len(member_cols),
            )

        ap_pred = x_ap.to_numpy(dtype=float) @ w
        final_pred.loc[q_mask] = ap_pred
        d = pd.to_datetime(x_ap.index.get_level_values(0)).strftime("%Y-%m-%d").to_numpy()
        inst = x_ap.index.get_level_values(1).astype(str).to_numpy()
        ic_mean, ic_ir = _daily_ic(ap_pred, y_ap.to_numpy(dtype=float), d)
        ann_proxy = _ann_proxy_topk(ap_pred, y_ap.to_numpy(dtype=float), d, topk=int(args.final_topk))
        to_proxy = _turnover_proxy(ap_pred, inst, d, topk=int(args.final_topk))

        period_rows.append(
            {
                "period_seq": i + 1,
                "period": str(q),
                "apply_start": str(pd.Timestamp(q_start).date()),
                "apply_end": str(pd.Timestamp(min(q_end, pd.Timestamp(args.end_date))).date()),
                "train_days": int(x_tr_df.index.get_level_values(0).nunique()),
                "apply_days": int(x_ap.index.get_level_values(0).nunique()),
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "ann_proxy": ann_proxy,
                "turnover_proxy": to_proxy,
                "alpha": chosen.get("alpha"),
                "corr_pen": chosen.get("corr_pen"),
                "prior_pen": chosen.get("prior_pen"),
                "turnover_pen": chosen.get("turnover_pen"),
                "max_members": chosen.get("max_members"),
                "cv_score": chosen.get("cv_score"),
                "cv_folds": chosen.get("cv_folds"),
            }
        )
        for k, wk in zip(member_cols, w):
            weight_rows.append(
                {
                    "period_seq": i + 1,
                    "period": str(q),
                    "member": k,
                    "weight": float(wk),
                    "abs_weight": float(abs(wk)),
                    "is_anchor": bool(k == anchor_key),
                    "is_expanded_factor": bool(expanded_key is not None and k == expanded_key),
                }
            )
        cv_diag_rows.append(
            {
                "period_seq": i + 1,
                "period": str(q),
                "selected_params": json.dumps({k: chosen.get(k) for k in ("alpha", "corr_pen", "prior_pen", "turnover_pen", "max_members")}, ensure_ascii=False),
                "cv_score": chosen.get("cv_score"),
                "cv_folds": chosen.get("cv_folds"),
            }
        )
        w_prev = w

    # Fill any warmup NaN with anchor signal to keep complete tradable score.
    nan_mask = final_pred.isna()
    if nan_mask.any():
        final_pred.loc[nan_mask] = anchor_s.loc[nan_mask]
        period_rows.append(
            {
                "period_seq": 0,
                "period": "warmup_fallback",
                "apply_start": str(pd.to_datetime(final_pred.loc[nan_mask].index.get_level_values(0)).min().date()),
                "apply_end": str(pd.to_datetime(final_pred.loc[nan_mask].index.get_level_values(0)).max().date()),
                "train_days": 0,
                "apply_days": int(pd.to_datetime(final_pred.loc[nan_mask].index.get_level_values(0)).nunique()),
                "ic_mean": float("nan"),
                "ic_ir": float("nan"),
                "ann_proxy": float("nan"),
                "turnover_proxy": float("nan"),
                "alpha": None,
                "corr_pen": None,
                "prior_pen": None,
                "turnover_pen": None,
                "max_members": None,
                "cv_score": None,
                "cv_folds": 0,
            }
        )

    pred_df = final_pred.rename("score").to_frame("score").dropna()
    pred_df = pred_df.loc[
        (pd.to_datetime(pred_df.index.get_level_values(0)) >= pd.Timestamp(args.start_date))
        & (pd.to_datetime(pred_df.index.get_level_values(0)) <= pd.Timestamp(args.end_date))
    ]
    if pred_df.empty:
        raise RuntimeError("final pred_df empty")

    full_metrics = _run_backtest_eval(
        signal_df=pred_df,
        base_port_cfg=base_port_cfg,
        start_date=args.start_date,
        end_date=args.end_date,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.final_topk),
        hold_topk=int(args.final_hold_topk),
    )
    slice_metrics = []
    for tag, st, ed in _year_slices(args.start_date, args.end_date):
        sm = _run_backtest_eval(
            signal_df=pred_df,
            base_port_cfg=base_port_cfg,
            start_date=st,
            end_date=ed,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            topk=int(args.final_topk),
            hold_topk=int(args.final_hold_topk),
        )
        sm["split"] = tag
        sm["start"] = st
        sm["end"] = ed
        slice_metrics.append(sm)

    # Baseline anchor for comparison.
    anchor_df = anchor_s.rename("score").to_frame("score").dropna()
    anchor_metrics = _run_backtest_eval(
        signal_df=anchor_df,
        base_port_cfg=base_port_cfg,
        start_date=args.start_date,
        end_date=args.end_date,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.final_topk),
        hold_topk=int(args.final_hold_topk),
    )

    # Reference snapshot metrics.
    ref_ir = None
    ref_ann = None
    snapshot = refs.get("sota_snapshot", {})
    if isinstance(snapshot, dict):
        cur = snapshot.get("current_sota", {})
        if isinstance(cur, dict):
            ref_ir = cur.get("costed_ir")
            ref_ann = cur.get("costed_annret")

    out_prefix = args.output_prefix
    summary_json = trans_dir / f"{out_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{out_prefix}_summary_{stamp}.md"
    weights_csv = trans_dir / f"{out_prefix}_weights_{stamp}.csv"
    periods_csv = trans_dir / f"{out_prefix}_periods_{stamp}.csv"
    cv_csv = trans_dir / f"{out_prefix}_cvdiag_{stamp}.csv"
    pred_pkl = trans_dir / f"{out_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = trans_dir / f"{out_prefix}_candidate_pred_{stamp}.csv"
    smoke_json = trans_dir / f"{out_prefix}_artifact_parse_smoke_{stamp}.json"

    _write_csv(weights_csv, weight_rows)
    _write_csv(periods_csv, period_rows)
    _write_csv(cv_csv, cv_diag_rows)
    with pred_pkl.open("wb") as f:
        pickle.dump(pred_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    pred_df.reset_index().to_csv(pred_csv, index=False)

    hard_gate_pass = bool(full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "WorkerF factor_augmented_meta_ensemble",
        "test_period": {"start": args.start_date, "end": args.end_date},
        "costs": {"open": float(args.open_cost), "close": float(args.close_cost)},
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "hard_gate_pass": hard_gate_pass,
        "protocol": {
            "selection": "locked_forward_quarterly",
            "note": "Per-quarter parameter selection uses only prior dates via blocked CV; no full-test global parameter pick.",
            "cv": {
                "min_train_days": int(args.min_train_days),
                "valid_days": int(args.cv_valid_days),
                "max_folds": int(args.cv_max_folds),
                "grid_size": len(param_grid),
            },
            "strategy_execution": {
                "class": "BufferedTopkWeightStrategy",
                "rebalance_mode": "weekly",
                "topk": int(args.final_topk),
                "hold_topk": int(args.final_hold_topk),
                "weight_mode": "equal",
            },
        },
        "members": [
            {
                "key": m.key,
                "source": m.source,
                "path": m.path,
                "rows": int(len(m.series)),
                "days": int(pd.to_datetime(m.series.index.get_level_values(0)).nunique()),
                "is_anchor": m.is_anchor,
                "is_expanded_factor": m.is_expanded_factor,
            }
            for m in member_specs
        ],
        "reference_artifacts_read": {
            "sota_snapshot_present": "sota_snapshot" in refs,
            "signal_conversion_summary_path": refs.get("signal_conversion_summary_path"),
            "nonlinear_summary_path": refs.get("nonlinear_summary_path"),
            "referenced_run_ids_count": len(ref_run_ids),
        },
        "metrics": {
            "meta_full": full_metrics,
            "meta_slices": slice_metrics,
            "anchor_full": anchor_metrics,
            "delta_vs_anchor": {
                "ir": float(full_metrics["ir"] - anchor_metrics["ir"]),
                "annret": float(full_metrics["annret"] - anchor_metrics["annret"]),
                "max_drawdown": float(full_metrics["max_drawdown"] - anchor_metrics["max_drawdown"]),
                "turnover": float(full_metrics["turnover"] - anchor_metrics["turnover"]),
            },
            "snapshot_current_sota": {"ir": ref_ir, "annret": ref_ann},
            "delta_vs_snapshot_sota": {
                "ir": float(full_metrics["ir"] - ref_ir) if ref_ir is not None else None,
                "annret": float(full_metrics["annret"] - ref_ann) if ref_ann is not None else None,
            },
        },
        "risk_notes": [
            "Training history starts at 2024-01 in this workspace, so protocol uses blocked pseudo-OOS CV + rolling forward lock instead of pre-2024 holdout.",
            "Meta weights include correlation and turnover penalties; sparse mask may drop weak members in some periods.",
            "Expanded factor is included as supplementary member, not standalone strategy.",
        ],
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "weights_csv": str(weights_csv),
            "periods_csv": str(periods_csv),
            "cvdiag_csv": str(cv_csv),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append(f"# Factor Augmented Meta Ensemble ({stamp})")
    md.append("")
    md.append("## Protocol")
    md.append("- Locked-forward quarterly meta (blocked CV on prior data only).")
    md.append("- Sparse ridge stacking with anchor prior + correlation penalty + turnover penalty.")
    md.append(f"- Costs: `open={args.open_cost}`, `close={args.close_cost}`.")
    md.append("")
    md.append("## Members")
    md.append("| key | source | anchor | expanded_factor |")
    md.append("|---|---|---:|---:|")
    for m in member_specs:
        md.append(f"| {m.key} | {m.source} | {int(m.is_anchor)} | {int(m.is_expanded_factor)} |")
    md.append("")
    md.append("## Full Test Metrics")
    md.append(
        f"- Meta: IR={full_metrics['ir']:.6f}, AnnRet={full_metrics['annret']:.6f}, "
        f"MaxDD={full_metrics['max_drawdown']:.6f}, Turnover={full_metrics['turnover']:.6f}"
    )
    md.append(
        f"- Anchor: IR={anchor_metrics['ir']:.6f}, AnnRet={anchor_metrics['annret']:.6f}, "
        f"MaxDD={anchor_metrics['max_drawdown']:.6f}, Turnover={anchor_metrics['turnover']:.6f}"
    )
    md.append(f"- Hard gate pass (`IR>{HARD_GATE_IR}`, `AnnRet>{HARD_GATE_ANNRET}`): `{hard_gate_pass}`")
    md.append("")
    md.append("## Slice Metrics")
    md.append("| split | IR | AnnRet | MaxDD | Turnover |")
    md.append("|---|---:|---:|---:|---:|")
    for sm in slice_metrics:
        md.append(
            f"| {sm['split']} | {sm['ir']:.6f} | {sm['annret']:.6f} | {sm['max_drawdown']:.6f} | {sm['turnover']:.6f} |"
        )
    summary_md.write_text("\n".join(md), encoding="utf-8")

    smoke = {
        "weights_rows": int(len(pd.read_csv(weights_csv))) if weights_csv.exists() and weights_csv.stat().st_size > 0 else 0,
        "period_rows": int(len(pd.read_csv(periods_csv))) if periods_csv.exists() and periods_csv.stat().st_size > 0 else 0,
        "cv_rows": int(len(pd.read_csv(cv_csv))) if cv_csv.exists() and cv_csv.stat().st_size > 0 else 0,
        "pred_rows": 0,
        "summary_keys": sorted(list(summary.keys())),
    }
    if pred_pkl.exists():
        obj = _load_pickle(pred_pkl)
        if isinstance(obj, pd.DataFrame):
            smoke["pred_rows"] = int(len(obj))
        elif isinstance(obj, pd.Series):
            smoke["pred_rows"] = int(len(obj))
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
