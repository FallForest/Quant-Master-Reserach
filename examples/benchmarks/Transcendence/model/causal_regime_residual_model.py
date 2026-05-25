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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import WeightStrategyBase


DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"
GLOBAL_START = "2024-01-01"
GLOBAL_END = "2026-04-30"
DEFAULT_EVAL_START = "2025-01-01"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

META_PRED_FILE = "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl"
META_SUMMARY_FILE = "factor_augmented_meta_summary_20260522T120515Z.json"
LH_PRED_FILE = "long_history_retrain_candidate_pred_20260522T134241Z.pkl"
LH_SUMMARY_FILE = "long_history_retrain_summary_20260522T134241Z.json"
LH2_SUMMARY_GLOB = "long_history_second_order_summary_*.json"


@dataclass(frozen=True)
class RegimeParams:
    residual_weight: float
    anchor_defense_weight: float
    dispersion_q: float
    agreement_q: float


@dataclass
class ProxyMetric:
    ic_mean: float
    ic_ir: float
    ann_proxy: float
    spread_ir_proxy: float
    max_drawdown_proxy: float
    turnover_proxy: float
    objective: float


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
        rebalance_mode: str = "weekly",
        rebalance_interval: int = 1,
        **kwargs,
    ):
        self.signal = signal
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.weight_mode = str(weight_mode).lower()
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

    def _calc_weights(self, target: List[str]) -> Dict[str, float]:
        if not target:
            return {}
        weight = 1.0 / len(target)
        return {code: weight for code in target}

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
        return self._calc_weights(target)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _load_pickle(path)


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
        return pred_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _as_label_series(label_obj: Any) -> pd.Series:
    if isinstance(label_obj, pd.Series):
        return label_obj.astype(float)
    if isinstance(label_obj, pd.DataFrame):
        if "label" in label_obj.columns:
            return label_obj["label"].astype(float)
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
    obj.index = obj.index.set_names(["datetime", "instrument"] + list(obj.index.names[2:]))
    return obj


def _slice_period(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    s = _normalize_mi_dt_inst(series.astype(float))
    dt = pd.to_datetime(s.index.get_level_values(0))
    mask = (dt >= pd.Timestamp(start_date)) & (dt <= pd.Timestamp(end_date))
    return s.loc[mask]


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(method="average", pct=True)


def _center_rank(series: pd.Series) -> pd.Series:
    return 2.0 * _cs_rank(series) - 1.0


def _build_residual_leg(main_s: pd.Series, aux_s: pd.Series) -> pd.Series:
    df = pd.concat({"main": main_s, "aux": aux_s}, axis=1).dropna()

    def _resid_one_day(g: pd.DataFrame) -> pd.Series:
        x = g["main"].to_numpy(dtype=float)
        y = g["aux"].to_numpy(dtype=float)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.dot(x, x))
        beta = 0.0 if denom <= 1e-12 else float(np.dot(x, y) / denom)
        return pd.Series(y - beta * x, index=g.index, dtype=float)

    resid = df.groupby(level=0, group_keys=False).apply(_resid_one_day)
    return _center_rank(resid.astype(float))


def _mean_daily_spearman(left: pd.Series, right: pd.Series) -> float:
    panel = pd.concat({"l": left, "r": right}, axis=1).dropna()
    vals = []
    for _, g in panel.groupby(level=0):
        corr = g["l"].corr(g["r"], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def _daily_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, g in panel.groupby(level=0):
        main = g["main"].astype(float)
        anchor = g["anchor"].astype(float)
        resid = g["resid"].astype(float)
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "main_dispersion": float(main.std(ddof=0)),
                "resid_dispersion": float(resid.std(ddof=0)),
                "main_anchor_spearman": float(main.corr(anchor, method="spearman")),
            }
        )
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def _causal_regime_gate(diag: pd.DataFrame, params: RegimeParams, warmup_days: int) -> pd.Series:
    flags = []
    idx = pd.DatetimeIndex(diag.index)
    for i, dt in enumerate(idx):
        if i < max(2, warmup_days):
            flags.append((dt, False, "warmup", np.nan, np.nan, np.nan, np.nan))
            continue
        hist = diag.iloc[:i]
        prev = diag.iloc[i - 1]
        disp_thr = float(hist["main_dispersion"].dropna().quantile(params.dispersion_q))
        agr_thr = float(hist["main_anchor_spearman"].dropna().quantile(params.agreement_q))
        prev_disp = float(prev["main_dispersion"])
        prev_agr = float(prev["main_anchor_spearman"])
        low_disp = np.isfinite(prev_disp) and np.isfinite(disp_thr) and prev_disp < disp_thr
        low_agree = np.isfinite(prev_agr) and np.isfinite(agr_thr) and prev_agr < agr_thr
        flag = bool(low_disp or low_agree)
        if low_disp and low_agree:
            reason = "low_dispersion_and_low_agreement"
        elif low_disp:
            reason = "low_dispersion"
        elif low_agree:
            reason = "low_agreement"
        else:
            reason = "normal"
        flags.append((dt, flag, reason, prev_disp, disp_thr, prev_agr, agr_thr))
    return pd.DataFrame(
        flags,
        columns=[
            "datetime",
            "stress_flag",
            "reason",
            "prev_main_dispersion",
            "dispersion_threshold",
            "prev_main_anchor_spearman",
            "agreement_threshold",
        ],
    ).set_index("datetime")


def _build_candidate_pred(panel: pd.DataFrame, diag: pd.DataFrame, params: RegimeParams, warmup_days: int) -> Tuple[pd.Series, pd.DataFrame]:
    gate = _causal_regime_gate(diag, params=params, warmup_days=warmup_days)
    date_index = pd.to_datetime(panel.index.get_level_values(0))
    active = pd.Series(date_index, index=panel.index).map(gate["stress_flag"]).fillna(False).astype(bool)
    active_f = active.astype(float)
    pred = (
        panel["main"].astype(float)
        + active_f * float(params.residual_weight) * panel["resid"].astype(float)
        + active_f * float(params.anchor_defense_weight) * (panel["anchor"].astype(float) - panel["main"].astype(float))
    )
    return _center_rank(pred.astype(float)), gate


def _spread_series_topk(pred_s: pd.Series, label_s: pd.Series, topk: int) -> pd.Series:
    aligned = pd.concat({"pred": pred_s, "label": label_s}, axis=1).dropna()
    vals = {}
    for dt, g in aligned.groupby(level=0):
        if len(g) < topk + 5:
            continue
        vals[pd.Timestamp(dt)] = float(g.nlargest(topk, "pred")["label"].mean() - g.nsmallest(topk, "pred")["label"].mean())
    return pd.Series(vals, dtype=float).sort_index()


def _daily_ic(pred: np.ndarray, y: np.ndarray, date_codes: np.ndarray) -> Tuple[float, float]:
    df = pd.DataFrame({"d": date_codes, "p": pred, "y": y})
    vals = []
    for _, g in df.groupby("d"):
        if len(g) < 30:
            continue
        corr = g["p"].corr(g["y"], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ic_ir = float(np.sqrt(252.0) * mean / std) if std > 1e-12 else float("nan")
    return mean, ic_ir


def _turnover_proxy(pred_s: pd.Series, topk: int) -> float:
    prev: Optional[set] = None
    turns = []
    df = pred_s.rename("pred").to_frame()
    for _, g in df.groupby(level=0):
        if len(g) < topk + 5:
            continue
        cur = set(g.nlargest(topk, "pred").index.get_level_values(1).astype(str).tolist())
        if prev is not None:
            turns.append(1.0 - len(cur.intersection(prev)) / max(1, topk))
        prev = cur
    return float(np.mean(turns)) if turns else float("nan")


def _proxy_metrics(pred_s: pd.Series, label_s: pd.Series, topk: int) -> ProxyMetric:
    aligned = pd.concat({"pred": pred_s, "label": label_s}, axis=1).dropna()
    if aligned.empty:
        return ProxyMetric(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -1e18)
    date_codes = pd.to_datetime(aligned.index.get_level_values(0)).strftime("%Y-%m-%d").to_numpy()
    pred = aligned["pred"].to_numpy(dtype=float)
    y = aligned["label"].to_numpy(dtype=float)
    ic_mean, ic_ir = _daily_ic(pred, y, date_codes)
    spread = _spread_series_topk(aligned["pred"], aligned["label"], topk=topk)
    ann_proxy = float(spread.mean() * 252.0) if len(spread) else float("nan")
    spread_std = float(spread.std(ddof=1)) if len(spread) > 1 else float("nan")
    spread_ir = float(np.sqrt(252.0) * spread.mean() / spread_std) if np.isfinite(spread_std) and spread_std > 1e-12 else float("nan")
    equity = (1.0 + spread.fillna(0.0)).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")
    turnover = _turnover_proxy(aligned["pred"], topk=topk)
    objective = (
        (float(ic_ir) if np.isfinite(ic_ir) else -999.0)
        + 0.20 * (float(ann_proxy) if np.isfinite(ann_proxy) else 0.0)
        + 0.15 * (float(spread_ir) if np.isfinite(spread_ir) else 0.0)
        - 0.45 * (float(turnover) if np.isfinite(turnover) else 0.0)
        + 0.20 * (float(max_dd) if np.isfinite(max_dd) else 0.0)
    )
    return ProxyMetric(
        ic_mean=float(ic_mean),
        ic_ir=float(ic_ir),
        ann_proxy=float(ann_proxy),
        spread_ir_proxy=float(spread_ir),
        max_drawdown_proxy=float(max_dd),
        turnover_proxy=float(turnover),
        objective=float(objective),
    )


def _build_cv_folds(train_dates: pd.DatetimeIndex, min_history_days: int, valid_days: int, max_folds: int) -> List[pd.DatetimeIndex]:
    uniq = pd.Index(sorted(pd.Index(train_dates).unique()))
    if len(uniq) < min_history_days + valid_days:
        return []
    folds: List[pd.DatetimeIndex] = []
    end = len(uniq)
    for _ in range(max_folds):
        v_start = end - valid_days
        if v_start < min_history_days:
            break
        folds.append(pd.DatetimeIndex(uniq[v_start:end]))
        end = v_start
    return list(reversed(folds))


def _param_grid(mode: str = "full") -> List[RegimeParams]:
    grid = []
    if mode == "tiny":
        residual_weights = (0.0, 0.10, 0.20)
        anchor_weights = (0.0, 0.10)
        quantiles = (0.35,)
    else:
        residual_weights = (0.0, 0.05, 0.10, 0.15, 0.20)
        anchor_weights = (0.0, 0.10, 0.20)
        quantiles = (0.25, 0.35)
    for rw in residual_weights:
        for aw in anchor_weights:
            for dq in quantiles:
                for aq in quantiles:
                    grid.append(RegimeParams(rw, aw, dq, aq))
    return grid


def _select_params_for_quarter(
    panel_hist: pd.DataFrame,
    diag_hist: pd.DataFrame,
    grid: Sequence[RegimeParams],
    min_history_days: int,
    valid_days: int,
    max_folds: int,
    warmup_days: int,
    topk: int,
) -> Tuple[RegimeParams, Dict[str, Any]]:
    hist_dates = pd.to_datetime(panel_hist.index.get_level_values(0))
    folds = _build_cv_folds(
        pd.DatetimeIndex(hist_dates.unique()),
        min_history_days=min_history_days,
        valid_days=valid_days,
        max_folds=max_folds,
    )
    zero = RegimeParams(0.0, 0.0, 0.25, 0.25)
    if not folds:
        return zero, {"selection_reason": "warmup_default_main_only", "cv_folds": 0, "cv_score": None, "cv_baseline_score": None}

    best_params = zero
    best_score = -1e18
    best_base = -1e18
    fold_count = 0
    for params in grid:
        scores = []
        base_scores = []
        for va_dates in folds:
            va_end = pd.Timestamp(va_dates.max())
            fold_panel = panel_hist.loc[pd.to_datetime(panel_hist.index.get_level_values(0)) <= va_end]
            fold_diag = diag_hist.loc[diag_hist.index <= va_end]
            pred, _ = _build_candidate_pred(fold_panel, fold_diag, params=params, warmup_days=warmup_days)
            va_mask = pd.to_datetime(pred.index.get_level_values(0)).isin(va_dates)
            cand_m = _proxy_metrics(pred.loc[va_mask], fold_panel.loc[va_mask, "label"], topk=topk)
            base_m = _proxy_metrics(fold_panel.loc[va_mask, "main"], fold_panel.loc[va_mask, "label"], topk=topk)
            if np.isfinite(cand_m.objective) and np.isfinite(base_m.objective):
                stress_rate = float(_causal_regime_gate(fold_diag, params=params, warmup_days=warmup_days).loc[va_dates, "stress_flag"].mean())
                scores.append(float(cand_m.objective - 0.05 * stress_rate))
                base_scores.append(float(base_m.objective))
        if not scores:
            continue
        score = float(np.mean(scores))
        base_score = float(np.mean(base_scores))
        if score > best_score:
            best_score = score
            best_base = base_score
            best_params = params
            fold_count = len(scores)

    margin = best_score - best_base
    if not np.isfinite(margin) or margin < 0.03:
        return zero, {
            "selection_reason": "prior_cv_no_material_edge_default_main_only",
            "cv_folds": int(fold_count),
            "cv_score": float(best_score) if np.isfinite(best_score) else None,
            "cv_baseline_score": float(best_base) if np.isfinite(best_base) else None,
            "cv_margin": float(margin) if np.isfinite(margin) else None,
            "best_nonzero_params": asdict(best_params),
        }
    return best_params, {
        "selection_reason": "blocked_cv_prior_only",
        "cv_folds": int(fold_count),
        "cv_score": float(best_score),
        "cv_baseline_score": float(best_base),
        "cv_margin": float(margin),
    }


def _get_day_report(pm: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in pm:
        return pm["1day"][0]
    if "day" in pm:
        return pm["day"][0]
    key = next(iter(pm.keys()))
    return pm[key][0]


def _eval_portfolio_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report_df["turnover"].mean()),
    }


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
    exch["exchange"] = get_exchange(
        freq=freq,
        start_time=start_date,
        end_time=end_date,
        deal_price=str(exch.get("deal_price", "close")),
        limit_threshold=float(exch.get("limit_threshold", 0.095)),
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=float(exch.get("min_cost", 5)),
    )
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
    metrics = _eval_portfolio_metrics(_get_day_report(pm))
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    return metrics


def _safe_backtest_eval(*args, **kwargs) -> Dict[str, Any]:
    try:
        return {"ok": True, "metrics": _run_backtest_eval(*args, **kwargs), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "metrics": None, "error": {"type": type(exc).__name__, "message": str(exc)}}


def _year_slices(start: str, end: str) -> List[Tuple[str, str, str]]:
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    rows = []
    for year in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{year}-01-01"), st)
        ye = min(pd.Timestamp(f"{year}-12-31"), ed)
        if ys > ye:
            continue
        tag = f"{year}_ytd" if ye < pd.Timestamp(f"{year}-12-31") else str(year)
        rows.append((tag, str(ys.date()), str(ye.date())))
    return rows


def _latest_json(trans_dir: Path, pattern: str) -> Optional[Path]:
    paths = sorted(trans_dir.glob(pattern))
    return paths[-1] if paths else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict past-only causal regime residual candidate for Transcendence.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--global-start", default=GLOBAL_START)
    p.add_argument("--global-end", default=GLOBAL_END)
    p.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    p.add_argument("--eval-end", default=GLOBAL_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--min-history-days", type=int, default=126)
    p.add_argument("--cv-valid-days", type=int, default=42)
    p.add_argument("--cv-max-folds", type=int, default=3)
    p.add_argument("--warmup-days", type=int, default=60)
    p.add_argument("--topk", type=int, default=55)
    p.add_argument("--hold-topk", type=int, default=85)
    p.add_argument("--grid-mode", choices=["tiny", "full"], default="full")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--output-prefix", default="causal_regime_residual")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    trans_dir = Path(__file__).resolve().parent
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    stamp = _stamp()

    base_run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    wf_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(wf_cfg)
    base_port_cfg = _extract_port_config(wf_cfg)

    anchor_pred = _slice_period(_as_score_series(_load_pickle(base_run_dir / "artifacts" / "pred.pkl")), args.global_start, args.global_end)
    anchor_label = _slice_period(_as_label_series(_load_pickle(base_run_dir / "artifacts" / "label.pkl")), args.global_start, args.global_end)
    meta_pred = _slice_period(_as_score_series(_load_pickle(trans_dir / META_PRED_FILE)), args.global_start, args.global_end)
    lh_pred = _slice_period(_as_score_series(_load_pickle(trans_dir / LH_PRED_FILE)), args.global_start, args.global_end)

    common_index = meta_pred.index.intersection(lh_pred.index).intersection(anchor_pred.index).intersection(anchor_label.index)
    if len(common_index) == 0:
        raise RuntimeError("no common aligned index across anchor/meta/long_history/label")

    main_leg = _center_rank(meta_pred.loc[common_index])
    aux_leg = _center_rank(lh_pred.loc[common_index])
    anchor_leg = _center_rank(anchor_pred.loc[common_index])
    label_leg = _center_rank(anchor_label.loc[common_index])
    residual_leg = _build_residual_leg(main_leg, aux_leg)
    panel = pd.concat(
        {
            "main": main_leg,
            "resid": residual_leg,
            "anchor": anchor_leg,
            "label": label_leg,
        },
        axis=1,
    ).dropna()
    if panel.empty:
        raise RuntimeError("aligned panel is empty after residualization")

    diag = _daily_diagnostics(panel)
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel.index.get_level_values(0)).unique()))
    effective_end = min(pd.Timestamp(args.eval_end), pd.Timestamp(all_dates.max()))
    effective_eval_start = max(pd.Timestamp(args.eval_start), pd.Timestamp(all_dates.min()))
    if effective_eval_start > effective_end:
        raise RuntimeError(f"invalid eval window: {effective_eval_start.date()}..{effective_end.date()}")

    grid = _param_grid(args.grid_mode)
    candidate_parts = []
    period_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    quarters = sorted(pd.PeriodIndex(all_dates, freq="Q").unique())
    for seq, quarter in enumerate(quarters, start=1):
        q_start = max(pd.Timestamp(quarter.start_time), pd.Timestamp(all_dates.min()))
        q_end = min(pd.Timestamp(quarter.end_time), pd.Timestamp(all_dates.max()))
        hist_mask = pd.to_datetime(panel.index.get_level_values(0)) < q_start
        apply_mask = (pd.to_datetime(panel.index.get_level_values(0)) >= q_start) & (pd.to_datetime(panel.index.get_level_values(0)) <= q_end)
        hist_days = int(pd.to_datetime(panel.loc[hist_mask].index.get_level_values(0)).nunique()) if hist_mask.any() else 0
        apply_days = int(pd.to_datetime(panel.loc[apply_mask].index.get_level_values(0)).nunique()) if apply_mask.any() else 0
        if hist_mask.any():
            params, sel = _select_params_for_quarter(
                panel_hist=panel.loc[hist_mask],
                diag_hist=diag.loc[diag.index < q_start],
                grid=grid,
                min_history_days=args.min_history_days,
                valid_days=args.cv_valid_days,
                max_folds=args.cv_max_folds,
                warmup_days=args.warmup_days,
                topk=args.topk,
            )
        else:
            params = RegimeParams(0.0, 0.0, 0.25, 0.25)
            sel = {"selection_reason": "warmup_default_main_only", "cv_folds": 0, "cv_score": None, "cv_baseline_score": None}

        pred_upto, gate_upto = _build_candidate_pred(
            panel.loc[pd.to_datetime(panel.index.get_level_values(0)) <= q_end],
            diag.loc[diag.index <= q_end],
            params=params,
            warmup_days=args.warmup_days,
        )
        pred_dates = pd.to_datetime(pred_upto.index.get_level_values(0))
        apply_pred = pred_upto.loc[(pred_dates >= q_start) & (pred_dates <= q_end)]
        candidate_parts.append(apply_pred)
        apply_panel = panel.loc[apply_mask]
        cand_pm = _proxy_metrics(apply_pred, apply_panel["label"], topk=args.topk)
        meta_pm = _proxy_metrics(apply_panel["main"], apply_panel["label"], topk=args.topk)
        gate_apply = gate_upto.loc[(gate_upto.index >= q_start) & (gate_upto.index <= q_end)]
        stress_rate = float(gate_apply["stress_flag"].mean()) if len(gate_apply) else 0.0

        row = {
            "period_seq": seq,
            "period": str(quarter),
            "apply_start": str(q_start.date()),
            "apply_end": str(q_end.date()),
            "train_days": hist_days,
            "apply_days": apply_days,
            "residual_weight": params.residual_weight,
            "anchor_defense_weight": params.anchor_defense_weight,
            "dispersion_q": params.dispersion_q,
            "agreement_q": params.agreement_q,
            "stress_rate": stress_rate,
            "selection_reason": sel.get("selection_reason"),
            "cv_folds": sel.get("cv_folds"),
            "cv_score": sel.get("cv_score"),
            "cv_baseline_score": sel.get("cv_baseline_score"),
            "cv_margin": sel.get("cv_margin"),
            "apply_ic_ir": cand_pm.ic_ir,
            "apply_ann_proxy": cand_pm.ann_proxy,
            "apply_spread_ir_proxy": cand_pm.spread_ir_proxy,
            "apply_turnover_proxy": cand_pm.turnover_proxy,
            "meta_apply_ic_ir": meta_pm.ic_ir,
            "meta_apply_ann_proxy": meta_pm.ann_proxy,
            "meta_apply_spread_ir_proxy": meta_pm.spread_ir_proxy,
            "meta_apply_turnover_proxy": meta_pm.turnover_proxy,
            "delta_apply_ic_ir": cand_pm.ic_ir - meta_pm.ic_ir,
            "delta_apply_ann_proxy": cand_pm.ann_proxy - meta_pm.ann_proxy,
        }
        period_rows.append(row)
        for dt, g in gate_apply.iterrows():
            diag_rows.append(
                {
                    "date": str(pd.Timestamp(dt).date()),
                    "period": str(quarter),
                    "stress_flag": bool(g["stress_flag"]),
                    "reason": g["reason"],
                    "prev_main_dispersion": g["prev_main_dispersion"],
                    "dispersion_threshold": g["dispersion_threshold"],
                    "prev_main_anchor_spearman": g["prev_main_anchor_spearman"],
                    "agreement_threshold": g["agreement_threshold"],
                    "residual_weight": params.residual_weight,
                    "anchor_defense_weight": params.anchor_defense_weight,
                }
            )

    candidate = pd.concat(candidate_parts).sort_index()
    candidate.name = "score"
    candidate_df = candidate.to_frame("score")
    meta_df = panel["main"].rename("score").to_frame()

    eval_mask = (pd.to_datetime(candidate.index.get_level_values(0)) >= effective_eval_start) & (
        pd.to_datetime(candidate.index.get_level_values(0)) <= effective_end
    )
    cand_eval_proxy = _proxy_metrics(candidate.loc[eval_mask], panel.loc[eval_mask, "label"], topk=args.topk)
    meta_eval_proxy = _proxy_metrics(panel.loc[eval_mask, "main"], panel.loc[eval_mask, "label"], topk=args.topk)
    anchor_eval_proxy = _proxy_metrics(panel.loc[eval_mask, "anchor"], panel.loc[eval_mask, "label"], topk=args.topk)

    if args.skip_backtest:
        cand_eval_bt = {"ok": False, "metrics": None, "error": {"type": "Skipped", "message": "--skip-backtest"}}
        meta_eval_bt = {"ok": False, "metrics": None, "error": {"type": "Skipped", "message": "--skip-backtest"}}
    else:
        cand_eval_bt = _safe_backtest_eval(
            candidate_df,
            base_port_cfg,
            str(effective_eval_start.date()),
            str(effective_end.date()),
            args.open_cost,
            args.close_cost,
            args.topk,
            args.hold_topk,
        )
        meta_eval_bt = _safe_backtest_eval(
            meta_df,
            base_port_cfg,
            str(effective_eval_start.date()),
            str(effective_end.date()),
            args.open_cost,
            args.close_cost,
            args.topk,
            args.hold_topk,
        )

    slice_rows: List[Dict[str, Any]] = []
    for tag, st, ed in _year_slices(str(effective_eval_start.date()), str(effective_end.date())):
        slice_mask = (pd.to_datetime(candidate.index.get_level_values(0)) >= pd.Timestamp(st)) & (
            pd.to_datetime(candidate.index.get_level_values(0)) <= pd.Timestamp(ed)
        )
        c_proxy = _proxy_metrics(candidate.loc[slice_mask], panel.loc[slice_mask, "label"], topk=args.topk)
        m_proxy = _proxy_metrics(panel.loc[slice_mask, "main"], panel.loc[slice_mask, "label"], topk=args.topk)
        if args.skip_backtest:
            c_bt = {"ok": False, "metrics": None, "error": {"type": "Skipped", "message": "--skip-backtest"}}
            m_bt = {"ok": False, "metrics": None, "error": {"type": "Skipped", "message": "--skip-backtest"}}
        else:
            c_bt = _safe_backtest_eval(candidate_df, base_port_cfg, st, ed, args.open_cost, args.close_cost, args.topk, args.hold_topk)
            m_bt = _safe_backtest_eval(meta_df, base_port_cfg, st, ed, args.open_cost, args.close_cost, args.topk, args.hold_topk)
        slice_rows.append(
            {
                "split": tag,
                "start": st,
                "end": ed,
                "candidate_backtest_ok": bool(c_bt["ok"]),
                "candidate_backtest_error": json.dumps(c_bt["error"], ensure_ascii=False) if c_bt["error"] else None,
                "candidate_ir": c_bt["metrics"]["ir"] if c_bt["ok"] else np.nan,
                "candidate_annret": c_bt["metrics"]["annret"] if c_bt["ok"] else np.nan,
                "candidate_max_drawdown": c_bt["metrics"]["max_drawdown"] if c_bt["ok"] else np.nan,
                "candidate_turnover": c_bt["metrics"]["turnover"] if c_bt["ok"] else np.nan,
                "meta_backtest_ok": bool(m_bt["ok"]),
                "meta_backtest_error": json.dumps(m_bt["error"], ensure_ascii=False) if m_bt["error"] else None,
                "meta_ir": m_bt["metrics"]["ir"] if m_bt["ok"] else np.nan,
                "meta_annret": m_bt["metrics"]["annret"] if m_bt["ok"] else np.nan,
                "meta_max_drawdown": m_bt["metrics"]["max_drawdown"] if m_bt["ok"] else np.nan,
                "meta_turnover": m_bt["metrics"]["turnover"] if m_bt["ok"] else np.nan,
                "candidate_proxy_ic_ir": c_proxy.ic_ir,
                "candidate_proxy_ann": c_proxy.ann_proxy,
                "meta_proxy_ic_ir": m_proxy.ic_ir,
                "meta_proxy_ann": m_proxy.ann_proxy,
                "delta_proxy_ic_ir": c_proxy.ic_ir - m_proxy.ic_ir,
                "delta_proxy_ann": c_proxy.ann_proxy - m_proxy.ann_proxy,
            }
        )

    ref_meta = _load_json(trans_dir / META_SUMMARY_FILE)
    ref_lh = _load_json(trans_dir / LH_SUMMARY_FILE)
    lh2_path = _latest_json(trans_dir, LH2_SUMMARY_GLOB)
    ref_lh2 = _load_json(lh2_path) if lh2_path else {}

    cand_bt_metrics = cand_eval_bt["metrics"] if cand_eval_bt["ok"] else {}
    promising = (
        cand_eval_bt["ok"]
        and float(cand_bt_metrics.get("ir", -999.0)) > 2.35
        and float(cand_bt_metrics.get("annret", -999.0)) > 0.22
        and cand_eval_proxy.objective > meta_eval_proxy.objective + 0.05
        and any(float(r["delta_proxy_ic_ir"]) > 0.1 for r in slice_rows)
    )
    hard_pass = bool(
        cand_eval_bt["ok"]
        and float(cand_bt_metrics.get("ir", -999.0)) > HARD_GATE_IR
        and float(cand_bt_metrics.get("annret", -999.0)) > HARD_GATE_ANNRET
    )
    verdict = "PROMISING_FOR_FULL_RUN" if promising else "NO_GO"

    out_pred_pkl = trans_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    out_pred_csv = trans_dir / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    out_summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    out_summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    out_periods_csv = trans_dir / f"{args.output_prefix}_periods_{stamp}.csv"
    out_diag_csv = trans_dir / f"{args.output_prefix}_selector_diag_{stamp}.csv"
    out_slices_csv = trans_dir / f"{args.output_prefix}_slices_{stamp}.csv"
    out_parse_smoke = trans_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    with out_pred_pkl.open("wb") as f:
        pickle.dump(candidate, f)
    candidate.reset_index().to_csv(out_pred_csv, index=False)
    _write_csv(out_periods_csv, period_rows)
    _write_csv(out_diag_csv, diag_rows)
    _write_csv(out_slices_csv, slice_rows)

    artifacts = {
        "summary_json": str(out_summary_json),
        "summary_md": str(out_summary_md),
        "candidate_pred_pkl": str(out_pred_pkl),
        "candidate_pred_csv": str(out_pred_csv),
        "periods_csv": str(out_periods_csv),
        "selector_diag_csv": str(out_diag_csv),
        "slices_csv": str(out_slices_csv),
        "artifact_parse_smoke_json": str(out_parse_smoke),
    }
    parse_smoke = {
        "candidate_rows": int(len(candidate)),
        "candidate_days": int(pd.to_datetime(candidate.index.get_level_values(0)).nunique()),
        "candidate_start": str(pd.to_datetime(candidate.index.get_level_values(0)).min().date()),
        "candidate_end": str(pd.to_datetime(candidate.index.get_level_values(0)).max().date()),
        "summary_json_exists_after_write": True,
    }
    out_parse_smoke.write_text(json.dumps(parse_smoke, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "strict_past_only_causal_regime_residual_candidate",
        "verdict": verdict,
        "hard_gate_pass": hard_pass,
        "protocol": {
            "candidate": "factor_meta main leg plus causal stress-only long_history residual and small anchor defense.",
            "selection": "quarterly_forward_locked_low_df_grid",
            "daily_regime_gate": "stress if previous-day main dispersion is below expanding quantile OR previous-day main-anchor agreement is below expanding quantile.",
            "parameter_grid": [asdict(p) for p in grid],
            "grid_mode": args.grid_mode,
            "cv": {
                "min_history_days": args.min_history_days,
                "valid_days": args.cv_valid_days,
                "max_folds": args.cv_max_folds,
                "warmup_days": args.warmup_days,
            },
            "execution": {
                "strategy": "BufferedTopkWeightStrategy",
                "topk": args.topk,
                "hold_topk": args.hold_topk,
                "rebalance_mode": "weekly",
                "runtime_sell_wrapper": "none",
                "skip_backtest": bool(args.skip_backtest),
            },
        },
        "leakage_boundary": {
            "past_only": [
                "Quarter parameters are chosen only from dates before the apply quarter.",
                "Blocked CV folds use validation dates that are all before the future apply quarter.",
                "Daily stress gate uses diagnostics shifted to t-1 through expanding thresholds.",
                "No test-window full-period parameter search is used.",
            ],
            "known_risk": "Anchor signal is used only as a small stress-regime stabilizer selected by prior-only CV; anchor portfolio parameters are not copied.",
        },
        "paths": {
            "anchor_pred": str(base_run_dir / "artifacts" / "pred.pkl"),
            "anchor_label": str(base_run_dir / "artifacts" / "label.pkl"),
            "meta_pred": str(trans_dir / META_PRED_FILE),
            "long_history_pred": str(trans_dir / LH_PRED_FILE),
            "factor_meta_summary": str(trans_dir / META_SUMMARY_FILE),
            "long_history_summary": str(trans_dir / LH_SUMMARY_FILE),
            "long_history_second_order_summary": str(lh2_path) if lh2_path else None,
        },
        "coverage": {
            "global_start_requested": args.global_start,
            "global_end_requested": args.global_end,
            "common_start": str(pd.to_datetime(panel.index.get_level_values(0)).min().date()),
            "common_end": str(pd.to_datetime(panel.index.get_level_values(0)).max().date()),
            "effective_eval_start": str(effective_eval_start.date()),
            "effective_eval_end": str(effective_end.date()),
            "common_rows": int(len(panel)),
            "common_days": int(pd.to_datetime(panel.index.get_level_values(0)).nunique()),
            "eval_rows": int(eval_mask.sum()),
            "eval_days": int(pd.to_datetime(candidate.loc[eval_mask].index.get_level_values(0)).nunique()),
        },
        "correlations": {
            "main_vs_aux_mean_daily_spearman": _mean_daily_spearman(main_leg, aux_leg),
            "main_vs_residual_mean_daily_spearman": _mean_daily_spearman(main_leg, residual_leg),
            "anchor_vs_residual_mean_daily_spearman": _mean_daily_spearman(anchor_leg, residual_leg),
        },
        "metrics": {
            "candidate_eval_backtest_status": cand_eval_bt,
            "factor_meta_eval_backtest_status": meta_eval_bt,
            "candidate_proxy_eval": asdict(cand_eval_proxy),
            "factor_meta_proxy_eval_same_window": asdict(meta_eval_proxy),
            "anchor_proxy_eval_same_window": asdict(anchor_eval_proxy),
            "delta_vs_factor_meta_proxy_eval": {
                "ic_ir": cand_eval_proxy.ic_ir - meta_eval_proxy.ic_ir,
                "ann_proxy": cand_eval_proxy.ann_proxy - meta_eval_proxy.ann_proxy,
                "spread_ir_proxy": cand_eval_proxy.spread_ir_proxy - meta_eval_proxy.spread_ir_proxy,
                "max_drawdown_proxy": cand_eval_proxy.max_drawdown_proxy - meta_eval_proxy.max_drawdown_proxy,
                "turnover_proxy": cand_eval_proxy.turnover_proxy - meta_eval_proxy.turnover_proxy,
                "objective": cand_eval_proxy.objective - meta_eval_proxy.objective,
            },
            "year_slices_same_window": slice_rows,
            "reference_published": {
                "factor_meta_full_2024_2026": ref_meta.get("metrics", {}).get("meta_full"),
                "long_history_full_2024_2026": ref_lh.get("metrics", {}).get("best_test_backtest"),
                "long_history_second_order_verdict": ref_lh2.get("verdict") if ref_lh2 else None,
                "long_history_second_order_proxy_delta": ref_lh2.get("metrics", {}).get("delta_vs_factor_meta_proxy_eval") if ref_lh2 else None,
            },
        },
        "selection_periods": period_rows,
        "hard_gate_reference": {
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
            "candidate_pass_eval_window": hard_pass,
            "note": "Smoke gate focuses on 2025-01-01 through common end migration window before any full run recommendation.",
        },
        "artifacts": artifacts,
        "runtime_sec_total": float(time.perf_counter() - started),
    }
    out_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# Causal Regime Residual Model {stamp}",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Protocol",
        "- Strict quarterly past-only selection.",
        "- Daily stress gate uses previous-day diagnostics only.",
        "- No runtime sell wrapper or execution-layer patch is applied in this script.",
        "",
        "## Key Metrics",
        f"- Eval window: {effective_eval_start.date()}..{effective_end.date()}",
        f"- Candidate backtest: {json.dumps(cand_eval_bt, ensure_ascii=False)}",
        f"- Factor-meta same-window backtest: {json.dumps(meta_eval_bt, ensure_ascii=False)}",
        f"- Candidate proxy objective: {cand_eval_proxy.objective:.6f}",
        f"- Factor-meta proxy objective: {meta_eval_proxy.objective:.6f}",
        f"- Proxy objective delta: {cand_eval_proxy.objective - meta_eval_proxy.objective:.6f}",
        "",
        "## Comparison",
        f"- factor_augmented_meta published full: {json.dumps(ref_meta.get('metrics', {}).get('meta_full'), ensure_ascii=False)}",
        f"- long_history_second_order verdict: {ref_lh2.get('verdict') if ref_lh2 else 'missing'}",
        "",
        "## Artifacts",
    ]
    for key, value in artifacts.items():
        md_lines.append(f"- {key}: `{value}`")
    out_summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out_parse_smoke.write_text(json.dumps({**parse_smoke, "summary_json_exists_after_write": out_summary_json.exists()}, indent=2), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "hard_gate_pass": hard_pass, "summary_json": str(out_summary_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
