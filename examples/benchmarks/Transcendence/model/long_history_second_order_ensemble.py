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
from dataclasses import dataclass
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
from quant_master.backtest.position import Position
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


@dataclass
class ProxyMetric:
    ic_mean: float
    ic_ir: float
    ann_proxy: float
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

    def _calc_weights(self, ranked_score: pd.Series, target: List[str]) -> Dict[str, float]:
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
            except Exception:
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


_EPSILON_PATCH_APPLIED = False


def _apply_runtime_sell_epsilon_patch() -> None:
    global _EPSILON_PATCH_APPLIED
    if _EPSILON_PATCH_APPLIED:
        return

    orig = Position._sell_stock

    def _patched_sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            raise KeyError(f"{stock_id} not in current position")

        current_amount = self.position[stock_id]["amount"]
        if trade_amount > current_amount:
            oversell_amount = trade_amount - current_amount
            oversell_tolerance = min(0.1, max(1e-5, 2e-5 * abs(current_amount)))
            if oversell_amount <= oversell_tolerance:
                trade_amount = current_amount
                trade_val = trade_amount * trade_price
            else:
                return orig(self, stock_id, trade_val, cost, trade_price)

        if np.isclose(current_amount, trade_amount):
            self._del_stock(stock_id)
        else:
            self.position[stock_id]["amount"] = current_amount - trade_amount
            if self.position[stock_id]["amount"] < -1e-5:
                raise ValueError(
                    "only have {} {}, require {}".format(
                        self.position[stock_id]["amount"] + trade_amount,
                        stock_id,
                        trade_amount,
                    ),
                )

        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    Position._sell_stock = _patched_sell_stock
    _EPSILON_PATCH_APPLIED = True


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
        resid = y - beta * x
        return pd.Series(resid, index=g.index, dtype=float)

    resid = df.groupby(level=0, group_keys=False).apply(_resid_one_day)
    return _center_rank(resid.astype(float))


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


def _ann_proxy_topk(pred: np.ndarray, y: np.ndarray, date_codes: np.ndarray, topk: int) -> float:
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


def _max_drawdown_proxy_topk(pred: np.ndarray, y: np.ndarray, date_codes: np.ndarray, topk: int) -> float:
    df = pd.DataFrame({"d": date_codes, "p": pred, "y": y})
    spreads = []
    for day, g in df.groupby("d"):
        if len(g) < topk + 5:
            continue
        spread = float(g.nlargest(topk, "p")["y"].mean() - g.nsmallest(topk, "p")["y"].mean())
        spreads.append((day, spread))
    if not spreads:
        return float("nan")
    ser = pd.Series({day: val for day, val in spreads}, dtype=float).sort_index()
    equity = (1.0 + ser).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _turnover_proxy(pred: np.ndarray, inst_codes: np.ndarray, date_codes: np.ndarray, topk: int) -> float:
    df = pd.DataFrame({"d": date_codes, "inst": inst_codes, "p": pred})
    prev: Optional[set] = None
    turns = []
    for _, g in df.groupby("d"):
        if len(g) < topk + 5:
            continue
        cur = set(g.nlargest(topk, "p")["inst"].tolist())
        if prev is not None:
            turns.append(1.0 - len(cur.intersection(prev)) / max(1, topk))
        prev = cur
    if not turns:
        return float("nan")
    return float(np.mean(turns))


def _proxy_metrics(pred_s: pd.Series, label_s: pd.Series, topk: int) -> ProxyMetric:
    aligned = pd.concat({"pred": pred_s, "label": label_s}, axis=1).dropna()
    if aligned.empty:
        return ProxyMetric(np.nan, np.nan, np.nan, np.nan, np.nan, -1e18)
    date_codes = pd.to_datetime(aligned.index.get_level_values(0)).strftime("%Y-%m-%d").to_numpy()
    inst_codes = aligned.index.get_level_values(1).astype(str).to_numpy()
    pred = aligned["pred"].to_numpy(dtype=float)
    y = aligned["label"].to_numpy(dtype=float)
    ic_mean, ic_ir = _daily_ic(pred, y, date_codes)
    ann_proxy = _ann_proxy_topk(pred, y, date_codes, topk=topk)
    max_drawdown_proxy = _max_drawdown_proxy_topk(pred, y, date_codes, topk=topk)
    turnover_proxy = _turnover_proxy(pred, inst_codes, date_codes, topk=topk)
    objective = (
        (float(ic_ir) if np.isfinite(ic_ir) else -999.0)
        + 0.25 * (float(ann_proxy) if np.isfinite(ann_proxy) else 0.0)
        - 0.35 * (float(turnover_proxy) if np.isfinite(turnover_proxy) else 0.0)
    )
    return ProxyMetric(
        ic_mean=float(ic_mean),
        ic_ir=float(ic_ir),
        ann_proxy=float(ann_proxy),
        max_drawdown_proxy=float(max_drawdown_proxy),
        turnover_proxy=float(turnover_proxy),
        objective=float(objective),
    )


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


def _safe_backtest_eval(
    signal_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    open_cost: float,
    close_cost: float,
    topk: int,
    hold_topk: int,
) -> Dict[str, Any]:
    try:
        metrics = _run_backtest_eval(
            signal_df=signal_df,
            base_port_cfg=base_port_cfg,
            start_date=start_date,
            end_date=end_date,
            open_cost=open_cost,
            close_cost=close_cost,
            topk=topk,
            hold_topk=hold_topk,
        )
        return {"ok": True, "metrics": metrics, "error": None}
    except Exception as exc:
        return {
            "ok": False,
            "metrics": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


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


def _mean_daily_spearman(left: pd.Series, right: pd.Series) -> float:
    panel = pd.concat({"l": left, "r": right}, axis=1).dropna()
    vals = []
    for _, g in panel.groupby(level=0):
        corr = g["l"].corr(g["r"], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict past-only second-order ensemble smoke for factor_meta + long_history.")
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
    p.add_argument("--topk", type=int, default=55)
    p.add_argument("--hold-topk", type=int, default=85)
    p.add_argument("--weight-grid", default="0.00,0.05,0.10,0.15,0.20")
    p.add_argument("--output-prefix", default="long_history_second_order")
    return p


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    trans_dir = Path(__file__).resolve().parent
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    stamp = _stamp()

    _apply_runtime_sell_epsilon_patch()

    base_run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    wf_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(wf_cfg)
    base_port_cfg = _extract_port_config(wf_cfg)

    meta_path = trans_dir / META_PRED_FILE
    lh_path = trans_dir / LH_PRED_FILE
    meta_summary_path = trans_dir / META_SUMMARY_FILE
    lh_summary_path = trans_dir / LH_SUMMARY_FILE

    anchor_pred = _slice_period(_as_score_series(_load_pickle(base_run_dir / "artifacts" / "pred.pkl")), args.global_start, args.global_end)
    anchor_label = _slice_period(_as_label_series(_load_pickle(base_run_dir / "artifacts" / "label.pkl")), args.global_start, args.global_end)
    meta_pred = _slice_period(_as_score_series(_load_pickle(meta_path)), args.global_start, args.global_end)
    lh_pred = _slice_period(_as_score_series(_load_pickle(lh_path)), args.global_start, args.global_end)
    meta_summary = _load_json(meta_summary_path)
    lh_summary = _load_json(lh_summary_path)

    common_index = meta_pred.index.intersection(lh_pred.index).intersection(anchor_pred.index).intersection(anchor_label.index)
    if len(common_index) == 0:
        raise RuntimeError("no common aligned index across anchor/meta/long_history/label")

    anchor_pred = anchor_pred.loc[common_index]
    anchor_label = anchor_label.loc[common_index]
    meta_pred = meta_pred.loc[common_index]
    lh_pred = lh_pred.loc[common_index]

    main_leg = _center_rank(meta_pred)
    aux_leg = _center_rank(lh_pred)
    anchor_leg = _center_rank(anchor_pred)
    label_leg = _center_rank(anchor_label)
    residual_leg = _build_residual_leg(main_s=main_leg, aux_s=aux_leg)

    panel = pd.concat(
        {
            "main": main_leg,
            "resid": residual_leg,
            "aux": aux_leg,
            "anchor": anchor_leg,
            "label": label_leg,
        },
        axis=1,
    ).dropna()
    if panel.empty:
        raise RuntimeError("aligned panel is empty after residualization")

    common_dates = pd.to_datetime(panel.index.get_level_values(0))
    effective_global_end = min(pd.Timestamp(args.global_end), pd.Timestamp(common_dates.max()))
    effective_eval_end = min(pd.Timestamp(args.eval_end), effective_global_end)
    effective_eval_start = max(pd.Timestamp(args.eval_start), pd.Timestamp(common_dates.min()))
    if effective_eval_start > effective_eval_end:
        raise RuntimeError(
            f"invalid eval window after common coverage trim: {effective_eval_start.date()}..{effective_eval_end.date()}"
        )

    panel = panel.loc[
        (pd.to_datetime(panel.index.get_level_values(0)) >= pd.Timestamp(args.global_start))
        & (pd.to_datetime(panel.index.get_level_values(0)) <= effective_global_end)
    ]
    dates = pd.to_datetime(panel.index.get_level_values(0))
    quarters = sorted(pd.PeriodIndex(dates, freq="Q").unique())
    weight_grid = [float(x.strip()) for x in str(args.weight_grid).split(",") if x.strip()]
    if not weight_grid:
        raise ValueError("weight_grid is empty")

    final_pred = pd.Series(np.nan, index=panel.index, dtype=float, name="score")
    period_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []

    for seq, quarter in enumerate(quarters, start=1):
        q_start = max(quarter.start_time, pd.Timestamp(args.global_start))
        q_end = min(quarter.end_time, effective_global_end)
        q_mask = (dates >= q_start) & (dates <= q_end)
        if not q_mask.any():
            continue

        train_mask = dates < q_start
        train_dates = pd.DatetimeIndex(sorted(pd.Index(dates[train_mask]).unique()))
        chosen_weight = 0.0
        chosen_cv_score = None
        chosen_cv_folds = 0
        chosen_reason = "warmup_default_main_only"
        eval_proxy = None

        if len(train_dates) >= int(args.min_history_days):
            folds = _build_cv_folds(
                train_dates=train_dates,
                min_history_days=int(args.min_history_days),
                valid_days=int(args.cv_valid_days),
                max_folds=int(args.cv_max_folds),
            )
            if folds:
                best_score = -1e18
                best_weight = 0.0
                fold_count = 0
                for weight in weight_grid:
                    fold_scores = []
                    for fold_dates in folds:
                        fold_mask = pd.to_datetime(panel.index.get_level_values(0)).isin(fold_dates)
                        pred_fold = panel.loc[fold_mask, "main"] + float(weight) * panel.loc[fold_mask, "resid"]
                        proxy = _proxy_metrics(pred_s=pred_fold, label_s=panel.loc[fold_mask, "label"], topk=int(args.topk))
                        if np.isfinite(proxy.objective):
                            fold_scores.append(proxy.objective)
                    if not fold_scores:
                        continue
                    mean_score = float(np.mean(fold_scores))
                    if (mean_score > best_score + 1e-12) or (
                        abs(mean_score - best_score) <= 1e-12 and float(weight) < float(best_weight)
                    ):
                        best_score = mean_score
                        best_weight = float(weight)
                        fold_count = len(fold_scores)
                if best_score > -1e17:
                    chosen_weight = best_weight
                    chosen_cv_score = float(best_score)
                    chosen_cv_folds = int(fold_count)
                    chosen_reason = "blocked_cv_prior_only"

        apply_pred = panel.loc[q_mask, "main"] + chosen_weight * panel.loc[q_mask, "resid"]
        final_pred.loc[q_mask] = apply_pred
        eval_proxy = _proxy_metrics(pred_s=apply_pred, label_s=panel.loc[q_mask, "label"], topk=int(args.topk))
        meta_apply_proxy = _proxy_metrics(
            pred_s=panel.loc[q_mask, "main"],
            label_s=panel.loc[q_mask, "label"],
            topk=int(args.topk),
        )

        period_rows.append(
            {
                "period_seq": seq,
                "period": str(quarter),
                "apply_start": str(pd.Timestamp(q_start).date()),
                "apply_end": str(pd.Timestamp(q_end).date()),
                "train_days": int(len(train_dates)),
                "apply_days": int(pd.to_datetime(panel.loc[q_mask].index.get_level_values(0)).nunique()),
                "selected_weight": float(chosen_weight),
                "selection_reason": chosen_reason,
                "cv_score": chosen_cv_score,
                "cv_folds": int(chosen_cv_folds),
                "apply_ic_ir": float(eval_proxy.ic_ir),
                "apply_ann_proxy": float(eval_proxy.ann_proxy),
                "apply_turnover_proxy": float(eval_proxy.turnover_proxy),
                "meta_apply_ic_ir": float(meta_apply_proxy.ic_ir),
                "meta_apply_ann_proxy": float(meta_apply_proxy.ann_proxy),
                "meta_apply_turnover_proxy": float(meta_apply_proxy.turnover_proxy),
                "delta_apply_ic_ir": float(eval_proxy.ic_ir - meta_apply_proxy.ic_ir)
                if np.isfinite(eval_proxy.ic_ir) and np.isfinite(meta_apply_proxy.ic_ir)
                else float("nan"),
                "delta_apply_ann_proxy": float(eval_proxy.ann_proxy - meta_apply_proxy.ann_proxy)
                if np.isfinite(eval_proxy.ann_proxy) and np.isfinite(meta_apply_proxy.ann_proxy)
                else float("nan"),
            }
        )
        weight_rows.append(
            {
                "period_seq": seq,
                "period": str(quarter),
                "selected_weight_long_history_residual": float(chosen_weight),
                "selected_weight_main_meta": float(1.0),
            }
        )

    final_pred = final_pred.dropna()
    if final_pred.empty:
        raise RuntimeError("final prediction is empty")

    eval_mask = (
        (pd.to_datetime(final_pred.index.get_level_values(0)) >= effective_eval_start)
        & (pd.to_datetime(final_pred.index.get_level_values(0)) <= effective_eval_end)
    )
    eval_pred = final_pred.loc[eval_mask].rename("score").to_frame("score")
    meta_eval = panel.loc[eval_mask, "main"].rename("score").to_frame("score")
    anchor_eval = panel.loc[eval_mask, "anchor"].rename("score").to_frame("score")
    label_eval = panel.loc[eval_mask, "label"]
    if eval_pred.empty:
        raise RuntimeError("evaluation prediction window is empty")

    candidate_proxy = _proxy_metrics(pred_s=eval_pred["score"], label_s=label_eval, topk=int(args.topk))
    meta_proxy = _proxy_metrics(pred_s=meta_eval["score"], label_s=label_eval, topk=int(args.topk))
    anchor_proxy = _proxy_metrics(pred_s=anchor_eval["score"], label_s=label_eval, topk=int(args.topk))

    candidate_bt_rs = _safe_backtest_eval(
        signal_df=eval_pred,
        base_port_cfg=base_port_cfg,
        start_date=str(effective_eval_start.date()),
        end_date=str(effective_eval_end.date()),
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.topk),
        hold_topk=int(args.hold_topk),
    )
    meta_bt_rs = _safe_backtest_eval(
        signal_df=meta_eval,
        base_port_cfg=base_port_cfg,
        start_date=str(effective_eval_start.date()),
        end_date=str(effective_eval_end.date()),
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.topk),
        hold_topk=int(args.hold_topk),
    )
    anchor_bt_rs = _safe_backtest_eval(
        signal_df=anchor_eval,
        base_port_cfg=base_port_cfg,
        start_date=str(effective_eval_start.date()),
        end_date=str(effective_eval_end.date()),
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.topk),
        hold_topk=int(args.hold_topk),
    )
    candidate_bt = candidate_bt_rs["metrics"]
    meta_bt = meta_bt_rs["metrics"]
    anchor_bt = anchor_bt_rs["metrics"]

    slice_rows = []
    for tag, st, ed in _year_slices(str(effective_eval_start.date()), str(effective_eval_end.date())):
        slice_mask = (
            (pd.to_datetime(final_pred.index.get_level_values(0)) >= pd.Timestamp(st))
            & (pd.to_datetime(final_pred.index.get_level_values(0)) <= pd.Timestamp(ed))
        )
        cand_slice = final_pred.loc[slice_mask].rename("score").to_frame("score")
        meta_slice = panel.loc[slice_mask, "main"].rename("score").to_frame("score")
        if cand_slice.empty or meta_slice.empty:
            continue
        cand_metrics_rs = _safe_backtest_eval(
            signal_df=cand_slice,
            base_port_cfg=base_port_cfg,
            start_date=st,
            end_date=ed,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            topk=int(args.topk),
            hold_topk=int(args.hold_topk),
        )
        meta_metrics_rs = _safe_backtest_eval(
            signal_df=meta_slice,
            base_port_cfg=base_port_cfg,
            start_date=st,
            end_date=ed,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            topk=int(args.topk),
            hold_topk=int(args.hold_topk),
        )
        cand_metrics = cand_metrics_rs["metrics"]
        meta_metrics = meta_metrics_rs["metrics"]
        slice_rows.append(
            {
                "split": tag,
                "start": st,
                "end": ed,
                "candidate_backtest_ok": bool(cand_metrics_rs["ok"]),
                "candidate_backtest_error": None if cand_metrics_rs["ok"] else json.dumps(cand_metrics_rs["error"], ensure_ascii=False),
                "candidate_ir": float(cand_metrics["ir"]) if cand_metrics else float("nan"),
                "candidate_annret": float(cand_metrics["annret"]) if cand_metrics else float("nan"),
                "candidate_max_drawdown": float(cand_metrics["max_drawdown"]) if cand_metrics else float("nan"),
                "candidate_turnover": float(cand_metrics["turnover"]) if cand_metrics else float("nan"),
                "meta_backtest_ok": bool(meta_metrics_rs["ok"]),
                "meta_backtest_error": None if meta_metrics_rs["ok"] else json.dumps(meta_metrics_rs["error"], ensure_ascii=False),
                "meta_ir": float(meta_metrics["ir"]) if meta_metrics else float("nan"),
                "meta_annret": float(meta_metrics["annret"]) if meta_metrics else float("nan"),
                "meta_max_drawdown": float(meta_metrics["max_drawdown"]) if meta_metrics else float("nan"),
                "meta_turnover": float(meta_metrics["turnover"]) if meta_metrics else float("nan"),
                "delta_ir": float(cand_metrics["ir"] - meta_metrics["ir"]) if cand_metrics and meta_metrics else float("nan"),
                "delta_annret": float(cand_metrics["annret"] - meta_metrics["annret"]) if cand_metrics and meta_metrics else float("nan"),
            }
        )

    verdict = "PROMISING_FOR_FULL_RUN"
    if not candidate_bt or not meta_bt:
        verdict = "NO_GO"
    elif candidate_bt["ir"] <= meta_bt["ir"] or candidate_bt["annret"] <= meta_bt["annret"]:
        verdict = "NO_GO"
    if len([r for r in period_rows if float(r["selected_weight"]) > 0.0]) == 0:
        verdict = "NO_GO"

    correlations = {
        "main_vs_aux_mean_daily_spearman": _mean_daily_spearman(panel["main"], panel["aux"]),
        "main_vs_residual_mean_daily_spearman": _mean_daily_spearman(panel["main"], panel["resid"]),
        "anchor_vs_aux_mean_daily_spearman": _mean_daily_spearman(panel["anchor"], panel["aux"]),
        "anchor_vs_residual_mean_daily_spearman": _mean_daily_spearman(panel["anchor"], panel["resid"]),
    }

    pred_pkl = trans_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = trans_dir / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    periods_csv = trans_dir / f"{args.output_prefix}_periods_{stamp}.csv"
    weights_csv = trans_dir / f"{args.output_prefix}_weights_{stamp}.csv"
    slices_csv = trans_dir / f"{args.output_prefix}_slices_{stamp}.csv"
    smoke_json = trans_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    with pred_pkl.open("wb") as f:
        pickle.dump(eval_pred.sort_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
    eval_pred.reset_index().to_csv(pred_csv, index=False)
    _write_csv(periods_csv, period_rows)
    _write_csv(weights_csv, weight_rows)
    _write_csv(slices_csv, slice_rows)

    meta_published = meta_summary.get("metrics", {}).get("meta_full", {})
    lh_published = lh_summary.get("metrics", {}).get("test_with_valid_selection", {})
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "strict_past_only_second_order_ensemble_smoke",
        "verdict": verdict,
        "protocol": {
            "selection": "quarterly_forward_locked_weight_grid",
            "main_leg": "factor_augmented_meta_candidate",
            "residual_leg": "cross_sectional_residualized_long_history",
            "weight_grid_long_history_residual": weight_grid,
            "selection_rule": "For each apply quarter, choose weight using prior-only blocked folds on earlier dates.",
            "warmup_default": "weight=0.0 when prior history is insufficient",
            "cv": {
                "min_history_days": int(args.min_history_days),
                "valid_days": int(args.cv_valid_days),
                "max_folds": int(args.cv_max_folds),
            },
            "execution": {
                "strategy": "BufferedTopkWeightStrategy",
                "topk": int(args.topk),
                "hold_topk": int(args.hold_topk),
                "rebalance_mode": "weekly",
            },
        },
        "paths": {
            "anchor_pred": str(base_run_dir / "artifacts" / "pred.pkl"),
            "anchor_label": str(base_run_dir / "artifacts" / "label.pkl"),
            "meta_pred": str(meta_path),
            "meta_summary": str(meta_summary_path),
            "long_history_pred": str(lh_path),
            "long_history_summary": str(lh_summary_path),
        },
        "coverage": {
            "global_start_requested": args.global_start,
            "global_end_requested": args.global_end,
            "eval_start_requested": args.eval_start,
            "eval_end_requested": args.eval_end,
            "common_start": str(pd.Timestamp(common_dates.min()).date()),
            "common_end": str(pd.Timestamp(common_dates.max()).date()),
            "effective_eval_start": str(effective_eval_start.date()),
            "effective_eval_end": str(effective_eval_end.date()),
            "common_rows": int(len(panel)),
            "common_days": int(pd.to_datetime(panel.index.get_level_values(0)).nunique()),
            "eval_rows": int(len(eval_pred)),
            "eval_days": int(pd.to_datetime(eval_pred.index.get_level_values(0)).nunique()),
            "trim_note": "Evaluation end is clipped to the common signal coverage; current long_history coverage ends on 2026-04-28.",
        },
        "correlations": correlations,
        "metrics": {
            "candidate_eval": candidate_bt,
            "factor_meta_eval_same_window": meta_bt,
            "anchor_eval_same_window": anchor_bt,
            "candidate_eval_backtest_status": candidate_bt_rs,
            "factor_meta_eval_backtest_status": meta_bt_rs,
            "anchor_eval_backtest_status": anchor_bt_rs,
            "candidate_proxy_eval": candidate_proxy.__dict__,
            "factor_meta_proxy_eval_same_window": meta_proxy.__dict__,
            "anchor_proxy_eval_same_window": anchor_proxy.__dict__,
            "delta_vs_factor_meta_eval": {
                "ir": float(candidate_bt["ir"] - meta_bt["ir"]) if candidate_bt and meta_bt else None,
                "annret": float(candidate_bt["annret"] - meta_bt["annret"]) if candidate_bt and meta_bt else None,
                "max_drawdown": float(candidate_bt["max_drawdown"] - meta_bt["max_drawdown"]) if candidate_bt and meta_bt else None,
                "turnover": float(candidate_bt["turnover"] - meta_bt["turnover"]) if candidate_bt and meta_bt else None,
            },
            "delta_vs_factor_meta_proxy_eval": {
                "ic_ir": float(candidate_proxy.ic_ir - meta_proxy.ic_ir)
                if np.isfinite(candidate_proxy.ic_ir) and np.isfinite(meta_proxy.ic_ir)
                else None,
                "ann_proxy": float(candidate_proxy.ann_proxy - meta_proxy.ann_proxy)
                if np.isfinite(candidate_proxy.ann_proxy) and np.isfinite(meta_proxy.ann_proxy)
                else None,
                "max_drawdown_proxy": float(candidate_proxy.max_drawdown_proxy - meta_proxy.max_drawdown_proxy)
                if np.isfinite(candidate_proxy.max_drawdown_proxy) and np.isfinite(meta_proxy.max_drawdown_proxy)
                else None,
                "turnover_proxy": float(candidate_proxy.turnover_proxy - meta_proxy.turnover_proxy)
                if np.isfinite(candidate_proxy.turnover_proxy) and np.isfinite(meta_proxy.turnover_proxy)
                else None,
            },
            "reference_published": {
                "factor_meta_full_2024_2026": meta_published,
                "long_history_full_2024_2026": lh_published,
            },
            "year_slices_same_window": slice_rows,
        },
        "selection_periods": period_rows,
        "hard_gate_reference": {
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
            "candidate_pass_same_window": bool(
                candidate_bt is not None and candidate_bt["ir"] > HARD_GATE_IR and candidate_bt["annret"] > HARD_GATE_ANNRET
            ),
            "note": "This smoke run is judged mainly by improvement vs factor_meta on the 2025-01-01..2026-04-28 migration window.",
        },
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "periods_csv": str(periods_csv),
            "weights_csv": str(weights_csv),
            "slices_csv": str(slices_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
        "runtime_sec_total": float(time.perf_counter() - t0_all),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Long History Second Order Ensemble ({stamp})",
        "",
        "## Verdict",
        f"- `{verdict}`",
        "",
        "## Protocol",
        "- Main leg: `factor_augmented_meta_candidate_pred_20260522T120515Z.pkl`.",
        "- Auxiliary leg: `long_history_retrain_candidate_pred_20260522T134241Z.pkl`.",
        "- Residualization: daily cross-sectional linear residual of long_history versus main leg, then rank-center.",
        "- Selection: quarterly forward, weight grid only, prior-only blocked folds.",
        f"- Effective evaluation window: `{effective_eval_start.date()}..{effective_eval_end.date()}`.",
        "",
        "## Same-Window Backtest",
        f"- Candidate backtest ok: `{candidate_bt_rs['ok']}`",
        f"- Candidate backtest metrics: `{json.dumps(candidate_bt, ensure_ascii=False) if candidate_bt else 'FAILED'}`",
        f"- Candidate backtest error: `{json.dumps(candidate_bt_rs['error'], ensure_ascii=False) if candidate_bt_rs['error'] else ''}`",
        f"- Factor meta backtest ok: `{meta_bt_rs['ok']}`",
        f"- Factor meta backtest metrics: `{json.dumps(meta_bt, ensure_ascii=False) if meta_bt else 'FAILED'}`",
        f"- Factor meta backtest error: `{json.dumps(meta_bt_rs['error'], ensure_ascii=False) if meta_bt_rs['error'] else ''}`",
        f"- Candidate proxy: ICIR=`{candidate_proxy.ic_ir:.6f}` AnnProxy=`{candidate_proxy.ann_proxy:.6f}` MDDProxy=`{candidate_proxy.max_drawdown_proxy:.6f}` TurnoverProxy=`{candidate_proxy.turnover_proxy:.6f}`",
        f"- Factor meta proxy: ICIR=`{meta_proxy.ic_ir:.6f}` AnnProxy=`{meta_proxy.ann_proxy:.6f}` MDDProxy=`{meta_proxy.max_drawdown_proxy:.6f}` TurnoverProxy=`{meta_proxy.turnover_proxy:.6f}`",
        f"- Proxy delta vs factor meta: ICIR=`{candidate_proxy.ic_ir - meta_proxy.ic_ir:.6f}` AnnProxy=`{candidate_proxy.ann_proxy - meta_proxy.ann_proxy:.6f}`",
        "",
        "## Selected Weights By Quarter",
        "",
        "| quarter | apply_start | apply_end | selected_weight | cv_folds | cv_score |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in period_rows:
        cv_score_str = "" if row["cv_score"] is None else f"{float(row['cv_score']):.6f}"
        md_lines.append(
            f"| {row['period']} | {row['apply_start']} | {row['apply_end']} | "
            f"{row['selected_weight']:.2f} | {row['cv_folds']} | {cv_score_str} |"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            f"- Common signal coverage ends on `{pd.Timestamp(common_dates.max()).date()}`, so this smoke run stops on `{effective_eval_end.date()}`.",
            f"- Published factor_meta full-window reference remains IR=`{float(meta_published.get('ir', float('nan'))):.6f}` / AnnRet=`{float(meta_published.get('annret', float('nan'))):.6f}` on `2024-01-01..2026-04-30`.",
        ]
    )
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    smoke = {
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "candidate_pred_rows": int(len(_load_pickle(pred_pkl))) if pred_pkl.exists() else 0,
        "period_rows": int(len(pd.read_csv(periods_csv))) if periods_csv.exists() and periods_csv.stat().st_size > 0 else 0,
        "weight_rows": int(len(pd.read_csv(weights_csv))) if weights_csv.exists() and weights_csv.stat().st_size > 0 else 0,
        "slice_rows": int(len(pd.read_csv(slices_csv))) if slices_csv.exists() and slices_csv.stat().st_size > 0 else 0,
        "verdict": verdict,
    }
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
