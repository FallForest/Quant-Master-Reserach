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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")


@dataclass
class Candidate:
    candidate_id: str
    target_mode: str
    alpha: float
    feature_mode: str
    train_rank_ic: float
    valid_rank_ic: float
    valid_rank_ic_ir: float


@dataclass
class BacktestMetric:
    split: str
    candidate_id: str
    topk: int
    n_drop: int
    annret: float
    ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float
    error: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


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


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.to_datetime(pd.Index(vals))


def _parse_instrument_intervals(inst_path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        out.setdefault(parts[0].lower(), []).append((pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    return out


def _read_feature_bin(path: Path, n_cal: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype="<f4")
    if raw.size <= 1:
        return None
    start_idx = int(raw[0])
    vals = raw[1:]
    arr = np.full(n_cal, np.nan, dtype=np.float32)
    s = max(0, start_idx)
    e = min(n_cal, start_idx + vals.size)
    if s < e:
        arr[s:e] = vals[(s - start_idx) : (e - start_idx)]
    return arr


def _interval_active_mask(dates: pd.DatetimeIndex, intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for st, ed in intervals:
        mask |= (dates >= st) & (dates <= ed)
    return mask


def _build_panel(
    provider_uri: Path,
    market: str,
    raw_start: str,
    end_date: str,
    fields: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal = _read_calendar(provider_uri)
    raw_mask = (cal >= pd.Timestamp(raw_start)) & (cal <= pd.Timestamp(end_date))
    if not raw_mask.any():
        raise RuntimeError(f"calendar has no overlap with {raw_start}..{end_date}")
    cal_sub = cal[raw_mask]
    start_pos = int(np.flatnonzero(raw_mask)[0])
    end_pos = int(np.flatnonzero(raw_mask)[-1])
    intervals = _parse_instrument_intervals(provider_uri / "instruments" / f"{market.lower()}.txt")

    frames: List[pd.DataFrame] = []
    coverage: List[Dict[str, Any]] = []
    for inst in sorted(intervals):
        feat_dir = provider_uri / "features" / inst
        arr_map: Dict[str, np.ndarray] = {}
        missing: List[str] = []
        for field in fields:
            arr = _read_feature_bin(feat_dir / f"{field}.day.bin", len(cal))
            if arr is None:
                arr = np.full(len(cal), np.nan, dtype=np.float32)
                missing.append(field)
            arr_map[field] = arr[start_pos : end_pos + 1]
        active = _interval_active_mask(cal_sub, intervals[inst])
        if not active.any():
            continue
        df_i = pd.DataFrame({k: v for k, v in arr_map.items()}, index=cal_sub)
        df_i = df_i.loc[active]
        df_i["instrument"] = inst.upper()
        frames.append(df_i)
        idx = cal_sub[active]
        coverage.append(
            {
                "instrument": inst.upper(),
                "first_date": str(idx.min().date()),
                "last_date": str(idx.max().date()),
                "rows_active": int(active.sum()),
                "rows_train": int(((idx >= TRAIN_START) & (idx <= TRAIN_END)).sum()),
                "rows_valid": int(((idx >= VALID_START) & (idx <= VALID_END)).sum()),
                "rows_test": int(((idx >= TEST_START) & (idx <= TEST_END)).sum()),
                "rows_all_fields_nonnull": int(
                    np.isfinite(np.column_stack([arr_map[f][active] for f in fields])).all(axis=1).sum()
                ),
                "missing_fields": ";".join(missing),
            }
        )
    if not frames:
        raise RuntimeError("no rows constructed from local feature bins")
    panel = pd.concat(frames, axis=0)
    panel.index.name = "datetime"
    panel = panel.reset_index().set_index(["datetime", "instrument"]).sort_index()
    return panel, pd.DataFrame(coverage).sort_values("instrument").reset_index(drop=True)


def _by_inst_pct(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1, sort=False).pct_change(w, fill_method=None)


def _by_inst_shift(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1, sort=False).shift(w)


def _by_inst_roll_mean(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).mean().reset_index(level=0, drop=True)


def _by_inst_roll_std(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(3, w // 3)).std().reset_index(level=0, drop=True)


def _by_inst_roll_min(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).min().reset_index(level=0, drop=True)


def _by_inst_roll_max(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).max().reset_index(level=0, drop=True)


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0, sort=False).rank(method="average", pct=True)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _build_features_and_targets(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    p = panel.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)

    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = p["close"] * factor
    open_ = p["open"] * factor
    high = p["high"] * factor
    low = p["low"] * factor
    vwap = p["vwap"] * factor
    volume = p["volume"].clip(lower=0.0)
    amount = p["amount"].clip(lower=0.0)

    ret1 = _by_inst_pct(close, 1)
    ret2 = _by_inst_pct(close, 2)
    ret5 = _by_inst_pct(close, 5)
    ret10 = _by_inst_pct(close, 10)
    ret20 = _by_inst_pct(close, 20)
    ret60 = _by_inst_pct(close, 60)
    ret120 = _by_inst_pct(close, 120)
    log_vol = np.log1p(volume)
    log_amt = np.log1p(amount)
    vol_chg1 = _by_inst_pct(volume.replace(0.0, np.nan), 1)
    vol_chg5 = _by_inst_pct(volume.replace(0.0, np.nan), 5)
    amt_chg5 = _by_inst_pct(amount.replace(0.0, np.nan), 5)
    intraday = close / (open_ + 1e-12) - 1.0
    overnight = open_ / (_by_inst_shift(close, 1) + 1e-12) - 1.0
    hl_range = (high - low) / (_by_inst_shift(close, 1).abs() + 1e-12)
    vwap_gap = close / (vwap + 1e-12) - 1.0
    vol10 = _by_inst_roll_std(ret1, 10)
    vol20 = _by_inst_roll_std(ret1, 20)
    vol60 = _by_inst_roll_std(ret1, 60)
    roll_min20 = _by_inst_roll_min(close, 20)
    roll_max20 = _by_inst_roll_max(close, 20)
    roll_max60 = _by_inst_roll_max(close, 60)
    rec20 = close / (roll_min20 + 1e-12) - 1.0
    dd60 = close / (roll_max60 + 1e-12) - 1.0
    price_pos20 = (close - roll_min20) / (roll_max20 - roll_min20 + 1e-12)
    mkt_ret = ret1.groupby(level=0).mean()
    mkt_ret_s = pd.Series(mkt_ret.reindex(ret1.index.get_level_values(0)).values, index=ret1.index)
    market_mom20 = _by_inst_roll_mean(mkt_ret_s, 20)

    raw_features: Dict[str, pd.Series] = {
        "rev_1": -ret1,
        "rev_5": -ret5,
        "mom_10": ret10,
        "mom_20": ret20,
        "mom_60": ret60,
        "mom_120": ret120,
        "mom_spread_5_20": ret5 - ret20,
        "mom_accel_10_20": ret10 - _by_inst_shift(ret10, 10),
        "ret2": ret2,
        "intraday": intraday,
        "overnight": overnight,
        "gap_reversal": -(overnight * np.sign(_by_inst_shift(ret1, 1)).fillna(0.0)),
        "intraday_minus_overnight": intraday - overnight,
        "vwap_gap": vwap_gap,
        "hl_range": hl_range,
        "vol_comp_10_60": -(vol10 / (vol60 + 1e-12)),
        "vol_exp_10_20": vol10 / (vol20 + 1e-12),
        "range_comp_10_40": -(_by_inst_roll_mean(hl_range.abs(), 10) / (_by_inst_roll_mean(hl_range.abs(), 40) + 1e-12)),
        "vol_break_5x20": _by_inst_roll_mean(ret1.abs(), 5) / (_by_inst_roll_mean(ret1.abs(), 20) + 1e-12),
        "liq_volume_z20": (log_vol - _by_inst_roll_mean(log_vol, 20)) / (_by_inst_roll_std(log_vol, 20) + 1e-12),
        "liq_amount_z20": (log_amt - _by_inst_roll_mean(log_amt, 20)) / (_by_inst_roll_std(log_amt, 20) + 1e-12),
        "liq_volume_shock_5": vol_chg5,
        "liq_amount_shock_5": amt_chg5,
        "vp_div_10": _by_inst_roll_mean(ret1, 10) - _by_inst_roll_mean(vol_chg1, 10),
        "vp_div_20": _by_inst_roll_mean(ret1, 20) - _by_inst_roll_mean(vol_chg1, 20),
        "vp_cov_proxy_20": -_by_inst_roll_mean(ret1 * vol_chg1, 20),
        "vp_lagcov_proxy_20": -_by_inst_roll_mean(ret1 * _by_inst_shift(vol_chg1, 1), 20),
        "price_pos20": price_pos20,
        "dd60": dd60,
        "recovery20": rec20,
        "recovery_speed": _by_inst_roll_mean(rec20.groupby(level=1, sort=False).diff(), 5),
        "mn_excess_ret20": _by_inst_roll_mean(ret1 - mkt_ret_s, 20),
        "mn_market_neutral_mom20": ret20 - market_mom20,
        "raw_change": p["change"],
        "vol20": vol20,
        "inv_vol20": 1.0 / (vol20 + 1e-4),
    }

    out = pd.DataFrame(index=p.index)
    feature_cols: List[str] = []
    for name, series in raw_features.items():
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{name}__z"] = _cs_z(clean)
        out[f"{name}__rank"] = (_cs_rank_pct(clean).fillna(0.5) - 0.5) * 2.0
        feature_cols.extend([f"{name}__z", f"{name}__rank"])

    label_raw = p.groupby(level=1, sort=False)["close"].shift(-2) / (
        p.groupby(level=1, sort=False)["close"].shift(-1) + 1e-12
    ) - 1.0
    vol_label = _by_inst_roll_std(ret1, 20).shift(1)
    out["label_raw"] = pd.to_numeric(label_raw, errors="coerce")
    out["label_rank"] = (_cs_rank_pct(out["label_raw"]) - 0.5) * 2.0
    out["label_volnorm_rank"] = (_cs_rank_pct(out["label_raw"] / (vol_label + 1e-4)) - 0.5) * 2.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feature_cols


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _daily_rank_ic_series(pred: pd.Series, label: pd.Series) -> pd.Series:
    panel = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    vals: List[Tuple[pd.Timestamp, float]] = []
    for dt, g in panel.groupby(level=0, sort=False):
        if len(g) < 20:
            continue
        corr = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(corr):
            vals.append((pd.Timestamp(dt), float(corr)))
    if not vals:
        return pd.Series(dtype=float)
    return pd.Series({dt: val for dt, val in vals}, dtype=float).sort_index()


def _mean_and_ir(s: pd.Series) -> Tuple[float, float]:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return float("nan"), float("nan")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    return mean, float(mean / (std + 1e-12) * np.sqrt(252.0))


def _fit_ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    yz = np.nan_to_num(y - np.nanmean(y), nan=0.0, posinf=0.0, neginf=0.0)
    xtx = xz.T @ xz
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    coef = np.linalg.solve(xtx + reg, xz.T @ yz)
    return coef.astype(np.float64), mu.astype(np.float64), sd.astype(np.float64)


def _predict_ridge(x: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return xz @ coef


def _get_report_for_day_freq(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    return portfolio_metric_dict[next(iter(portfolio_metric_dict.keys()))][0]


def _calc_costed_metrics(report_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    return (
        float(risk_df.loc["annualized_return", "risk"]),
        float(risk_df.loc["information_ratio", "risk"]),
        float(risk_df.loc["max_drawdown", "risk"]),
        float(report_df["turnover"].mean()),
    )


def _run_bt(
    signal_df: pd.DataFrame,
    split_name: str,
    candidate_id: str,
    start_time: str,
    end_time: str,
    topk: int,
    n_drop: int,
    port_cfg_template: Dict[str, Any],
    benchmark: str,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, int, int], Any],
) -> BacktestMetric:
    t0 = time.perf_counter()
    try:
        cfg = copy.deepcopy(port_cfg_template)
        bcfg = cfg["backtest"]
        bcfg["start_time"] = start_time
        bcfg["end_time"] = end_time
        exch_cfg = dict(bcfg.get("exchange_kwargs", {}))
        cache_key = (start_time, end_time, int(topk), int(n_drop))
        if cache_key not in exchange_cache:
            exchange_cache[cache_key] = get_exchange(
                freq="day",
                start_time=start_time,
                end_time=end_time,
                deal_price=str(exch_cfg.get("deal_price", "close")),
                limit_threshold=float(exch_cfg.get("limit_threshold", 0.095)),
                open_cost=float(open_cost),
                close_cost=float(close_cost),
                min_cost=float(exch_cfg.get("min_cost", 5)),
            )
        strategy = TopkDropoutStrategy(
            signal=signal_df,
            topk=int(topk),
            n_drop=int(n_drop),
            method_sell="bottom",
            method_buy="top",
            hold_thresh=1,
            only_tradable=False,
            forbid_all_trade_at_limit=True,
        )
        exch_cfg["open_cost"] = float(open_cost)
        exch_cfg["close_cost"] = float(close_cost)
        exch_cfg["exchange"] = exchange_cache[cache_key]
        executor_cfg = cfg.get(
            "executor",
            {
                "class": "SimulatorExecutor",
                "module_path": "quant_master.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
        )
        pm, _ = run_backtest(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy,
            executor=executor_cfg,
            benchmark=benchmark,
            account=bcfg.get("account", 100000000),
            exchange_kwargs=exch_cfg,
            pos_type=bcfg.get("pos_type", "Position"),
        )
        report = _get_report_for_day_freq(pm)
        annret, ir, maxdd, turnover = _calc_costed_metrics(report)
        return BacktestMetric(split_name, candidate_id, topk, n_drop, annret, ir, maxdd, turnover, time.perf_counter() - t0, "")
    except Exception as exc:  # noqa: BLE001
        return BacktestMetric(
            split_name,
            candidate_id,
            topk,
            n_drop,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            time.perf_counter() - t0,
            f"{type(exc).__name__}: {exc}",
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-2024 strict ridge rank retrain with one-shot 2024-2026 evaluation.")
    p.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default=str(
            Path(__file__).resolve().parent
            / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
        ),
    )
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--alpha-grid", default="1,10,100,1000,10000")
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="pre2024_train_new_model_lockstep")
    p.add_argument("--smoke", action="store_true", help="Use a smaller grid for fast plumbing validation.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    out_dir = Path(__file__).resolve().parent
    stamp = _stamp()
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
    quant_master.init(provider_uri=str(provider_uri), region="cn")
    wf_cfg = _load_config(Path(args.workflow_config).expanduser().resolve())
    port_cfg = _extract_port_config(wf_cfg)
    benchmark = str(wf_cfg.get("benchmark", "SH000300"))

    panel_raw, coverage_df = _build_panel(provider_uri, str(args.market), RAW_START, TEST_END, BASE_FIELDS)
    dataset, feature_cols = _build_features_and_targets(panel_raw)
    dataset = dataset.dropna(subset=["label_raw", "label_rank", "label_volnorm_rank"])
    day_counts = dataset.groupby(level=0)["label_raw"].count()
    good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
    dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    train_df = dataset.loc[_mask(dt_idx, TRAIN_START, TRAIN_END)]
    valid_df = dataset.loc[_mask(dt_idx, VALID_START, VALID_END)]
    test_df = dataset.loc[_mask(dt_idx, TEST_START, TEST_END)]
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(f"empty split train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

    alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
    topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
    ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
    if args.smoke:
        alpha_grid = alpha_grid[:2]
        topk_grid = topk_grid[:1]
        ndrop_grid = ndrop_grid[:1]

    target_modes = ["label_rank", "label_volnorm_rank"]
    feature_modes = ["expanded_rank_z", "expanded_rank_only"]
    x_cols_by_mode = {
        "expanded_rank_z": feature_cols,
        "expanded_rank_only": [c for c in feature_cols if c.endswith("__rank")],
    }

    candidate_rows: List[Dict[str, Any]] = []
    candidates: List[Candidate] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}
    for feature_mode in feature_modes:
        cols = x_cols_by_mode[feature_mode]
        x_train = train_df[cols].astype(np.float64).values
        x_valid = valid_df[cols].astype(np.float64).values
        x_test = test_df[cols].astype(np.float64).values
        for target_mode in target_modes:
            y_train = train_df[target_mode].astype(np.float64).values
            for alpha in alpha_grid:
                cid = f"ridge_{feature_mode}_{target_mode}_a{alpha:g}"
                coef, mu, sd = _fit_ridge_closed_form(x_train, y_train, alpha)
                pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
                pred_valid = _cs_z(pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
                pred_test = _cs_z(pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"))
                train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
                train_ic, _ = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                candidates.append(Candidate(cid, target_mode, alpha, feature_mode, train_ic, valid_ic, valid_ic_ir))
                predictions[cid] = {"train": pred_train, "valid": pred_valid, "test": pred_test}
                candidate_rows.append(
                    {
                        "candidate_id": cid,
                        "feature_mode": feature_mode,
                        "target_mode": target_mode,
                        "alpha": alpha,
                        "feature_count": len(cols),
                        "train_rank_ic": train_ic,
                        "valid_rank_ic": valid_ic,
                        "valid_rank_ic_ir": valid_ic_ir,
                    }
                )

    preselect = sorted(
        candidates,
        key=lambda x: (
            x.valid_rank_ic_ir if np.isfinite(x.valid_rank_ic_ir) else -1e9,
            x.valid_rank_ic if np.isfinite(x.valid_rank_ic) else -1e9,
        ),
        reverse=True,
    )[: max(1, min(6, len(candidates)))]

    exchange_cache: Dict[Tuple[str, str, int, int], Any] = {}
    valid_bt_rows: List[BacktestMetric] = []
    combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk]
    for cand in preselect:
        sig = predictions[cand.candidate_id]["valid"].rename("score").to_frame("score")
        for topk, n_drop in combos:
            valid_bt_rows.append(
                _run_bt(
                    sig,
                    "valid_2023",
                    cand.candidate_id,
                    VALID_START,
                    VALID_END,
                    topk,
                    n_drop,
                    port_cfg,
                    benchmark,
                    float(args.open_cost),
                    float(args.close_cost),
                    exchange_cache,
                )
            )

    ok_valid = [r for r in valid_bt_rows if not r.error and np.isfinite(r.ir) and np.isfinite(r.annret)]
    if ok_valid:
        selected = sorted(ok_valid, key=lambda r: (r.ir, r.annret), reverse=True)[0]
    else:
        fallback = preselect[0]
        selected = BacktestMetric(
            "valid_2023",
            fallback.candidate_id,
            combos[0][0] if combos else 40,
            combos[0][1] if combos else 2,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            "no valid backtest combo",
        )

    test_metric = _run_bt(
        predictions[selected.candidate_id]["test"].rename("score").to_frame("score"),
        "test_2024_2026",
        selected.candidate_id,
        TEST_START,
        TEST_END,
        int(selected.topk),
        int(selected.n_drop),
        port_cfg,
        benchmark,
        float(args.open_cost),
        float(args.close_cost),
        exchange_cache,
    )
    selected_cand = next(c for c in candidates if c.candidate_id == selected.candidate_id)
    test_ic_s = _daily_rank_ic_series(predictions[selected.candidate_id]["test"], test_df["label_raw"])
    test_rank_ic, test_rank_ic_ir = _mean_and_ir(test_ic_s)

    hard_gate_pass = bool(
        np.isfinite(selected.ir)
        and np.isfinite(selected.annret)
        and selected.ir > HARD_GATE_IR
        and selected.annret > HARD_GATE_ANNRET
    )

    coverage_csv = out_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    candidates_csv = out_dir / f"{args.output_prefix}_candidates_{stamp}.csv"
    valid_bt_csv = out_dir / f"{args.output_prefix}_valid_backtests_{stamp}.csv"
    split_metrics_csv = out_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    pred_pkl = out_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = out_dir / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    smoke_json = out_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    coverage_df.to_csv(coverage_csv, index=False)
    _write_csv(candidates_csv, candidate_rows)
    _write_csv(
        valid_bt_csv,
        [
            {
                "split": r.split,
                "candidate_id": r.candidate_id,
                "topk": r.topk,
                "n_drop": r.n_drop,
                "costed_ir": r.ir,
                "costed_annret": r.annret,
                "max_drawdown": r.max_drawdown,
                "turnover": r.turnover,
                "elapsed_sec": r.elapsed_sec,
                "error": r.error,
            }
            for r in valid_bt_rows
        ],
    )
    split_rows = [
        {
            "split": "train_2020_2022",
            "candidate_id": selected.candidate_id,
            "rank_ic": selected_cand.train_rank_ic,
            "rank_ic_ir": "",
            "costed_ir": "",
            "costed_annret": "",
            "max_drawdown": "",
            "turnover": "",
            "topk": "",
            "n_drop": "",
            "error": "",
        },
        {
            "split": "valid_2023",
            "candidate_id": selected.candidate_id,
            "rank_ic": selected_cand.valid_rank_ic,
            "rank_ic_ir": selected_cand.valid_rank_ic_ir,
            "costed_ir": selected.ir,
            "costed_annret": selected.annret,
            "max_drawdown": selected.max_drawdown,
            "turnover": selected.turnover,
            "topk": selected.topk,
            "n_drop": selected.n_drop,
            "error": selected.error,
        },
        {
            "split": "test_2024_2026",
            "candidate_id": selected.candidate_id,
            "rank_ic": test_rank_ic,
            "rank_ic_ir": test_rank_ic_ir,
            "costed_ir": test_metric.ir,
            "costed_annret": test_metric.annret,
            "max_drawdown": test_metric.max_drawdown,
            "turnover": test_metric.turnover,
            "topk": test_metric.topk,
            "n_drop": test_metric.n_drop,
            "error": test_metric.error,
        },
    ]
    _write_csv(split_metrics_csv, split_rows)

    pred_test = predictions[selected.candidate_id]["test"].rename("score").to_frame("score").sort_index()
    with pred_pkl.open("wb") as f:
        pickle.dump(pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    pred_test.reset_index().to_csv(pred_csv, index=False)

    summary = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "strict breakthrough search: train only on 2020-2022, select on 2023, evaluate 2024-2026 once",
        "provider_uri": str(provider_uri),
        "market": str(args.market),
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "splits": {
            "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": int(len(train_df))},
            "valid": {"start": VALID_START, "end": VALID_END, "rows": int(len(valid_df))},
            "test": {"start": TEST_START, "end": TEST_END, "rows": int(len(test_df))},
        },
        "training_protocol": {
            "raw_start": RAW_START,
            "test_selection": "none; selected candidate/topk/n_drop fixed by valid_2023 before test backtest",
            "model_family": "closed_form_ridge_linear",
            "target_modes": target_modes,
            "feature_modes": feature_modes,
            "alpha_grid": alpha_grid,
            "valid_backtested_candidates": [c.candidate_id for c in preselect],
            "topk_grid": topk_grid,
            "ndrop_grid": ndrop_grid,
            "features": "expanded OHLCV/factor past-only signals with cross-sectional z/rank transforms",
            "label_raw": "per-instrument shift(-2)/shift(-1)-1 on close",
            "selection_rule": "max valid_2023 real net-cost IR, tie by AnnRet",
        },
        "selected_model": {
            "candidate_id": selected.candidate_id,
            "feature_mode": selected_cand.feature_mode,
            "target_mode": selected_cand.target_mode,
            "alpha": selected_cand.alpha,
            "topk": selected.topk,
            "n_drop": selected.n_drop,
        },
        "metrics": {
            "train_2020_2022": {"rank_ic": selected_cand.train_rank_ic},
            "valid_2023": {
                "rank_ic": selected_cand.valid_rank_ic,
                "rank_ic_ir": selected_cand.valid_rank_ic_ir,
                "costed_ir": selected.ir,
                "costed_annret": selected.annret,
                "max_drawdown": selected.max_drawdown,
                "turnover": selected.turnover,
                "error": selected.error,
            },
            "test_2024_2026": {
                "rank_ic": test_rank_ic,
                "rank_ic_ir": test_rank_ic_ir,
                "costed_ir": test_metric.ir,
                "costed_annret": test_metric.annret,
                "max_drawdown": test_metric.max_drawdown,
                "turnover": test_metric.turnover,
                "error": test_metric.error,
            },
        },
        "hard_gate": {
            "rule": {
                "scope": "strict non-test-selected valid_2023",
                "ir_gt": HARD_GATE_IR,
                "annret_gt": HARD_GATE_ANNRET,
            },
            "passed": hard_gate_pass,
        },
        "runtime_sec_total": time.perf_counter() - t0_all,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "candidates_csv": str(candidates_csv),
            "valid_backtests_csv": str(valid_bt_csv),
            "split_metrics_csv": str(split_metrics_csv),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        f"# Pre-2024 Train New Model Lockstep ({stamp})",
        "",
        f"- hard_gate_pass: `{hard_gate_pass}`",
        f"- selected: `{selected.candidate_id}` topk/n_drop `{selected.topk}/{selected.n_drop}`",
        f"- valid_2023 IR/AnnRet: `{_safe_float(selected.ir):.6f}` / `{_safe_float(selected.annret):.6f}`",
        f"- test_2024_2026 IR/AnnRet: `{_safe_float(test_metric.ir):.6f}` / `{_safe_float(test_metric.annret):.6f}`",
        f"- costs: open `{args.open_cost}` close `{args.close_cost}`",
        f"- protocol: train `{TRAIN_START}..{TRAIN_END}`, select `{VALID_START}..{VALID_END}`, test once `{TEST_START}..{TEST_END}`",
    ]
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    smoke = {
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "coverage_csv_rows": int(len(pd.read_csv(coverage_csv))) if coverage_csv.exists() else 0,
        "candidates_csv_rows": int(len(pd.read_csv(candidates_csv))) if candidates_csv.exists() else 0,
        "valid_backtests_csv_rows": int(len(pd.read_csv(valid_bt_csv))) if valid_bt_csv.exists() and valid_bt_csv.stat().st_size else 0,
        "split_metrics_csv_rows": int(len(pd.read_csv(split_metrics_csv))) if split_metrics_csv.exists() else 0,
        "candidate_pred_rows": int(len(_load_pickle(pred_pkl))) if pred_pkl.exists() else 0,
        "hard_gate_passed": hard_gate_pass,
    }
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

