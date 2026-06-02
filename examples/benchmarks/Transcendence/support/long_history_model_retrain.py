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
from lightgbm import LGBMRegressor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
RAW_START = "2019-01-01"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")


@dataclass
class BacktestMetric:
    split: str
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
        for row in rows:
            writer.writerow(row)


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


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


def _init_quant_master(provider_uri: str) -> None:
    quant_master.init(provider_uri=provider_uri, region="cn")


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.to_datetime(pd.Index(vals))


def _parse_instrument_intervals(inst_path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        inst = parts[0].lower()
        st = pd.Timestamp(parts[1])
        ed = pd.Timestamp(parts[2])
        out.setdefault(inst, []).append((st, ed))
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
    m = np.zeros(len(dates), dtype=bool)
    if not intervals:
        return m
    for st, ed in intervals:
        m |= (dates >= st) & (dates <= ed)
    return m


def _build_long_history_panel(
    provider_uri: Path,
    market: str,
    raw_start: str,
    end_date: str,
    fields: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal = _read_calendar(provider_uri)
    raw_st = pd.Timestamp(raw_start)
    raw_ed = pd.Timestamp(end_date)
    raw_mask = (cal >= raw_st) & (cal <= raw_ed)
    if not raw_mask.any():
        raise RuntimeError(f"calendar has no overlap with {raw_start}..{end_date}")
    cal_sub = cal[raw_mask]
    start_pos = int(np.flatnonzero(raw_mask)[0])
    end_pos = int(np.flatnonzero(raw_mask)[-1])

    inst_path = provider_uri / "instruments" / f"{market.lower()}.txt"
    intervals = _parse_instrument_intervals(inst_path)
    instruments = sorted(intervals.keys())

    rows: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, Any]] = []

    for inst in instruments:
        feat_dir = provider_uri / "features" / inst
        arr_map: Dict[str, np.ndarray] = {}
        missing_fields: List[str] = []
        for f in fields:
            p = feat_dir / f"{f}.day.bin"
            arr = _read_feature_bin(p, n_cal=len(cal))
            if arr is None:
                missing_fields.append(f)
                arr = np.full(len(cal), np.nan, dtype=np.float32)
            arr_map[f] = arr[start_pos : end_pos + 1]

        active_mask = _interval_active_mask(cal_sub, intervals.get(inst, []))
        if not active_mask.any():
            continue
        df_i = pd.DataFrame({k: v for k, v in arr_map.items()}, index=cal_sub)
        df_i = df_i.loc[active_mask]
        df_i["instrument"] = inst.upper()
        rows.append(df_i)

        close_nonnull = int(np.isfinite(arr_map["close"][active_mask]).sum())
        all_nonnull = int(np.isfinite(np.column_stack([arr_map[f][active_mask] for f in fields])).all(axis=1).sum())
        date_idx = cal_sub[active_mask]
        year_set = set(int(x.year) for x in date_idx)
        train_m = (date_idx >= pd.Timestamp(TRAIN_START)) & (date_idx <= pd.Timestamp(TRAIN_END))
        valid_m = (date_idx >= pd.Timestamp(VALID_START)) & (date_idx <= pd.Timestamp(VALID_END))
        test_m = (date_idx >= pd.Timestamp(TEST_START)) & (date_idx <= pd.Timestamp(TEST_END))
        coverage_rows.append(
            {
                "instrument": inst.upper(),
                "first_date": str(date_idx.min().date()),
                "last_date": str(date_idx.max().date()),
                "rows_active": int(active_mask.sum()),
                "rows_close_nonnull": close_nonnull,
                "rows_all_fields_nonnull": all_nonnull,
                "rows_train": int(train_m.sum()),
                "rows_valid": int(valid_m.sum()),
                "rows_test": int(test_m.sum()),
                "has_2020": int(2020 in year_set),
                "has_2021": int(2021 in year_set),
                "has_2022": int(2022 in year_set),
                "has_2023": int(2023 in year_set),
                "has_2024": int(2024 in year_set),
                "has_2025": int(2025 in year_set),
                "has_2026": int(2026 in year_set),
                "missing_fields": ";".join(missing_fields),
            }
        )

    if not rows:
        raise RuntimeError("no instrument rows constructed from local feature bins")

    panel = pd.concat(rows, axis=0)
    panel.index.name = "datetime"
    panel = panel.reset_index().set_index(["datetime", "instrument"]).sort_index()
    cov_df = pd.DataFrame(coverage_rows).sort_values("instrument").reset_index(drop=True)
    return panel, cov_df


def _build_features_and_label(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    p = panel.copy()
    for c in ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)

    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    adj_close = p["close"] * factor
    adj_open = p["open"] * factor
    adj_high = p["high"] * factor
    adj_low = p["low"] * factor
    adj_vwap = p["vwap"] * factor

    g = p.groupby(level=1, sort=False)
    p["ret1"] = g["close"].pct_change(1, fill_method=None)
    p["ret2"] = g["close"].pct_change(2, fill_method=None)
    p["ret5"] = g["close"].pct_change(5, fill_method=None)
    p["ret10"] = g["close"].pct_change(10, fill_method=None)
    p["ret20"] = g["close"].pct_change(20, fill_method=None)
    p["ret60"] = g["close"].pct_change(60, fill_method=None)
    p["mom120"] = g["close"].pct_change(120, fill_method=None)
    p["rev5"] = -p["ret5"]
    p["rev20"] = -p["ret20"]

    p["intraday"] = adj_close / (adj_open + 1e-12) - 1.0
    p["overnight"] = adj_open / (g["close"].shift(1) * factor + 1e-12) - 1.0
    p["hl_range"] = (adj_high - adj_low) / (adj_close.abs() + 1e-12)
    p["vwap_gap"] = adj_close / (adj_vwap + 1e-12) - 1.0

    p["log_volume"] = np.log1p(np.clip(p["volume"], 0.0, None))
    p["log_amount"] = np.log1p(np.clip(p["amount"], 0.0, None))
    p["vol_chg1"] = g["volume"].pct_change(1, fill_method=None)
    p["vol_chg5"] = g["volume"].pct_change(5, fill_method=None)
    p["amt_chg5"] = g["amount"].pct_change(5, fill_method=None)
    p["amt_chg20"] = g["amount"].pct_change(20, fill_method=None)

    p["ma_gap5"] = p["close"] / (g["close"].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0
    p["ma_gap20"] = p["close"] / (g["close"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0
    p["ma_gap60"] = p["close"] / (g["close"].rolling(60, min_periods=15).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0

    p["vol5"] = g["ret1"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    p["vol20"] = g["ret1"].rolling(20, min_periods=8).std().reset_index(level=0, drop=True)
    p["vol60"] = g["ret1"].rolling(60, min_periods=20).std().reset_index(level=0, drop=True)
    p["ret_skew20"] = g["ret1"].rolling(20, min_periods=10).skew().reset_index(level=0, drop=True)

    roll_min20 = g["close"].rolling(20, min_periods=5).min().reset_index(level=0, drop=True)
    roll_max20 = g["close"].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    p["price_pos20"] = (p["close"] - roll_min20) / (roll_max20 - roll_min20 + 1e-12)
    p["dd120"] = p["close"] / (g["close"].rolling(120, min_periods=30).max().reset_index(level=0, drop=True) + 1e-12) - 1.0

    p["corr_ret1_logvol20"] = np.nan
    for _, sub in p[["ret1", "log_volume"]].groupby(level=1, sort=False):
        corr = sub["ret1"].rolling(20, min_periods=10).corr(sub["log_volume"])
        p.loc[sub.index, "corr_ret1_logvol20"] = corr.values

    p["label"] = g["close"].shift(-2) / (g["close"].shift(-1) + 1e-12) - 1.0

    feature_cols = [
        "ret1",
        "ret2",
        "ret5",
        "ret10",
        "ret20",
        "ret60",
        "mom120",
        "rev5",
        "rev20",
        "intraday",
        "overnight",
        "hl_range",
        "vwap_gap",
        "log_volume",
        "log_amount",
        "vol_chg1",
        "vol_chg5",
        "amt_chg5",
        "amt_chg20",
        "ma_gap5",
        "ma_gap20",
        "ma_gap60",
        "vol5",
        "vol20",
        "vol60",
        "ret_skew20",
        "price_pos20",
        "dd120",
        "corr_ret1_logvol20",
        "change",
    ]

    # Cross-sectional normalization to stabilize across regimes.
    x = p[feature_cols].copy()
    for c in feature_cols:
        mu = x[c].groupby(level=0).transform("mean")
        sd = x[c].groupby(level=0).transform("std")
        z = (x[c] - mu) / (sd + 1e-12)
        x[c] = z.clip(-8.0, 8.0).fillna(0.0)

    out = x.copy()
    out["label"] = pd.to_numeric(p["label"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feature_cols


def _daily_rank_ic(pred: pd.Series, y: pd.Series) -> float:
    panel = pd.concat([pred.rename("pred"), y.rename("label")], axis=1).dropna()
    if panel.empty:
        return float("nan")
    vals = []
    for _, g in panel.groupby(level=0):
        if len(g) < 8:
            continue
        c = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(c):
            vals.append(float(c))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _get_report_for_day_freq(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    k0 = next(iter(portfolio_metric_dict.keys()))
    return portfolio_metric_dict[k0][0]


def _calc_costed_metrics(report_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    maxdd = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, maxdd, turnover


def _run_bt(
    signal_df: pd.DataFrame,
    split_name: str,
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
        return BacktestMetric(
            split=split_name,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=float(annret),
            ir=float(ir),
            max_drawdown=float(maxdd),
            turnover=float(turnover),
            elapsed_sec=float(time.perf_counter() - t0),
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return BacktestMetric(
            split=split_name,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=float("nan"),
            ir=float("nan"),
            max_drawdown=float("nan"),
            turnover=float("nan"),
            elapsed_sec=float(time.perf_counter() - t0),
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_mask(idx_dt: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(idx_dt)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def main() -> int:
    p = argparse.ArgumentParser(description="Worker G long-history retrain from the local QuantMaster CN data store feature bins")
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
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-prefix", default="long_history_retrain")
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample-bytree", type=float, default=0.80)
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    args = p.parse_args()

    t0_all = time.perf_counter()
    out_dir = Path(__file__).resolve().parent
    stamp = _stamp()
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))

    _init_quant_master(str(provider_uri))

    wf_cfg = _load_config(Path(args.workflow_config).expanduser().resolve())
    port_cfg = _extract_port_config(wf_cfg)
    benchmark = str(wf_cfg.get("benchmark", "SH000300"))

    panel_raw, cov_df = _build_long_history_panel(
        provider_uri=provider_uri,
        market=str(args.market),
        raw_start=RAW_START,
        end_date=TEST_END,
        fields=BASE_FIELDS,
    )
    dataset, feature_cols = _build_features_and_label(panel_raw)

    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    train_m = _build_mask(dt_idx, TRAIN_START, TRAIN_END)
    valid_m = _build_mask(dt_idx, VALID_START, VALID_END)
    test_m = _build_mask(dt_idx, TEST_START, TEST_END)

    dataset = dataset[train_m | valid_m | test_m]
    dataset = dataset.dropna(subset=["label"])

    # Ensure each date has enough names after drops.
    count_per_day = dataset.groupby(level=0)["label"].count()
    good_days = count_per_day[count_per_day >= 40].index
    dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)]

    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    train_m = _build_mask(dt_idx, TRAIN_START, TRAIN_END)
    valid_m = _build_mask(dt_idx, VALID_START, VALID_END)
    test_m = _build_mask(dt_idx, TEST_START, TEST_END)

    train_df = dataset.loc[train_m]
    valid_df = dataset.loc[valid_m]
    test_df = dataset.loc[test_m]

    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(
            f"split empty: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}; cannot continue"
        )

    x_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df["label"].astype(np.float32).values
    x_valid = valid_df[feature_cols].astype(np.float32).values
    y_valid = valid_df["label"].astype(np.float32).values
    x_test = test_df[feature_cols].astype(np.float32).values
    y_test = test_df["label"].astype(np.float32).values

    train_year = pd.to_datetime(train_df.index.get_level_values(0)).year.values
    sample_weight = np.ones(len(train_df), dtype=np.float32)
    sample_weight += (train_year >= 2022).astype(np.float32) * 0.5

    model = LGBMRegressor(
        objective="regression",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=-1,
        min_child_samples=80,
        reg_alpha=0.2,
        reg_lambda=1.2,
        subsample=float(args.subsample),
        subsample_freq=1,
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.random_state),
        n_jobs=8,
        verbosity=-1,
    )
    fit_t0 = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_valid, y_valid)],
        eval_metric="l2",
        callbacks=[],
    )
    fit_sec = time.perf_counter() - fit_t0

    pred_train = pd.Series(model.predict(x_train), index=train_df.index, name="score").astype(float)
    pred_valid = pd.Series(model.predict(x_valid), index=valid_df.index, name="score").astype(float)
    pred_test = pd.Series(model.predict(x_test), index=test_df.index, name="score").astype(float)

    # Cross-sectional z-score as final score for portfolio ranking.
    def _cs_z(s: pd.Series) -> pd.Series:
        mu = s.groupby(level=0).transform("mean")
        sd = s.groupby(level=0).transform("std")
        return ((s - mu) / (sd + 1e-12)).fillna(0.0)

    pred_train = _cs_z(pred_train)
    pred_valid = _cs_z(pred_valid)
    pred_test = _cs_z(pred_test)

    ic_train = _daily_rank_ic(pred_train, train_df["label"])
    ic_valid = _daily_rank_ic(pred_valid, valid_df["label"])
    ic_test = _daily_rank_ic(pred_test, test_df["label"])

    topk_grid = [int(x.strip()) for x in str(args.topk_grid).split(",") if x.strip()]
    ndrop_grid = [int(x.strip()) for x in str(args.ndrop_grid).split(",") if x.strip()]
    combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk]
    if not combos:
        combos = [(40, 3)]

    exchange_cache: Dict[Tuple[str, str, int, int], Any] = {}
    bt_rows: List[BacktestMetric] = []
    for tk, nd in combos:
        bt_rows.append(
            _run_bt(
                signal_df=pred_valid.rename("score").to_frame("score"),
                split_name="valid_2023",
                start_time=VALID_START,
                end_time=VALID_END,
                topk=tk,
                n_drop=nd,
                port_cfg_template=port_cfg,
                benchmark=benchmark,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=exchange_cache,
            )
        )

    ok_valid = [x for x in bt_rows if not x.error and np.isfinite(x.ir) and np.isfinite(x.annret)]
    if ok_valid:
        best_valid = sorted(ok_valid, key=lambda x: (x.ir, x.annret), reverse=True)[0]
    else:
        best_valid = BacktestMetric(
            split="valid_2023",
            topk=40,
            n_drop=3,
            annret=float("nan"),
            ir=float("nan"),
            max_drawdown=float("nan"),
            turnover=float("nan"),
            elapsed_sec=0.0,
            error="no valid backtest combo",
        )

    test_metric = _run_bt(
        signal_df=pred_test.rename("score").to_frame("score"),
        split_name="test_2024_2026",
        start_time=TEST_START,
        end_time=TEST_END,
        topk=int(best_valid.topk),
        n_drop=int(best_valid.n_drop),
        port_cfg_template=port_cfg,
        benchmark=benchmark,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        exchange_cache=exchange_cache,
    )
    bt_rows.append(test_metric)

    pass_hard_gate = bool(
        np.isfinite(best_valid.ir)
        and np.isfinite(best_valid.annret)
        and float(best_valid.ir) > HARD_GATE_IR
        and float(best_valid.annret) > HARD_GATE_ANNRET
    )

    coverage_csv = out_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    metrics_csv = out_dir / f"{args.output_prefix}_metrics_{stamp}.csv"
    pred_pkl = out_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = out_dir / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    smoke_json = out_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    cov_df.to_csv(coverage_csv, index=False)
    with pred_pkl.open("wb") as f:
        pickle.dump(pred_test.rename("score").to_frame("score").sort_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
    pred_test.rename("score").to_frame("score").reset_index().to_csv(pred_csv, index=False)

    metrics_rows: List[Dict[str, Any]] = []
    metrics_rows.append(
        {
            "split": "train_2020_2022",
            "metric": "daily_rank_ic",
            "value": float(ic_train),
            "topk": "",
            "n_drop": "",
            "error": "",
        }
    )
    metrics_rows.append(
        {
            "split": "valid_2023",
            "metric": "daily_rank_ic",
            "value": float(ic_valid),
            "topk": "",
            "n_drop": "",
            "error": "",
        }
    )
    metrics_rows.append(
        {
            "split": "test_2024_2026",
            "metric": "daily_rank_ic",
            "value": float(ic_test),
            "topk": "",
            "n_drop": "",
            "error": "",
        }
    )
    for r in bt_rows:
        metrics_rows.append(
            {
                "split": r.split,
                "metric": "costed_ir",
                "value": float(r.ir),
                "topk": int(r.topk),
                "n_drop": int(r.n_drop),
                "error": r.error,
            }
        )
        metrics_rows.append(
            {
                "split": r.split,
                "metric": "costed_annret",
                "value": float(r.annret),
                "topk": int(r.topk),
                "n_drop": int(r.n_drop),
                "error": r.error,
            }
        )
        metrics_rows.append(
            {
                "split": r.split,
                "metric": "max_drawdown",
                "value": float(r.max_drawdown),
                "topk": int(r.topk),
                "n_drop": int(r.n_drop),
                "error": r.error,
            }
        )
        metrics_rows.append(
            {
                "split": r.split,
                "metric": "turnover",
                "value": float(r.turnover),
                "topk": int(r.topk),
                "n_drop": int(r.n_drop),
                "error": r.error,
            }
        )
    _write_csv(metrics_csv, metrics_rows)

    feature_importance = sorted(
        [
            {"feature": f, "importance": float(v)}
            for f, v in zip(feature_cols, getattr(model, "feature_importances_", np.zeros(len(feature_cols))))
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    test_signal_df = pred_test.rename("score").to_frame("score").sort_index()
    test_dt = pd.to_datetime(test_signal_df.index.get_level_values(0))
    summary = {
        "scan_time_utc": _now_utc(),
        "provider_uri": str(provider_uri),
        "market": str(args.market),
        "coverage": {
            "instrument_count": int(cov_df["instrument"].nunique()),
            "calendar_start": str(_read_calendar(provider_uri).min().date()),
            "calendar_end": str(_read_calendar(provider_uri).max().date()),
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "valid_start": VALID_START,
            "valid_end": VALID_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "coverage_csv": str(coverage_csv),
        },
        "model": {
            "name": "LightGBMRegressor",
            "params": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "random_state": int(args.random_state),
            },
            "fit_seconds": float(fit_sec),
            "feature_count": len(feature_cols),
            "top_features": feature_importance[:20],
        },
        "features": {
            "base_fields": list(BASE_FIELDS),
            "derived_feature_count": len(feature_cols),
            "derived_feature_names": feature_cols,
            "label_expr_local": "shift(-2)/shift(-1)-1 on close (per instrument)",
        },
        "splits": {
            "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": int(len(train_df))},
            "valid": {"start": VALID_START, "end": VALID_END, "rows": int(len(valid_df))},
            "test": {"start": TEST_START, "end": TEST_END, "rows": int(len(test_df))},
        },
        "prediction": {
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "rows": int(len(test_signal_df)),
            "start": str(test_dt.min().date()) if len(test_signal_df) else "",
            "end": str(test_dt.max().date()) if len(test_signal_df) else "",
        },
        "backtest_costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "metrics": {
            "daily_rank_ic": {
                "train_2020_2022": float(ic_train),
                "valid_2023": float(ic_valid),
                "test_2024_2026": float(ic_test),
            },
            "valid_best_non_test": {
                "topk": int(best_valid.topk),
                "n_drop": int(best_valid.n_drop),
                "costed_ir": float(best_valid.ir),
                "costed_annret": float(best_valid.annret),
                "max_drawdown": float(best_valid.max_drawdown),
                "turnover": float(best_valid.turnover),
                "error": best_valid.error,
            },
            "test_with_valid_selection": {
                "topk": int(test_metric.topk),
                "n_drop": int(test_metric.n_drop),
                "costed_ir": float(test_metric.ir),
                "costed_annret": float(test_metric.annret),
                "max_drawdown": float(test_metric.max_drawdown),
                "turnover": float(test_metric.turnover),
                "error": test_metric.error,
            },
            "metrics_csv": str(metrics_csv),
        },
        "hard_gate": {
            "rule": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET, "scope": "non-test(valid_2023)"},
            "passed": pass_hard_gate,
        },
        "runtime_sec_total": float(time.perf_counter() - t0_all),
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "metrics_csv": str(metrics_csv),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Long History Retrain Summary ({stamp})",
        "",
        f"- provider_uri: `{provider_uri}`",
        f"- market: `{args.market}`",
        f"- train/valid/test: `{TRAIN_START}..{TRAIN_END}` / `{VALID_START}..{VALID_END}` / `{TEST_START}..{TEST_END}`",
        f"- backtest costs: `open={args.open_cost}` `close={args.close_cost}`",
        f"- hard gate (non-test valid): `IR > {HARD_GATE_IR}` and `AnnRet > {HARD_GATE_ANNRET}` => `{'PASS' if pass_hard_gate else 'FAIL'}`",
        "",
        "## Best Non-Test (Valid 2023)",
        "",
        f"- topk/n_drop: `{best_valid.topk}/{best_valid.n_drop}`",
        f"- IR: `{_safe_float(best_valid.ir):.6f}`",
        f"- AnnRet: `{_safe_float(best_valid.annret):.6f}`",
        f"- MaxDD: `{_safe_float(best_valid.max_drawdown):.6f}`",
        f"- Turnover: `{_safe_float(best_valid.turnover):.6f}`",
        "",
        "## Test (2024-01-01..2026-04-30)",
        "",
        f"- topk/n_drop: `{test_metric.topk}/{test_metric.n_drop}`",
        f"- IR: `{_safe_float(test_metric.ir):.6f}`",
        f"- AnnRet: `{_safe_float(test_metric.annret):.6f}`",
        f"- MaxDD: `{_safe_float(test_metric.max_drawdown):.6f}`",
        f"- Turnover: `{_safe_float(test_metric.turnover):.6f}`",
        "",
        "## Coverage",
        "",
        f"- instruments: `{int(cov_df['instrument'].nunique())}`",
        f"- calendar range: `{summary['coverage']['calendar_start']}..{summary['coverage']['calendar_end']}`",
        f"- coverage csv: `{coverage_csv}`",
    ]
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    smoke = {
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "coverage_csv_rows": int(len(pd.read_csv(coverage_csv))) if coverage_csv.exists() else 0,
        "metrics_csv_rows": int(len(pd.read_csv(metrics_csv))) if metrics_csv.exists() else 0,
        "candidate_pred_rows": int(len(_load_pickle(pred_pkl))) if pred_pkl.exists() else 0,
        "hard_gate_passed": bool(pass_hard_gate),
    }
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

