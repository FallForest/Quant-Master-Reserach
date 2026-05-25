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
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import quant_master
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
    objective_mode: str
    alpha: float
    train_rank_ic: float
    train_rank_ic_ir: float
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
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        inst = parts[0].lower()
        out.setdefault(inst, []).append((pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
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


def _build_long_history_panel(
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
        year_set = set(int(x.year) for x in idx)
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
                "has_2020": int(2020 in year_set),
                "has_2021": int(2021 in year_set),
                "has_2022": int(2022 in year_set),
                "has_2023": int(2023 in year_set),
                "has_2024": int(2024 in year_set),
                "has_2025": int(2025 in year_set),
                "has_2026": int(2026 in year_set),
                "missing_fields": ";".join(missing),
            }
        )
    if not frames:
        raise RuntimeError("no rows constructed from local feature bins")
    panel = pd.concat(frames, axis=0)
    panel.index.name = "datetime"
    panel = panel.reset_index().set_index(["datetime", "instrument"]).sort_index()
    return panel, pd.DataFrame(coverage).sort_values("instrument").reset_index(drop=True)


def _build_features_and_label(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    p = panel.copy()
    for c in BASE_FIELDS:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)

    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = p["close"] * factor
    open_ = p["open"] * factor
    high = p["high"] * factor
    low = p["low"] * factor
    vwap = p["vwap"] * factor
    g = close.groupby(level=1, sort=False)

    out = pd.DataFrame(index=p.index)
    out["ret1"] = g.pct_change(1, fill_method=None)
    out["ret2"] = g.pct_change(2, fill_method=None)
    out["ret5"] = g.pct_change(5, fill_method=None)
    out["ret10"] = g.pct_change(10, fill_method=None)
    out["ret20"] = g.pct_change(20, fill_method=None)
    out["ret60"] = g.pct_change(60, fill_method=None)
    out["mom120"] = g.pct_change(120, fill_method=None)
    out["rev5"] = -out["ret5"]
    out["rev20"] = -out["ret20"]
    out["intraday"] = close / (open_ + 1e-12) - 1.0
    out["overnight"] = open_ / (g.shift(1) + 1e-12) - 1.0
    out["hl_range"] = (high - low) / (close.abs() + 1e-12)
    out["vwap_gap"] = close / (vwap + 1e-12) - 1.0
    out["log_volume"] = np.log1p(np.clip(p["volume"], 0.0, None))
    out["log_amount"] = np.log1p(np.clip(p["amount"], 0.0, None))
    out["vol_chg1"] = p["volume"].groupby(level=1, sort=False).pct_change(1, fill_method=None)
    out["vol_chg5"] = p["volume"].groupby(level=1, sort=False).pct_change(5, fill_method=None)
    out["amt_chg5"] = p["amount"].groupby(level=1, sort=False).pct_change(5, fill_method=None)
    out["amt_chg20"] = p["amount"].groupby(level=1, sort=False).pct_change(20, fill_method=None)
    out["ma_gap5"] = close / (g.rolling(5, min_periods=2).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0
    out["ma_gap20"] = close / (g.rolling(20, min_periods=5).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0
    out["ma_gap60"] = close / (g.rolling(60, min_periods=15).mean().reset_index(level=0, drop=True) + 1e-12) - 1.0
    ret1g = out["ret1"].groupby(level=1, sort=False)
    out["vol5"] = ret1g.rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    out["vol20"] = ret1g.rolling(20, min_periods=8).std().reset_index(level=0, drop=True)
    out["vol60"] = ret1g.rolling(60, min_periods=20).std().reset_index(level=0, drop=True)
    out["price_pos20"] = (close - g.rolling(20, min_periods=5).min().reset_index(level=0, drop=True)) / (
        g.rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
        - g.rolling(20, min_periods=5).min().reset_index(level=0, drop=True)
        + 1e-12
    )
    out["dd120"] = close / (g.rolling(120, min_periods=30).max().reset_index(level=0, drop=True) + 1e-12) - 1.0
    out["change"] = p["change"]

    feature_cols = [c for c in out.columns]
    for c in feature_cols:
        out[c] = _cs_z(out[c], clip=8.0)
    out["label"] = g.shift(-2) / (g.shift(-1) + 1e-12) - 1.0
    return out.replace([np.inf, -np.inf], np.nan), feature_cols


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _cs_rank(s: pd.Series, center: bool = False) -> pd.Series:
    r = s.groupby(level=0, sort=False).rank(method="average", pct=True)
    if center:
        return (2.0 * r - 1.0).fillna(0.0)
    return r.fillna(0.0)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    yz = np.nan_to_num(y - np.nanmean(y), nan=0.0, posinf=0.0, neginf=0.0)
    xtx = xz.T @ xz
    coef = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=np.float64) * float(alpha), xz.T @ yz)
    return coef.astype(np.float64), mu.astype(np.float64), sd.astype(np.float64)


def _predict_ridge(x: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return xz @ coef


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
        return BacktestMetric(
            split=split_name,
            candidate_id=candidate_id,
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
            candidate_id=candidate_id,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=float("nan"),
            ir=float("nan"),
            max_drawdown=float("nan"),
            turnover=float("nan"),
            elapsed_sec=float(time.perf_counter() - t0),
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_targets(raw_panel: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    p = raw_panel.copy()
    for c in BASE_FIELDS:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)

    close = p["close"] * p["factor"].replace(0.0, np.nan).fillna(1.0)
    ret1 = close.groupby(level=1, sort=False).pct_change(1, fill_method=None)
    vol20 = ret1.groupby(level=1, sort=False).rolling(20, min_periods=8).std().reset_index(level=0, drop=True)
    vol60 = ret1.groupby(level=1, sort=False).rolling(60, min_periods=20).std().reset_index(level=0, drop=True)

    label = pd.to_numeric(feat_df["label"], errors="coerce")
    out = pd.DataFrame(index=feat_df.index)
    out["label_raw"] = label
    out["label_rank"] = 2.0 * _cs_rank(label, center=False) - 1.0
    out["label_zscore"] = _cs_z(label)
    out["label_volnorm"] = label / (vol20.reindex(feat_df.index).astype(float) + 1e-12)
    out["label_volnorm_zscore"] = _cs_z(out["label_volnorm"])
    out["ret1_vol20"] = vol20.reindex(feat_df.index).astype(float)
    out["ret1_vol60"] = vol60.reindex(feat_df.index).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _objective_target(df: pd.DataFrame, objective_mode: str) -> pd.Series:
    if objective_mode == "rank":
        return df["label_rank"].astype(float)
    if objective_mode == "zscore":
        return df["label_zscore"].astype(float)
    if objective_mode == "volnorm":
        return df["label_volnorm"].astype(float)
    raise ValueError(f"unsupported objective_mode={objective_mode}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact pre-2024 ridge/rank model using long-history utilities.")
    p.add_argument("--provider-uri", default=".qmData/cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default=str(
            THIS_DIR / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
        ),
    )
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--alpha-grid", default="0.1,1,10,100,1000,10000")
    p.add_argument("--objective-grid", default="rank,zscore,volnorm")
    p.add_argument("--topk-grid", default="40,45")
    p.add_argument("--ndrop-grid", default="3,4")
    p.add_argument("--preselect", type=int, default=3)
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="compact_linear_rank_pre2024")
    return p


def _feature_coverage_rows(feat_df: pd.DataFrame, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for col in [c for c in feat_df.columns if c != "label"]:
        rows.append(
            {
                "feature": col,
                "train_nonnull": int(train_df[col].notna().sum()),
                "valid_nonnull": int(valid_df[col].notna().sum()),
                "test_nonnull": int(test_df[col].notna().sum()),
                "train_mean": float(pd.to_numeric(train_df[col], errors="coerce").mean()),
                "valid_mean": float(pd.to_numeric(valid_df[col], errors="coerce").mean()),
                "test_mean": float(pd.to_numeric(test_df[col], errors="coerce").mean()),
            }
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    provider_uri = Path(args.provider_uri).expanduser().resolve()

    output_dir = THIS_DIR
    coverage_csv = output_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    feature_cov_csv = output_dir / f"{args.output_prefix}_feature_coverage_{stamp}.csv"
    candidates_csv = output_dir / f"{args.output_prefix}_candidates_{stamp}.csv"
    selection_csv = output_dir / f"{args.output_prefix}_validation_selection_{stamp}.csv"
    split_metrics_csv = output_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    pred_pkl = output_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = output_dir / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    summary_json = output_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = output_dir / f"{args.output_prefix}_summary_{stamp}.md"
    smoke_json = output_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "status": "started",
        "blocker": "",
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "feature_coverage_csv": str(feature_cov_csv),
            "candidates_csv": str(candidates_csv),
            "validation_selection_csv": str(selection_csv),
            "split_metrics_csv": str(split_metrics_csv),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
    }

    try:
        quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = _load_config(Path(args.workflow_config).expanduser().resolve())
        port_cfg = _extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))

        raw_panel, coverage_df = _build_long_history_panel(
            provider_uri=provider_uri,
            market=str(args.market),
            raw_start=RAW_START,
            end_date=TEST_END,
            fields=BASE_FIELDS,
        )
        feat_df, feature_cols = _build_features_and_label(raw_panel)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        targets = _build_targets(raw_panel, feat_df)
        dataset = feat_df.join(targets)
        dataset = dataset.dropna(subset=["label_raw"])
        day_counts = dataset.groupby(level=0)["label_raw"].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()
        if dataset.empty:
            raise RuntimeError("no rows left after minimum daily universe filter")

        train_mask = _mask(dataset.index.get_level_values(0), TRAIN_START, TRAIN_END)
        valid_mask = _mask(dataset.index.get_level_values(0), VALID_START, VALID_END)
        test_mask = _mask(dataset.index.get_level_values(0), TEST_START, TEST_END)
        train_df = dataset.loc[train_mask].copy()
        valid_df = dataset.loc[valid_mask].copy()
        test_df = dataset.loc[test_mask].copy()
        if train_df.empty or valid_df.empty or test_df.empty:
            raise RuntimeError(
                f"empty split after filtering train={len(train_df)} valid={len(valid_df)} test={len(test_df)}"
            )

        alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
        objective_grid = [x.strip() for x in str(args.objective_grid).split(",") if x.strip()]
        topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
        ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
        combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk] or [(40, 2)]

        candidate_rows: List[Dict[str, Any]] = []
        predictions: Dict[str, Dict[str, pd.Series]] = {}
        x_train = train_df[feature_cols].astype(np.float64).values
        x_valid = valid_df[feature_cols].astype(np.float64).values
        x_test = test_df[feature_cols].astype(np.float64).values

        for objective_mode in objective_grid:
            y_train_all = _objective_target(train_df, objective_mode).astype(np.float64).values
            for alpha in alpha_grid:
                candidate_id = f"ridge_{objective_mode}_a{alpha:g}"
                fit_t0 = time.perf_counter()
                coef, mu, sd = _fit_ridge(x_train, y_train_all, alpha)
                fit_sec = float(time.perf_counter() - fit_t0)

                pred_train = _cs_rank(
                    pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"), center=True
                )
                pred_valid = _cs_rank(
                    pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"), center=True
                )
                pred_test = _cs_rank(
                    pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"), center=True
                )

                train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
                train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                predictions[candidate_id] = {"train": pred_train, "valid": pred_valid, "test": pred_test}
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "objective_mode": objective_mode,
                        "alpha": float(alpha),
                        "feature_count": len(feature_cols),
                        "fit_sec": fit_sec,
                        "train_rank_ic": train_ic,
                        "train_rank_ic_ir": train_ic_ir,
                        "valid_rank_ic": valid_ic,
                        "valid_rank_ic_ir": valid_ic_ir,
                    }
                )

        if not candidate_rows:
            raise RuntimeError("no ridge candidates were generated")

        preselected = sorted(
            candidate_rows,
            key=lambda r: (
                _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
                _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
            ),
            reverse=True,
        )[: max(1, int(args.preselect))]

        valid_bt_rows: List[Dict[str, Any]] = []
        exchange_cache: Dict[Tuple[str, str, int, int], Any] = {}
        for cand in preselected:
            signal_valid = predictions[str(cand["candidate_id"])]["valid"].rename("score").to_frame("score")
            for topk, n_drop in combos:
                metric = _run_bt(
                    signal_valid,
                    "valid_2023",
                    str(cand["candidate_id"]),
                    VALID_START,
                    VALID_END,
                    int(topk),
                    int(n_drop),
                    port_cfg,
                    benchmark,
                    float(args.open_cost),
                    float(args.close_cost),
                    exchange_cache,
                )
                valid_bt_rows.append(
                    {
                        "split": metric.split,
                        "candidate_id": metric.candidate_id,
                        "topk": metric.topk,
                        "n_drop": metric.n_drop,
                        "costed_ir": metric.ir,
                        "costed_annret": metric.annret,
                        "max_drawdown": metric.max_drawdown,
                        "turnover": metric.turnover,
                        "elapsed_sec": metric.elapsed_sec,
                        "error": metric.error,
                    }
                )

        ok_valid = [
            r
            for r in valid_bt_rows
            if not r["error"]
            and np.isfinite(_safe_float(r["costed_ir"]))
            and np.isfinite(_safe_float(r["costed_annret"]))
        ]
        if not ok_valid:
            raise RuntimeError("no valid 2023 backtest combo succeeded; cannot select a frozen portfolio")

        selected = sorted(
            ok_valid,
            key=lambda r: (_safe_float(r["costed_ir"]), _safe_float(r["costed_annret"])),
            reverse=True,
        )[0]
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])

        test_signal = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score")
        test_metric = _run_bt(
            test_signal,
            "test_2024_2026",
            str(selected["candidate_id"]),
            TEST_START,
            TEST_END,
            int(selected["topk"]),
            int(selected["n_drop"]),
            port_cfg,
            benchmark,
            float(args.open_cost),
            float(args.close_cost),
            exchange_cache,
        )
        test_ic_s = _daily_rank_ic_series(predictions[str(selected["candidate_id"])]["test"], test_df["label_raw"])
        test_rank_ic, test_rank_ic_ir = _mean_and_ir(test_ic_s)

        hard_gate_pass = bool(
            np.isfinite(_safe_float(selected["costed_ir"]))
            and np.isfinite(_safe_float(selected["costed_annret"]))
            and _safe_float(selected["costed_ir"]) > HARD_GATE_IR
            and _safe_float(selected["costed_annret"]) > HARD_GATE_ANNRET
        )

        coverage_df.to_csv(coverage_csv, index=False)
        pd.DataFrame(_feature_coverage_rows(feat_df, train_df, valid_df, test_df)).to_csv(feature_cov_csv, index=False)
        _write_csv(candidates_csv, candidate_rows)
        _write_csv(selection_csv, valid_bt_rows)

        split_rows = [
            {
                "split": "train_2020_2022",
                "candidate_id": selected["candidate_id"],
                "objective_mode": selected_candidate["objective_mode"],
                "alpha": selected_candidate["alpha"],
                "rank_ic": selected_candidate["train_rank_ic"],
                "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
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
                "candidate_id": selected["candidate_id"],
                "objective_mode": selected_candidate["objective_mode"],
                "alpha": selected_candidate["alpha"],
                "rank_ic": selected_candidate["valid_rank_ic"],
                "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                "costed_ir": selected["costed_ir"],
                "costed_annret": selected["costed_annret"],
                "max_drawdown": selected["max_drawdown"],
                "turnover": selected["turnover"],
                "topk": selected["topk"],
                "n_drop": selected["n_drop"],
                "error": selected["error"],
            },
            {
                "split": "test_2024_2026",
                "candidate_id": selected["candidate_id"],
                "objective_mode": selected_candidate["objective_mode"],
                "alpha": selected_candidate["alpha"],
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

        pred_test = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
        with pred_pkl.open("wb") as f:
            pickle.dump(pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_test.reset_index().to_csv(pred_csv, index=False)

        summary.update(
            {
                "status": "ok",
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "training_protocol": {
                    "raw_start": RAW_START,
                    "train": {"start": TRAIN_START, "end": TRAIN_END},
                    "valid": {"start": VALID_START, "end": VALID_END},
                    "test": {"start": TEST_START, "end": TEST_END},
                    "objective_grid": objective_grid,
                    "alpha_grid": alpha_grid,
                    "selection_rule": "2023-only shortlist by daily rank IC IR, then 2023-only net-cost backtest selection",
                    "topk_grid": topk_grid,
                    "ndrop_grid": ndrop_grid,
                    "selected_pre_backtest_by": "2023 daily rank IC IR",
                    "preselected_candidates": [r["candidate_id"] for r in preselected],
                },
                "coverage": {
                    "instrument_count": int(coverage_df["instrument"].nunique()),
                    "calendar_start": str(pd.to_datetime(raw_panel.index.get_level_values(0)).min().date()),
                    "calendar_end": str(pd.to_datetime(raw_panel.index.get_level_values(0)).max().date()),
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "test_rows": int(len(test_df)),
                    "coverage_csv": str(coverage_csv),
                    "feature_coverage_csv": str(feature_cov_csv),
                },
                "selected_model": {
                    "candidate_id": selected["candidate_id"],
                    "objective_mode": selected_candidate["objective_mode"],
                    "alpha": selected_candidate["alpha"],
                    "feature_count": selected_candidate["feature_count"],
                    "topk": selected["topk"],
                    "n_drop": selected["n_drop"],
                },
                "metrics": {
                    "train_2020_2022": {
                        "rank_ic": selected_candidate["train_rank_ic"],
                        "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
                    },
                    "valid_2023": {
                        "rank_ic": selected_candidate["valid_rank_ic"],
                        "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                        "costed_ir": selected["costed_ir"],
                        "costed_annret": selected["costed_annret"],
                        "max_drawdown": selected["max_drawdown"],
                        "turnover": selected["turnover"],
                        "error": selected["error"],
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
                    "rule": {"scope": "valid_2023_non_test_selection", "ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
                    "passed": hard_gate_pass,
                },
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )

        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_md.write_text(
            "\n".join(
                [
                    f"# Compact Linear Rank Pre-2024 ({stamp})",
                    "",
                    f"- hard_gate_pass: `{hard_gate_pass}`",
                    f"- selected: `{selected['candidate_id']}` topk/n_drop `{selected['topk']}/{selected['n_drop']}`",
                    f"- objective/alpha: `{selected_candidate['objective_mode']}` / `{selected_candidate['alpha']}`",
                    f"- valid_2023 IR/AnnRet: `{_safe_float(selected['costed_ir']):.6f}` / `{_safe_float(selected['costed_annret']):.6f}`",
                    f"- test_2024_2026 IR/AnnRet: `{_safe_float(test_metric.ir):.6f}` / `{_safe_float(test_metric.annret):.6f}`",
                    f"- costs: open `{args.open_cost}` close `{args.close_cost}`",
                    f"- protocol: train `{TRAIN_START}..{TRAIN_END}`, select `{VALID_START}..{VALID_END}`, test once `{TEST_START}..{TEST_END}`",
                    f"- artifacts: `{summary_json}`",
                ]
            ),
            encoding="utf-8",
        )
        smoke = {
            "summary_json_exists": summary_json.exists(),
            "summary_md_exists": summary_md.exists(),
            "coverage_csv_rows": int(len(pd.read_csv(coverage_csv))) if coverage_csv.exists() else 0,
            "feature_coverage_csv_rows": int(len(pd.read_csv(feature_cov_csv))) if feature_cov_csv.exists() else 0,
            "candidates_csv_rows": int(len(pd.read_csv(candidates_csv))) if candidates_csv.exists() else 0,
            "validation_selection_csv_rows": int(len(pd.read_csv(selection_csv))) if selection_csv.exists() else 0,
            "split_metrics_csv_rows": int(len(pd.read_csv(split_metrics_csv))) if split_metrics_csv.exists() else 0,
            "candidate_pred_rows": int(len(_load_pickle(pred_pkl))) if pred_pkl.exists() else 0,
            "hard_gate_passed": hard_gate_pass,
        }
        smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_md.write_text(
            "\n".join(
                [
                    f"# Compact Linear Rank Pre-2024 ({stamp})",
                    "",
                    f"- status: `failed`",
                    f"- blocker: `{summary['blocker']}`",
                    f"- artifacts: `{summary_json}`",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
