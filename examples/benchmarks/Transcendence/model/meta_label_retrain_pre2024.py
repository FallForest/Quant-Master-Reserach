#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import quant_master
from quant_master.config import resolve_provider_uri
import pre2024_train_new_model_lockstep as base
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
HARD_GATE_ROWS = 562


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _as_score_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, pd.Series):
        df = obj.rename("score").to_frame()
    else:
        raise TypeError(f"unsupported score object type {type(obj)}")
    if "score" not in df.columns:
        df = df.iloc[:, [0]].copy()
        df.columns = ["score"]
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("score frame index is not MultiIndex(datetime, instrument)")
    df = df[["score"]].copy()
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)), df.index.get_level_values(1).astype(str)],
        names=["datetime", "instrument"],
    )
    return df.sort_index()


def _probe_prediction_artifacts(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        row: Dict[str, Any] = {
            "path": str(path),
            "exists": bool(path.exists()),
            "usable_pre2024": False,
            "rows": 0,
            "date_min": "",
            "date_max": "",
            "n_dates": 0,
            "reason": "",
        }
        if not path.exists():
            row["reason"] = "missing"
            rows.append(row)
            continue
        try:
            df = _as_score_df(_load_pickle(path))
            dates = pd.to_datetime(df.index.get_level_values(0))
            row.update(
                {
                    "rows": int(len(df)),
                    "date_min": str(dates.min().date()),
                    "date_max": str(dates.max().date()),
                    "n_dates": int(dates.nunique()),
                    "usable_pre2024": bool((dates < pd.Timestamp("2024-01-01")).any()),
                }
            )
            if not row["usable_pre2024"]:
                row["reason"] = "no rows before 2024-01-01"
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0, sort=False).rank(method="average", pct=True)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    steps: int,
    lr: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    yv = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    w = np.zeros(xz.shape[1], dtype=np.float64)
    b = float(np.log((yv.mean() + 1e-6) / (1.0 - yv.mean() + 1e-6)))
    n = max(1, len(yv))
    reg = float(alpha) / n
    for _ in range(int(steps)):
        p = _sigmoid(xz @ w + b)
        err = p - yv
        w -= float(lr) * ((xz.T @ err) / n + reg * w)
        b -= float(lr) * float(err.mean())
    return w, b, mu.astype(np.float64), sd.astype(np.float64)


def _predict_logistic(x: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return _sigmoid(xz @ w + float(b))


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


def _report_metrics(report: pd.DataFrame) -> Dict[str, Any]:
    needed = ["return", "bench", "cost", "turnover"]
    missing = [c for c in needed if c not in report.columns]
    if missing:
        raise ValueError(f"report missing columns {missing}")
    finite_mask = np.isfinite(report[needed].astype(float)).all(axis=1)
    finite_report = report.loc[finite_mask].copy()
    excess = finite_report["return"].astype(float) - finite_report["bench"].astype(float) - finite_report["cost"].astype(float)
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(finite_report["turnover"].astype(float).mean()),
        "report_rows": int(len(report)),
        "finite_report_rows": int(len(finite_report)),
        "first_report_date": str(pd.to_datetime(report.index).min().date()) if len(report) else "",
        "last_report_date": str(pd.to_datetime(report.index).max().date()) if len(report) else "",
    }


def _run_bt_report(
    signal_df: pd.DataFrame,
    start_time: str,
    end_time: str,
    topk: int,
    n_drop: int,
    port_cfg_template: Dict[str, Any],
    benchmark: str,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, int, int, float, float], Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.perf_counter()
    cfg = copy.deepcopy(port_cfg_template)
    bcfg = cfg["backtest"]
    bcfg["start_time"] = start_time
    bcfg["end_time"] = end_time
    exch_cfg = dict(bcfg.get("exchange_kwargs", {}))
    cache_key = (start_time, end_time, int(topk), int(n_drop), float(open_cost), float(close_cost))
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
    report = base._get_report_for_day_freq(pm).copy()
    metrics = _report_metrics(report)
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    return report, metrics


def _add_meta_features(dataset: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = dataset.copy()
    rank_cols = [c for c in feature_cols if c.endswith("__rank")]
    z_cols = [c for c in feature_cols if c.endswith("__z")]
    df["base_linear_rank_mean"] = df[rank_cols].mean(axis=1)
    df["base_linear_z_mean"] = df[z_cols].mean(axis=1)
    df["base_linear_rank_std"] = df[rank_cols].std(axis=1).fillna(0.0)
    df["quality_low_vol"] = -df.get("vol20__rank", pd.Series(0.0, index=df.index))
    df["quality_liq"] = df.get("liq_amount_z20__rank", pd.Series(0.0, index=df.index))
    df["reversal_pack"] = df[[c for c in ["rev_1__rank", "rev_5__rank", "gap_reversal__rank"] if c in df.columns]].mean(axis=1)
    df["momentum_pack"] = df[[c for c in ["mom_20__rank", "mom_60__rank", "mom_120__rank"] if c in df.columns]].mean(axis=1)
    df["risk_pack"] = df[[c for c in ["vol20__rank", "hl_range__rank", "vol_break_5x20__rank"] if c in df.columns]].mean(axis=1)
    df["flow_pack"] = df[[c for c in ["vp_cov_proxy_20__rank", "vp_lagcov_proxy_20__rank", "liq_amount_shock_5__rank"] if c in df.columns]].mean(axis=1)
    pack_cols = [
        "base_linear_rank_mean",
        "base_linear_z_mean",
        "base_linear_rank_std",
        "quality_low_vol",
        "quality_liq",
        "reversal_pack",
        "momentum_pack",
        "risk_pack",
        "flow_pack",
    ]
    for col in pack_cols:
        df[f"{col}__csrank"] = (_cs_rank_pct(pd.to_numeric(df[col], errors="coerce")).fillna(0.5) - 0.5) * 2.0
        df[f"{col}__csz"] = _cs_z(pd.to_numeric(df[col], errors="coerce"))
    meta_cols = [f"{c}__csrank" for c in pack_cols] + [f"{c}__csz" for c in pack_cols]
    return df.replace([np.inf, -np.inf], np.nan), list(meta_cols)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict pre-2024 meta-label retrain; select 2023, test 2024-2026 once.")
    p.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default=str(THIS_DIR / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"),
    )
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--alpha-grid", default="0.1,1,10,100,1000")
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    p.add_argument("--preselect", type=int, default=8)
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--logistic-steps", type=int, default=220)
    p.add_argument("--logistic-lr", type=float, default=0.25)
    p.add_argument("--output-prefix", default="meta_label_retrain_pre2024")
    p.add_argument("--verify", action="store_true", help="Re-run the selected 2024-2026 backtest after a hard-gate pass.")
    p.add_argument("--smoke", action="store_true")
    return p


def _paths(prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "pred_probe_csv": THIS_DIR / f"{prefix}_pred_probe_{stamp}.csv",
        "coverage_csv": THIS_DIR / f"{prefix}_coverage_{stamp}.csv",
        "candidates_csv": THIS_DIR / f"{prefix}_candidates_{stamp}.csv",
        "validation_selection_csv": THIS_DIR / f"{prefix}_validation_selection_{stamp}.csv",
        "split_metrics_csv": THIS_DIR / f"{prefix}_split_metrics_{stamp}.csv",
        "daily_report_csv": THIS_DIR / f"{prefix}_daily_report_{stamp}.csv",
        "candidate_pred_pkl": THIS_DIR / f"{prefix}_candidate_pred_{stamp}.pkl",
        "candidate_pred_csv": THIS_DIR / f"{prefix}_candidate_pred_{stamp}.csv",
        "summary_json": THIS_DIR / f"{prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{prefix}_summary_{stamp}.md",
        "artifact_parse_smoke_json": THIS_DIR / f"{prefix}_artifact_parse_smoke_{stamp}.json",
    }


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _paths(str(args.output_prefix), stamp)
    summary: Dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "new lightweight pre-2024 meta-label candidate from historical provider features",
        "status": "started",
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    try:
        provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
        pred_probe_paths = [
            REPO_ROOT / "mlruns/984329077332834218/7406e47063e9479cb34d300b9ed03bad/artifacts/pred.pkl",
            REPO_ROOT / "mlruns/984329077332834218/587bba6200be43e68cf02f59d1b7f890/artifacts/pred.pkl",
            REPO_ROOT / "mlruns/984329077332834218/5003b27b4683400e8d67a3209433cc24/artifacts/pred.pkl",
            REPO_ROOT / "mlruns/663421623398938249/d4526da7854245af954fc99cf02963f0/artifacts/pred.pkl",
        ]
        pred_probe_rows = _probe_prediction_artifacts(pred_probe_paths)
        _write_csv(paths["pred_probe_csv"], pred_probe_rows)

        quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = base._load_config(Path(args.workflow_config).expanduser().resolve())
        port_cfg = base._extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))
        panel_raw, coverage_df = base._build_panel(provider_uri, str(args.market), RAW_START, TEST_END, base.BASE_FIELDS)
        coverage_df.to_csv(paths["coverage_csv"], index=False)
        dataset, feature_cols = base._build_features_and_targets(panel_raw)
        dataset, meta_cols = _add_meta_features(dataset, feature_cols)
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

        feature_sets = {
            "meta_pack": meta_cols,
            "meta_plus_core": meta_cols
            + [
                c
                for c in [
                    "rev_1__rank",
                    "rev_5__rank",
                    "mom_20__rank",
                    "mom_60__rank",
                    "vol20__rank",
                    "liq_amount_z20__rank",
                    "vp_cov_proxy_20__rank",
                    "price_pos20__rank",
                    "dd60__rank",
                ]
                if c in dataset.columns
            ],
        }
        target_defs = {
            "top40_binary": (_cs_rank_pct(train_df["label_raw"]) >= 0.60).astype(float),
            "top35_binary": (_cs_rank_pct(train_df["label_raw"]) >= 0.65).astype(float),
            "rank_regression": train_df["label_rank"].astype(float),
            "volnorm_regression": train_df["label_volnorm_rank"].astype(float),
        }
        candidate_rows: List[Dict[str, Any]] = []
        predictions: Dict[str, Dict[str, pd.Series]] = {}
        for feature_mode, cols in feature_sets.items():
            cols = list(dict.fromkeys(cols))
            x_train = train_df[cols].astype(np.float64).values
            x_valid = valid_df[cols].astype(np.float64).values
            x_test = test_df[cols].astype(np.float64).values
            for target_mode, y_train_s in target_defs.items():
                for alpha in alpha_grid:
                    cid = f"{'logit' if target_mode.endswith('binary') else 'ridge'}_{feature_mode}_{target_mode}_a{alpha:g}"
                    fit_t0 = time.perf_counter()
                    if target_mode.endswith("binary"):
                        w, b, mu, sd = _fit_logistic_ridge(
                            x_train,
                            y_train_s.astype(float).values,
                            alpha=alpha,
                            steps=int(args.logistic_steps),
                            lr=float(args.logistic_lr),
                        )
                        raw_train = _predict_logistic(x_train, w, b, mu, sd)
                        raw_valid = _predict_logistic(x_valid, w, b, mu, sd)
                        raw_test = _predict_logistic(x_test, w, b, mu, sd)
                    else:
                        coef, mu, sd = _fit_ridge(x_train, y_train_s.astype(float).values, alpha=alpha)
                        raw_train = _predict_ridge(x_train, coef, mu, sd)
                        raw_valid = _predict_ridge(x_valid, coef, mu, sd)
                        raw_test = _predict_ridge(x_test, coef, mu, sd)
                    pred_train = _cs_z(pd.Series(raw_train, index=train_df.index, name="score"))
                    pred_valid = _cs_z(pd.Series(raw_valid, index=valid_df.index, name="score"))
                    pred_test = _cs_z(pd.Series(raw_test, index=test_df.index, name="score"))
                    train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
                    valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
                    train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
                    valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                    predictions[cid] = {"train": pred_train, "valid": pred_valid, "test": pred_test}
                    candidate_rows.append(
                        {
                            "candidate_id": cid,
                            "model_family": "logistic_meta_label" if target_mode.endswith("binary") else "ridge_meta_rank",
                            "feature_mode": feature_mode,
                            "target_mode": target_mode,
                            "alpha": float(alpha),
                            "feature_count": int(len(cols)),
                            "fit_sec": float(time.perf_counter() - fit_t0),
                            "train_rank_ic": train_ic,
                            "train_rank_ic_ir": train_ic_ir,
                            "valid_rank_ic": valid_ic,
                            "valid_rank_ic_ir": valid_ic_ir,
                        }
                    )

        _write_csv(paths["candidates_csv"], candidate_rows)
        preselected = sorted(
            candidate_rows,
            key=lambda r: (
                _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
                _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
            ),
            reverse=True,
        )[: max(1, int(args.preselect))]

        exchange_cache: Dict[Tuple[str, str, int, int, float, float], Any] = {}
        combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk] or [(40, 2)]
        valid_rows: List[Dict[str, Any]] = []
        for cand in preselected:
            sig = predictions[str(cand["candidate_id"])]["valid"].rename("score").to_frame("score")
            for topk, n_drop in combos:
                row = {
                    "split": "valid_2023",
                    "candidate_id": cand["candidate_id"],
                    "topk": int(topk),
                    "n_drop": int(n_drop),
                    "error": "",
                }
                try:
                    _, metrics = _run_bt_report(
                        sig,
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
                    row.update({f"costed_{k}" if k in {"annret", "ir"} else k: v for k, v in metrics.items()})
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                valid_rows.append(row)
        _write_csv(paths["validation_selection_csv"], valid_rows)
        ok_valid = [r for r in valid_rows if not r.get("error") and np.isfinite(_safe_float(r.get("costed_ir")))]
        if not ok_valid:
            raise RuntimeError("no successful 2023 validation backtests")
        selected = sorted(ok_valid, key=lambda r: (_safe_float(r["costed_ir"]), _safe_float(r["costed_annret"])), reverse=True)[0]
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])

        test_signal = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
        test_report, test_metrics = _run_bt_report(
            test_signal,
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
        test_rank_ic, test_rank_ic_ir = _mean_and_ir(_daily_rank_ic_series(test_signal["score"], test_df["label_raw"]))
        test_metrics["rank_ic"] = test_rank_ic
        test_metrics["rank_ic_ir"] = test_rank_ic_ir
        test_report.to_csv(paths["daily_report_csv"])
        with paths["candidate_pred_pkl"].open("wb") as f:
            pickle.dump(test_signal, f, protocol=pickle.HIGHEST_PROTOCOL)
        test_signal.reset_index().to_csv(paths["candidate_pred_csv"], index=False)

        verification: Dict[str, Any] = {"requested": bool(args.verify), "ran": False}
        hard_gate_pass = bool(
            int(test_metrics["finite_report_rows"]) == HARD_GATE_ROWS
            and _safe_float(test_metrics["ir"]) > HARD_GATE_IR
            and _safe_float(test_metrics["annret"]) > HARD_GATE_ANNRET
        )
        if hard_gate_pass and args.verify:
            verify_report, verify_metrics = _run_bt_report(
                test_signal,
                TEST_START,
                TEST_END,
                int(selected["topk"]),
                int(selected["n_drop"]),
                port_cfg,
                benchmark,
                float(args.open_cost),
                float(args.close_cost),
                {},
            )
            verification = {
                "requested": True,
                "ran": True,
                "finite_report_rows": int(verify_metrics["finite_report_rows"]),
                "costed_ir": float(verify_metrics["ir"]),
                "costed_annret": float(verify_metrics["annret"]),
                "matches_initial": bool(
                    int(verify_metrics["finite_report_rows"]) == int(test_metrics["finite_report_rows"])
                    and abs(float(verify_metrics["ir"]) - float(test_metrics["ir"])) < 1e-12
                    and abs(float(verify_metrics["annret"]) - float(test_metrics["annret"])) < 1e-12
                    and len(verify_report) == len(test_report)
                ),
            }

        split_rows = [
            {
                "split": "train_2020_2022",
                "candidate_id": selected["candidate_id"],
                "rank_ic": selected_candidate["train_rank_ic"],
                "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
                "rows": int(len(train_df)),
            },
            {
                "split": "valid_2023_selection_only",
                "candidate_id": selected["candidate_id"],
                "rank_ic": selected_candidate["valid_rank_ic"],
                "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                "costed_ir": selected["costed_ir"],
                "costed_annret": selected["costed_annret"],
                "finite_report_rows": selected.get("finite_report_rows"),
                "topk": selected["topk"],
                "n_drop": selected["n_drop"],
            },
            {
                "split": "test_2024_2026_hard_gate",
                "candidate_id": selected["candidate_id"],
                "rank_ic": test_metrics["rank_ic"],
                "rank_ic_ir": test_metrics["rank_ic_ir"],
                "costed_ir": test_metrics["ir"],
                "costed_annret": test_metrics["annret"],
                "max_drawdown": test_metrics["max_drawdown"],
                "turnover": test_metrics["turnover"],
                "report_rows": test_metrics["report_rows"],
                "finite_report_rows": test_metrics["finite_report_rows"],
                "topk": selected["topk"],
                "n_drop": selected["n_drop"],
            },
        ]
        _write_csv(paths["split_metrics_csv"], split_rows)

        summary.update(
            {
                "status": "ok",
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "historical_data_evidence": {
                    "provider_features": "local QuantMaster CN data store OHLCV/factor bins loaded via pre2024_train_new_model_lockstep._build_panel",
                    "pred_probe_csv": str(paths["pred_probe_csv"]),
                    "pre2024_prediction_features_available": any(bool(r["usable_pre2024"]) for r in pred_probe_rows),
                    "base_prediction_feature_decision": "not used because inspected mlruns pred.pkl files had no pre-2024 rows",
                },
                "training_protocol": {
                    "raw_start": RAW_START,
                    "train": [TRAIN_START, TRAIN_END],
                    "selection": [VALID_START, VALID_END],
                    "test_once": [TEST_START, TEST_END],
                    "selection_rule": "2023-only real net-cost backtest max IR, tie by AnnRet; no 2024-2026 tuning",
                    "preselection_rule": "2023 daily rank IC IR shortlist before 2023 backtests",
                    "feature_sets": {k: len(v) for k, v in feature_sets.items()},
                    "alpha_grid": alpha_grid,
                    "topk_grid": topk_grid,
                    "ndrop_grid": ndrop_grid,
                    "smoke": bool(args.smoke),
                },
                "splits": {
                    "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": int(len(train_df))},
                    "valid": {"start": VALID_START, "end": VALID_END, "rows": int(len(valid_df))},
                    "test": {"start": TEST_START, "end": TEST_END, "rows": int(len(test_df))},
                },
                "coverage": {"instrument_count": int(coverage_df["instrument"].nunique()), "coverage_csv": str(paths["coverage_csv"])},
                "selected_model": {
                    "candidate_id": selected["candidate_id"],
                    "model_family": selected_candidate["model_family"],
                    "feature_mode": selected_candidate["feature_mode"],
                    "target_mode": selected_candidate["target_mode"],
                    "alpha": selected_candidate["alpha"],
                    "feature_count": selected_candidate["feature_count"],
                    "topk": int(selected["topk"]),
                    "n_drop": int(selected["n_drop"]),
                },
                "metrics": {
                    "train_2020_2022": {
                        "rank_ic": selected_candidate["train_rank_ic"],
                        "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
                    },
                    "valid_2023_selection_only": {
                        "rank_ic": selected_candidate["valid_rank_ic"],
                        "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                        "costed_ir": selected["costed_ir"],
                        "costed_annret": selected["costed_annret"],
                        "finite_report_rows": selected.get("finite_report_rows"),
                    },
                    "test_2024_2026_hard_gate": {
                        "rank_ic": test_metrics["rank_ic"],
                        "rank_ic_ir": test_metrics["rank_ic_ir"],
                        "costed_ir": test_metrics["ir"],
                        "costed_annret": test_metrics["annret"],
                        "max_drawdown": test_metrics["max_drawdown"],
                        "turnover": test_metrics["turnover"],
                        "report_rows": test_metrics["report_rows"],
                        "finite_report_rows": test_metrics["finite_report_rows"],
                    },
                },
                "hard_gate": {
                    "rule": {
                        "scope": "2024-01-01..2026-04-30",
                        "finite_report_rows_eq": HARD_GATE_ROWS,
                        "ir_gt": HARD_GATE_IR,
                        "annret_gt": HARD_GATE_ANNRET,
                        "open_cost": float(args.open_cost),
                        "close_cost": float(args.close_cost),
                    },
                    "passed": hard_gate_pass,
                },
                "verification": verification,
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        paths["summary_json"].write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Meta Label Retrain Pre-2024 ({stamp})",
                    "",
                    f"- hard_gate_pass: `{hard_gate_pass}`",
                    f"- selected: `{selected['candidate_id']}` topk/n_drop `{selected['topk']}/{selected['n_drop']}`",
                    f"- valid_2023 selection IR/AnnRet: `{_safe_float(selected['costed_ir']):.6f}` / `{_safe_float(selected['costed_annret']):.6f}`",
                    f"- test_2024_2026 hard-gate IR/AnnRet: `{_safe_float(test_metrics['ir']):.6f}` / `{_safe_float(test_metrics['annret']):.6f}`",
                    f"- finite report rows: `{int(test_metrics['finite_report_rows'])}`",
                    f"- base pre-2024 preds usable: `{summary['historical_data_evidence']['pre2024_prediction_features_available']}`",
                    f"- artifacts: `{paths['summary_json']}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        smoke = {
            "summary_json_exists": paths["summary_json"].exists(),
            "summary_md_exists": paths["summary_md"].exists(),
            "pred_probe_csv_rows": int(len(pd.read_csv(paths["pred_probe_csv"]))),
            "coverage_csv_rows": int(len(pd.read_csv(paths["coverage_csv"]))),
            "candidates_csv_rows": int(len(pd.read_csv(paths["candidates_csv"]))),
            "validation_selection_csv_rows": int(len(pd.read_csv(paths["validation_selection_csv"]))),
            "split_metrics_csv_rows": int(len(pd.read_csv(paths["split_metrics_csv"]))),
            "daily_report_csv_rows": int(len(pd.read_csv(paths["daily_report_csv"]))),
            "candidate_pred_rows": int(len(_load_pickle(paths["candidate_pred_pkl"]))),
            "finite_report_rows": int(test_metrics["finite_report_rows"]),
            "hard_gate_passed": hard_gate_pass,
            "hard_gate_scope": "2024_01_01_to_2026_04_30_only",
        }
        paths["artifact_parse_smoke_json"].write_text(json.dumps(_json_safe(smoke), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(_json_safe(summary), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        paths["summary_json"].write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary_md"].write_text(
            "\n".join(["# Meta Label Retrain Pre-2024", "", "- status: `failed`", f"- blocker: `{summary['blocker']}`"])
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_json_safe(summary), ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

