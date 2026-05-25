#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass, asdict
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
EXPECTED_TEST_ROWS = 562
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_family: str
    feature_mode: str
    target_mode: str
    alpha: float
    train_rank_ic: float
    valid_rank_ic: float
    valid_rank_ic_ir: float
    note: str = ""


@dataclass(frozen=True)
class BtSpec:
    split: str
    candidate_id: str
    topk: int
    n_drop: int
    annret: float
    ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float
    row_count: int
    finite_rows: int
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
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0, sort=False).rank(method="average", pct=True)


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


def _load_config(path: Path) -> Dict[str, Any]:
    return base._load_config(path)


def _count_calendar_rows(provider_uri: Path, start: str, end: str) -> int:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = pd.to_datetime(vals)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return int(mask.sum())


def _build_tail_targets(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = dataset.copy()
    train_mask = _mask(out.index.get_level_values(0), TRAIN_START, TRAIN_END)
    train_raw = out.loc[train_mask, "label_raw"].astype(float)
    downside_floor = float(train_raw.quantile(0.10))
    tail_hit_threshold = float(max(0.0, train_raw.quantile(0.70)))

    out["label_downside_clip"] = out["label_raw"].astype(float).clip(lower=downside_floor)
    out["label_downside_rank"] = (_cs_rank_pct(out["label_downside_clip"]) - 0.5) * 2.0
    out["label_tail_hit"] = (out["label_raw"].astype(float) > tail_hit_threshold).astype(float)
    out["label_tail_margin"] = np.maximum(out["label_raw"].astype(float) - tail_hit_threshold, 0.0)
    out["label_tail_margin_rank"] = (_cs_rank_pct(out["label_tail_margin"]) - 0.5) * 2.0

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, {"downside_floor": downside_floor, "tail_hit_threshold": tail_hit_threshold}


def _make_predictions(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_modes: Dict[str, List[str]],
    target_modes: Sequence[str],
    alpha_grid: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    candidate_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}
    for feature_mode, cols in feature_modes.items():
        x_train = train_df[cols].astype(np.float64).values
        x_valid = valid_df[cols].astype(np.float64).values
        x_test = test_df[cols].astype(np.float64).values
        for target_mode in target_modes:
            y_train = train_df[target_mode].astype(np.float64).values
            for alpha in alpha_grid:
                candidate_id = f"ridge_{feature_mode}_{target_mode}_a{alpha:g}"
                fit_t0 = time.perf_counter()
                coef, mu, sd = _fit_ridge(x_train, y_train, float(alpha))
                fit_sec = float(time.perf_counter() - fit_t0)
                pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
                pred_valid = _cs_z(pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
                pred_test = _cs_z(pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"))
                train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
                train_ic, _ = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_family": "closed_form_ridge_tail_aware",
                        "feature_mode": feature_mode,
                        "target_mode": target_mode,
                        "alpha": float(alpha),
                        "feature_count": int(len(cols)),
                        "fit_sec": fit_sec,
                        "train_rank_ic": train_ic,
                        "valid_rank_ic": valid_ic,
                        "valid_rank_ic_ir": valid_ic_ir,
                    }
                )
                predictions[candidate_id] = {
                    "train": pred_train,
                    "valid": pred_valid,
                    "test": pred_test,
                }
    return candidate_rows, predictions


def _append_ensemble_candidate(
    candidate_rows: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, pd.Series]],
    base_candidates: Sequence[Dict[str, Any]],
    top_n: int = 3,
) -> None:
    if len(base_candidates) < top_n:
        return
    members = [str(r["candidate_id"]) for r in base_candidates[:top_n]]
    candidate_id = "ensemble_" + "_".join(members)
    if candidate_id in predictions:
        return
    preds: Dict[str, pd.Series] = {}
    for split in ("train", "valid", "test"):
        panel = pd.concat([predictions[m][split].rename(m) for m in members], axis=1)
        blend = panel.mean(axis=1)
        preds[split] = _cs_z(blend.rename("score"))
    predictions[candidate_id] = preds
    candidate_rows.append(
        {
            "candidate_id": candidate_id,
            "model_family": "ensemble_equal_weight",
            "feature_mode": "ensemble_top3",
            "target_mode": "mixed",
            "alpha": float("nan"),
            "feature_count": int(len(members)),
            "fit_sec": 0.0,
            "train_rank_ic": float("nan"),
            "valid_rank_ic": float("nan"),
            "valid_rank_ic_ir": float("nan"),
            "ensemble_members": ";".join(members),
        }
    )


def _update_ensemble_metrics(
    candidate_rows: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, pd.Series]],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> None:
    for row in candidate_rows:
        if row.get("model_family") != "ensemble_equal_weight":
            continue
        cid = str(row["candidate_id"])
        pred_train = predictions[cid]["train"]
        pred_valid = predictions[cid]["valid"]
        train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
        valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
        train_ic, _ = _mean_and_ir(train_ic_s)
        valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
        row["train_rank_ic"] = train_ic
        row["valid_rank_ic"] = valid_ic
        row["valid_rank_ic_ir"] = valid_ic_ir


def _run_backtest_with_report(
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
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[BtSpec, pd.DataFrame]:
    t0 = time.perf_counter()
    try:
        cfg = copy.deepcopy(port_cfg_template)
        bcfg = cfg["backtest"]
        bcfg["start_time"] = str(pd.Timestamp(start_time).date())
        bcfg["end_time"] = str(pd.Timestamp(end_time).date())
        executor_cfg = cfg.get(
            "executor",
            {
                "class": "SimulatorExecutor",
                "module_path": "quant_master.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
        )
        pred_slice = signal_df.loc[_mask(signal_df.index.get_level_values(0), start_time, end_time)].copy()
        if pred_slice.empty:
            raise ValueError(f"empty signal slice for {candidate_id}: {start_time}..{end_time}")

        exchange_kwargs = dict(bcfg.get("exchange_kwargs", {}))
        exchange_kwargs["open_cost"] = float(open_cost)
        exchange_kwargs["close_cost"] = float(close_cost)
        freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
        deal_price = str(exchange_kwargs.get("deal_price", "close"))
        limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
        min_cost = float(exchange_kwargs.get("min_cost", 5))
        cache_key = (
            str(bcfg["start_time"]),
            str(bcfg["end_time"]),
            float(open_cost),
            float(close_cost),
            limit_threshold,
            deal_price,
        )
        if cache_key not in exchange_cache:
            exchange_cache[cache_key] = get_exchange(
                freq=freq,
                start_time=bcfg["start_time"],
                end_time=bcfg["end_time"],
                deal_price=deal_price,
                limit_threshold=limit_threshold,
                open_cost=float(open_cost),
                close_cost=float(close_cost),
                min_cost=min_cost,
            )
        exchange_kwargs["exchange"] = exchange_cache[cache_key]

        strategy = TopkDropoutStrategy(
            signal=pred_slice,
            topk=int(topk),
            n_drop=int(n_drop),
            method_sell="bottom",
            method_buy="top",
            hold_thresh=1,
            only_tradable=False,
            forbid_all_trade_at_limit=True,
        )
        pm, _ = run_backtest(
            start_time=bcfg["start_time"],
            end_time=bcfg["end_time"],
            strategy=strategy,
            executor=executor_cfg,
            benchmark=benchmark,
            account=bcfg.get("account", 100000000),
            exchange_kwargs=exchange_kwargs,
            pos_type=bcfg.get("pos_type", "Position"),
        )
        report = base._get_report_for_day_freq(pm)
        report = report.sort_index()
        if report.empty:
            raise ValueError(f"empty report for {candidate_id}: {start_time}..{end_time}")
        excess = (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")
        risk_df = risk_analysis(excess, freq="1day")
        annret = float(risk_df.loc["annualized_return", "risk"])
        ir = float(risk_df.loc["information_ratio", "risk"])
        max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
        turnover = float(report["turnover"].astype(float).mean())
        finite_rows = int(np.isfinite(report[["return", "bench", "cost", "turnover"]].to_numpy(dtype=float)).all(axis=1).sum())
        metric = BtSpec(
            split=split_name,
            candidate_id=candidate_id,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=annret,
            ir=ir,
            max_drawdown=max_drawdown,
            turnover=turnover,
            elapsed_sec=float(time.perf_counter() - t0),
            row_count=int(len(report)),
            finite_rows=finite_rows,
            error="",
        )
        return metric, report
    except Exception as exc:  # noqa: BLE001
        metric = BtSpec(
            split=split_name,
            candidate_id=candidate_id,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=float("nan"),
            ir=float("nan"),
            max_drawdown=float("nan"),
            turnover=float("nan"),
            elapsed_sec=float(time.perf_counter() - t0),
            row_count=0,
            finite_rows=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return metric, pd.DataFrame()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tail-risk-aware pre-2024 model search with 2023-only selection.")
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
    p.add_argument("--alpha-grid", default="0.1,1,10,100")
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3")
    p.add_argument("--preselect", type=int, default=6)
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="tail_risk_aware_pre2024_model")
    p.add_argument("--smoke", action="store_true", help="Shrink candidate grid for a quick plumbing check.")
    return p


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    names = {
        "coverage_csv": "coverage",
        "candidates_csv": "candidates",
        "valid_backtests_csv": "valid_backtests",
        "split_metrics_csv": "split_metrics",
        "candidate_pred_pkl": "candidate_pred",
        "candidate_pred_csv": "candidate_pred",
        "summary_json": "summary",
        "summary_md": "summary",
        "hard_gate_pass_json": "hard_gate_pass",
        "verification_json": "verification",
    }
    out: Dict[str, Path] = {}
    for key, suffix in names.items():
        ext = ".pkl" if key.endswith("_pkl") else ".json" if key.endswith("_json") else ".md" if key.endswith("_md") else ".csv"
        out[key] = THIS_DIR / f"{output_prefix}_{suffix}_{stamp}{ext}"
    return out


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(args.provider_uri).expanduser().resolve()

    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "tail-risk-aware target engineering; train 2020-2022, select on 2023 only, evaluate 2024-2026 once",
        "status": "started",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
    }

    try:
        quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = _load_config(Path(args.workflow_config).expanduser().resolve())
        port_cfg = base._extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))

        panel_raw, coverage_df = base._build_panel(provider_uri, str(args.market), RAW_START, TEST_END, BASE_FIELDS)
        dataset, feature_cols = base._build_features_and_targets(panel_raw)
        dataset = dataset.dropna(subset=["label_raw", "label_rank", "label_volnorm_rank"])
        day_counts = dataset.groupby(level=0)["label_raw"].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()
        dataset, label_meta = _build_tail_targets(dataset)
        dataset = dataset.dropna(subset=["label_raw", "label_rank", "label_volnorm_rank", "label_downside_rank", "label_tail_hit"])

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
            alpha_grid = alpha_grid[:1]
            topk_grid = topk_grid[:1]
            ndrop_grid = ndrop_grid[:1]

        feature_modes = {
            "rank_only": [c for c in feature_cols if c.endswith("__rank")],
            "rank_and_z": list(feature_cols),
        }
        target_modes = ["label_rank", "label_volnorm_rank", "label_downside_rank", "label_tail_hit"]
        candidate_rows, predictions = _make_predictions(train_df, valid_df, test_df, feature_modes, target_modes, alpha_grid)
        if not candidate_rows:
            raise RuntimeError("no base candidates generated")

        sorted_base_rows = sorted(
            candidate_rows,
            key=lambda r: (
                _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
                _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
            ),
            reverse=True,
        )
        _append_ensemble_candidate(candidate_rows, predictions, sorted_base_rows, top_n=3)
        _update_ensemble_metrics(candidate_rows, predictions, train_df, valid_df)

        sorted_rows = sorted(
            candidate_rows,
            key=lambda r: (
                _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
                _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
            ),
            reverse=True,
        )
        preselected = sorted_rows[: max(1, min(int(args.preselect), len(sorted_rows)))]
        combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk]
        if not combos:
            raise RuntimeError("no valid topk/n_drop combos")

        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
        valid_bt_rows: List[Dict[str, Any]] = []
        for cand in preselected:
            sig = predictions[str(cand["candidate_id"])]["valid"].rename("score").to_frame("score")
            for topk, n_drop in combos:
                metric, _ = _run_backtest_with_report(
                    sig,
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
                valid_bt_rows.append(asdict(metric))

        ok_valid = [
            r
            for r in valid_bt_rows
            if not r["error"] and np.isfinite(_safe_float(r["ir"])) and np.isfinite(_safe_float(r["annret"]))
        ]
        if not ok_valid:
            raise RuntimeError("no valid 2023 backtest combo succeeded; cannot freeze a portfolio")

        selected = sorted(
            ok_valid,
            key=lambda r: (_safe_float(r["ir"]), _safe_float(r["annret"])),
            reverse=True,
        )[0]
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])

        test_metric, test_report = _run_backtest_with_report(
            predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score"),
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
        if test_metric.error:
            raise RuntimeError(f"test backtest failed: {test_metric.error}")
        expected_test_rows = _count_calendar_rows(provider_uri, TEST_START, TEST_END)
        if expected_test_rows != EXPECTED_TEST_ROWS:
            raise RuntimeError(f"calendar expectation mismatch: computed={expected_test_rows} expected={EXPECTED_TEST_ROWS}")
        if int(test_metric.row_count) != expected_test_rows or int(test_metric.finite_rows) != expected_test_rows:
            raise RuntimeError(
                f"finite-row gate failed: rows={test_metric.row_count} finite={test_metric.finite_rows} expected={expected_test_rows}"
            )

        test_ic_s = _daily_rank_ic_series(predictions[str(selected["candidate_id"])]["test"], test_df["label_raw"])
        test_rank_ic, test_rank_ic_ir = _mean_and_ir(test_ic_s)
        hard_gate_pass = bool(
            np.isfinite(test_metric.ir)
            and np.isfinite(test_metric.annret)
            and test_metric.ir > HARD_GATE_IR
            and test_metric.annret > HARD_GATE_ANNRET
        )

        verification_metric, verification_report = _run_backtest_with_report(
            predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score"),
            "verification_2024_2026",
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
        verification_ok = bool(
            not verification_metric.error
            and verification_metric.row_count == test_metric.row_count
            and verification_metric.finite_rows == test_metric.finite_rows
            and abs(verification_metric.ir - test_metric.ir) <= 1e-12
            and abs(verification_metric.annret - test_metric.annret) <= 1e-12
        )
        if hard_gate_pass and not verification_ok:
            raise RuntimeError("verification rerun did not match the test backtest")

        coverage_df.to_csv(paths["coverage_csv"], index=False)
        _write_csv(paths["candidates_csv"], candidate_rows)
        _write_csv(paths["valid_backtests_csv"], valid_bt_rows)
        _write_csv(
            paths["split_metrics_csv"],
            [
                {
                    "split": "train_2020_2022",
                    "candidate_id": selected["candidate_id"],
                    "rank_ic": selected_candidate["train_rank_ic"],
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
                    "candidate_id": selected["candidate_id"],
                    "rank_ic": selected_candidate["valid_rank_ic"],
                    "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                    "costed_ir": selected["ir"],
                    "costed_annret": selected["annret"],
                    "max_drawdown": selected["max_drawdown"],
                    "turnover": selected["turnover"],
                    "topk": selected["topk"],
                    "n_drop": selected["n_drop"],
                    "error": selected["error"],
                },
                {
                    "split": "test_2024_2026",
                    "candidate_id": selected["candidate_id"],
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
            ],
        )

        pred_test = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
        with paths["candidate_pred_pkl"].open("wb") as f:
            pickle.dump(pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_test.reset_index().to_csv(paths["candidate_pred_csv"], index=False)

        hard_gate_payload = {
            "passed": hard_gate_pass,
            "verification_ok": verification_ok,
            "expected_test_rows": expected_test_rows,
            "test_rows": int(test_metric.row_count),
            "test_finite_rows": int(test_metric.finite_rows),
            "selected_candidate_id": selected["candidate_id"],
            "metrics": {
                "valid_2023": {"ir": selected["ir"], "annret": selected["annret"]},
                "test_2024_2026": {"ir": test_metric.ir, "annret": test_metric.annret},
                "verification_2024_2026": {"ir": verification_metric.ir, "annret": verification_metric.annret},
            },
        }
        paths["hard_gate_pass_json"].write_text(json.dumps(hard_gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["verification_json"].write_text(
            json.dumps(
                {
                    "verification_ok": verification_ok,
                    "metric": asdict(verification_metric),
                    "rows": int(len(verification_report)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary.update(
            {
                "status": "ok" if hard_gate_pass else "gate_failed",
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "training_protocol": {
                    "raw_start": RAW_START,
                    "train_window": [TRAIN_START, TRAIN_END],
                    "valid_window": [VALID_START, VALID_END],
                    "test_window": [TEST_START, TEST_END],
                    "feature_modes": list(feature_modes.keys()),
                    "target_modes": target_modes,
                    "alpha_grid": alpha_grid,
                    "topk_grid": topk_grid,
                    "ndrop_grid": ndrop_grid,
                    "selection_rule": "2023-only real net-cost backtest max IR, tie by AnnRet; no 2024-2026 metric used for selection",
                    "tail_target_meta": label_meta,
                    "preselected_candidates": [r["candidate_id"] for r in preselected],
                    "ensemble_members": next((r.get("ensemble_members", "") for r in candidate_rows if r["candidate_id"] == selected["candidate_id"]), ""),
                    "smoke": bool(args.smoke),
                },
                "coverage": {
                    "instrument_count": int(coverage_df["instrument"].nunique()),
                    "coverage_csv": str(paths["coverage_csv"]),
                },
                "selected_model": {
                    "candidate_id": selected["candidate_id"],
                    "feature_mode": selected_candidate["feature_mode"],
                    "target_mode": selected_candidate["target_mode"],
                    "alpha": selected_candidate["alpha"],
                    "feature_count": selected_candidate["feature_count"],
                    "topk": selected["topk"],
                    "n_drop": selected["n_drop"],
                },
                "metrics": {
                    "train_2020_2022": {
                        "rank_ic": selected_candidate["train_rank_ic"],
                    },
                    "valid_2023_selection_only": {
                        "rank_ic": selected_candidate["valid_rank_ic"],
                        "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                        "costed_ir": selected["ir"],
                        "costed_annret": selected["annret"],
                        "max_drawdown": selected["max_drawdown"],
                        "turnover": selected["turnover"],
                        "error": selected["error"],
                    },
                    "test_2024_2026_hard_gate": {
                        "rank_ic": test_rank_ic,
                        "rank_ic_ir": test_rank_ic_ir,
                        "costed_ir": test_metric.ir,
                        "costed_annret": test_metric.annret,
                        "max_drawdown": test_metric.max_drawdown,
                        "turnover": test_metric.turnover,
                        "error": test_metric.error,
                        "row_count": test_metric.row_count,
                        "finite_rows": test_metric.finite_rows,
                    },
                    "verification_2024_2026": {
                        "costed_ir": verification_metric.ir,
                        "costed_annret": verification_metric.annret,
                        "row_count": verification_metric.row_count,
                        "finite_rows": verification_metric.finite_rows,
                    },
                },
                "hard_gate": {
                    "rule": {
                        "scope": "test_2024_01_01_to_2026_04_30_only",
                        "ir_gt": HARD_GATE_IR,
                        "annret_gt": HARD_GATE_ANNRET,
                        "expected_rows": expected_test_rows,
                        "open_cost": float(args.open_cost),
                        "close_cost": float(args.close_cost),
                    },
                    "passed": hard_gate_pass,
                    "verification_ok": verification_ok,
                },
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Tail Risk Aware Pre-2024 Model ({stamp})",
                    "",
                    f"- status: `{summary['status']}`",
                    f"- selected_candidate: `{selected['candidate_id']}`",
                    f"- valid_2023 IR / AnnRet: `{selected['ir']:.6f}` / `{selected['annret']:.6f}`",
                    f"- test_2024_2026 IR / AnnRet: `{test_metric.ir:.6f}` / `{test_metric.annret:.6f}`",
                    f"- finite_rows: `{test_metric.finite_rows}` / `{expected_test_rows}`",
                    f"- hard_gate_pass: `{hard_gate_pass}`",
                    f"- verification_ok: `{verification_ok}`",
                    f"- summary_json: `{paths['summary_json']}`",
                    f"- hard_gate_pass_json: `{paths['hard_gate_pass_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        return 0 if hard_gate_pass and verification_ok else 2
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Tail Risk Aware Pre-2024 Model ({stamp})",
                    "",
                    f"- status: `failed`",
                    f"- blocker: `{summary['blocker']}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        paths["hard_gate_pass_json"].write_text(
            json.dumps(
                {
                    "passed": False,
                    "verification_ok": False,
                    "blocker": summary["blocker"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
