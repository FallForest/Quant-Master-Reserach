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
import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri_in_config
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy
from examples.benchmarks.Transcendence._bootstrap import init_quant_master_from_config, load_config_with_resolved_provider


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TARGET_7406_COSTED_IR = 2.799983676714277
TARGET_7406_COSTED_ANNRET = 0.24466463608994535

RUN_ALIAS = {
    "7406e470": "7406e47063e9479cb34d300b9ed03bad",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "1a085ff": "1a085ff9b5a34f408a44ad74055fc5da",
}


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    candidates = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not candidates:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id}")
    if len(candidates) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_config(path: Path) -> Dict[str, Any]:
    return load_config_with_resolved_provider(
        path,
        loader=lambda config_path: yaml.safe_load(config_path.read_text(encoding="utf-8")),
        binary_fallback=_load_pickle,
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


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


def _as_score_df(pred_obj: Any) -> pd.DataFrame:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float).to_frame("score")
    if isinstance(pred_obj, pd.DataFrame):
        if pred_obj.empty:
            return pd.DataFrame(columns=["score"], index=pred_obj.index)
        if "score" in pred_obj.columns:
            return pred_obj[["score"]].astype(float)
        if pred_obj.shape[1] == 1:
            c0 = pred_obj.columns[0]
            return pred_obj[[c0]].rename(columns={c0: "score"}).astype(float)
        return pred_obj.iloc[:, [0]].rename(columns={pred_obj.columns[0]: "score"}).astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _cross_section_rank(score: pd.Series) -> pd.Series:
    idx = score.index
    if isinstance(idx, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(idx).rank(method="average", pct=True)


def _slice_pred(pred_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        d = pd.to_datetime(idx.get_level_values(0))
        m = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
        return pred_df.loc[m]
    d = pd.to_datetime(idx)
    return pred_df.loc[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def _pred_dates(pred_df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    return pd.to_datetime(idx)


def _build_rank_ensemble_pred(
    *,
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    if len(run_ids) != len(weights):
        raise ValueError("run_ids and weights length mismatch")
    series_list: List[pd.Series] = []
    for run_id in run_ids:
        run_dir = _find_run_dir(tracking_dir, run_id)
        pred = _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
        pred = _slice_pred(pred, start_time, end_time)
        if pred.empty:
            raise ValueError(f"empty pred for run_id={run_id}")
        rank_s = _cross_section_rank(pred["score"].astype(float))
        rank_s.name = run_id
        series_list.append(rank_s)
    panel = pd.concat(series_list, axis=1)
    w = pd.Series([float(x) for x in weights], index=panel.columns, dtype=float)
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


def _signal_dispersion(pred_df: pd.DataFrame) -> pd.Series:
    score = pred_df["score"].astype(float)
    idx = score.index
    if isinstance(idx, pd.MultiIndex):
        return score.groupby(level=0).std().sort_index()
    return score.groupby(idx).std().sort_index()


def _calc_costed_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    excess = report_df["return"] - report_df["bench"] - report_df["cost"]
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "costed_annret": float(risk_df.loc["annualized_return", "risk"]),
        "costed_ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report_df["turnover"].mean()),
    }


def _eval_topk_strategy(
    *,
    pred_df: pd.DataFrame,
    topk: int,
    n_drop: int,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_time).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_time).date())
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    pred_slice = _slice_pred(pred_df, start_time, end_time)
    if pred_slice.empty:
        raise ValueError(f"empty signal slice in {start_time}~{end_time}")

    strategy = TopkDropoutStrategy(
        signal=pred_slice,
        topk=int(topk),
        n_drop=int(n_drop),
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
    exchange_kwargs["benchmark"] = "SH000300"

    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = (
        str(backtest_cfg["start_time"]),
        str(backtest_cfg["end_time"]),
        float(open_cost),
        float(close_cost),
        limit_threshold,
        deal_price,
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy,
        executor=executor_cfg,
        benchmark="SH000300",
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    if "1day" in portfolio_metric_dict:
        report_df = portfolio_metric_dict["1day"][0]
    else:
        key = next(iter(portfolio_metric_dict.keys()))
        report_df = portfolio_metric_dict[key][0]
    metrics = _calc_costed_metrics(report_df)
    return {
        "metrics": metrics,
        "report_df": report_df,
        "excess_series": (report_df["return"] - report_df["bench"] - report_df["cost"]).astype(float),
        "elapsed_sec": float(elapsed),
    }


def _year_slices(coverage_start: pd.Timestamp, coverage_end: pd.Timestamp) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    out: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for y in (2024, 2025, 2026):
        st = max(pd.Timestamp(f"{y}-01-01"), coverage_start)
        ed = min(pd.Timestamp(f"{y}-12-31"), coverage_end)
        if st > ed:
            continue
        tag = f"{y}_ytd" if ed < pd.Timestamp(f"{y}-12-31") else str(y)
        out.append((tag, st, ed))
    out.append(("2024_2026_full", coverage_start, coverage_end))
    return out


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.astype(float), freq="1day")
    return {
        "costed_annret": float(risk_df.loc["annualized_return", "risk"]),
        "costed_ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _slice_series(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.to_datetime(series.index)
    return series.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]


def _safe_quantile(hist: pd.Series, q: float) -> float:
    s = hist.dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def _regime_switch(
    *,
    strategy_reports: Dict[str, Dict[str, Any]],
    strategy_dispersion: Dict[str, pd.Series],
    warmup_days: int,
) -> Tuple[pd.Series, pd.Series, List[Dict[str, Any]]]:
    base_id = "baseline_7406_default_tk45_nd4_daily"
    idx = pd.DatetimeIndex(strategy_reports[base_id]["report_df"].index)
    bench = strategy_reports[base_id]["report_df"]["bench"].astype(float)
    bench_vol20 = bench.rolling(20).std().shift(1)
    selected = pd.Series(index=idx, dtype=object)
    meta_excess = pd.Series(index=idx, dtype=float)
    diag_rows: List[Dict[str, Any]] = []

    prev_sid = base_id
    for i, dt in enumerate(idx):
        if i < warmup_days:
            sid = prev_sid
            reason = "warmup"
        else:
            vol_now = float(bench_vol20.iloc[i]) if pd.notna(bench_vol20.iloc[i]) else float("nan")
            vol_thr = _safe_quantile(bench_vol20.iloc[:i], 0.75)

            base_disp_s = strategy_dispersion[base_id].reindex(idx)
            base_disp_now = float(base_disp_s.iloc[i]) if pd.notna(base_disp_s.iloc[i]) else float("nan")
            base_disp_thr = _safe_quantile(base_disp_s.iloc[:i], 0.35)

            score_map: Dict[str, float] = {}
            for sid0, rec in strategy_reports.items():
                ex = rec["excess_series"]
                report = rec["report_df"]
                hist5 = ex.iloc[max(0, i - 5) : i]
                hist20 = ex.iloc[max(0, i - 20) : i]
                ret5 = float(hist5.mean() * 252.0) if len(hist5) > 0 else 0.0
                ret20 = float(hist20.mean() * 252.0) if len(hist20) > 0 else 0.0
                vol20 = float(hist20.std(ddof=0) * np.sqrt(252.0)) if len(hist20) > 1 else 0.0
                turn5 = float(report["turnover"].iloc[max(0, i - 5) : i].mean()) if i > 0 else 0.0

                disp_s = strategy_dispersion[sid0].reindex(idx)
                disp_now = float(disp_s.iloc[i]) if pd.notna(disp_s.iloc[i]) else 0.0
                disp_med20 = float(disp_s.iloc[max(0, i - 20) : i].median()) if i > 0 else 0.0
                disp_ratio = disp_now / (disp_med20 + 1e-12) if np.isfinite(disp_med20) else 1.0

                # All inputs use history up to t-1, no t realization included.
                score = 0.8 * ret5 + 0.7 * ret20 - 0.55 * vol20 - 0.14 * turn5 + 0.06 * (disp_ratio - 1.0)
                score_map[sid0] = float(score)

            high_vol = np.isfinite(vol_now) and np.isfinite(vol_thr) and vol_now > vol_thr
            low_disp = np.isfinite(base_disp_now) and np.isfinite(base_disp_thr) and base_disp_now < base_disp_thr
            candidate = "fixed_topk40_nd2_daily" if (high_vol or low_disp) else max(score_map, key=score_map.get)
            if candidate != prev_sid and score_map[candidate] < score_map[prev_sid] + 0.02:
                sid = prev_sid
                reason = "hysteresis_hold"
            else:
                sid = candidate
                reason = "rule_select"

            diag_rows.append(
                {
                    "date": str(pd.Timestamp(dt).date()),
                    "selected_strategy_id": sid,
                    "prev_strategy_id": prev_sid,
                    "candidate_strategy_id": candidate,
                    "reason": reason,
                    "bench_vol20_prevday": vol_now,
                    "bench_vol20_q75_expanding": vol_thr,
                    "base_disp_prevday": base_disp_now,
                    "base_disp_q35_expanding": base_disp_thr,
                    "high_vol_trigger": bool(high_vol),
                    "low_disp_trigger": bool(low_disp),
                    "candidate_score": float(score_map.get(candidate, float("nan"))),
                    "prev_score": float(score_map.get(prev_sid, float("nan"))),
                }
            )
        selected.loc[dt] = sid
        meta_excess.loc[dt] = float(strategy_reports[sid]["excess_series"].loc[dt])
        prev_sid = sid
    return selected, meta_excess, diag_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime switch stability eval without future leakage (Transcendence).")
    p.add_argument("--run-id", default=TARGET_RUN_ID)
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--config-path", default="")
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--output-prefix", default="regime_switch_stability")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    run_dir = _find_run_dir(tracking_dir, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    pred_path = artifacts_dir / "pred.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing pred artifact: {pred_path}")

    run_cfg_path = Path(args.config_path).expanduser().resolve() if args.config_path else artifacts_dir / "config"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"missing config file: {run_cfg_path}")

    workflow_cfg = _load_config(run_cfg_path)
    _init_quant_master(workflow_cfg)
    base_port_cfg = _extract_port_config(workflow_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base_pred_df = _as_score_df(_load_pickle(pred_path))
    coverage_dates = _pred_dates(base_pred_df)
    coverage_start = pd.Timestamp(coverage_dates.min())
    coverage_end = pd.Timestamp(coverage_dates.max())
    slices = _year_slices(coverage_start, coverage_end)

    rank_pred_df = _build_rank_ensemble_pred(
        tracking_dir=tracking_dir,
        run_ids=[RUN_ALIAS["7406e470"], RUN_ALIAS["773bd6d"], RUN_ALIAS["bc641"]],
        weights=[0.6, 0.2, 0.2],
        start_time=coverage_start,
        end_time=coverage_end,
    )
    gru_meta_pred_df = _build_rank_ensemble_pred(
        tracking_dir=tracking_dir,
        run_ids=[RUN_ALIAS["7406e470"], RUN_ALIAS["1a085ff"], RUN_ALIAS["773bd6d"]],
        weights=[0.4, 0.2, 0.4],
        start_time=coverage_start,
        end_time=coverage_end,
    )

    strategy_specs = [
        {
            "strategy_id": "baseline_7406_default_tk45_nd4_daily",
            "signal_key": "base",
            "signal_df": base_pred_df,
            "topk": 45,
            "n_drop": 4,
        },
        {
            "strategy_id": "fixed_topk40_nd2_daily",
            "signal_key": "base",
            "signal_df": base_pred_df,
            "topk": 40,
            "n_drop": 2,
        },
        {
            "strategy_id": "fixed_topk50_nd5_daily",
            "signal_key": "base",
            "signal_df": base_pred_df,
            "topk": 50,
            "n_drop": 5,
        },
        {
            "strategy_id": "fixed_rank_ensemble_tk45_nd4_daily",
            "signal_key": "rank_ens_7406_773_bc641",
            "signal_df": rank_pred_df,
            "topk": 45,
            "n_drop": 4,
        },
        {
            "strategy_id": "fixed_rank_ensemble_tk50_nd5_daily",
            "signal_key": "rank_ens_7406_773_bc641",
            "signal_df": rank_pred_df,
            "topk": 50,
            "n_drop": 5,
        },
        {
            "strategy_id": "fixed_gru_meta_rank_ens_tk45_nd4_daily",
            "signal_key": "rank_ens_7406_1a085ff_773",
            "signal_df": gru_meta_pred_df,
            "topk": 45,
            "n_drop": 4,
        },
    ]

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    slice_rows: List[Dict[str, Any]] = []
    for sp in strategy_specs:
        sid = str(sp["strategy_id"])
        for slice_name, st, ed in slices:
            ev = _eval_topk_strategy(
                pred_df=sp["signal_df"],
                topk=sp["topk"],
                n_drop=sp["n_drop"],
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                start_time=st,
                end_time=ed,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=exchange_cache,
            )
            rec = {
                "strategy_id": sid,
                "signal_key": sp["signal_key"],
                "topk": int(sp["topk"]),
                "n_drop": int(sp["n_drop"]),
                "slice": slice_name,
                "start_time": str(pd.Timestamp(st).date()),
                "end_time": str(pd.Timestamp(ed).date()),
                "costed_annret": float(ev["metrics"]["costed_annret"]),
                "costed_ir": float(ev["metrics"]["costed_ir"]),
                "max_drawdown": float(ev["metrics"]["max_drawdown"]),
                "turnover": float(ev["metrics"]["turnover"]),
                "elapsed_sec": float(ev["elapsed_sec"]),
            }
            slice_rows.append(rec)
            print(
                f"[slice][{sid}][{slice_name}] IR={rec['costed_ir']:.6f} AnnRet={rec['costed_annret']:.6f} TO={rec['turnover']:.6f}",
                flush=True,
            )

    # Continuous walk-forward reports for switching logic.
    full_reports: Dict[str, Dict[str, Any]] = {}
    for sp in strategy_specs:
        sid = str(sp["strategy_id"])
        ev = _eval_topk_strategy(
            pred_df=sp["signal_df"],
            topk=sp["topk"],
            n_drop=sp["n_drop"],
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            start_time=coverage_start,
            end_time=coverage_end,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        full_reports[sid] = ev
        print(
            f"[full][{sid}] IR={ev['metrics']['costed_ir']:.6f} AnnRet={ev['metrics']['costed_annret']:.6f}",
            flush=True,
        )

    # Strategy-level signal diagnostics.
    disp_map = {
        "base": _signal_dispersion(_slice_pred(base_pred_df, coverage_start, coverage_end)),
        "rank_ens_7406_773_bc641": _signal_dispersion(_slice_pred(rank_pred_df, coverage_start, coverage_end)),
        "rank_ens_7406_1a085ff_773": _signal_dispersion(_slice_pred(gru_meta_pred_df, coverage_start, coverage_end)),
    }
    strategy_disp = {
        sp["strategy_id"]: disp_map[sp["signal_key"]]
        for sp in strategy_specs
    }

    selected_sid, regime_excess, diag_rows = _regime_switch(
        strategy_reports=full_reports,
        strategy_dispersion=strategy_disp,
        warmup_days=25,
    )

    regime_full_metrics = _metrics_from_excess(regime_excess)
    yearly_eval_rows: List[Dict[str, Any]] = []
    baseline_id = "baseline_7406_default_tk45_nd4_daily"
    baseline_excess = full_reports[baseline_id]["excess_series"]
    for slice_name, st, ed in slices:
        rg = _metrics_from_excess(_slice_series(regime_excess, st, ed))
        bs = _metrics_from_excess(_slice_series(baseline_excess, st, ed))
        yearly_eval_rows.append(
            {
                "slice": slice_name,
                "start_time": str(pd.Timestamp(st).date()),
                "end_time": str(pd.Timestamp(ed).date()),
                "regime_costed_ir": float(rg["costed_ir"]),
                "regime_costed_annret": float(rg["costed_annret"]),
                "regime_max_drawdown": float(rg["max_drawdown"]),
                "baseline_costed_ir": float(bs["costed_ir"]),
                "baseline_costed_annret": float(bs["costed_annret"]),
                "baseline_max_drawdown": float(bs["max_drawdown"]),
                "delta_ir": float(rg["costed_ir"] - bs["costed_ir"]),
                "delta_annret": float(rg["costed_annret"] - bs["costed_annret"]),
            }
        )

    baseline_full = full_reports[baseline_id]["metrics"]
    slice_map = {r["slice"]: r for r in yearly_eval_rows}
    no_sacrifice = all(
        slice_map[x]["delta_ir"] >= -1e-12 and slice_map[x]["delta_annret"] >= -1e-12
        for x in ("2024", "2025")
        if x in slice_map
    )
    improved_2026 = (
        "2026_ytd" in slice_map
        and slice_map["2026_ytd"]["delta_ir"] > 0
        and slice_map["2026_ytd"]["delta_annret"] > 0
    )
    beats_7406_full = (
        float(regime_full_metrics["costed_ir"]) > float(baseline_full["costed_ir"])
        and float(regime_full_metrics["costed_annret"]) > float(baseline_full["costed_annret"])
    )
    has_breakthrough = bool(no_sacrifice and improved_2026 and beats_7406_full)

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_short = args.run_id[:8]
    slice_csv = out_dir / f"{args.output_prefix}_slice_metrics_{run_short}_{stamp}.csv"
    regime_year_csv = out_dir / f"{args.output_prefix}_yearly_vs_baseline_{run_short}_{stamp}.csv"
    selector_csv = out_dir / f"{args.output_prefix}_selector_diag_{run_short}_{stamp}.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"

    _write_csv(slice_csv, slice_rows)
    _write_csv(regime_year_csv, yearly_eval_rows)
    _write_csv(selector_csv, diag_rows)

    strategy_selection_counts = {
        str(k): int(v) for k, v in selected_sid.value_counts(dropna=False).to_dict().items()
    }
    summary = {
        "run_id": args.run_id,
        "tracking_uri": args.tracking_uri,
        "artifact_dir": str(artifacts_dir),
        "config_path": str(run_cfg_path),
        "scan_time_utc": _now_utc(),
        "signal_coverage": {
            "start": str(pd.Timestamp(coverage_start).date()),
            "end": str(pd.Timestamp(coverage_end).date()),
            "rows": int(len(base_pred_df)),
            "unique_trade_days": int(pd.Index(coverage_dates).nunique()),
        },
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "strategies_predeclared": [
            {
                "strategy_id": sp["strategy_id"],
                "signal_key": sp["signal_key"],
                "topk": int(sp["topk"]),
                "n_drop": int(sp["n_drop"]),
            }
            for sp in strategy_specs
        ],
        "signal_diagnostics_rule": {
            "description": "Causal regime switch using previous-day diagnostics only.",
            "warmup_days": 25,
            "features_prevday_only": [
                "strategy_excess_return_5d_annualized",
                "strategy_excess_return_20d_annualized",
                "strategy_excess_vol_20d_annualized",
                "strategy_turnover_5d_mean",
                "strategy_signal_dispersion_ratio_vs_20d_median",
                "benchmark_volatility_20d",
                "base_signal_dispersion",
            ],
            "score_formula": "score = 0.8*ret5 + 0.7*ret20 - 0.55*vol20 - 0.14*turn5 + 0.06*(disp_ratio-1)",
            "hard_regime_trigger": "if benchmark_vol20 > expanding_q75 OR base_dispersion < expanding_q35, choose fixed_topk40_nd2_daily",
            "hysteresis": "switch only when candidate_score >= prev_score + 0.02",
            "default_during_warmup": baseline_id,
        },
        "leakage_boundary": {
            "guardrails": [
                "All regime features are shifted to t-1 (no t-day realized return/cost/bench in decisions).",
                "Quantile thresholds use expanding history ending at t-1 only.",
                "Strategy universe and rule coefficients are fixed before evaluating 2026_ytd.",
                "No per-slice future pick: single rule runs through full 2024-01-02~2026-04-30 timeline.",
            ],
            "train_validation_split_used_for_rule_choice": {
                "rule_constants_fixed_from": "2024-01-02~2025-12-31 diagnostics only",
                "oos_focus_slice": "2026-01-01~2026-04-30",
            },
        },
        "slice_diagnostics_independent": slice_rows,
        "continuous_regime_metrics_full": regime_full_metrics,
        "continuous_baseline_metrics_full": baseline_full,
        "continuous_yearly_vs_baseline": yearly_eval_rows,
        "selector_counts": strategy_selection_counts,
        "selector_records": int(len(diag_rows)),
        "thresholds": {
            "legacy_7406_costed_ir": TARGET_7406_COSTED_IR,
            "legacy_7406_costed_annret": TARGET_7406_COSTED_ANNRET,
        },
        "breakthrough_checks": {
            "beats_7406_full_ir_and_annret": bool(beats_7406_full),
            "improves_2026_ytd": bool(improved_2026),
            "no_sacrifice_2024_2025": bool(no_sacrifice),
            "has_breakthrough": bool(has_breakthrough),
        },
        "artifacts": {
            "slice_metrics_csv": str(slice_csv),
            "yearly_vs_baseline_csv": str(regime_year_csv),
            "selector_diag_csv": str(selector_csv),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

