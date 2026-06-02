#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri_in_config
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from examples.benchmarks.Transcendence._bootstrap import init_quant_master_from_config, load_config_with_resolved_provider


TRANS_DIR = Path(__file__).resolve().parent
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
OPEN_COST = 0.0001
CLOSE_COST = 0.0006
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

RUNS = {
    "7406e470": "7406e47063e9479cb34d300b9ed03bad",
    "1a085ff9": "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
}


@dataclass
class ActionSpec:
    name: str
    signal: pd.Series
    strategy: str
    strategy_kwargs: Dict[str, Any]
    source_paths: List[str]
    notes: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return (REPO_ROOT / tracking_uri).resolve() if not Path(tracking_uri).is_absolute() else Path(tracking_uri)


def _find_run_dir(tracking_dir: Path, run_id_or_prefix: str) -> Path:
    token = str(run_id_or_prefix).strip()
    cands = [p for p in tracking_dir.glob(f"*/{token}") if (p / "artifacts").exists()]
    if not cands:
        cands = [p for p in tracking_dir.glob(f"*/{token}*") if (p / "artifacts").exists()]
    if not cands:
        raise FileNotFoundError(f"run not found under {tracking_dir}: {run_id_or_prefix}")
    if len(cands) > 1:
        exact = [p for p in cands if p.name == token]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"run token matched multiple dirs: {run_id_or_prefix}")
    return cands[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_config(path: Path) -> Dict[str, Any]:
    return load_config_with_resolved_provider(
        path,
        loader=lambda config_path: yaml.safe_load(config_path.read_text(encoding="utf-8")),
        binary_fallback=_load_pickle,
    )


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
    init_quant_master_from_config(config, base_dir=REPO_ROOT, region="cn")


def _as_score_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.astype(float)
    elif isinstance(obj, pd.DataFrame):
        col = "score" if "score" in obj.columns else obj.columns[0]
        s = obj[col].astype(float)
    else:
        raise TypeError(f"unsupported prediction object: {type(obj)}")
    return _normalize_signal_index(s)


def _normalize_signal_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise TypeError("expected MultiIndex signal with datetime/instrument")
    idx = series.index
    dt0 = pd.to_datetime(pd.Index(idx.get_level_values(0)[:64]), errors="coerce")
    dt1 = pd.to_datetime(pd.Index(idx.get_level_values(1)[:64]), errors="coerce")
    if dt0.notna().mean() < dt1.notna().mean():
        series = series.swaplevel(0, 1)
    series = series.sort_index()
    series.index = series.index.set_names(["datetime", "instrument"] + list(series.index.names[2:]))
    series.name = "score"
    return series.astype(float)


def _slice_signal(series: pd.Series, start: str, end: str) -> pd.Series:
    dt = pd.to_datetime(series.index.get_level_values(0))
    return series.loc[(dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))].dropna()


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(method="average", pct=True)


def _rank_blend(signals: Sequence[pd.Series], weights: Sequence[float]) -> pd.Series:
    base_index = signals[0].index
    for s in signals[1:]:
        base_index = base_index.union(s.index)
    cols = [_cs_rank(s).reindex(base_index) for s in signals]
    panel = pd.concat(cols, axis=1)
    w = pd.Series([float(x) for x in weights], index=panel.columns)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = panel.fillna(0.0).mul(w, axis=1).sum(axis=1).div(denom.where(denom > 0))
    blend.name = "score"
    return blend.dropna()


def _day_result(portfolio_metric: Dict[str, Any]) -> Tuple[str, pd.DataFrame, Dict[pd.Timestamp, Any]]:
    if "1day" in portfolio_metric:
        freq = "1day"
    elif "day" in portfolio_metric:
        freq = "day"
    else:
        freq = next(iter(portfolio_metric.keys()))
    report, positions = portfolio_metric[freq]
    return freq, report, positions


def _calc_metrics(report: pd.DataFrame) -> Dict[str, float]:
    risk = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="1day")
    return {
        "annret": float(risk.loc["annualized_return", "risk"]),
        "ir": float(risk.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].mean()),
        "rows": int(len(report)),
    }


def _positions_to_frame(positions: Dict[pd.Timestamp, Any]) -> pd.DataFrame:
    rows = []
    for dt, pos in sorted(positions.items(), key=lambda x: x[0]):
        data = copy.deepcopy(getattr(pos, "position", {}))
        cash = data.pop("cash", None)
        account_value = data.pop("now_account_value", None)
        for instrument, item in data.items():
            if not isinstance(item, dict):
                continue
            row = {"datetime": dt, "instrument": instrument, "cash": cash, "now_account_value": account_value}
            row.update(item)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()


def _build_exchange(
    backtest_cfg: Dict[str, Any],
    executor_cfg: Dict[str, Any],
    start: str,
    end: str,
    exchange_cache: Dict[Tuple[str, str, str, float, float, float, str, float], Any],
) -> Dict[str, Any]:
    exch = dict(backtest_cfg.get("exchange_kwargs", {}))
    exch["open_cost"] = OPEN_COST
    exch["close_cost"] = CLOSE_COST
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exch.get("limit_threshold", 0.095))
    deal_price = str(exch.get("deal_price", "close"))
    min_cost = float(exch.get("min_cost", 5))
    key = (freq, start, end, OPEN_COST, CLOSE_COST, limit_threshold, deal_price, min_cost)
    if key not in exchange_cache:
        exchange_cache[key] = get_exchange(
            freq=freq,
            start_time=start,
            end_time=end,
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=OPEN_COST,
            close_cost=CLOSE_COST,
            min_cost=min_cost,
        )
    exch["exchange"] = exchange_cache[key]
    return exch


def _run_action(
    action: ActionSpec,
    base_port_cfg: Dict[str, Any],
    start: str,
    end: str,
    out_dir: Path,
    exchange_cache: Dict[Tuple[str, str, str, float, float, float, str, float], Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = start
    backtest_cfg["end_time"] = end
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exchange_kwargs = _build_exchange(backtest_cfg, executor_cfg, start, end, exchange_cache)

    signal = _slice_signal(action.signal, start, end)
    if signal.empty:
        raise RuntimeError(f"{action.name} has no signal rows for {start}..{end}")

    if action.strategy == "TopkDropoutStrategy":
        strategy_cfg = copy.deepcopy(cfg["strategy"])
        strategy_cfg["class"] = "TopkDropoutStrategy"
        strategy_cfg["module_path"] = "quant_master.contrib.strategy"
        strategy_cfg.setdefault("kwargs", {})
        strategy_cfg["kwargs"]["signal"] = signal
        strategy_cfg["kwargs"].update(action.strategy_kwargs)
        strategy_obj_or_cfg: Any = strategy_cfg
    elif action.strategy == "BufferedTopkWeightStrategy":
        from factor_augmented_meta_ensemble import BufferedTopkWeightStrategy

        strategy_obj_or_cfg = BufferedTopkWeightStrategy(signal=signal.to_frame("score"), **action.strategy_kwargs)
    else:
        raise ValueError(f"unsupported strategy: {action.strategy}")

    t0 = time.perf_counter()
    pm, indicators = run_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy_obj_or_cfg,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    freq, report, positions = _day_result(pm)
    metrics = _calc_metrics(report)
    metrics["elapsed_sec"] = float(elapsed)

    tag = f"{action.name}_{start}_{end}".replace("-", "")
    report_pkl = out_dir / f"{tag}_report.pkl"
    report_csv = out_dir / f"{tag}_report.csv"
    positions_pkl = out_dir / f"{tag}_positions.pkl"
    positions_csv = out_dir / f"{tag}_positions.csv"
    indicator_pkl = out_dir / f"{tag}_indicators.pkl"
    _dump_pickle(report_pkl, report)
    report.to_csv(report_csv)
    _dump_pickle(positions_pkl, positions)
    _dump_pickle(indicator_pkl, indicators)
    try:
        pos_df = _positions_to_frame(positions)
        pos_df.to_csv(positions_csv)
        positions_csv_status = str(positions_csv)
        positions_rows = int(len(pos_df))
    except Exception as exc:  # noqa: BLE001
        positions_csv_status = f"parse_failed: {type(exc).__name__}: {exc}"
        positions_rows = None

    return {
        "action": action.name,
        "status": "ok",
        "start": start,
        "end": end,
        "freq": freq,
        "strategy": action.strategy,
        "strategy_kwargs": action.strategy_kwargs,
        "source_paths": action.source_paths,
        "signal_rows": int(len(signal)),
        "signal_days": int(pd.Index(signal.index.get_level_values(0)).nunique()),
        "metrics": metrics,
        "hard_gate_pass": bool(metrics["ir"] > HARD_GATE_IR and metrics["annret"] > HARD_GATE_ANNRET),
        "artifacts": {
            "report_pkl": str(report_pkl),
            "report_csv": str(report_csv),
            "positions_pkl": str(positions_pkl),
            "positions_csv": positions_csv_status,
            "indicator_pkl": str(indicator_pkl),
        },
        "notes": action.notes,
    }


def _load_actions(tracking_dir: Path, start: str, end: str) -> Tuple[Dict[str, ActionSpec], Dict[str, Any]]:
    run_dirs = {k: _find_run_dir(tracking_dir, v) for k, v in RUNS.items()}
    run_signals = {
        k: _slice_signal(_as_score_series(_load_pickle(run_dirs[k] / "artifacts" / "pred.pkl")), start, end)
        for k in RUNS
    }
    factor_summary_path = TRANS_DIR / "factor_augmented_meta_summary_20260522T120515Z.json"
    factor_summary: Dict[str, Any] = {}
    if factor_summary_path.exists():
        factor_summary = json.loads(factor_summary_path.read_text(encoding="utf-8"))
    candidate_path = Path(
        factor_summary.get("artifacts", {}).get(
            "candidate_pred_pkl", str(TRANS_DIR / "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl")
        )
    )
    if not candidate_path.is_absolute():
        candidate_path = (TRANS_DIR / candidate_path).resolve()
    strategy_execution = factor_summary.get("protocol", {}).get("strategy_execution", {})
    factor_kwargs = {
        "topk": int(strategy_execution.get("topk", 55)),
        "hold_topk": int(strategy_execution.get("hold_topk", 85)),
        "weight_mode": strategy_execution.get("weight_mode", "equal"),
        "rebalance_mode": strategy_execution.get("rebalance_mode", "weekly"),
        "rebalance_interval": int(strategy_execution.get("rebalance_interval", 1)),
    }

    actions = {
        "base40": ActionSpec(
            name="base40",
            signal=run_signals["7406e470"],
            strategy="TopkDropoutStrategy",
            strategy_kwargs={"topk": 40, "n_drop": 2},
            source_paths=[str(run_dirs["7406e470"] / "artifacts" / "pred.pkl")],
            notes="base run 7406e470 TopkDropoutStrategy topk=40 n_drop=2",
        ),
        "gru45": ActionSpec(
            name="gru45",
            signal=_rank_blend(
                [run_signals["7406e470"], run_signals["1a085ff9"], run_signals["773bd6d"]],
                [0.4, 0.2, 0.4],
            ),
            strategy="TopkDropoutStrategy",
            strategy_kwargs={"topk": 45, "n_drop": 4},
            source_paths=[str(run_dirs[k] / "artifacts" / "pred.pkl") for k in ("7406e470", "1a085ff9", "773bd6d")],
            notes="rank ensemble 7406e470/1a085ff9/773bd6d weights=0.4/0.2/0.4 topk=45 n_drop=4",
        ),
    }
    if candidate_path.exists():
        actions["factor_augmented_meta"] = ActionSpec(
            name="factor_augmented_meta",
            signal=_as_score_series(_load_pickle(candidate_path)),
            strategy="BufferedTopkWeightStrategy",
            strategy_kwargs=factor_kwargs,
            source_paths=[str(candidate_path), str(factor_summary_path)],
            notes="candidate_pred_pkl and strategy_execution loaded from factor_augmented_meta_summary_20260522T120515Z.json",
        )
    else:
        raise FileNotFoundError(f"factor_augmented_meta candidate_pred_pkl missing: {candidate_path}")
    return actions, {"run_dirs": {k: str(v) for k, v in run_dirs.items()}, "factor_summary_path": str(factor_summary_path)}


def _parse_windows(values: Sequence[str]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for text in values:
        parts = [x.strip() for x in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"window must be name:start:end, got {text}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay predeclared strict breakthrough actions into real backtest artifacts.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--actions", default="base40,gru45,factor_augmented_meta")
    p.add_argument("--windows", nargs="*", default=[])
    p.add_argument("--output-prefix", default="replay_action_reports_cache")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    stamp = _stamp()
    out_dir = TRANS_DIR / f"{args.output_prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    selected = [x.strip() for x in args.actions.split(",") if x.strip()]
    windows = [("full", args.start_date, args.end_date)] + _parse_windows(args.windows)
    summary_path = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    metrics_csv = out_dir / f"{args.output_prefix}_metrics_{stamp}.csv"

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "tracking_dir": str(tracking_dir),
        "test_period": {"start": args.start_date, "end": args.end_date},
        "costs": {"open": OPEN_COST, "close": CLOSE_COST},
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "windows": [{"name": n, "start": s, "end": e} for n, s, e in windows],
        "results": [],
        "errors": [],
    }

    try:
        base_run_dir = _find_run_dir(tracking_dir, RUNS["7406e470"])
        base_cfg = _load_config(base_run_dir / "artifacts" / "config")
        _init_quant_master(base_cfg)
        base_port_cfg = _extract_port_config(base_cfg)
        actions, load_meta = _load_actions(tracking_dir, args.start_date, args.end_date)
        summary["load_meta"] = load_meta
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append({"stage": "load", "error": f"{type(exc).__name__}: {exc}"})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    exchange_cache: Dict[Tuple[str, str, str, float, float, float, str, float], Any] = {}
    for action_name in selected:
        action = actions.get(action_name)
        if action is None:
            err = {"action": action_name, "stage": "select", "error": "unknown action"}
            summary["errors"].append(err)
            rows.append(err)
            continue
        for window_name, start, end in windows:
            try:
                result = _run_action(action, base_port_cfg, start, end, out_dir, exchange_cache)
                result["window"] = window_name
                summary["results"].append(result)
                row = {
                    "action": action.name,
                    "window": window_name,
                    "status": "ok",
                    "start": start,
                    "end": end,
                    "annret": result["metrics"]["annret"],
                    "ir": result["metrics"]["ir"],
                    "max_drawdown": result["metrics"]["max_drawdown"],
                    "turnover": result["metrics"]["turnover"],
                    "rows": result["metrics"]["rows"],
                    "signal_rows": result["signal_rows"],
                    "signal_days": result["signal_days"],
                    "hard_gate_pass": result["hard_gate_pass"],
                    "report_pkl": result["artifacts"]["report_pkl"],
                    "report_csv": result["artifacts"]["report_csv"],
                    "positions_pkl": result["artifacts"]["positions_pkl"],
                    "positions_csv": result["artifacts"]["positions_csv"],
                    "error": "",
                }
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                err = {
                    "action": action_name,
                    "window": window_name,
                    "status": "error",
                    "start": start,
                    "end": end,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                summary["errors"].append(err)
                rows.append(err)

    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    summary["artifacts"] = {"summary_json": str(summary_path), "metrics_csv": str(metrics_csv), "out_dir": str(out_dir)}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "metrics_csv": str(metrics_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

