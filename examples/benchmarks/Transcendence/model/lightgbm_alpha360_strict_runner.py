#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "hard_gate_pass"
LOCAL_PROVIDER = (REPO_ROOT / ".qmData" / "cn_data").resolve()
LIGHTGBM_ALPHA360_CONFIG = (
    REPO_ROOT / "examples" / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha360.yaml"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.model.gbdt import LGBModel
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy
from quant_master.data.dataset import DatasetH

TRAIN_RANGE = ["2020-01-01", "2022-12-31"]
VALID_RANGE = ["2023-01-01", "2023-12-31"]
TEST_RANGE = ["2024-01-01", "2026-04-30"]
SMOKE_TEST_RANGE = ["2024-01-01", "2024-03-31"]
RAW_START = "2019-01-01"
SMOKE_RAW_START = "2022-10-01"
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
HARD_GATE_ROWS = 562
MARKET = "csi300"
BENCHMARK = "SH000300"

BASE_MODEL_KWARGS: Dict[str, Any] = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}
FULL_MODEL_EXTRAS = {"num_boost_round": 1000, "early_stopping_rounds": 50}
SMOKE_MODEL_EXTRAS = {"num_boost_round": 80, "early_stopping_rounds": 20}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(_jsonable(obj), f)


def _extract_lgbm_alpha360_evidence(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    task = cfg.get("task", {})
    dataset_cfg = task.get("dataset", {}).get("kwargs", {})
    handler_cfg = dataset_cfg.get("handler", {})
    model_cfg = task.get("model", {})
    port_cfg = cfg.get("port_analysis_config", {})
    return {
        "source_config": str(config_path),
        "source_model": copy.deepcopy(model_cfg),
        "source_handler_class": handler_cfg.get("class"),
        "source_handler_module": handler_cfg.get("module_path"),
        "source_label": copy.deepcopy(handler_cfg.get("kwargs", {}).get("label")),
        "source_segments": copy.deepcopy(dataset_cfg.get("segments")),
        "source_strategy": copy.deepcopy(port_cfg.get("strategy")),
        "source_backtest": copy.deepcopy(port_cfg.get("backtest")),
    }


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    vals = [x.strip() for x in (provider_uri / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.to_datetime(pd.Index(vals))


def _calendar_coverage(provider_uri: Path) -> Dict[str, Any]:
    cal = _read_calendar(provider_uri)
    out: Dict[str, Any] = {}
    for name, rng in {"train": TRAIN_RANGE, "valid": VALID_RANGE, "test": TEST_RANGE, "smoke_test": SMOKE_TEST_RANGE}.items():
        mask = (cal >= pd.Timestamp(rng[0])) & (cal <= pd.Timestamp(rng[1]))
        sub = cal[mask]
        out[name] = {
            "requested_start": rng[0],
            "requested_end": rng[1],
            "trading_rows": int(mask.sum()),
            "first_trading_day": str(sub.min().date()) if len(sub) else "",
            "last_trading_day": str(sub.max().date()) if len(sub) else "",
        }
    return out


def _build_dataset(test_range: List[str], mode: str) -> DatasetH:
    raw_start = SMOKE_RAW_START if mode == "smoke" else RAW_START
    data_end = test_range[1] if mode == "smoke" else TEST_RANGE[1]
    handler = {
        "class": "Alpha360",
        "module_path": "quant_master.contrib.data.handler",
        "kwargs": {
            "start_time": raw_start,
            "end_time": data_end,
            "fit_start_time": TRAIN_RANGE[0],
            "fit_end_time": TRAIN_RANGE[1],
            "instruments": MARKET,
            "infer_processors": [],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
            "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
        },
    }
    return DatasetH(
        handler=handler,
        segments={"train": list(TRAIN_RANGE), "valid": list(VALID_RANGE), "test": list(test_range)},
    )


def _init_quant_master() -> None:
    quant_master.init(
        provider_uri=str(LOCAL_PROVIDER),
        region="cn",
        kernels=1,
        joblib_backend="threading",
        clear_mem_cache=True,
    )


def _train_model(dataset: DatasetH, mode: str, seed: int) -> Tuple[LGBModel, Dict[str, Any]]:
    kwargs = dict(BASE_MODEL_KWARGS)
    kwargs.update(SMOKE_MODEL_EXTRAS if mode == "smoke" else FULL_MODEL_EXTRAS)
    kwargs["num_threads"] = min(int(kwargs["num_threads"]), 4 if mode == "smoke" else int(kwargs["num_threads"]))
    kwargs["seed"] = int(seed)
    kwargs["feature_fraction_seed"] = int(seed)
    kwargs["bagging_seed"] = int(seed)
    kwargs["data_random_seed"] = int(seed)
    model = LGBModel(**kwargs)
    evals_result: Dict[str, Any] = {}
    model.fit(dataset, evals_result=evals_result, verbose_eval=20 if mode == "smoke" else 50)
    return model, {"model_kwargs": kwargs, "evals_result": evals_result}


def _predict(model: LGBModel, dataset: DatasetH, segment: str) -> pd.DataFrame:
    pred = model.predict(dataset, segment=segment).astype(float)
    pred.name = "score"
    return pred.to_frame("score").sort_index()


def _daily_rank_ic(pred: pd.DataFrame, label: pd.Series) -> Dict[str, Any]:
    panel = pd.concat([pred["score"].rename("pred"), label.rename("label")], axis=1).dropna()
    vals: List[Tuple[pd.Timestamp, float]] = []
    if isinstance(panel.index, pd.MultiIndex):
        for dt, g in panel.groupby(level=0, sort=False):
            if len(g) < 20:
                continue
            corr = g["pred"].corr(g["label"], method="spearman")
            if pd.notna(corr) and np.isfinite(corr):
                vals.append((pd.Timestamp(dt), float(corr)))
    if len(vals) < 2:
        return {"rank_ic": float("nan"), "rank_ic_ir": float("nan"), "rank_ic_days": len(vals)}
    s = pd.Series({dt: val for dt, val in vals}, dtype=float).sort_index()
    return {
        "rank_ic": float(s.mean()),
        "rank_ic_ir": float(s.mean() / (s.std(ddof=1) + 1e-12) * np.sqrt(252.0)),
        "rank_ic_days": int(len(s)),
    }


def _get_label(dataset: DatasetH, segment: str) -> pd.Series:
    df = dataset.prepare(segment, col_set="label")
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            ser = df.iloc[:, 0]
        else:
            ser = df[df.columns[0]]
    else:
        ser = pd.Series(df)
    ser.name = "label"
    return ser.astype(float).sort_index()


def _report_from_portfolio_metric(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    return portfolio_metric_dict[next(iter(portfolio_metric_dict.keys()))][0]


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, Any]:
    missing = [c for c in ["return", "bench", "cost"] if c not in report.columns]
    if missing:
        raise KeyError(f"report missing required columns: {missing}")
    excess = pd.to_numeric(report["return"] - report["bench"] - report["cost"], errors="coerce")
    finite_mask = excess.notna() & np.isfinite(excess.astype(float))
    finite_excess = excess.loc[finite_mask].astype(float)
    if finite_excess.empty:
        raise ValueError("report has no finite net-cost excess return rows")
    risk = risk_analysis(finite_excess.sort_index(), freq="1day")
    turnover = pd.to_numeric(report.get("turnover", pd.Series(index=report.index, dtype=float)), errors="coerce")
    return {
        "costed_annret": float(risk.loc["annualized_return", "risk"]),
        "costed_ir": float(risk.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk.loc["max_drawdown", "risk"]),
        "turnover": float(turnover.loc[finite_mask].dropna().mean()) if finite_mask.any() else float("nan"),
        "rows": int(len(report)),
        "finite_rows": int(finite_mask.sum()),
        "nonfinite_rows": int(len(report) - finite_mask.sum()),
        "start": str(pd.to_datetime(report.index).min().date()) if len(report) else "",
        "end": str(pd.to_datetime(report.index).max().date()) if len(report) else "",
    }


def _run_backtest(
    signal_df: pd.DataFrame,
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    exchange_cache: Dict[Tuple[str, str], Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_key = (start, end)
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq="day",
            start_time=start,
            end_time=end,
            codes=MARKET,
            deal_price="close",
            limit_threshold=0.095,
            open_cost=OPEN_COST,
            close_cost=CLOSE_COST,
            min_cost=5,
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
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "quant_master.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    pm, _ = run_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=BENCHMARK,
        account=100000000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "min_cost": 5,
            "exchange": exchange_cache[cache_key],
        },
        pos_type="Position",
    )
    report = _report_from_portfolio_metric(pm)
    return report, _metrics_from_report(report)


def _strategy_grid(mode: str) -> Iterable[Tuple[int, int]]:
    if mode == "smoke":
        return [(20, 1)]
    return [(35, 2), (35, 3), (40, 2), (40, 3), (45, 3), (50, 5)]


def _artifact_paths(prefix: str, stamp: str) -> Dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "summary_json": ARTIFACT_DIR / f"{prefix}_summary_{stamp}.json",
        "summary_md": ARTIFACT_DIR / f"{prefix}_summary_{stamp}.md",
        "run_config_yaml": ARTIFACT_DIR / f"{prefix}_run_config_{stamp}.yaml",
        "valid_selection_csv": ARTIFACT_DIR / f"{prefix}_valid_selection_{stamp}.csv",
        "test_report_csv": ARTIFACT_DIR / f"{prefix}_test_report_{stamp}.csv",
        "prediction_pkl": ARTIFACT_DIR / f"{prefix}_test_prediction_{stamp}.pkl",
        "hard_gate_json": ARTIFACT_DIR / f"{prefix}_hard_gate_{stamp}.json",
        "verification_json": ARTIFACT_DIR / f"{prefix}_verification_{stamp}.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict LightGBM Alpha360 local-data hard-gate runner.")
    parser.add_argument("--mode", choices=["smoke", "full", "verify"], default="smoke")
    parser.add_argument("--output-prefix", default="lightgbm_alpha360_strict")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    prefix = f"{args.output_prefix}_{args.mode}"
    paths = _artifact_paths(prefix, stamp)
    test_range = SMOKE_TEST_RANGE if args.mode == "smoke" else TEST_RANGE

    summary: Dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "mode": args.mode,
        "status": "started",
        "verdict": "ERROR",
        "hard_gate_pass": False,
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
        "protocol": {
            "objective": "strict LightGBM Alpha360 tree baseline using local .qmData",
            "train": TRAIN_RANGE,
            "valid_select": VALID_RANGE,
            "test": TEST_RANGE,
            "evaluated_test_range": test_range,
            "no_2024_2026_tuning": True,
            "provider_uri": str(LOCAL_PROVIDER),
            "market": MARKET,
            "benchmark": BENCHMARK,
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET, "finite_rows_eq": HARD_GATE_ROWS},
        },
        "lightgbm_alpha360_config_evidence": _extract_lgbm_alpha360_evidence(LIGHTGBM_ALPHA360_CONFIG),
        "calendar_coverage": _calendar_coverage(LOCAL_PROVIDER),
    }

    try:
        _init_quant_master()
        dataset = _build_dataset(test_range, args.mode)
        run_config = {
            "quant_master_init": {"provider_uri": str(LOCAL_PROVIDER), "region": "cn", "kernels": 1, "joblib_backend": "threading"},
            "model_kwargs": {**BASE_MODEL_KWARGS, **(SMOKE_MODEL_EXTRAS if args.mode == "smoke" else FULL_MODEL_EXTRAS), "seed": args.seed},
            "dataset_segments": dataset.segments,
            "strategy_grid": [{"topk": topk, "n_drop": n_drop} for topk, n_drop in _strategy_grid(args.mode)],
        }
        _dump_yaml(run_config, paths["run_config_yaml"])

        model, train_info = _train_model(dataset, args.mode, args.seed)
        valid_pred = _predict(model, dataset, "valid")
        test_pred = _predict(model, dataset, "test")
        valid_label = _get_label(dataset, "valid")
        test_label = _get_label(dataset, "test")

        valid_signal_metrics = _daily_rank_ic(valid_pred, valid_label)
        test_signal_metrics = _daily_rank_ic(test_pred, test_label)
        exchange_cache: Dict[Tuple[str, str], Any] = {}
        valid_rows: List[Dict[str, Any]] = []
        best_valid: Dict[str, Any] | None = None
        for topk, n_drop in _strategy_grid(args.mode):
            t0 = time.perf_counter()
            try:
                _, metrics = _run_backtest(valid_pred, VALID_RANGE[0], VALID_RANGE[1], topk, n_drop, exchange_cache)
                row = {"topk": topk, "n_drop": n_drop, **metrics, "elapsed_sec": time.perf_counter() - t0, "error": ""}
            except Exception as exc:  # noqa: BLE001
                row = {
                    "topk": topk,
                    "n_drop": n_drop,
                    "costed_annret": float("nan"),
                    "costed_ir": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "rows": 0,
                    "finite_rows": 0,
                    "nonfinite_rows": 0,
                    "elapsed_sec": time.perf_counter() - t0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            valid_rows.append(row)
            if not row["error"] and np.isfinite(_safe_float(row["costed_ir"])) and np.isfinite(_safe_float(row["costed_annret"])):
                if best_valid is None or (_safe_float(row["costed_ir"]), _safe_float(row["costed_annret"])) > (
                    _safe_float(best_valid["costed_ir"]),
                    _safe_float(best_valid["costed_annret"]),
                ):
                    best_valid = row

        pd.DataFrame(valid_rows).to_csv(paths["valid_selection_csv"], index=False)
        if best_valid is None:
            raise RuntimeError("no valid 2023 strategy candidate produced finite net-cost metrics")

        test_report, test_metrics = _run_backtest(
            test_pred,
            test_range[0],
            test_range[1],
            int(best_valid["topk"]),
            int(best_valid["n_drop"]),
            exchange_cache,
        )
        test_report.to_csv(paths["test_report_csv"])
        test_pred.to_pickle(paths["prediction_pkl"])

        verification_metrics: Dict[str, Any] = {}
        verification_ok = False
        hard_gate_pass = bool(
            args.mode != "smoke"
            and int(test_metrics.get("finite_rows", -1)) == HARD_GATE_ROWS
            and int(test_metrics.get("nonfinite_rows", -1)) == 0
            and _safe_float(test_metrics.get("costed_ir")) > HARD_GATE_IR
            and _safe_float(test_metrics.get("costed_annret")) > HARD_GATE_ANNRET
        )
        if hard_gate_pass:
            _, verification_metrics = _run_backtest(
                test_pred,
                test_range[0],
                test_range[1],
                int(best_valid["topk"]),
                int(best_valid["n_drop"]),
                {},
            )
            verification_ok = bool(
                int(verification_metrics.get("finite_rows", -1)) == int(test_metrics.get("finite_rows", -2))
                and abs(_safe_float(verification_metrics.get("costed_ir")) - _safe_float(test_metrics.get("costed_ir"))) <= 1e-12
                and abs(_safe_float(verification_metrics.get("costed_annret")) - _safe_float(test_metrics.get("costed_annret"))) <= 1e-12
            )
            if not verification_ok:
                raise RuntimeError("hard gate passed but verification rerun did not match")

        summary.update(
            {
                "status": "ok",
                "verdict": "PASS" if hard_gate_pass else "NO_GO",
                "hard_gate_pass": hard_gate_pass,
                "train_info": train_info,
                "validation_signal_metrics": valid_signal_metrics,
                "test_signal_metrics": test_signal_metrics,
                "valid_selection_rule": "max valid_2023 real net-cost IR, tie by AnnRet; no test metrics used",
                "selected_strategy": {"topk": int(best_valid["topk"]), "n_drop": int(best_valid["n_drop"]), "valid_2023": best_valid},
                "test_metrics": test_metrics,
                "verification_ok": verification_ok,
                "verification_metrics": verification_metrics,
                "runtime_sec": float(time.perf_counter() - started),
            }
        )
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "verdict": "ERROR",
                "hard_gate_pass": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec": float(time.perf_counter() - started),
            }
        )

    hard_gate_payload = {
        "passed": bool(summary.get("hard_gate_pass")),
        "verification_ok": bool(summary.get("verification_ok", False)),
        "status": summary.get("status"),
        "verdict": summary.get("verdict"),
        "mode": args.mode,
        "costs": {"open_cost": OPEN_COST, "close_cost": CLOSE_COST},
        "gate": summary["protocol"]["hard_gate"],
        "selected_strategy": summary.get("selected_strategy", {}),
        "test_metrics": summary.get("test_metrics", {}),
        "summary_json": str(paths["summary_json"]),
    }
    paths["hard_gate_json"].write_text(json.dumps(_jsonable(hard_gate_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["verification_json"].write_text(
        json.dumps(
            _jsonable(
                {
                    "verification_ok": bool(summary.get("verification_ok", False)),
                    "verification_metrics": summary.get("verification_metrics", {}),
                    "hard_gate_pass": bool(summary.get("hard_gate_pass")),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["summary_json"].write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        f"# LightGBM Alpha360 Strict {args.mode} {stamp}",
        "",
        f"- status: `{summary.get('status')}`",
        f"- verdict: `{summary.get('verdict')}`",
        f"- hard_gate_pass: `{summary.get('hard_gate_pass')}`",
        f"- verification_ok: `{summary.get('verification_ok', False)}`",
        f"- finite_rows: `{summary.get('test_metrics', {}).get('finite_rows')}`",
        f"- costed_ir: `{summary.get('test_metrics', {}).get('costed_ir')}`",
        f"- costed_annret: `{summary.get('test_metrics', {}).get('costed_annret')}`",
        f"- selected_strategy: `{summary.get('selected_strategy', {})}`",
        f"- blocker: `{summary.get('blocker', '')}`",
        f"- summary_json: `{paths['summary_json']}`",
        f"- hard_gate_json: `{paths['hard_gate_json']}`",
    ]
    paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "verdict": summary.get("verdict"),
                "hard_gate_pass": summary.get("hard_gate_pass"),
                "summary_json": str(paths["summary_json"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
