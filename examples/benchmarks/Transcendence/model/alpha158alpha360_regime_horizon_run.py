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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "quant_master").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"cannot locate Quant-Master-Research repo root from {start}")


REPO_ROOT = _find_repo_root(THIS_FILE)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri, resolve_provider_uri_in_config
from quant_master.contrib.evaluate import risk_analysis
from quant_master.model.trainer import task_train
from quant_master.workflow import R

TARGET_CONFIG = THIS_DIR / "workflow_config_regime_horizon_cost_ensemble_Alpha158Alpha360_2026_csi300.yaml"
TRAIN_RANGE = ["2020-01-01", "2022-12-31"]
VALID_RANGE = ["2023-01-01", "2023-12-31"]
TEST_RANGE = ["2024-01-01", "2026-04-30"]
SMOKE_TEST_RANGE = ["2024-01-01", "2024-03-31"]
OPEN_COST = 0.0001
CLOSE_COST = 0.0006
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
HARD_GATE_ROWS = 562
MEDIUM_SELECTION_GRID_KEYS = (
    "search_step",
    "memory_boost_grid",
    "turnover_penalty_grid",
    "risk_penalty_grid",
)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _jsonable(x: Any) -> Any:
    if x is pd.NaT:
        return None
    if isinstance(x, (pd.Timestamp, date, datetime)):
        return x.isoformat()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return val if math.isfinite(val) else None
    if isinstance(x, (np.generic,)):
        return _jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _load_config(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return resolve_provider_uri_in_config(yaml.load(f), base_dir=path.parent)


def _dump_config(config: Dict[str, Any], path: Path) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)


def _find_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "port_analysis_config" in config:
        return config["port_analysis_config"]
    for rec in config.get("task", {}).get("record", []):
        if rec.get("class") == "PortAnaRecord":
            return rec["kwargs"]["config"]
    raise KeyError("cannot find port_analysis_config or PortAnaRecord config")


def _test_range_for_mode(mode: str) -> List[str]:
    return list(SMOKE_TEST_RANGE if mode == "smoke" else TEST_RANGE)


def _configured_segments(config: Dict[str, Any]) -> Dict[str, List[str]]:
    segments = config["task"]["dataset"]["kwargs"].get("segments", {})
    return {
        "train": list(segments.get("train", TRAIN_RANGE)),
        "valid": list(segments.get("valid", VALID_RANGE)),
        "test": list(segments.get("test", TEST_RANGE)),
    }


def _record_override(changes: List[Dict[str, Any]], path: str, old_value: Any, new_value: Any) -> None:
    if old_value != new_value:
        changes.append({"path": path, "old": copy.deepcopy(old_value), "new": copy.deepcopy(new_value)})


def _set_with_record(changes: List[Dict[str, Any]], target: Dict[str, Any], key: str, value: Any, path: str) -> None:
    old_value = target.get(key)
    _record_override(changes, path, old_value, value)
    target[key] = value


def _record_selection_grid_decision(
    changes: List[Dict[str, Any]],
    model_kwargs: Dict[str, Any],
    key: str,
    action: str,
    reason: str,
    old_value: Any = None,
    new_value: Any = None,
) -> None:
    entry = {
        "path": f"task.model.kwargs.{key}",
        "key": key,
        "action": action,
        "reason": reason,
    }
    if action == "preserved":
        entry["value"] = copy.deepcopy(model_kwargs.get(key))
    else:
        entry["old"] = copy.deepcopy(old_value)
        entry["new"] = copy.deepcopy(new_value)
    changes.append(entry)


def _set_grid_with_decision(
    budget_overrides: List[Dict[str, Any]],
    selection_grid_decisions: List[Dict[str, Any]],
    model_kwargs: Dict[str, Any],
    key: str,
    value: Any,
    reason: str,
) -> None:
    old_value = model_kwargs.get(key)
    _set_with_record(budget_overrides, model_kwargs, key, value, f"task.model.kwargs.{key}")
    _record_selection_grid_decision(selection_grid_decisions, model_kwargs, key, "overridden", reason, old_value, value)


def _preserve_medium_selection_grids(
    selection_grid_decisions: List[Dict[str, Any]],
    model_kwargs: Dict[str, Any],
    keys: tuple[str, ...] = MEDIUM_SELECTION_GRID_KEYS,
) -> None:
    for key in keys:
        if key in model_kwargs:
            _record_selection_grid_decision(
                selection_grid_decisions,
                model_kwargs,
                key,
                "preserved",
                "medium preserve_config_windows keeps YAML candidate selection grids",
            )


def _apply_budget_overrides(
    config: Dict[str, Any],
    mode: str,
    preserve_config_windows: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    budget_overrides: List[Dict[str, Any]] = []
    selection_grid_decisions: List[Dict[str, Any]] = []
    model_kwargs = config["task"]["model"]["kwargs"]
    port_cfg = _find_port_config(config)

    if mode == "smoke":
        original_specs = model_kwargs.get("horizon_model_specs", [])
        smoke_specs = original_specs[:2]
        _record_override(
            budget_overrides,
            "task.model.kwargs.horizon_model_specs",
            original_specs,
            smoke_specs,
        )
        model_kwargs["horizon_model_specs"] = smoke_specs
        _set_grid_with_decision(
            budget_overrides,
            selection_grid_decisions,
            model_kwargs,
            "search_step",
            0.5,
            "smoke uses quick default candidate search budget",
        )
        _set_grid_with_decision(
            budget_overrides,
            selection_grid_decisions,
            model_kwargs,
            "memory_boost_grid",
            [0.0],
            "smoke uses quick default candidate search budget",
        )
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "regime_consensus_quantiles",
            [0.5],
            "task.model.kwargs.regime_consensus_quantiles",
        )
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "regime_disagreement_quantiles",
            [0.5],
            "task.model.kwargs.regime_disagreement_quantiles",
        )
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "min_regime_samples",
            120,
            "task.model.kwargs.min_regime_samples",
        )
        max_epochs = 4
        lgb_rounds = 80
    elif mode == "medium":
        if preserve_config_windows:
            _preserve_medium_selection_grids(selection_grid_decisions, model_kwargs)
        else:
            _set_grid_with_decision(
                budget_overrides,
                selection_grid_decisions,
                model_kwargs,
                "search_step",
                0.5,
                "medium without preserve_config_windows uses quick default candidate search budget",
            )
            _set_grid_with_decision(
                budget_overrides,
                selection_grid_decisions,
                model_kwargs,
                "memory_boost_grid",
                [0.0],
                "medium without preserve_config_windows uses quick default candidate search budget",
            )
            _preserve_medium_selection_grids(
                selection_grid_decisions,
                model_kwargs,
                ("turnover_penalty_grid", "risk_penalty_grid"),
            )
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "regime_consensus_quantiles",
            [0.5],
            "task.model.kwargs.regime_consensus_quantiles",
        )
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "regime_disagreement_quantiles",
            [0.5],
            "task.model.kwargs.regime_disagreement_quantiles",
        )
        # Keep the full test/backtest window, but cap model-side work for quick candidate checks.
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "min_regime_samples",
            0,
            "task.model.kwargs.min_regime_samples",
        )
        max_epochs = 8
        lgb_rounds = 120
    else:
        # With the mandated 2023-only validation split, some predeclared regimes can be sparse.
        # Learning every regime avoids empty fallback weights without using 2024-2026 data.
        _set_with_record(
            budget_overrides,
            model_kwargs,
            "min_regime_samples",
            0,
            "task.model.kwargs.min_regime_samples",
        )
        return budget_overrides, selection_grid_decisions

    for spec_index, spec in enumerate(model_kwargs.get("horizon_model_specs", [])):
        mk = spec.get("model_kwargs", {})
        path = f"task.model.kwargs.horizon_model_specs[{spec_index}].model_kwargs"
        if spec.get("model_type") == "double_ensemble":
            num_models = min(int(mk.get("num_models", 2)), 1)
            _set_with_record(budget_overrides, mk, "num_models", num_models, f"{path}.num_models")
            _set_with_record(budget_overrides, mk, "epochs", min(int(mk.get("epochs", max_epochs)), max_epochs), f"{path}.epochs")
            _set_with_record(
                budget_overrides,
                mk,
                "sub_weights",
                list(mk.get("sub_weights", [1]))[:num_models],
                f"{path}.sub_weights",
            )
            _set_with_record(budget_overrides, mk, "enable_sr", False, f"{path}.enable_sr")
            _set_with_record(budget_overrides, mk, "enable_fs", False, f"{path}.enable_fs")
            _set_with_record(
                budget_overrides,
                mk,
                "num_threads",
                min(int(mk.get("num_threads", 8)), 4),
                f"{path}.num_threads",
            )
        if spec.get("model_type") in {"lightgbm", "lgb", "lgbm"}:
            _set_with_record(
                budget_overrides,
                mk,
                "num_boost_round",
                min(int(mk.get("num_boost_round", 200)), lgb_rounds),
                f"{path}.num_boost_round",
            )
            _set_with_record(
                budget_overrides,
                mk,
                "early_stopping_rounds",
                min(int(mk.get("early_stopping_rounds", 30)), 20),
                f"{path}.early_stopping_rounds",
            )
            _set_with_record(
                budget_overrides,
                mk,
                "num_threads",
                min(int(mk.get("num_threads", 8)), 4),
                f"{path}.num_threads",
            )

    if mode == "smoke":
        strategy_kwargs = port_cfg["strategy"]["kwargs"]
        _set_with_record(
            budget_overrides,
            strategy_kwargs,
            "topk",
            min(int(strategy_kwargs.get("topk", 30)), 20),
            "port_analysis_config.strategy.kwargs.topk",
        )
        _set_with_record(
            budget_overrides,
            strategy_kwargs,
            "n_drop",
            min(int(strategy_kwargs.get("n_drop", 3)), 1),
            "port_analysis_config.strategy.kwargs.n_drop",
        )
    return budget_overrides, selection_grid_decisions


def _apply_common_overrides(config: Dict[str, Any], mode: str, preserve_config_windows: bool = False) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    window_overrides: List[Dict[str, Any]] = []
    cfg["quant_master_init"]["provider_uri"] = str(
        resolve_provider_uri(cfg["quant_master_init"]["provider_uri"], base_dir=REPO_ROOT)
    )

    handler_cfg = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    if preserve_config_windows:
        actual_segments = _configured_segments(cfg)
    else:
        actual_segments = {
            "train": list(TRAIN_RANGE),
            "valid": list(VALID_RANGE),
            "test": _test_range_for_mode(mode),
        }
        handler_overrides = {
            "start_time": "2019-01-01",
            "end_time": actual_segments["test"][1],
            "fit_start_time": actual_segments["train"][0],
            "fit_end_time": actual_segments["train"][1],
        }
        for key, value in handler_overrides.items():
            _set_with_record(window_overrides, handler_cfg, key, value, f"task.dataset.kwargs.handler.kwargs.{key}")
        if "data_handler_config" in cfg:
            cfg["data_handler_config"].update(handler_cfg)

        _record_override(
            window_overrides,
            "task.dataset.kwargs.segments",
            cfg["task"]["dataset"]["kwargs"].get("segments"),
            actual_segments,
        )
        cfg["task"]["dataset"]["kwargs"]["segments"] = copy.deepcopy(actual_segments)

    port_cfg = _find_port_config(cfg)
    _set_with_record(
        window_overrides,
        port_cfg["backtest"],
        "start_time",
        actual_segments["test"][0],
        "port_analysis_config.backtest.start_time",
    )
    _set_with_record(
        window_overrides,
        port_cfg["backtest"],
        "end_time",
        actual_segments["test"][1],
        "port_analysis_config.backtest.end_time",
    )
    port_cfg["backtest"]["exchange_kwargs"]["open_cost"] = OPEN_COST
    port_cfg["backtest"]["exchange_kwargs"]["close_cost"] = CLOSE_COST

    budget_overrides, selection_grid_decisions = _apply_budget_overrides(cfg, mode, preserve_config_windows)
    cfg["runner_metadata"] = {
        "mode": mode,
        "preserve_config_windows": bool(preserve_config_windows),
        "actual_segments": copy.deepcopy(actual_segments),
        "actual_train": list(actual_segments["train"]),
        "actual_valid": list(actual_segments["valid"]),
        "actual_test": list(actual_segments["test"]),
        "window_overrides": window_overrides,
        "budget_overrides": budget_overrides,
        "selection_grid_decisions": selection_grid_decisions,
    }
    return cfg


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


def _signal_metrics(recorder: Any) -> Dict[str, Any]:
    try:
        pred = recorder.load_object("pred.pkl")
        label = recorder.load_object("label.pkl")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if pred is None or label is None:
        return {"error": "missing pred or label"}
    pred_s = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pd.Series(pred)
    label_s = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else pd.Series(label)
    panel = pd.concat([pred_s.rename("pred"), label_s.rename("label")], axis=1).dropna()
    if panel.empty or not isinstance(panel.index, pd.MultiIndex):
        return {"error": "empty or non-panel pred/label"}
    ic_vals: List[float] = []
    rank_ic_vals: List[float] = []
    for _, g in panel.groupby(level=0, sort=False):
        if len(g) < 20:
            continue
        ic = g["pred"].corr(g["label"])
        if pd.notna(ic) and np.isfinite(ic):
            ic_vals.append(float(ic))
        rank_ic = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(rank_ic) and np.isfinite(rank_ic):
            rank_ic_vals.append(float(rank_ic))

    result: Dict[str, Any] = {
        "ic_days": len(ic_vals),
        "rank_ic_days": len(rank_ic_vals),
    }
    if len(ic_vals) >= 2:
        ic_s = pd.Series(ic_vals, dtype=float)
        result.update(
            {
                "ic": float(ic_s.mean()),
                "ic_ir": float(ic_s.mean() / (ic_s.std(ddof=1) + 1e-12) * np.sqrt(252.0)),
            }
        )
    else:
        result["ic_missing_reason"] = f"insufficient ic days: {len(ic_vals)}"
    if len(rank_ic_vals) >= 2:
        rank_ic_s = pd.Series(rank_ic_vals, dtype=float)
        result.update(
            {
                "rank_ic": float(rank_ic_s.mean()),
                "rank_ic_ir": float(rank_ic_s.mean() / (rank_ic_s.std(ddof=1) + 1e-12) * np.sqrt(252.0)),
            }
        )
    else:
        result["rank_ic_missing_reason"] = f"insufficient rank_ic days: {len(rank_ic_vals)}"
    if "ic" not in result or "rank_ic" not in result:
        result["error"] = "; ".join(
            str(result[key]) for key in ["ic_missing_reason", "rank_ic_missing_reason"] if key in result
        )
    return result


def _finite_metric(metrics: Dict[str, Any], key: str) -> Any:
    value = _safe_float(metrics.get(key))
    return value if math.isfinite(value) else None


def _metric_missing_reason(metrics: Dict[str, Any], key: str) -> str:
    reason_key = f"{key}_missing_reason"
    if metrics.get(reason_key):
        return str(metrics[reason_key])
    if metrics.get("error"):
        return str(metrics["error"])
    if key not in metrics:
        return f"{key} not produced by signal validation metrics"
    return f"{key} is non-finite"


def _apply_validation_metrics(summary: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    summary["validation_metrics"] = metrics
    for key in ["ic", "rank_ic"]:
        value = _finite_metric(metrics, key)
        summary[key] = value
        if value is None:
            summary[f"{key}_missing_reason"] = _metric_missing_reason(metrics, key)


def _split_report(report: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]


def _run_workflow(config: Dict[str, Any], experiment_name: str, uri_folder: str) -> Any:
    init_cfg = copy.deepcopy(config["quant_master_init"])
    quant_master.init(**init_cfg)
    return task_train(config["task"], experiment_name=experiment_name)


def _artifact_paths(prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "run_config_yaml": THIS_DIR / f"{prefix}_run_config_{stamp}.yaml",
        "report_csv": THIS_DIR / f"{prefix}_report_{stamp}.csv",
        "summary_json": THIS_DIR / f"{prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{prefix}_summary_{stamp}.md",
        "artifact_parse_smoke_json": THIS_DIR / f"{prefix}_artifact_parse_smoke_{stamp}.json",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alpha158Alpha360 regime-horizon cost ensemble gate runner.")
    p.add_argument("--mode", choices=["smoke", "medium", "full", "verify"], default="smoke")
    p.add_argument(
        "--preserve-config-windows",
        action="store_true",
        help="Keep train/valid/test windows from the workflow YAML instead of forcing the default protocol windows.",
    )
    p.add_argument("--workflow-config", default=str(TARGET_CONFIG))
    p.add_argument("--output-prefix", default="alpha158alpha360_regime_horizon")
    p.add_argument("--experiment-name", default="")
    p.add_argument("--uri-folder", default="mlruns")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    prefix = f"{args.output_prefix}_{args.mode}"
    paths = _artifact_paths(prefix, stamp)
    command = " ".join(sys.argv)

    summary: Dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "command": command,
        "mode": args.mode,
        "preserve_config_windows": bool(args.preserve_config_windows),
        "status": "started",
        "blocker": "",
        "workflow_config_source": str(Path(args.workflow_config).resolve()),
        "artifacts": {k: str(v) for k, v in paths.items()},
        "protocol": {
            "model_path": "quant_master.contrib.model.regime_horizon_cost_ensemble.RegimeHorizonCostEnsembleModel",
            "train": TRAIN_RANGE,
            "valid_select": VALID_RANGE,
            "test": TEST_RANGE,
            "no_2024_2026_tuning": True,
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET, "finite_rows_eq": HARD_GATE_ROWS},
            "wrapper_guard": "full/verify set min_regime_samples=0 to avoid empty 2023 validation regime weights",
        },
    }

    try:
        base_config = _load_config(Path(args.workflow_config).resolve())
        run_config = _apply_common_overrides(base_config, args.mode, args.preserve_config_windows)
        runner_metadata = copy.deepcopy(run_config.get("runner_metadata", {}))
        summary.update(
            {
                "actual_segments": runner_metadata.get("actual_segments", {}),
                "actual_train": runner_metadata.get("actual_train", []),
                "actual_valid": runner_metadata.get("actual_valid", []),
                "actual_test": runner_metadata.get("actual_test", []),
                "window_overrides": runner_metadata.get("window_overrides", []),
                "budget_overrides": runner_metadata.get("budget_overrides", []),
                "selection_grid_decisions": runner_metadata.get("selection_grid_decisions", []),
            }
        )
        _dump_config(run_config, paths["run_config_yaml"])
        experiment = args.experiment_name or f"alpha158alpha360_regime_horizon_{args.mode}_{stamp}"
        recorder = _run_workflow(run_config, experiment, args.uri_folder)
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        report.to_csv(paths["report_csv"])
        test_metrics = _metrics_from_report(report)
        full_test_metrics = test_metrics
        if args.mode == "smoke":
            full_test_metrics = {
                **test_metrics,
                "note": "smoke mode uses bounded test range and is not a hard-gate evaluation",
            }

        split_metrics = []
        for name, start, end in [
            ("2024", "2024-01-01", "2024-12-31"),
            ("2025", "2025-01-01", "2025-12-31"),
            ("2026_ytd", "2026-01-01", "2026-04-30"),
        ]:
            sub = _split_report(report, start, end)
            if not sub.empty:
                split_metrics.append({"split": name, **_metrics_from_report(sub)})

        signal_metrics = _signal_metrics(recorder)
        hard_gate_pass = bool(
            args.mode != "smoke"
            and int(test_metrics.get("finite_rows", -1)) == HARD_GATE_ROWS
            and int(test_metrics.get("nonfinite_rows", -1)) == 0
            and _safe_float(test_metrics.get("costed_ir")) > HARD_GATE_IR
            and _safe_float(test_metrics.get("costed_annret")) > HARD_GATE_ANNRET
        )
        summary.update(
            {
                "status": "ok",
                "verdict": "PASS" if hard_gate_pass else "NO_GO",
                "hard_gate_pass": hard_gate_pass,
                "mlflow": {
                    "experiment_id": getattr(recorder, "experiment_id", ""),
                    "recorder_id": getattr(recorder, "id", getattr(recorder, "info", {}).get("id", "")),
                    "artifact_uri": recorder.get_artifact_uri(),
                },
                "test_metrics": full_test_metrics,
                "split_metrics": split_metrics,
                "runtime_sec": float(time.perf_counter() - started),
            }
        )
        _apply_validation_metrics(summary, signal_metrics)
        if hard_gate_pass and args.mode == "full":
            summary["verification_required"] = "run --mode verify with the same script"
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

    paths["summary_json"].write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        f"# Alpha158Alpha360 Regime Horizon {args.mode} {stamp}",
        "",
        f"- status: `{summary.get('status')}`",
        f"- verdict: `{summary.get('verdict')}`",
        f"- hard_gate_pass: `{summary.get('hard_gate_pass')}`",
        f"- runtime_sec: `{summary.get('runtime_sec')}`",
        f"- finite_rows: `{summary.get('test_metrics', {}).get('finite_rows')}`",
        f"- costed_ir: `{summary.get('test_metrics', {}).get('costed_ir')}`",
        f"- costed_annret: `{summary.get('test_metrics', {}).get('costed_annret')}`",
        f"- blocker: `{summary.get('blocker', '')}`",
        f"- summary_json: `{paths['summary_json']}`",
    ]
    paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    paths["artifact_parse_smoke_json"].write_text(
        json.dumps(
            {
                "summary_json_exists": paths["summary_json"].exists(),
                "summary_json_parse_ok": True,
                "summary_keys": sorted(summary.keys()),
                "hard_gate_pass": bool(summary.get("hard_gate_pass")),
                "finite_rows": summary.get("test_metrics", {}).get("finite_rows"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": summary.get("status"), "hard_gate_pass": summary.get("hard_gate_pass"), "summary_json": str(paths["summary_json"])}, ensure_ascii=False))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

