#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.contrib.evaluate import risk_analysis
from quant_master.model.trainer import task_train
from quant_master.workflow import R

TARGET_CONFIG = THIS_DIR / "workflow_config_regime_horizon_cost_ensemble_Alpha158Alpha360_2026_csi300.yaml"
TRAIN_RANGE = ["2020-01-01", "2022-12-31"]
VALID_RANGE = ["2023-01-01", "2023-12-31"]
TEST_RANGE = ["2024-01-01", "2026-04-30"]
SMOKE_TEST_RANGE = ["2024-01-01", "2024-03-31"]
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
HARD_GATE_ROWS = 562


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
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return val if math.isfinite(val) else None
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _load_config(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


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


def _apply_common_overrides(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    provider_uri = Path(str(cfg["quant_master_init"]["provider_uri"])).expanduser()
    if not provider_uri.is_absolute():
        provider_uri = (REPO_ROOT / provider_uri).resolve()
    cfg["quant_master_init"]["provider_uri"] = str(provider_uri)

    handler_cfg = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_cfg.update(
        {
            "start_time": "2019-01-01",
            "end_time": TEST_RANGE[1],
            "fit_start_time": TRAIN_RANGE[0],
            "fit_end_time": TRAIN_RANGE[1],
        }
    )
    if "data_handler_config" in cfg:
        cfg["data_handler_config"].update(handler_cfg)

    cfg["task"]["dataset"]["kwargs"]["segments"] = {
        "train": list(TRAIN_RANGE),
        "valid": list(VALID_RANGE),
        "test": list(TEST_RANGE if mode != "smoke" else SMOKE_TEST_RANGE),
    }

    port_cfg = _find_port_config(cfg)
    port_cfg["backtest"]["start_time"] = TEST_RANGE[0] if mode != "smoke" else SMOKE_TEST_RANGE[0]
    port_cfg["backtest"]["end_time"] = TEST_RANGE[1] if mode != "smoke" else SMOKE_TEST_RANGE[1]
    port_cfg["backtest"]["exchange_kwargs"]["open_cost"] = OPEN_COST
    port_cfg["backtest"]["exchange_kwargs"]["close_cost"] = CLOSE_COST

    if mode == "smoke":
        model_kwargs = cfg["task"]["model"]["kwargs"]
        model_kwargs["horizon_model_specs"] = model_kwargs.get("horizon_model_specs", [])[:2]
        model_kwargs["search_step"] = 0.5
        model_kwargs["memory_boost_grid"] = [0.0]
        model_kwargs["regime_consensus_quantiles"] = [0.5]
        model_kwargs["regime_disagreement_quantiles"] = [0.5]
        model_kwargs["min_regime_samples"] = 120
        for spec in model_kwargs["horizon_model_specs"]:
            mk = spec.get("model_kwargs", {})
            if spec.get("model_type") == "double_ensemble":
                mk["num_models"] = min(int(mk.get("num_models", 2)), 1)
                mk["epochs"] = min(int(mk.get("epochs", 8)), 4)
                mk["sub_weights"] = list(mk.get("sub_weights", [1]))[: int(mk["num_models"])]
                mk["enable_sr"] = False
                mk["enable_fs"] = False
                mk["num_threads"] = min(int(mk.get("num_threads", 8)), 4)
            if spec.get("model_type") in {"lightgbm", "lgb", "lgbm"}:
                mk["num_boost_round"] = min(int(mk.get("num_boost_round", 200)), 80)
                mk["early_stopping_rounds"] = min(int(mk.get("early_stopping_rounds", 30)), 20)
                mk["num_threads"] = min(int(mk.get("num_threads", 8)), 4)
        port_cfg["strategy"]["kwargs"]["topk"] = min(int(port_cfg["strategy"]["kwargs"].get("topk", 30)), 20)
        port_cfg["strategy"]["kwargs"]["n_drop"] = min(int(port_cfg["strategy"]["kwargs"].get("n_drop", 3)), 1)
    else:
        # With the mandated 2023-only validation split, some predeclared regimes can be sparse.
        # Learning every regime avoids empty fallback weights without using 2024-2026 data.
        cfg["task"]["model"]["kwargs"]["min_regime_samples"] = 0
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
    vals: List[float] = []
    for _, g in panel.groupby(level=0, sort=False):
        if len(g) < 20:
            continue
        corr = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(corr) and np.isfinite(corr):
            vals.append(float(corr))
    if len(vals) < 2:
        return {"error": "insufficient rank_ic days", "rank_ic_days": len(vals)}
    s = pd.Series(vals, dtype=float)
    return {
        "rank_ic": float(s.mean()),
        "rank_ic_ir": float(s.mean() / (s.std(ddof=1) + 1e-12) * np.sqrt(252.0)),
        "rank_ic_days": int(len(s)),
    }


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
    p.add_argument("--mode", choices=["smoke", "full", "verify"], default="smoke")
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
        run_config = _apply_common_overrides(base_config, args.mode)
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
                "validation_metrics": signal_metrics,
                "test_metrics": full_test_metrics,
                "split_metrics": split_metrics,
                "runtime_sec": float(time.perf_counter() - started),
            }
        )
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
