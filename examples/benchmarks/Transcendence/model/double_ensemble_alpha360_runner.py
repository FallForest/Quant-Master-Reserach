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
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "hard_gate_pass"
LOCAL_PROVIDER = (REPO_ROOT / ".qmData" / "cn_data").resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.contrib.evaluate import risk_analysis
from quant_master.model.trainer import task_train

CONFIGS = {
    "alpha158_baseline": REPO_ROOT
    / "examples"
    / "benchmarks"
    / "DoubleEnsemble"
    / "workflow_config_doubleensemble_Alpha158_2026_local.yaml",
    "alpha158_cost_aware": REPO_ROOT
    / "examples"
    / "benchmarks"
    / "BeatDoubleEnsemble"
    / "workflow_config_cost_aware_doubleensemble_Alpha158_2026_local.yaml",
    "alpha158_regime_de_only": THIS_DIR
    / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_windowC_2016_2022_valid2023_Alpha158_2026_csi300.yaml",
    "alpha360_de_only": THIS_DIR
    / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_universe_Alpha360_2026_csi300.yaml",
}

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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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


def _iter_model_kwargs(model_kwargs: Dict[str, Any]):
    yield model_kwargs
    nested = model_kwargs.get("double_ensemble_kwargs")
    if isinstance(nested, dict):
        yield nested
    for spec in model_kwargs.get("horizon_model_specs", []) or []:
        spec_kwargs = spec.get("model_kwargs", {})
        if isinstance(spec_kwargs, dict):
            yield spec_kwargs


def _apply_common_overrides(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("quant_master_init", {})
    cfg["quant_master_init"]["provider_uri"] = str(LOCAL_PROVIDER)
    cfg["quant_master_init"].setdefault("region", "cn")
    cfg["quant_master_init"]["kernels"] = 1
    cfg["quant_master_init"]["joblib_backend"] = "threading"

    handler_cfg = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_cfg["start_time"] = TRAIN_RANGE[0]
    handler_cfg["end_time"] = TEST_RANGE[1]
    handler_cfg["fit_start_time"] = TRAIN_RANGE[0]
    handler_cfg["fit_end_time"] = TRAIN_RANGE[1]
    if "data_handler_config" in cfg:
        cfg["data_handler_config"].update(handler_cfg)

    test_range = SMOKE_TEST_RANGE if mode == "smoke" else TEST_RANGE
    cfg["task"]["dataset"]["kwargs"]["segments"] = {
        "train": list(TRAIN_RANGE),
        "valid": list(VALID_RANGE),
        "test": list(test_range),
    }

    port_cfg = _find_port_config(cfg)
    port_cfg["backtest"]["start_time"] = test_range[0]
    port_cfg["backtest"]["end_time"] = test_range[1]
    port_cfg["backtest"]["exchange_kwargs"]["open_cost"] = OPEN_COST
    port_cfg["backtest"]["exchange_kwargs"]["close_cost"] = CLOSE_COST

    if mode == "smoke":
        port_cfg["strategy"]["kwargs"]["topk"] = min(int(port_cfg["strategy"]["kwargs"].get("topk", 50)), 20)
        port_cfg["strategy"]["kwargs"]["n_drop"] = min(int(port_cfg["strategy"]["kwargs"].get("n_drop", 5)), 1)

        for kwargs in _iter_model_kwargs(cfg["task"]["model"]["kwargs"]):
            if "num_models" in kwargs:
                kwargs["num_models"] = min(int(kwargs["num_models"]), 1)
            if "sub_weights" in kwargs:
                kwargs["sub_weights"] = list(kwargs["sub_weights"])[: int(kwargs.get("num_models", 1))]
            if "epochs" in kwargs:
                kwargs["epochs"] = min(int(kwargs["epochs"]), 4)
            if "enable_sr" in kwargs:
                kwargs["enable_sr"] = False
            if "enable_fs" in kwargs:
                kwargs["enable_fs"] = False
            if "num_threads" in kwargs:
                kwargs["num_threads"] = min(int(kwargs["num_threads"]), 4)
            if "residual_num_boost_round" in kwargs:
                kwargs["residual_num_boost_round"] = min(int(kwargs["residual_num_boost_round"]), 80)

        model_kwargs = cfg["task"]["model"]["kwargs"]
        if "horizon_model_specs" in model_kwargs:
            model_kwargs["horizon_model_specs"] = list(model_kwargs["horizon_model_specs"])[:1]
            model_kwargs["search_step"] = 1.0
            model_kwargs["memory_boost_grid"] = [0.0]
            model_kwargs["regime_consensus_quantiles"] = []
            model_kwargs["regime_disagreement_quantiles"] = []
            model_kwargs["min_regime_samples"] = 0
    else:
        model_kwargs = cfg["task"]["model"]["kwargs"]
        if "min_regime_samples" in model_kwargs:
            model_kwargs["min_regime_samples"] = 0

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


def _run_workflow(config: Dict[str, Any], experiment_name: str) -> Any:
    quant_master.init(**copy.deepcopy(config["quant_master_init"]))
    return task_train(config["task"], experiment_name=experiment_name)


def _artifact_paths(prefix: str, stamp: str) -> Dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "run_config_yaml": ARTIFACT_DIR / f"{prefix}_run_config_{stamp}.yaml",
        "report_csv": ARTIFACT_DIR / f"{prefix}_report_{stamp}.csv",
        "summary_json": ARTIFACT_DIR / f"{prefix}_summary_{stamp}.json",
        "summary_md": ARTIFACT_DIR / f"{prefix}_summary_{stamp}.md",
        "hard_gate_json": ARTIFACT_DIR / f"{prefix}_hard_gate_{stamp}.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DoubleEnsemble/BeatDoubleEnsemble hard-gate runner.")
    parser.add_argument("--mode", choices=["smoke", "full", "verify"], default="smoke")
    parser.add_argument("--candidate", choices=sorted(CONFIGS), default="alpha158_cost_aware")
    parser.add_argument("--workflow-config", default="")
    parser.add_argument("--output-prefix", default="double_ensemble_alpha360")
    parser.add_argument("--experiment-name", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    prefix = f"{args.output_prefix}_{args.candidate}_{args.mode}"
    paths = _artifact_paths(prefix, stamp)
    config_path = Path(args.workflow_config).resolve() if args.workflow_config else CONFIGS[args.candidate].resolve()

    summary: Dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "candidate": args.candidate,
        "mode": args.mode,
        "status": "started",
        "blocker": "",
        "workflow_config_source": str(config_path),
        "artifacts": {k: str(v) for k, v in paths.items()},
        "protocol": {
            "train": TRAIN_RANGE,
            "valid_select": VALID_RANGE,
            "test": TEST_RANGE,
            "smoke_test": SMOKE_TEST_RANGE,
            "local_provider": str(LOCAL_PROVIDER),
            "no_2024_2026_tuning": True,
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "hard_gate": {
                "ir_gt": HARD_GATE_IR,
                "annret_gt": HARD_GATE_ANNRET,
                "finite_rows_eq": HARD_GATE_ROWS,
            },
        },
    }

    try:
        base_config = _load_config(config_path)
        run_config = _apply_common_overrides(base_config, args.mode)
        _dump_config(run_config, paths["run_config_yaml"])
        experiment = args.experiment_name or f"double_ensemble_alpha360_{args.candidate}_{args.mode}_{stamp}"
        recorder = _run_workflow(run_config, experiment)
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        report.to_csv(paths["report_csv"])
        test_metrics = _metrics_from_report(report)

        split_metrics = []
        if args.mode != "smoke":
            for name, start, end in [
                ("2024", "2024-01-01", "2024-12-31"),
                ("2025", "2025-01-01", "2025-12-31"),
                ("2026_ytd", "2026-01-01", "2026-04-30"),
            ]:
                sub = _split_report(report, start, end)
                if not sub.empty:
                    split_metrics.append({"split": name, **_metrics_from_report(sub)})
        else:
            test_metrics = {
                **test_metrics,
                "note": "smoke mode uses a bounded test range and reduced model settings; not hard-gate eligible",
            }

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
                "test_metrics": test_metrics,
                "split_metrics": split_metrics,
                "runtime_sec": float(time.perf_counter() - started),
            }
        )
        if hard_gate_pass and args.mode == "full":
            summary["verification_required"] = "run --mode verify with the same candidate and workflow config"
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
        "status": summary.get("status"),
        "verdict": summary.get("verdict"),
        "candidate": args.candidate,
        "mode": args.mode,
        "metrics": summary.get("test_metrics", {}),
        "gate": summary["protocol"]["hard_gate"],
        "costs": {"open_cost": OPEN_COST, "close_cost": CLOSE_COST},
        "summary_json": str(paths["summary_json"]),
    }
    paths["hard_gate_json"].write_text(json.dumps(_jsonable(hard_gate_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        f"# DoubleEnsemble Alpha360 Runner {args.candidate} {args.mode} {stamp}",
        "",
        f"- status: `{summary.get('status')}`",
        f"- verdict: `{summary.get('verdict')}`",
        f"- hard_gate_pass: `{summary.get('hard_gate_pass')}`",
        f"- runtime_sec: `{summary.get('runtime_sec')}`",
        f"- finite_rows: `{summary.get('test_metrics', {}).get('finite_rows')}`",
        f"- costed_ir: `{summary.get('test_metrics', {}).get('costed_ir')}`",
        f"- costed_annret: `{summary.get('test_metrics', {}).get('costed_annret')}`",
        f"- blocker: `{summary.get('blocker', '')}`",
        f"- workflow_config_source: `{config_path}`",
        f"- summary_json: `{paths['summary_json']}`",
        f"- hard_gate_json: `{paths['hard_gate_json']}`",
    ]
    paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary.get("status"), "hard_gate_pass": summary.get("hard_gate_pass"), "summary_json": str(paths["summary_json"])}, ensure_ascii=False))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
