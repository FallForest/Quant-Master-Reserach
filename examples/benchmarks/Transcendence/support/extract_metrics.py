#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd


@dataclass
class RunContext:
    run_id: str
    artifact_uri: Optional[str]
    metrics: Dict[str, float]
    runtime_sec: Optional[float]


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _find_artifact_file(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.exists():
        return direct
    for p in root.rglob(name):
        return p
    return None


def _risk_from_port_analysis(df: pd.DataFrame, key: str) -> Optional[float]:
    if not isinstance(df.index, pd.MultiIndex):
        return None
    idx = ("excess_return_with_cost", key)
    if idx in df.index and "risk" in df.columns:
        return _to_float(df.loc[idx, "risk"])
    return None


def _turnover_from_report(df: pd.DataFrame) -> Optional[float]:
    if "turnover" not in df.columns:
        return None
    return _to_float(df["turnover"].mean())


def _runtime_sec(run) -> Optional[float]:
    st = run.info.start_time
    ed = run.info.end_time
    if st is None or ed is None:
        return None
    return (ed - st) / 1000.0


def _build_client(tracking_uri: str) -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def _pick_run(client: mlflow.MlflowClient, experiment_name: str, recorder_name: Optional[str]) -> RunContext:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"experiment not found: {experiment_name}")
    filter_string = ""
    if recorder_name:
        filter_string = f"tags.`mlflow.runName` = '{recorder_name}'"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=1,
        order_by=["attribute.start_time DESC"],
    )
    if not runs:
        raise ValueError(f"no runs found in experiment: {experiment_name}")
    run = runs[0]
    return RunContext(
        run_id=run.info.run_id,
        artifact_uri=run.info.artifact_uri,
        metrics=dict(run.data.metrics),
        runtime_sec=_runtime_sec(run),
    )


def _get_run(client: mlflow.MlflowClient, run_id: str) -> RunContext:
    run = client.get_run(run_id)
    return RunContext(
        run_id=run.info.run_id,
        artifact_uri=run.info.artifact_uri,
        metrics=dict(run.data.metrics),
        runtime_sec=_runtime_sec(run),
    )


def _resolve_local_artifact_dir(args, run_ctx: Optional[RunContext], client: Optional[mlflow.MlflowClient]) -> Optional[Path]:
    if args.artifact_dir:
        p = Path(args.artifact_dir).expanduser().resolve()
        return p if p.exists() else None
    if run_ctx and client:
        try:
            local = client.download_artifacts(run_ctx.run_id, "")
            return Path(local).resolve()
        except Exception:
            return None
    return None


def _extract(args) -> Dict[str, Any]:
    run_ctx: Optional[RunContext] = None
    client: Optional[mlflow.MlflowClient] = None

    if args.run_id or args.experiment_name:
        client = _build_client(args.tracking_uri)
        if args.run_id:
            run_ctx = _get_run(client, args.run_id)
        else:
            run_ctx = _pick_run(client, args.experiment_name, args.recorder_name)

    metrics = run_ctx.metrics if run_ctx else {}

    ic = _to_float(metrics.get("IC"))
    rank_ic = _to_float(metrics.get("Rank IC"))
    costed_annret = _to_float(metrics.get("1day.excess_return_with_cost.annualized_return"))
    costed_ir = _to_float(metrics.get("1day.excess_return_with_cost.information_ratio"))
    max_drawdown = _to_float(metrics.get("1day.excess_return_with_cost.max_drawdown"))
    turnover = _to_float(metrics.get("1day.turnover"))

    artifact_root = _resolve_local_artifact_dir(args, run_ctx, client)
    if artifact_root:
        port_path = _find_artifact_file(artifact_root, "port_analysis_1day.pkl")
        if port_path and (max_drawdown is None or costed_annret is None or costed_ir is None):
            port_df = _load_pickle(port_path)
            max_drawdown = max_drawdown if max_drawdown is not None else _risk_from_port_analysis(port_df, "max_drawdown")
            costed_annret = (
                costed_annret
                if costed_annret is not None
                else _risk_from_port_analysis(port_df, "annualized_return")
            )
            costed_ir = (
                costed_ir if costed_ir is not None else _risk_from_port_analysis(port_df, "information_ratio")
            )

        report_path = _find_artifact_file(artifact_root, "report_normal_1day.pkl")
        if report_path and turnover is None:
            report_df = _load_pickle(report_path)
            turnover = _turnover_from_report(report_df)

    result = {
        "run_id": run_ctx.run_id if run_ctx else args.run_id,
        "model_name": args.model_name,
        "workflow_config": args.workflow_config,
        "ic": ic,
        "rank_ic": rank_ic,
        "costed_annret": costed_annret,
        "costed_ir": costed_ir,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "runtime_sec": run_ctx.runtime_sec if run_ctx else None,
    }
    return result


def _write_csv(row: Dict[str, Any], path: Path) -> None:
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Transcendence leaderboard metrics from QuantMaster recorder artifacts."
    )
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", help="Experiment name to query latest run.")
    parser.add_argument("--recorder-name", default=None, help="Optional run name filter (mlflow.runName).")
    parser.add_argument("--run-id", help="Run ID to extract.")
    parser.add_argument("--artifact-dir", help="Local artifact directory (alternative to MLflow query).")
    parser.add_argument("--model-name", default="", help="Model name for leaderboard row.")
    parser.add_argument("--workflow-config", default="", help="Workflow config path for leaderboard row.")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV output file path.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.run_id and not args.experiment_name and not args.artifact_dir:
        parser.error("one of --run-id / --experiment-name / --artifact-dir is required")

    row = _extract(args)
    if args.format == "json":
        print(json.dumps(row, ensure_ascii=False))
    else:
        if args.csv_path:
            out = Path(args.csv_path).expanduser().resolve()
            _write_csv(row, out)
            print(str(out))
        else:
            print(",".join(row.keys()))
            print(",".join("" if v is None else str(v) for v in row.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
