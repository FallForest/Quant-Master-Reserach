#!/usr/bin/env python
from __future__ import annotations

import argparse
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

import factor_meta_gru_base_fusion_lockstep as fm
import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
TARGET_ROWS = 562
BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
GRU45_RUN_IDS = (
    "7406e47063e9479cb34d300b9ed03bad",
    "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d8413b4bb0b388a63a6b5b6a86",
)
GRU45_RUN_WEIGHTS = (0.4, 0.2, 0.4)
FACTOR_PRED_NAME = "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl"
FACTOR_SUMMARY_NAME = "factor_augmented_meta_summary_20260522T120515Z.json"
FIXED_CANDIDATE_ID = "f80_g20"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _finite_metrics(report: pd.DataFrame) -> Dict[str, Any]:
    required = ["return", "bench", "cost", "turnover"]
    missing = [c for c in required if c not in report.columns]
    if missing:
        raise KeyError(f"report missing required columns: {missing}")
    numeric = report[required].apply(pd.to_numeric, errors="coerce")
    excess = numeric["return"] - numeric["bench"] - numeric["cost"]
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        excess.to_numpy(dtype=float)
    )
    finite_rows = int(finite_mask.sum())
    nonfinite_rows = int(len(report) - finite_rows)
    if int(len(report)) != TARGET_ROWS:
        raise ValueError(f"expected {TARGET_ROWS} report rows, found {len(report)}")
    if finite_rows != TARGET_ROWS:
        raise ValueError(f"expected {TARGET_ROWS} finite rows, found {finite_rows}")
    risk_df = risk_analysis(excess.loc[finite_mask].astype(float).sort_index(), freq="1day")
    return {
        "costed_annret": float(risk_df.loc["annualized_return", "risk"]),
        "costed_ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(numeric.loc[finite_mask, "turnover"].mean()),
        "rows": int(len(report)),
        "finite_rows": int(finite_rows),
        "nonfinite_rows": int(nonfinite_rows),
        "excess": excess.loc[finite_mask].astype(float),
    }


def _split_metrics(report: pd.DataFrame) -> List[Dict[str, Any]]:
    split_defs = [
        ("test_full", TEST_START, TEST_END),
        ("2024", TEST_START, pd.Timestamp("2024-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]
    rows: List[Dict[str, Any]] = []
    dates = pd.to_datetime(report.index)
    for split, start, end in split_defs:
        mask = (dates >= start) & (dates <= end)
        part = report.loc[mask].copy()
        required = ["return", "bench", "cost", "turnover"]
        numeric = part[required].apply(pd.to_numeric, errors="coerce")
        excess = numeric["return"] - numeric["bench"] - numeric["cost"]
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
            excess.to_numpy(dtype=float)
        )
        if int(finite_mask.sum()) != int(len(part)):
            raise ValueError(f"non-finite split rows detected for {split}: {len(part) - int(finite_mask.sum())}")
        risk_df = risk_analysis(excess.loc[finite_mask].astype(float).sort_index(), freq="1day")
        rows.append(
            {
                "candidate_id": FIXED_CANDIDATE_ID,
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "days": int(len(part)),
                "costed_annret": float(risk_df.loc["annualized_return", "risk"]),
                "costed_ir": float(risk_df.loc["information_ratio", "risk"]),
                "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
                "turnover": float(numeric.loc[finite_mask, "turnover"].mean()),
            }
        )
    return rows


def _scan_pred_coverage(tracking_dir: Path) -> Dict[str, Any]:
    earliest: pd.Timestamp | None = None
    earliest_path = ""
    scanned = 0
    for pred_path in tracking_dir.rglob("pred.pkl"):
        scanned += 1
        try:
            pred_obj = conv._load_pickle(pred_path)
            pred_df = conv._as_score_df(pred_obj)
            idx = pred_df.index
            if isinstance(idx, pd.MultiIndex):
                dates = pd.to_datetime(idx.get_level_values(0))
            else:
                dates = pd.to_datetime(idx)
            if len(dates) == 0:
                continue
            local_earliest = pd.Timestamp(dates.min()).normalize()
            if earliest is None or local_earliest < earliest:
                earliest = local_earliest
                earliest_path = str(pred_path)
        except Exception:
            continue
    return {
        "files_scanned": int(scanned),
        "earliest_pred_date": str(earliest.date()) if earliest is not None else "",
        "earliest_pred_path": earliest_path,
        "pre2024_diagnostics_available": bool(earliest is not None and earliest < pd.Timestamp("2024-01-01")),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed predeclared replay for regime-switch stability fallback.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--factor-pred-pkl", default="")
    p.add_argument("--factor-summary-json", default="")
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--selected-candidate-id", default=FIXED_CANDIDATE_ID)
    p.add_argument("--output-prefix", default="pre2024_regime_switch_replay")
    p.add_argument("--verify-summary-json", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.perf_counter()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(base_cfg)
    base_port_cfg = conv._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    factor_pred_path = Path(args.factor_pred_pkl).expanduser().resolve() if args.factor_pred_pkl else THIS_DIR / FACTOR_PRED_NAME
    factor_summary_path = (
        Path(args.factor_summary_json).expanduser().resolve()
        if args.factor_summary_json
        else THIS_DIR / FACTOR_SUMMARY_NAME
    )

    factor_df = conv._as_score_df(conv._load_pickle(factor_pred_path)).sort_index()
    base_df = conv._as_score_df(conv._load_pickle(base_dir / "artifacts" / "pred.pkl")).sort_index()
    coverage_start = min(pd.Timestamp(fm._coverage(factor_df)["start"]), pd.Timestamp(fm._coverage(base_df)["start"]))
    coverage_end = max(pd.Timestamp(fm._coverage(factor_df)["end"]), pd.Timestamp(fm._coverage(base_df)["end"]))
    gru_df = fm._rank_ensemble(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, coverage_start, coverage_end)

    specs = fm._candidate_family()
    spec_map = {spec.candidate_id: spec for spec in specs}
    if args.selected_candidate_id not in spec_map:
        raise KeyError(f"unknown fixed candidate: {args.selected_candidate_id}")

    signals = {
        spec.candidate_id: fm._build_rank_fusion(spec, factor_df, gru_df, base_df)
        for spec in specs
    }
    selected_spec = spec_map[args.selected_candidate_id]
    selected_signal = signals[args.selected_candidate_id]

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    selected_eval = fm._eval_signal(
        pred_df=selected_signal,
        base_port_cfg=base_port_cfg,
        base_strategy_kwargs=base_strategy_kwargs,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        start_time=TEST_START,
        end_time=TEST_END,
        exchange_cache=exchange_cache,
    )
    report = selected_eval["report"].copy()
    report.index = pd.to_datetime(report.index)
    report = report.loc[(report.index >= TEST_START) & (report.index <= TEST_END)].copy()
    full_metrics = _finite_metrics(report)
    split_rows = _split_metrics(report)

    legacy_summary = {}
    if factor_summary_path.exists():
        try:
            legacy_summary = json.loads(factor_summary_path.read_text(encoding="utf-8"))
        except Exception:
            legacy_summary = {}

    scan_info = _scan_pred_coverage(tracking_dir)
    hard_gate_pass = bool(
        full_metrics["finite_rows"] == TARGET_ROWS
        and float(full_metrics["costed_ir"]) > HARD_GATE_IR
        and float(full_metrics["costed_annret"]) > HARD_GATE_ANNRET
    )
    admissible = bool(not scan_info["pre2024_diagnostics_available"])

    reference_candidate_id = "factor_only"
    reference_metrics = None
    if reference_candidate_id in signals:
        reference_eval = fm._eval_signal(
            pred_df=signals[reference_candidate_id],
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_time=TEST_START,
            end_time=TEST_END,
            exchange_cache=exchange_cache,
        )
        ref_report = reference_eval["report"].copy()
        ref_report.index = pd.to_datetime(ref_report.index)
        ref_report = ref_report.loc[(ref_report.index >= TEST_START) & (ref_report.index <= TEST_END)].copy()
        ref_metrics = _finite_metrics(ref_report)
        reference_metrics = {
            "candidate_id": reference_candidate_id,
            "costed_annret": float(ref_metrics["costed_annret"]),
            "costed_ir": float(ref_metrics["costed_ir"]),
            "max_drawdown": float(ref_metrics["max_drawdown"]),
            "turnover": float(ref_metrics["turnover"]),
            "rows": int(ref_metrics["rows"]),
            "finite_rows": int(ref_metrics["finite_rows"]),
        }

    out_dir = THIS_DIR / f"{args.output_prefix}_{_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = out_dir / f"{args.output_prefix}_report.csv"
    report_pkl = out_dir / f"{args.output_prefix}_report.pkl"
    split_csv = out_dir / f"{args.output_prefix}_split_metrics.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary.json"
    summary_md = out_dir / f"{args.output_prefix}_summary.md"
    selected_signal_pkl = out_dir / f"{args.output_prefix}_selected_signal.pkl"

    report.to_csv(report_csv)
    with report_pkl.open("wb") as f:
        pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)
    with selected_signal_pkl.open("wb") as f:
        pickle.dump(selected_signal, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_csv(split_csv, split_rows)

    verification: Dict[str, Any] = {
        "mode": bool(args.verify_summary_json),
        "matched": None,
        "reference_summary_json": args.verify_summary_json,
    }
    if args.verify_summary_json:
        ref_summary = json.loads(Path(args.verify_summary_json).read_text(encoding="utf-8"))
        ref_full = ref_summary.get("full_metrics", {})
        verification["matched"] = bool(
            abs(float(ref_full.get("costed_ir", float("nan"))) - float(full_metrics["costed_ir"])) <= 1e-12
            and abs(float(ref_full.get("costed_annret", float("nan"))) - float(full_metrics["costed_annret"]))
            <= 1e-12
            and int(ref_full.get("rows", -1)) == int(full_metrics["rows"])
            and int(ref_full.get("finite_rows", -1)) == int(full_metrics["finite_rows"])
        )
        verification["reference_full_metrics"] = {
            "costed_ir": float(ref_full.get("costed_ir", float("nan"))),
            "costed_annret": float(ref_full.get("costed_annret", float("nan"))),
            "rows": int(ref_full.get("rows", -1)),
            "finite_rows": int(ref_full.get("finite_rows", -1)),
        }

    summary = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "fixed predeclared replay for old regime_switch_stability fallback",
        "rule_source": {
            "family_script": str(THIS_DIR / "factor_meta_gru_base_fusion_lockstep.py"),
            "selected_candidate_id": str(args.selected_candidate_id),
            "selection_policy": "hardcoded predeclared candidate; no test-window tuning",
            "pre2024_diagnostics_available": bool(scan_info["pre2024_diagnostics_available"]),
            "pre2024_scan": scan_info,
            "limitation": (
                "No pre-2024 diagnostics or predictions were found in scanned artifacts, so this replay "
                "is a fixed predeclared fallback rather than a pre-2024-frozen regime rule."
            ),
        },
        "constants": {
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "candidate_weights": {
                "factor": float(selected_spec.factor_weight),
                "gru": float(selected_spec.gru_weight),
                "base": float(selected_spec.base_weight),
            },
            "strategy_execution": fm._strategy_combo(),
        },
        "coverage": {
            "factor_augmented_meta": fm._coverage(factor_df),
            "base40": fm._coverage(base_df),
            "gru45_rank_source": fm._coverage(gru_df),
        },
        "full_metrics": {
            "candidate_id": str(args.selected_candidate_id),
            **{k: float(v) for k, v in full_metrics.items() if k not in {"excess"}},
        },
        "split_metrics": split_rows,
        "reference_metrics": reference_metrics,
        "hard_gate": {
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
            "passed": bool(hard_gate_pass),
        },
        "admissible": bool(admissible),
        "verification": verification,
        "selected_signal_rows": int(len(selected_signal)),
        "selected_signal_start": str(pd.Timestamp(selected_signal.index.get_level_values(0).min()).date())
        if isinstance(selected_signal.index, pd.MultiIndex)
        else str(pd.Timestamp(selected_signal.index.min()).date()),
        "selected_signal_end": str(pd.Timestamp(selected_signal.index.get_level_values(0).max()).date())
        if isinstance(selected_signal.index, pd.MultiIndex)
        else str(pd.Timestamp(selected_signal.index.max()).date()),
        "artifacts": {
            "report_csv": str(report_csv),
            "report_pkl": str(report_pkl),
            "split_metrics_csv": str(split_csv),
            "selected_signal_pkl": str(selected_signal_pkl),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
        "runtime_sec": float(time.perf_counter() - t0),
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                "# Pre2024 Regime Switch Replay",
                "",
                f"- selected candidate: `{args.selected_candidate_id}`",
                f"- hard gate pass: `{hard_gate_pass}`",
                f"- admissible: `{admissible}`",
                f"- pre-2024 diagnostics available: `{scan_info['pre2024_diagnostics_available']}`",
                f"- full IR: `{full_metrics['costed_ir']:.6f}`",
                f"- full AnnRet: `{full_metrics['costed_annret']:.6f}`",
                f"- report rows: `{full_metrics['rows']}`",
                f"- finite rows: `{full_metrics['finite_rows']}`",
                f"- summary: `{summary_json}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

