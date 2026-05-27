#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TRANS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PRE2024_START = pd.Timestamp("2020-01-01")
PRE2024_END_EXCLUSIVE = pd.Timestamp("2024-01-01")
TARGET_YEARS = (2020, 2021, 2022, 2023)
KEYWORDS = ("long_history", "second_order", "factor_meta", "factor_augmented")
PRED_TOKENS = ("candidate_pred", "selected_signal", "pred")
CORR_GATE = 0.85


@dataclass
class SignalArtifact:
    key: str
    path: Path
    source_kind: str
    companion_csv: Optional[Path] = None


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _norm_key(path: Path) -> str:
    name = path.stem
    for suffix in ("_candidate_pred", "_selected_signal"):
        if suffix in name:
            name = name.split(suffix)[0]
    return name


def _is_relevant(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".csv", ".pkl"}:
        return False
    return any(k in name for k in KEYWORDS) and any(t in name for t in PRED_TOKENS)


def _discover_artifacts(trans_dir: Path) -> List[SignalArtifact]:
    paths = sorted(p for p in trans_dir.rglob("*") if p.is_file() and _is_relevant(p))
    by_stem: Dict[str, Dict[str, Path]] = {}
    for p in paths:
        by_stem.setdefault(str(p.with_suffix("")), {})[p.suffix.lower()] = p

    artifacts: List[SignalArtifact] = []
    seen: set[Path] = set()
    for stem, parts in sorted(by_stem.items()):
        csv_path = parts.get(".csv")
        pkl_path = parts.get(".pkl")
        if csv_path is not None:
            artifacts.append(SignalArtifact(key=_norm_key(csv_path), path=csv_path, source_kind="csv"))
            seen.add(csv_path)
            if pkl_path is not None:
                seen.add(pkl_path)
        elif pkl_path is not None:
            artifacts.append(SignalArtifact(key=_norm_key(pkl_path), path=pkl_path, source_kind="pkl_unsafe", companion_csv=None))
            seen.add(pkl_path)
    return artifacts


def _csv_cols(fieldnames: Optional[Sequence[str]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    fields = list(fieldnames or [])
    lower = {c.lower(): c for c in fields}
    dt_col = lower.get("datetime") or lower.get("date") or lower.get("time")
    inst_col = lower.get("instrument") or lower.get("symbol") or lower.get("code")
    score_col = lower.get("score") or lower.get("pred") or lower.get("prediction")
    if score_col is None:
        for col in fields:
            if col not in {dt_col, inst_col}:
                score_col = col
                break
    return dt_col, inst_col, score_col


def _empty_row(artifact: SignalArtifact, status: str, note: str) -> Dict[str, Any]:
    return {
        "key": artifact.key,
        "path": str(artifact.path.resolve()),
        "source_kind": artifact.source_kind,
        "status": status,
        "note": note,
        "min_date": "",
        "max_date": "",
        "rows_total_seen_until_stop": 0,
        "rows_pre2024_finite": 0,
        "days_pre2024": 0,
        "names_pre2024": 0,
        "has_2020": 0,
        "has_2021": 0,
        "has_2022": 0,
        "has_2023": 0,
        "complete_2020_2023": 0,
        "first_2024_or_later_date_seen": "",
        "used_for_gate": 0,
    }


def _read_csv_pre2024(artifact: SignalArtifact) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
    row = _empty_row(artifact, "ok", "")
    values: List[Tuple[pd.Timestamp, str, float]] = []
    years: set[int] = set()
    dates: set[pd.Timestamp] = set()
    names: set[str] = set()
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None
    first_2024: Optional[pd.Timestamp] = None
    rows_seen = 0

    try:
        with artifact.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            dt_col, inst_col, score_col = _csv_cols(reader.fieldnames)
            if dt_col is None or inst_col is None or score_col is None:
                return _empty_row(artifact, "bad_schema", f"columns={reader.fieldnames}"), None

            for raw in reader:
                rows_seen += 1
                dt = pd.to_datetime(raw.get(dt_col, ""), errors="coerce")
                if pd.isna(dt):
                    continue
                dt = pd.Timestamp(dt).normalize()
                min_date = dt if min_date is None else min(min_date, dt)
                max_date = dt if max_date is None else max(max_date, dt)

                if dt >= PRE2024_END_EXCLUSIVE:
                    first_2024 = dt
                    # Candidate prediction CSVs are date sorted in this workspace. Stop before
                    # reading any 2024+ scores into the audit panel.
                    break
                if dt < PRE2024_START:
                    continue

                score = pd.to_numeric(raw.get(score_col, ""), errors="coerce")
                if not math.isfinite(float(score)):
                    continue
                inst = str(raw.get(inst_col, "")).strip()
                if not inst:
                    continue
                values.append((dt, inst, float(score)))
                years.add(int(dt.year))
                dates.add(dt)
                names.add(inst)
    except Exception as exc:  # noqa: BLE001
        return _empty_row(artifact, type(exc).__name__, str(exc)), None

    row.update(
        {
            "min_date": "" if min_date is None else str(min_date.date()),
            "max_date": "" if max_date is None else str(max_date.date()),
            "rows_total_seen_until_stop": rows_seen,
            "rows_pre2024_finite": len(values),
            "days_pre2024": len(dates),
            "names_pre2024": len(names),
            "has_2020": int(2020 in years),
            "has_2021": int(2021 in years),
            "has_2022": int(2022 in years),
            "has_2023": int(2023 in years),
            "complete_2020_2023": int(all(y in years for y in TARGET_YEARS)),
            "first_2024_or_later_date_seen": "" if first_2024 is None else str(first_2024.date()),
            "note": "csv scan stopped at first 2024+ row; no 2024+ score panel used" if first_2024 else "",
        }
    )

    if not values:
        return row, None

    idx = pd.MultiIndex.from_tuples([(dt, inst) for dt, inst, _ in values], names=["datetime", "instrument"])
    series = pd.Series([score for _, _, score in values], index=idx, name=artifact.key).sort_index()
    return row, series


def _safe_pickle_metadata(artifact: SignalArtifact) -> Dict[str, Any]:
    row = _empty_row(
        artifact,
        "skipped_pkl_no_csv",
        "pkl-only artifact skipped: cannot safely filter before loading; no 2024+ read/eval",
    )
    return row


def _mean_daily_rank_corr(a: pd.Series, b: pd.Series) -> Tuple[float, int, int, int]:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if both.empty:
        return float("nan"), 0, 0, 0
    corrs: List[float] = []
    overlap_names: set[str] = set()
    for _, g in both.groupby(level="datetime", sort=True):
        if len(g) < 2:
            continue
        corr = g["a"].rank(method="average").corr(g["b"].rank(method="average"), method="pearson")
        if pd.notna(corr):
            corrs.append(float(corr))
            overlap_names.update(g.index.get_level_values("instrument").astype(str))
    return (float(np.mean(corrs)) if corrs else float("nan"), len(corrs), len(overlap_names), len(both))


def _correlation_rows(signals: Dict[str, pd.Series], eligible_keys: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = list(eligible_keys)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            corr, days, names, rows_n = _mean_daily_rank_corr(signals[left], signals[right])
            rows.append(
                {
                    "signal_a": left,
                    "signal_b": right,
                    "overlap_days": days,
                    "overlap_names": names,
                    "overlap_rows": rows_n,
                    "mean_daily_cross_sectional_rank_corr": corr,
                    "below_0_85": int(math.isfinite(corr) and abs(corr) < CORR_GATE),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def _write_md(path: Path, summary: Dict[str, Any], coverage_rows: Sequence[Dict[str, Any]], corr_rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Pre-2024 Second-Order Coverage Smoke",
        "",
        f"- timestamp_utc: `{summary['timestamp_utc']}`",
        f"- verdict: `{summary['verdict']}`",
        f"- gate_pass: `{summary['gate_pass']}`",
        f"- legal_pre2024_selection_possible: `{summary['legal_pre2024_selection_possible']}`",
        f"- no_2024_plus_used_for_gate: `{summary['no_2024_plus_used_for_gate']}`",
        f"- eligible_complete_signals: `{summary['eligible_complete_signal_count']}`",
        f"- max_abs_pairwise_corr: `{summary['max_abs_pairwise_corr']}`",
        "",
        "## Coverage",
        "",
        "| key | status | min_date | max_date | rows_pre2024_finite | days | names | 2020 | 2021 | 2022 | 2023 | complete | first_2024+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage_rows:
        lines.append(
            "| {key} | {status} | {min_date} | {max_date} | {rows_pre2024_finite} | {days_pre2024} | "
            "{names_pre2024} | {has_2020} | {has_2021} | {has_2022} | {has_2023} | {complete_2020_2023} | "
            "{first_2024_or_later_date_seen} |".format(**row)
        )
    lines.extend(["", "## Pairwise Correlation", ""])
    if corr_rows:
        lines.extend(
            [
                "| signal_a | signal_b | overlap_days | overlap_names | corr | below_0_85 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in corr_rows:
            lines.append(
                f"| {row['signal_a']} | {row['signal_b']} | {row['overlap_days']} | {row['overlap_names']} | "
                f"{row['mean_daily_cross_sectional_rank_corr']} | {row['below_0_85']} |"
            )
    else:
        lines.append("No compliant complete 2020-2023 signal pair was available for correlation.")
    lines.extend(["", "## Recommendation", "", summary["next_step_recommendation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-2024 coverage/correlation smoke for long-history second-order candidates.")
    parser.add_argument("--output-prefix", default="pre2024_second_order_coverage_smoke")
    parser.add_argument("--artifact-root", default=str(TRANS_DIR), help="Root to scan for relevant prediction artifacts.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = Path(args.artifact_root).resolve()
    output_prefix = Path(args.output_prefix)
    if output_prefix.is_absolute():
        out_base = output_prefix
    else:
        out_base = SCRIPT_DIR / output_prefix.name
    out_base.parent.mkdir(parents=True, exist_ok=True)

    stamp = _stamp()
    coverage_csv = out_base.parent / f"{out_base.name}_coverage_{stamp}.csv"
    corr_csv = out_base.parent / f"{out_base.name}_correlations_{stamp}.csv"
    summary_json = out_base.parent / f"{out_base.name}_summary_{stamp}.json"
    summary_md = out_base.parent / f"{out_base.name}_summary_{stamp}.md"

    artifacts = _discover_artifacts(artifact_root)
    coverage_rows: List[Dict[str, Any]] = []
    signals: Dict[str, pd.Series] = {}
    for artifact in artifacts:
        if artifact.source_kind == "csv":
            row, series = _read_csv_pre2024(artifact)
            coverage_rows.append(row)
            if series is not None:
                signals[artifact.key] = series
        else:
            coverage_rows.append(_safe_pickle_metadata(artifact))

    eligible = [
        row["key"]
        for row in coverage_rows
        if int(row.get("complete_2020_2023", 0)) == 1 and int(row.get("rows_pre2024_finite", 0)) > 0 and row["key"] in signals
    ]
    for row in coverage_rows:
        row["used_for_gate"] = int(row["key"] in eligible)

    corr_rows = _correlation_rows(signals, eligible) if len(eligible) >= 2 else []
    finite_corrs = [
        abs(float(row["mean_daily_cross_sectional_rank_corr"]))
        for row in corr_rows
        if math.isfinite(float(row["mean_daily_cross_sectional_rank_corr"]))
    ]
    max_abs_corr = max(finite_corrs) if finite_corrs else float("nan")
    low_corr_pair_exists = any(int(row["below_0_85"]) == 1 for row in corr_rows)
    gate_pass = len(eligible) >= 2 and low_corr_pair_exists
    legal_possible = gate_pass
    no_2024_plus_used_for_gate = True

    if gate_pass:
        verdict = "GO_SMOKE_ONLY"
        rec = (
            "Continue this line only with a pre-declared pre-2024 weight-selection protocol; "
            "do not run full 2024-2026 from this smoke."
        )
    else:
        verdict = "NO_GO"
        rec = (
            "Do not continue to weight selection: no two distinct compliant signals with complete 2020-2023 "
            "coverage and pairwise rank correlation below 0.85 were found. Generate a low-cost csi300 "
            "2020-2021 OOS panel only if the lead approves a bounded prediction job."
        )

    summary = {
        "timestamp_utc": _now_utc(),
        "task_id": "Q-LONG-HISTORY-PRE2024-COVERAGE-SMOKE",
        "artifact_root": str(artifact_root),
        "verdict": verdict,
        "gate_pass": gate_pass,
        "legal_pre2024_selection_possible": legal_possible,
        "no_2024_plus_used_for_gate": no_2024_plus_used_for_gate,
        "selection_window": {"start": "2020-01-01", "end_exclusive": "2024-01-01"},
        "gate": {
            "required_complete_signal_count": 2,
            "required_years": list(TARGET_YEARS),
            "pairwise_abs_rank_corr_lt": CORR_GATE,
        },
        "artifact_count": len(coverage_rows),
        "eligible_complete_signal_count": len(eligible),
        "eligible_complete_signals": eligible,
        "pairwise_correlation_count": len(corr_rows),
        "max_abs_pairwise_corr": None if not math.isfinite(max_abs_corr) else max_abs_corr,
        "paths": {
            "coverage_csv": str(coverage_csv.resolve()),
            "correlations_csv": str(corr_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
        },
        "next_step_recommendation": rec,
        "audit_safety": {
            "pkl_only_policy": "skipped because pkl cannot be filtered before loading",
            "csv_policy": "scan only pre-2024 rows and stop at first 2024+ row",
            "entry_scripts_reviewed": [
                str((TRANS_DIR / "support" / "long_history_model_retrain.py").resolve()),
                str((SCRIPT_DIR / "long_history_second_order_ensemble.py").resolve()),
                str((TRANS_DIR / "support" / "factor_augmented_meta_ensemble.py").resolve()),
            ],
            "entry_script_risk": "reviewed scripts target 2024-2026 or high-cost ensemble/backtest paths; smoke did not run them",
        },
    }

    _write_csv(coverage_csv, coverage_rows)
    _write_csv(corr_csv, corr_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_md(summary_md, summary, coverage_rows, corr_rows)

    print(json.dumps({"gate_pass": gate_pass, "verdict": verdict, "summary_json": str(summary_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
