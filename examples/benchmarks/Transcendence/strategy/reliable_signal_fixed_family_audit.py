#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import finite_price_universe_replay as replay


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
OPEN_COST = 0.0001
CLOSE_COST = 0.0006
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
EXPECTED_REPORT_DAYS = 562


@dataclass(frozen=True)
class FamilyMember:
    candidate_id: str
    signal_key: str
    topk: int
    n_drop: int
    protocol_class: str
    strict_eligible: bool
    rationale: str
    prior_script_evidence: str
    blend_gru_weight: float | None = None


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


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
        for row in rows:
            writer.writerow(_jsonable(row))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _required_finite_report(report: pd.DataFrame, expected_days: int) -> Tuple[bool, Dict[str, Any], pd.Series]:
    required = ["return", "bench", "cost", "turnover"]
    check: Dict[str, Any] = {
        "expected_report_days": int(expected_days),
        "rows": int(len(report)),
        "finite_rows": 0,
        "nonfinite_rows": int(len(report)),
        "report_days_match": False,
        "complete_finite_report": False,
        "first_report_day": "",
        "last_report_day": "",
        "missing_columns": [],
    }
    if report.empty:
        return False, check, pd.Series(dtype=float, name="excess")
    idx = pd.to_datetime(report.index)
    check["first_report_day"] = str(idx.min().date())
    check["last_report_day"] = str(idx.max().date())
    missing = [c for c in required if c not in report.columns]
    check["missing_columns"] = missing
    if missing:
        return False, check, pd.Series(dtype=float, name="excess")

    numeric = report[required].apply(pd.to_numeric, errors="coerce")
    excess = (numeric["return"] - numeric["bench"] - numeric["cost"]).astype(float).rename("excess")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1) & np.isfinite(excess.to_numpy(dtype=float))
    finite_rows = int(finite_mask.sum())
    check["finite_rows"] = finite_rows
    check["nonfinite_rows"] = int(len(report) - finite_rows)
    check["report_days_match"] = bool(len(report) == int(expected_days))
    check["complete_finite_report"] = bool(check["report_days_match"] and finite_rows == len(report))
    return bool(check["complete_finite_report"]), check, excess


def _split_rows(candidate_id: str, protocol_class: str, excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, start, end in [
        ("test_full", "2024-01-01", TEST_END),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", TEST_END),
    ]:
        part = excess.loc[(excess.index >= pd.Timestamp(start)) & (excess.index <= pd.Timestamp(end))]
        if part.empty:
            continue
        metrics = replay._metrics_from_excess(part)
        rows.append(
            {
                "candidate_id": candidate_id,
                "protocol_class": protocol_class,
                "split": split,
                "start": start,
                "end": end,
                "days": int(len(part)),
                "annret": float(metrics["annret"]),
                "ir": float(metrics["ir"]),
                "max_drawdown": float(metrics["max_drawdown"]),
            }
        )
    return rows


def _fixed_family() -> List[FamilyMember]:
    return [
        FamilyMember(
            candidate_id="base40_control",
            signal_key="base40",
            topk=40,
            n_drop=2,
            protocol_class="strict_fixed_non_test_selected",
            strict_eligible=True,
            rationale="Fixed base run 7406e470 TopkDropoutStrategy topk=40,n_drop=2; no test-period candidate selection.",
            prior_script_evidence="finite_price_universe_replay.py and base_gru_strict_blend_search.py use base40_control.",
        ),
        FamilyMember(
            candidate_id="gru45_control",
            signal_key="gru45",
            topk=45,
            n_drop=4,
            protocol_class="strict_fixed_non_test_selected",
            strict_eligible=True,
            rationale="Fixed GRU/rank ensemble weights 0.40/0.20/0.40 with topk=45,n_drop=4; no test-period candidate selection.",
            prior_script_evidence="finite_price_universe_replay.py and intra2024_forward_blend_gate.py use gru45.",
        ),
        FamilyMember(
            candidate_id="blend25_audit",
            signal_key="blend25",
            topk=40,
            n_drop=2,
            protocol_class="audit_only_fixed_fusion_no_pre2024_selection",
            strict_eligible=False,
            blend_gru_weight=0.25,
            rationale="Fixed base40/gru45 rank blend from prior code, but no unavailable pre-2024 selection proof for fusion weight.",
            prior_script_evidence="intra2024_forward_blend_gate.py predeclares blend25.",
        ),
        FamilyMember(
            candidate_id="blend50_audit",
            signal_key="blend50",
            topk=40,
            n_drop=2,
            protocol_class="audit_only_fixed_fusion_no_pre2024_selection",
            strict_eligible=False,
            blend_gru_weight=0.50,
            rationale="Fixed base40/gru45 rank blend from prior code, but no unavailable pre-2024 selection proof for fusion weight.",
            prior_script_evidence="intra2024_forward_blend_gate.py predeclares blend50.",
        ),
        FamilyMember(
            candidate_id="blend75_audit",
            signal_key="blend75",
            topk=40,
            n_drop=2,
            protocol_class="audit_only_fixed_fusion_no_pre2024_selection",
            strict_eligible=False,
            blend_gru_weight=0.75,
            rationale="Fixed base40/gru45 rank blend from prior code, but no unavailable pre-2024 selection proof for fusion weight.",
            prior_script_evidence="intra2024_forward_blend_gate.py predeclares blend75.",
        ),
        FamilyMember(
            candidate_id="base40_tk35_nd2_audit",
            signal_key="base40",
            topk=35,
            n_drop=2,
            protocol_class="audit_only_base_variation_no_pre2024_selection",
            strict_eligible=False,
            rationale="Base signal topk/n_drop variation is in prior sensitivity grid, but this full-window audit does not select it from pre-2024.",
            prior_script_evidence="topk_sensitivity_past_selected.py predeclares topk in (35,40,45) and n_drop in (2,4,6).",
        ),
        FamilyMember(
            candidate_id="base40_tk45_nd4_audit",
            signal_key="base40",
            topk=45,
            n_drop=4,
            protocol_class="audit_only_base_variation_no_pre2024_selection",
            strict_eligible=False,
            rationale="Base signal topk/n_drop variation is in prior sensitivity grid, but this full-window audit does not select it from pre-2024.",
            prior_script_evidence="topk_sensitivity_past_selected.py predeclares topk in (35,40,45) and n_drop in (2,4,6).",
        ),
    ]


def _make_replay_spec(
    member: FamilyMember,
    signals: Dict[str, pd.DataFrame],
    source_paths: Dict[str, Tuple[str, ...]],
) -> replay.CandidateSpec:
    return replay.CandidateSpec(
        candidate_id=member.candidate_id,
        signal_kind="topk_dropout",
        topk=int(member.topk),
        n_drop=int(member.n_drop),
        signal=signals[member.signal_key],
        source_paths=source_paths[member.signal_key],
        notes=f"{member.protocol_class}; {member.rationale}",
    )


def _evaluate_member(
    *,
    member: FamilyMember,
    spec: replay.CandidateSpec,
    port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str, Tuple[str, ...]], Any],
    universe: Sequence[str],
    out_dir: Path,
    expected_report_days: int,
    phase: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    result = replay._eval_candidate(
        candidate=spec,
        port_cfg=port_cfg,
        base_strategy_kwargs=base_strategy_kwargs,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        start=TEST_START,
        end=TEST_END,
        exchange_cache=exchange_cache,
        universe=universe,
        out_dir=out_dir,
    )
    row: Dict[str, Any] = {
        "candidate_id": member.candidate_id,
        "phase": phase,
        "protocol_class": member.protocol_class,
        "strict_eligible": bool(member.strict_eligible),
        "topk": int(member.topk),
        "n_drop": int(member.n_drop),
        "blend_gru_weight": member.blend_gru_weight if member.blend_gru_weight is not None else "",
        "rationale": member.rationale,
        "prior_script_evidence": member.prior_script_evidence,
        "status": result.get("status", "error"),
    }
    split_rows: List[Dict[str, Any]] = []
    if result.get("status") != "ok":
        row.update(
            {
                "complete_finite_report": False,
                "hard_gate_pass": False,
                "strict_pass": False,
                "audit_only_pass": False,
                "error_type": result.get("error_type", ""),
                "error_message": result.get("error", ""),
            }
        )
        return row, split_rows

    report_path = Path(result["artifacts"]["report_pkl"])
    report = _load_pickle(report_path)
    report.index = pd.to_datetime(report.index)
    complete, finite_check, excess = _required_finite_report(report, expected_report_days)
    if complete:
        split_rows = _split_rows(member.candidate_id, member.protocol_class, excess)
    metrics = result["metrics"]
    hard_gate = bool(
        complete
        and float(metrics["ir"]) > HARD_GATE_IR
        and float(metrics["annret"]) > HARD_GATE_ANNRET
    )
    row.update(
        {
            "complete_finite_report": bool(complete),
            **finite_check,
            "annret": float(metrics["annret"]),
            "ir": float(metrics["ir"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "turnover": float(metrics["turnover"]),
            "elapsed_sec": float(metrics["elapsed_sec"]),
            "hard_gate_pass": hard_gate,
            "strict_pass": bool(hard_gate and member.strict_eligible),
            "audit_only_pass": bool(hard_gate and not member.strict_eligible),
            "report_pkl": str(report_path),
            "report_csv": str(result["artifacts"]["report_csv"]),
            "positions_pkl": str(result["artifacts"]["positions_pkl"]),
            "indicators_pkl": str(result["artifacts"]["indicators_pkl"]),
            "error_type": "" if complete else "IncompleteFiniteReport",
            "error_message": "" if complete else json.dumps(finite_check, ensure_ascii=False),
        }
    )
    return row, split_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bounded fixed-family audit for reliable complete finite signals only.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--output-prefix", default="reliable_signal_fixed_family_audit")
    p.add_argument("--market", default=replay.MARKET)
    p.add_argument("--audit-start", default=replay.AUDIT_START)
    p.add_argument("--audit-end", default=replay.AUDIT_END)
    p.add_argument("--open-cost", type=float, default=OPEN_COST)
    p.add_argument("--close-cost", type=float, default=CLOSE_COST)
    p.add_argument("--expected-report-days", type=int, default=EXPECTED_REPORT_DAYS)
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = SCRIPT_DIR / f"{args.output_prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    candidates_csv = out_dir / f"{args.output_prefix}_candidates_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    verification_csv = out_dir / f"{args.output_prefix}_verification_{stamp}.csv"
    coverage_csv = out_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    excluded_csv = out_dir / f"{args.output_prefix}_excluded_{stamp}.csv"
    family_csv = out_dir / f"{args.output_prefix}_family_{stamp}.csv"

    tracking_dir = replay._parse_tracking_dir(args.tracking_uri)
    base_run_dir = replay._find_run_dir(tracking_dir, replay.BASE_RUN_ID)
    base_cfg = replay._load_config(base_run_dir / "artifacts" / "config")
    replay._init_quant_master(base_cfg)
    port_cfg = replay._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base_signal_all = replay._load_score_df(base_run_dir)
    coverage_start, coverage_end, coverage_days = replay._date_range(base_signal_all)
    blend_start = min(coverage_start, pd.Timestamp(TEST_START))
    blend_end = max(coverage_end, pd.Timestamp(TEST_END))
    gru_signal_all = replay._rank_ensemble(tracking_dir, replay.GRU45_RUN_IDS, replay.GRU45_RUN_WEIGHTS, blend_start, blend_end)

    base_signal = replay._slice_df(base_signal_all, TEST_START, TEST_END)
    gru_signal = replay._slice_df(gru_signal_all, TEST_START, TEST_END)
    signals = {
        "base40": base_signal,
        "gru45": gru_signal,
        "blend25": replay._blend_base_gru(base_signal, gru_signal, 0.25),
        "blend50": replay._blend_base_gru(base_signal, gru_signal, 0.50),
        "blend75": replay._blend_base_gru(base_signal, gru_signal, 0.75),
    }
    source_paths = {
        "base40": (str(base_run_dir / "artifacts" / "pred.pkl"),),
        "gru45": tuple(str(replay._find_run_dir(tracking_dir, rid) / "artifacts" / "pred.pkl") for rid in replay.GRU45_RUN_IDS),
        "blend25": tuple(str(replay._find_run_dir(tracking_dir, rid) / "artifacts" / "pred.pkl") for rid in replay.GRU45_RUN_IDS),
        "blend50": tuple(str(replay._find_run_dir(tracking_dir, rid) / "artifacts" / "pred.pkl") for rid in replay.GRU45_RUN_IDS),
        "blend75": tuple(str(replay._find_run_dir(tracking_dir, rid) / "artifacts" / "pred.pkl") for rid in replay.GRU45_RUN_IDS),
    }

    universe, coverage_rows, excluded_rows, audit_meta = replay._build_universe_audit(
        market=args.market,
        audit_start=args.audit_start,
        audit_end=args.audit_end,
    )
    replay._write_csv(coverage_csv, coverage_rows)
    replay._write_csv(excluded_csv, excluded_rows)

    family = _fixed_family()
    _write_csv(family_csv, [member.__dict__ for member in family])

    exchange_cache: Dict[Tuple[str, str, float, float, float, str, Tuple[str, ...]], Any] = {}
    candidate_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    for member in family:
        spec = _make_replay_spec(member, signals, source_paths)
        row, member_splits = _evaluate_member(
            member=member,
            spec=spec,
            port_cfg=port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
            universe=universe,
            out_dir=out_dir,
            expected_report_days=int(args.expected_report_days),
            phase="primary",
        )
        candidate_rows.append(row)
        split_rows.extend(member_splits)

    strict_pass_ids = [r["candidate_id"] for r in candidate_rows if bool(r.get("strict_pass"))]
    audit_pass_ids = [r["candidate_id"] for r in candidate_rows if bool(r.get("audit_only_pass"))]
    verification_rows: List[Dict[str, Any]] = []
    if strict_pass_ids:
        verify_dir = out_dir / "verification"
        verify_dir.mkdir(parents=True, exist_ok=False)
        for member in family:
            if member.candidate_id not in strict_pass_ids:
                continue
            verify_member = FamilyMember(
                candidate_id=f"{member.candidate_id}_verify",
                signal_key=member.signal_key,
                topk=member.topk,
                n_drop=member.n_drop,
                protocol_class=member.protocol_class,
                strict_eligible=member.strict_eligible,
                rationale=member.rationale,
                prior_script_evidence=member.prior_script_evidence,
                blend_gru_weight=member.blend_gru_weight,
            )
            spec = _make_replay_spec(verify_member, signals, source_paths)
            verify_cache: Dict[Tuple[str, str, float, float, float, str, Tuple[str, ...]], Any] = {}
            row, _ = _evaluate_member(
                member=verify_member,
                spec=spec,
                port_cfg=port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=verify_cache,
                universe=universe,
                out_dir=verify_dir,
                expected_report_days=int(args.expected_report_days),
                phase="strict_pass_verification",
            )
            verification_rows.append(row)

    _write_csv(candidates_csv, candidate_rows)
    _write_csv(splits_csv, split_rows)
    _write_csv(verification_csv, verification_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "reliable_signal_fixed_family_audit",
        "objective": "One bounded strict candidate generation pass using reliable complete finite reports only.",
        "verdict": "STRICT_PASS" if strict_pass_ids else ("AUDIT_ONLY_PASS" if audit_pass_ids else "NO_PASS"),
        "strict_hard_gate_pass": bool(strict_pass_ids),
        "strict_pass_candidates": strict_pass_ids,
        "audit_only_hard_gate_pass_candidates": audit_pass_ids,
        "hard_gate": {
            "test_start": TEST_START,
            "test_end": TEST_END,
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "expected_report_days": int(args.expected_report_days),
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
        },
        "protocol": {
            "strict_definition": "strict_fixed_non_test_selected candidates only; no full-test or unavailable pre-2024 selection.",
            "audit_only_definition": "fixed candidates present in prior scripts but not admissibly selected before the 2024-2026 test window.",
            "finite_report_gate": "candidate cannot pass unless return/bench/cost/turnover/excess are finite for exactly 562 report rows.",
            "reliable_signals_included": ["base40 run pred.pkl", "gru45 rank ensemble 7406e470/1a085ff9/773bd6d"],
            "explicitly_excluded": {
                "factor_augmented_meta": "excluded until its finite report issue is solved; this script does not load or evaluate it"
            },
        },
        "filter_definition": audit_meta,
        "universe": {"market": args.market, "size": int(len(universe))},
        "source_meta": {
            "tracking_dir": str(tracking_dir),
            "base_run_id": replay.BASE_RUN_ID,
            "base_pred_pkl": str(base_run_dir / "artifacts" / "pred.pkl"),
            "gru45_run_ids": list(replay.GRU45_RUN_IDS),
            "gru45_run_weights": list(replay.GRU45_RUN_WEIGHTS),
            "base_coverage_start": str(coverage_start.date()),
            "base_coverage_end": str(coverage_end.date()),
            "base_coverage_days": int(coverage_days),
        },
        "fixed_family": [member.__dict__ for member in family],
        "candidate_results": candidate_rows,
        "split_metrics": split_rows,
        "verification_results": verification_rows,
        "counts": {
            "family_size": int(len(family)),
            "candidate_rows": int(len(candidate_rows)),
            "complete_finite_candidates": int(sum(bool(r.get("complete_finite_report")) for r in candidate_rows)),
            "strict_pass_count": int(len(strict_pass_ids)),
            "audit_only_pass_count": int(len(audit_pass_ids)),
            "verification_rows": int(len(verification_rows)),
        },
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "family_csv": str(family_csv),
            "candidates_csv": str(candidates_csv),
            "splits_csv": str(splits_csv),
            "verification_csv": str(verification_csv),
            "coverage_csv": str(coverage_csv),
            "excluded_csv": str(excluded_csv),
        },
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Reliable Signal Fixed Family Audit {stamp}",
                "",
                f"- verdict: `{summary['verdict']}`",
                f"- strict_hard_gate_pass: `{bool(strict_pass_ids)}`",
                f"- strict_pass_candidates: `{json.dumps(strict_pass_ids, ensure_ascii=False)}`",
                f"- audit_only_hard_gate_pass_candidates: `{json.dumps(audit_pass_ids, ensure_ascii=False)}`",
                f"- complete_finite_candidates: `{summary['counts']['complete_finite_candidates']}` / `{len(family)}`",
                f"- summary_json: `{summary_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "strict_hard_gate_pass": bool(strict_pass_ids),
                "strict_pass_candidates": strict_pass_ids,
                "audit_only_hard_gate_pass_candidates": audit_pass_ids,
                "complete_finite_candidates": summary["counts"]["complete_finite_candidates"],
                "summary_json": str(summary_json),
                "candidates_csv": str(candidates_csv),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

