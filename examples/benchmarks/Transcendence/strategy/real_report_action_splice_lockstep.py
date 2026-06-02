#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_master.contrib.evaluate import risk_analysis


TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
OPEN_COST = 0.0001
CLOSE_COST = 0.0006
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
ACTION_IDS = ("base40", "gru45", "factor_augmented_meta")
SCORE_RULES = ("max_ir", "annret_minus_abs_mdd", "annret_gt_0p27_then_ir")
CAVEAT = (
    "Report-level splice across independently real action backtests; admissible only as an audit/search "
    "screen. A hard-gate pass needs full trade replay verification because boundary holdings, turnover, "
    "and costs are not re-executed after switching actions."
)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "replay_action_reports_cache_20260524T045740Z"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "return", "bench", "cost", "turnover"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required report columns: {sorted(missing)}")
    df = df.set_index("datetime").sort_index()
    return df


def _slice_report(report: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= s) & (idx <= e)].copy()


def _daily_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")


def _metrics(report: pd.DataFrame) -> Dict[str, Any]:
    report_rows = int(len(report))
    if report.empty:
        return {
            "report_rows": report_rows,
            "metric_days": 0,
            "days": 0,
            "coverage_complete": False,
            "annret": float("nan"),
            "ir": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
        }
    excess = _daily_excess(report).dropna().sort_index()
    metric_days = int(len(excess))
    coverage_complete = bool(metric_days == report_rows)
    if excess.empty:
        return {
            "report_rows": report_rows,
            "metric_days": metric_days,
            "days": metric_days,
            "coverage_complete": coverage_complete,
            "annret": float("nan"),
            "ir": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float(report["turnover"].astype(float).mean()),
        }
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "report_rows": report_rows,
        "metric_days": metric_days,
        "days": metric_days,
        "coverage_complete": coverage_complete,
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].astype(float).reindex(excess.index).mean()),
    }


def _finite_number(value: Any, default: float = -1.0e18) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _selection_windows() -> List[Dict[str, Any]]:
    return [
        {
            "apply_tag": "2024H2",
            "train_tag": "2024H1",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2024-06-30"),
            "apply_start": pd.Timestamp("2024-07-01"),
            "apply_end": pd.Timestamp("2024-12-31"),
        },
        {
            "apply_tag": "2025",
            "train_tag": "2024",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2024-12-31"),
            "apply_start": pd.Timestamp("2025-01-01"),
            "apply_end": pd.Timestamp("2025-12-31"),
        },
        {
            "apply_tag": "2026_ytd",
            "train_tag": "2024_2025",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2025-12-31"),
            "apply_start": pd.Timestamp("2026-01-01"),
            "apply_end": pd.Timestamp("2026-04-30"),
        },
    ]


def _split_specs() -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("full", TEST_START, TEST_END),
        ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]


def _select_action(
    train_rows: Sequence[Dict[str, Any]],
    score_rule: str,
    action_order: Dict[str, int],
) -> Tuple[str, List[Dict[str, Any]], str]:
    ranked: List[Dict[str, Any]] = []
    for row in train_rows:
        annret = _finite_number(row.get("annret"))
        ir = _finite_number(row.get("ir"))
        mdd = _finite_number(row.get("max_drawdown"), default=0.0)
        out = dict(row)
        if score_rule == "max_ir":
            out["eligible"] = True
            out["selection_score"] = ir
            key = (ir, annret, -abs(mdd), -action_order[str(row["action"])])
        elif score_rule == "annret_minus_abs_mdd":
            out["eligible"] = True
            out["selection_score"] = annret - abs(mdd)
            key = (annret - abs(mdd), ir, annret, -action_order[str(row["action"])])
        elif score_rule == "annret_gt_0p27_then_ir":
            eligible = bool(annret > HARD_GATE_ANNRET)
            out["eligible"] = eligible
            out["selection_score"] = ir if eligible else float("nan")
            key = (1 if eligible else 0, ir if eligible else annret, annret, -abs(mdd), -action_order[str(row["action"])])
        else:
            raise ValueError(f"unknown score_rule: {score_rule}")
        out["_rank_key"] = key
        ranked.append(out)
    ranked = sorted(ranked, key=lambda x: x["_rank_key"], reverse=True)
    selected = str(ranked[0]["action"])
    reason = {
        "max_ir": "predeclared_max_prior_window_ir",
        "annret_minus_abs_mdd": "predeclared_max_prior_window_annret_minus_abs_mdd",
        "annret_gt_0p27_then_ir": "predeclared_prior_annret_gt_0p27_then_max_ir_else_best_annret",
    }[score_rule]
    clean_rows = []
    for rank, row in enumerate(ranked, start=1):
        row = dict(row)
        row.pop("_rank_key", None)
        row["train_rank"] = rank
        clean_rows.append(row)
    return selected, clean_rows, reason


def _build_policy(
    reports: Dict[str, pd.DataFrame],
    fixed_first_action: str,
    score_rule: str,
    action_order: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    plan: List[Dict[str, Any]] = [
        {
            "apply_tag": "2024H1",
            "train_tag": "fixed_predeclared_no_prior_window",
            "train_start": "",
            "train_end": "",
            "apply_start": str(pd.Timestamp("2024-01-01").date()),
            "apply_end": str(pd.Timestamp("2024-06-30").date()),
            "selected_action": fixed_first_action,
            "selection_rule": "fixed_predeclared_2024H1_action",
            "score_rule": score_rule,
            "strict_non_test_selected": True,
        }
    ]
    selection_rows: List[Dict[str, Any]] = []
    for spec in _selection_windows():
        train_rows: List[Dict[str, Any]] = []
        for action in ACTION_IDS:
            row = {
                "fixed_first_action": fixed_first_action,
                "score_rule": score_rule,
                "apply_tag": spec["apply_tag"],
                "train_tag": spec["train_tag"],
                "train_start": str(spec["train_start"].date()),
                "train_end": str(spec["train_end"].date()),
                "action": action,
                **_metrics(_slice_report(reports[action], spec["train_start"], spec["train_end"])),
            }
            train_rows.append(row)
        selected, ranked_rows, reason = _select_action(train_rows, score_rule, action_order)
        selection_rows.extend(ranked_rows)
        plan.append(
            {
                "apply_tag": spec["apply_tag"],
                "train_tag": spec["train_tag"],
                "train_start": str(spec["train_start"].date()),
                "train_end": str(spec["train_end"].date()),
                "apply_start": str(spec["apply_start"].date()),
                "apply_end": str(spec["apply_end"].date()),
                "selected_action": selected,
                "selection_rule": reason,
                "score_rule": score_rule,
                "strict_non_test_selected": True,
            }
        )

    parts: List[pd.DataFrame] = []
    for row in plan:
        part = _slice_report(reports[str(row["selected_action"])], row["apply_start"], row["apply_end"])
        if part.empty:
            continue
        part = part.copy()
        part["selected_action"] = str(row["selected_action"])
        part["apply_tag"] = str(row["apply_tag"])
        part["selection_rule"] = str(row["selection_rule"])
        part["score_rule"] = score_rule
        part["fixed_first_action"] = fixed_first_action
        parts.append(part)
    stitched = pd.concat(parts).sort_index() if parts else pd.DataFrame()
    return plan, selection_rows, stitched


def _selected_counts(report: pd.DataFrame) -> str:
    if "selected_action" not in report:
        return "{}"
    return json.dumps({str(k): int(v) for k, v in report["selected_action"].value_counts().to_dict().items()}, sort_keys=True)


def _load_reports(cache_dir: Path) -> Dict[str, pd.DataFrame]:
    reports: Dict[str, pd.DataFrame] = {}
    for action in ACTION_IDS:
        path = cache_dir / f"{action}_20240101_20260430_report.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing real daily report CSV for {action}: {path}")
        reports[action] = _load_report(path)
    return reports


def _source_report_rows(reports: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    rows = []
    for action, report in reports.items():
        m = _metrics(_slice_report(report, TEST_START, TEST_END))
        complete_gate_data = bool(m["coverage_complete"])
        rows.append(
            {
                "action": action,
                "start": str(report.index.min().date()),
                "end": str(report.index.max().date()),
                "rows": int(len(report)),
                "hard_gate_pass": bool(complete_gate_data and m["ir"] > HARD_GATE_IR and m["annret"] > HARD_GATE_ANNRET),
                **m,
            }
        )
    return rows


def _validate_cache_summary(cache_dir: Path) -> Dict[str, Any]:
    path = cache_dir / "replay_action_reports_cache_summary_20260524T045740Z.json"
    if not path.exists():
        return {"summary_path": str(path), "summary_found": False}
    summary = _load_json(path)
    costs = summary.get("costs", {})
    test_period = summary.get("test_period", {})
    return {
        "summary_path": str(path),
        "summary_found": True,
        "test_start": test_period.get("start"),
        "test_end": test_period.get("end"),
        "open_cost": costs.get("open"),
        "close_cost": costs.get("close"),
        "matches_required_gate": bool(
            str(test_period.get("start")) == str(TEST_START.date())
            and str(test_period.get("end")) == str(TEST_END.date())
            and float(costs.get("open", float("nan"))) == OPEN_COST
            and float(costs.get("close", float("nan"))) == CLOSE_COST
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict report-level action-splice audit using persisted real backtest reports.")
    p.add_argument("--cache-dir", default=str(_default_cache_dir()))
    p.add_argument("--output-prefix", default="real_report_action_splice_lockstep")
    p.add_argument(
        "--max-policies",
        type=int,
        default=0,
        help="Optional bounded smoke limit over fixed-action/score policies; 0 evaluates all predeclared policies.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent
    cache_dir = Path(args.cache_dir).resolve()

    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    source_csv = out_dir / f"{args.output_prefix}_source_reports_{stamp}.csv"
    policies_csv = out_dir / f"{args.output_prefix}_policies_{stamp}.csv"
    selections_csv = out_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    plan_csv = out_dir / f"{args.output_prefix}_plan_{stamp}.csv"
    stitched_csv = out_dir / f"{args.output_prefix}_stitched_daily_{stamp}.csv"

    cache_meta = _validate_cache_summary(cache_dir)
    reports = _load_reports(cache_dir)
    source_rows = _source_report_rows(reports)
    action_order = {action: i for i, action in enumerate(ACTION_IDS)}

    policy_specs = [(fixed, score) for fixed in ACTION_IDS for score in SCORE_RULES]
    if int(args.max_policies) > 0:
        policy_specs = policy_specs[: int(args.max_policies)]

    policy_rows: List[Dict[str, Any]] = []
    all_selection_rows: List[Dict[str, Any]] = []
    all_plan_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    daily_rows: List[pd.DataFrame] = []

    for fixed_first_action, score_rule in policy_specs:
        policy_id = f"fixed_{fixed_first_action}__score_{score_rule}"
        plan, selection_rows, stitched = _build_policy(reports, fixed_first_action, score_rule, action_order)
        all_selection_rows.extend({"policy_id": policy_id, **row} for row in selection_rows)
        all_plan_rows.extend({"policy_id": policy_id, **row} for row in plan)

        full_metrics = _metrics(_slice_report(stitched, TEST_START, TEST_END))
        hard_gate_pass = bool(
            full_metrics["coverage_complete"]
            and full_metrics["ir"] > HARD_GATE_IR
            and full_metrics["annret"] > HARD_GATE_ANNRET
        )
        policy_rows.append(
            {
                "policy_id": policy_id,
                "fixed_first_action": fixed_first_action,
                "score_rule": score_rule,
                "start": str(TEST_START.date()),
                "end": str(TEST_END.date()),
                "hard_gate_pass": hard_gate_pass,
                "admissible": False,
                "requires_full_trade_replay_verification": True,
                "strict_non_test_selected": True,
                "caveat": CAVEAT,
                "selected_counts": _selected_counts(stitched),
                **full_metrics,
            }
        )

        for split, start, end in _split_specs():
            part = _slice_report(stitched, start, end)
            split_rows.append(
                {
                    "policy_id": policy_id,
                    "fixed_first_action": fixed_first_action,
                    "score_rule": score_rule,
                    "split": split,
                    "start": str(start.date()),
                    "end": str(end.date()),
                    "hard_gate_pass_on_full_policy": hard_gate_pass,
                    "admissible": False,
                    "requires_full_trade_replay_verification": True,
                    "selected_counts": _selected_counts(part),
                    **_metrics(part),
                }
            )

        daily = stitched.reset_index().rename(columns={"index": "datetime"})
        daily.insert(0, "policy_id", policy_id)
        daily_rows.append(daily)

    any_splice_pass = any(bool(row["hard_gate_pass"]) for row in policy_rows)
    admissible_breakthrough = any(bool(row["hard_gate_pass"]) and bool(row["admissible"]) for row in policy_rows)
    verdict = "FULL_REPLAY_REQUIRED" if any_splice_pass else "NO_GO"

    _write_csv(source_csv, source_rows)
    _write_csv(policies_csv, policy_rows)
    _write_csv(selections_csv, all_selection_rows)
    _write_csv(splits_csv, split_rows)
    _write_csv(plan_csv, all_plan_rows)
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_csv(stitched_csv, index=False)
    else:
        stitched_csv.write_text("", encoding="utf-8")

    summary = {
        "timestamp_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "task": "strict_report_level_action_splice_lockstep",
        "cache_dir": str(cache_dir),
        "cache_meta": cache_meta,
        "candidate_actions": list(ACTION_IDS),
        "score_rules": list(SCORE_RULES),
        "hard_gate": {
            "start": str(TEST_START.date()),
            "end": str(TEST_END.date()),
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
        },
        "selection_protocol": {
            "fixed_2024H1_actions_audited": list(ACTION_IDS),
            "windows": [
                {
                    "train": "fixed_predeclared_no_prior_window",
                    "apply": "2024H1",
                    "apply_start": "2024-01-01",
                    "apply_end": "2024-06-30",
                },
                {
                    "train": "2024H1",
                    "apply": "2024H2",
                    "train_start": "2024-01-01",
                    "train_end": "2024-06-30",
                    "apply_start": "2024-07-01",
                    "apply_end": "2024-12-31",
                },
                {
                    "train": "2024",
                    "apply": "2025",
                    "train_start": "2024-01-01",
                    "train_end": "2024-12-31",
                    "apply_start": "2025-01-01",
                    "apply_end": "2025-12-31",
                },
                {
                    "train": "2024_2025",
                    "apply": "2026_ytd",
                    "train_start": "2024-01-01",
                    "train_end": "2025-12-31",
                    "apply_start": "2026-01-01",
                    "apply_end": "2026-04-30",
                },
            ],
            "strict_non_test_selected": True,
        },
        "admissibility": {
            "admissible": False,
            "requires_full_trade_replay_verification": True,
            "caveat": CAVEAT,
        },
        "source_reports": source_rows,
        "policy_count": len(policy_rows),
        "any_report_level_splice_hard_gate_pass": any_splice_pass,
        "admissible_breakthrough": admissible_breakthrough,
        "verdict": verdict,
        "best_by_ir": sorted(policy_rows, key=lambda r: _finite_number(r.get("ir")), reverse=True)[:5],
        "best_by_annret": sorted(policy_rows, key=lambda r: _finite_number(r.get("annret")), reverse=True)[:5],
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "summary_json": str(summary_json),
            "source_reports_csv": str(source_csv),
            "policies_csv": str(policies_csv),
            "selections_csv": str(selections_csv),
            "splits_csv": str(splits_csv),
            "plan_csv": str(plan_csv),
            "stitched_daily_csv": str(stitched_csv),
        },
    }
    _write_json(summary_json, summary)

    print(json.dumps(_jsonable({
        "verdict": verdict,
        "any_report_level_splice_hard_gate_pass": any_splice_pass,
        "admissible_breakthrough": admissible_breakthrough,
        "policy_count": len(policy_rows),
        "summary_json": str(summary_json),
        "policies_csv": str(policies_csv),
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

