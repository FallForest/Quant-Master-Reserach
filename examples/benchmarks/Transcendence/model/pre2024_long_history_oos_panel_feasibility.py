#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
TRANS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TASK_ID = "Q-LONG-HISTORY-OOS-PANEL-FEASIBILITY"


@dataclass
class EntrypointAudit:
    name: str
    path: Path
    available: bool
    can_generate_2020_2021_oos_now: bool
    action: str
    blockers: List[str]
    required_changes: List[str]
    estimated_runtime: str
    estimated_memory: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _regex_default(text: str, name: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return m.group(1) if m else None


def _parser_defaults(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in (
        "n-estimators",
        "num-leaves",
        "learning-rate",
        "subsample",
        "colsample-bytree",
        "topk-grid",
        "ndrop-grid",
        "global-start",
        "global-end",
        "eval-start",
        "eval-end",
        "min-history-days",
        "cv-valid-days",
        "cv-max-folds",
        "weight-grid",
    ):
        pattern = rf"add_argument\(\s*[\"']--{re.escape(name)}[\"'][\s\S]*?default=([^,\)\n]+)"
        m = re.search(pattern, text)
        if m:
            out[name.replace("-", "_")] = m.group(1).strip().strip("\"'")
    return out


def _latest_coverage_summary() -> Optional[Path]:
    summaries = sorted(
        SCRIPT_DIR.glob("pre2024_second_order_coverage_smoke_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def _load_coverage_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "path": "",
            "gate_pass": False,
            "eligible_complete_signal_count": 0,
            "verdict": "UNKNOWN",
            "note": "No pre2024 coverage summary artifact found.",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "path": str(path.resolve()),
            "gate_pass": False,
            "eligible_complete_signal_count": 0,
            "verdict": "READ_ERROR",
            "note": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "path": str(path.resolve()),
        "gate_pass": bool(data.get("gate_pass", False)),
        "eligible_complete_signal_count": int(data.get("eligible_complete_signal_count", 0) or 0),
        "verdict": str(data.get("verdict", "")),
        "legal_pre2024_selection_possible": bool(data.get("legal_pre2024_selection_possible", False)),
        "no_2024_plus_used_for_gate": bool(data.get("no_2024_plus_used_for_gate", False)),
        "next_step_recommendation": str(data.get("next_step_recommendation", "")),
    }


def _audit_long_history_retrain() -> EntrypointAudit:
    support_path = TRANS_DIR / "support" / "long_history_model_retrain.py"
    wrapper_path = TRANS_DIR / "long_history_model_retrain.py"
    text = _read_text(support_path)
    available = support_path.exists() and wrapper_path.exists()
    defaults = {
        "train_start": _regex_default(text, "TRAIN_START"),
        "train_end": _regex_default(text, "TRAIN_END"),
        "valid_start": _regex_default(text, "VALID_START"),
        "valid_end": _regex_default(text, "VALID_END"),
        "test_start": _regex_default(text, "TEST_START"),
        "test_end": _regex_default(text, "TEST_END"),
        "raw_start": _regex_default(text, "RAW_START"),
        **_parser_defaults(text),
    }

    blockers = []
    if defaults.get("test_start") == "2024-01-01":
        blockers.append("Hard-coded TEST_START=2024-01-01; candidate_pred is generated only for the 2024-2026 test split.")
    if defaults.get("test_end") == "2026-04-30":
        blockers.append("Hard-coded TEST_END=2026-04-30; _build_long_history_panel reads feature bins through 2026-04-30.")
    if "model.fit(" in text:
        blockers.append("Entrypoint trains a LightGBMRegressor before prediction; no load-pretrained or predict-only mode is present.")
    if "n_jobs=8" in text:
        blockers.append("Training uses n_jobs=8 and defaults to n_estimators=800 / num_leaves=127, which is not a bounded smoke job.")
    if "_run_bt(" in text and "test_2024_2026" in text:
        blockers.append("Entrypoint runs validation and 2024-2026 test backtests after fitting.")

    required_changes = [
        "Add CLI split overrides for train/valid/apply windows and raw end date, with an enforce-pre2024 guard.",
        "Add predict-panel-only output mode that skips portfolio backtests and never opens 2024+ windows.",
        "Add smoke limits such as max instruments, max dates, n_estimators <= 20, n_jobs <= 2, and a wall-clock budget.",
        "Optionally add a train-before-2020/apply-2020 or train-before-2021/apply-2021 OOS fold definition.",
    ]

    return EntrypointAudit(
        name="long_history_model_retrain",
        path=support_path,
        available=available,
        can_generate_2020_2021_oos_now=False,
        action="NO_RUN",
        blockers=blockers,
        required_changes=required_changes,
        estimated_runtime="Current defaults are a large local LightGBM + backtest run; prior artifact took long enough to produce multi-MB outputs, not an approved smoke.",
        estimated_memory="Unbounded by CLI; constructs csi300 multi-year panel and dense float32 matrices for train/valid/test.",
    )


def _audit_second_order() -> EntrypointAudit:
    path = SCRIPT_DIR / "long_history_second_order_ensemble.py"
    text = _read_text(path)
    defaults = {
        "global_start": _regex_default(text, "GLOBAL_START"),
        "global_end": _regex_default(text, "GLOBAL_END"),
        "default_eval_start": _regex_default(text, "DEFAULT_EVAL_START"),
        **_parser_defaults(text),
    }

    blockers = []
    if defaults.get("global_start") == "2024-01-01":
        blockers.append("Default GLOBAL_START=2024-01-01; entrypoint is written around 2024-2026 common signal coverage.")
    if defaults.get("global_end") == "2026-04-30":
        blockers.append("Default GLOBAL_END=2026-04-30; loading published candidate pkls targets post-2023 outputs.")
    if "META_PRED_FILE" in text and "LH_PRED_FILE" in text:
        blockers.append("Uses fixed published factor_meta and long_history candidate prediction files, not a generator for new 2020-2021 OOS panels.")
    if "_load_pickle(meta_path)" in text or "_load_pickle(lh_path)" in text:
        blockers.append("Loads full prediction pickle artifacts before slicing, which is not acceptable for a strict no-2024+ feasibility run.")
    if "_safe_backtest_eval(" in text:
        blockers.append("Runs same-window portfolio backtests after creating the ensemble signal.")

    required_changes = [
        "Refactor loading so CSV rows can be date-filtered before any score panel is materialized, or require pre-filtered pre-2024 artifacts.",
        "Parameterize candidate signal paths and require their declared coverage to be pre-2024 only.",
        "Add a no-backtest panel-export mode for 2020-2021 OOS only.",
    ]

    return EntrypointAudit(
        name="long_history_second_order_ensemble",
        path=path,
        available=path.exists(),
        can_generate_2020_2021_oos_now=False,
        action="NO_RUN",
        blockers=blockers,
        required_changes=required_changes,
        estimated_runtime="Static blend itself is modest, but current entrypoint loads full post-2023 pkl panels and runs backtests; not safe under this task.",
        estimated_memory="Depends on full published pkl panel size; not bounded before load.",
    )


def _audit_config() -> Dict[str, Any]:
    path = TRANS_DIR / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
    text = _read_text(path)
    return {
        "path": str(path.resolve()),
        "available": path.exists(),
        "contains_2024_2026_windows": "2024-01-01" in text and "2026-04-30" in text,
        "contains_heavy_double_ensemble_defaults": all(
            token in text for token in ("num_models: 3", "epochs: 28", "num_threads: 20")
        ),
        "audit_note": "Config is useful for port_analysis defaults only; its task dataset/test windows and model defaults are not a light 2020-2021 OOS generator.",
    }


def _artifact_inventory() -> List[Dict[str, Any]]:
    names = [
        "factor_augmented_meta_candidate_pred_20260522T120515Z.csv",
        "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl",
        "long_history_retrain_candidate_pred_20260522T134241Z.csv",
        "long_history_retrain_candidate_pred_20260522T134241Z.pkl",
        "long_history_second_order_candidate_pred_20260522T153041Z.csv",
        "long_history_second_order_candidate_pred_20260522T153041Z.pkl",
    ]
    rows: List[Dict[str, Any]] = []
    for name in names:
        path = TRANS_DIR / name
        rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "policy": "not_loaded",
                "note": "Prediction artifacts were intentionally not opened; existing coverage smoke reports no complete 2020-2023 compliant signal.",
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


def _entrypoint_rows(audits: Sequence[EntrypointAudit]) -> List[Dict[str, Any]]:
    rows = []
    for audit in audits:
        rows.append(
            {
                "name": audit.name,
                "path": str(audit.path.resolve()),
                "available": int(audit.available),
                "can_generate_2020_2021_oos_now": int(audit.can_generate_2020_2021_oos_now),
                "action": audit.action,
                "blockers": " | ".join(audit.blockers),
                "required_changes": " | ".join(audit.required_changes),
                "estimated_runtime": audit.estimated_runtime,
                "estimated_memory": audit.estimated_memory,
            }
        )
    return rows


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj.resolve())
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-2024 long-history OOS panel feasibility audit.")
    parser.add_argument("--output-prefix", default="pre2024_long_history_oos_panel_feasibility")
    args = parser.parse_args()

    stamp = _stamp()
    prefix = Path(args.output_prefix)
    if prefix.is_absolute() or prefix.parent != Path("."):
        raise ValueError("--output-prefix must be a bare filename prefix for this worker-owned output set")
    if not prefix.name.startswith("pre2024_long_history_oos_panel_feasibility"):
        raise ValueError("--output-prefix must start with pre2024_long_history_oos_panel_feasibility")

    summary_json = SCRIPT_DIR / f"{prefix.name}_summary_{stamp}.json"
    summary_md = SCRIPT_DIR / f"{prefix.name}_summary_{stamp}.md"
    entrypoints_csv = SCRIPT_DIR / f"{prefix.name}_entrypoints_{stamp}.csv"
    artifacts_csv = SCRIPT_DIR / f"{prefix.name}_artifacts_{stamp}.csv"

    audits = [_audit_long_history_retrain(), _audit_second_order()]
    config_audit = _audit_config()
    coverage_summary = _load_coverage_summary(_latest_coverage_summary())
    artifact_rows = _artifact_inventory()

    feasible_now = bool(
        coverage_summary.get("gate_pass", False)
        and any(a.can_generate_2020_2021_oos_now for a in audits)
    )
    gate_pass = False
    panel_generated = False

    verdict = "NO_GO"
    no_go_reasons = [
        "No existing audited entrypoint can generate a bounded 2020-2021 OOS panel without code changes.",
        "Current long_history_model_retrain hard-codes 2024-2026 test generation and performs full LightGBM training/backtests.",
        "Current long_history_second_order_ensemble consumes fixed post-2023 prediction pickle artifacts and backtests the eval window.",
        "Existing pre-2024 coverage smoke found zero complete compliant signals for legal weight selection.",
    ]

    summary = {
        "timestamp_utc": _now_utc(),
        "task_id": TASK_ID,
        "repo_root": str(REPO_ROOT.resolve()),
        "script": str(Path(__file__).resolve()),
        "verdict": verdict,
        "gate_pass": gate_pass,
        "panel_generation_feasible_now": feasible_now,
        "panel_generated": panel_generated,
        "panel_path": None,
        "no_2024_plus_data_loaded_or_evaluated": True,
        "run_policy": {
            "action": "NO_RUN_TRAINING_NO_PANEL",
            "reason": "Existing interfaces require large training/backtest work or unsafe full prediction artifact loads.",
            "time_limit_for_this_audit": "static audit only; expected < 5 seconds",
            "memory_limit_for_this_audit": "small text/json/csv metadata only; no prediction panels loaded",
        },
        "available_entrypoints": [audit.__dict__ | {"path": str(audit.path.resolve())} for audit in audits],
        "related_config": config_audit,
        "coverage_smoke": coverage_summary,
        "artifact_inventory": artifact_rows,
        "required_changes": {
            "minimum_to_make_feasible": [
                "Add an explicit pre-2024 OOS panel mode to long_history_model_retrain.",
                "Make train/valid/apply windows CLI-controlled with a hard assert max_date < 2024-01-01.",
                "Add smoke bounds: max instruments/days, n_estimators <= 20, n_jobs <= 2, skip backtests, and write only candidate_pred CSV/summary.",
                "Use one legal fold such as train <= 2019-12-31 and apply 2020, or train <= 2020-12-31 and apply 2021, depending on available data.",
            ],
            "not_required_for_feasibility": [
                "Do not run 2024-2026 backtests.",
                "Do not run full double ensemble / GPU / multi-model training.",
                "Do not load full post-2023 prediction pickle artifacts.",
            ],
        },
        "estimated_runtime_memory": {
            "this_audit": {"runtime": "< 5 seconds", "memory": "< 100 MB"},
            "current_retrain_entrypoint": {
                "runtime": "NO-GO estimate: unbounded large LightGBM/backtest job under defaults",
                "memory": "NO-GO estimate: csi300 multi-year feature panel plus dense train/valid/test arrays",
            },
            "hypothetical_bounded_smoke_after_changes": {
                "runtime": "target <= 5-10 minutes with <=20 estimators, <=2 jobs, one year apply, no backtest",
                "memory": "target <= 2 GB with instrument/date caps and no full 2024+ materialization",
            },
        },
        "no_go_reasons": no_go_reasons,
        "next_step_recommendation": (
            "Do not generate a panel through current interfaces. If lead approves code changes, add a bounded "
            "pre-2024 panel-export mode to long_history_model_retrain and run a single csi300 2020 or 2021 fold."
        ),
        "artifacts": {
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
            "entrypoints_csv": str(entrypoints_csv.resolve()),
            "artifacts_csv": str(artifacts_csv.resolve()),
        },
    }

    _write_csv(entrypoints_csv, _entrypoint_rows(audits))
    _write_csv(artifacts_csv, artifact_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    md_lines = [
        "# Pre-2024 Long-History OOS Panel Feasibility",
        "",
        f"- task_id: `{TASK_ID}`",
        f"- verdict: `{verdict}`",
        f"- gate_pass: `{gate_pass}`",
        f"- panel_generation_feasible_now: `{feasible_now}`",
        f"- panel_generated: `{panel_generated}`",
        f"- no_2024_plus_data_loaded_or_evaluated: `True`",
        "",
        "## Available Entrypoints",
        "",
        "| entrypoint | action | feasible now | key blockers |",
        "|---|---|---:|---|",
    ]
    for audit in audits:
        md_lines.append(
            f"| `{audit.name}` | `{audit.action}` | `{audit.can_generate_2020_2021_oos_now}` | "
            f"{'; '.join(audit.blockers)} |"
        )
    md_lines.extend(
        [
            "",
            "## Required Changes",
            "",
            "- Add CLI-controlled pre-2024 train/apply windows with hard max-date guards.",
            "- Add panel-export-only smoke mode that skips all portfolio backtests.",
            "- Add explicit resource bounds: max instruments/dates, <=20 estimators, <=2 jobs, and one fold.",
            "- Avoid full post-2023 pickle loads; use pre-filtered CSV or freshly generated pre-2024-only output.",
            "",
            "## Estimates",
            "",
            "- this audit runtime/memory: `<5s` / `<100MB`.",
            "- current retrain entrypoint: `NO-GO`, large LightGBM plus backtests and unbounded dense panel memory.",
            "- bounded smoke after changes: target `<=5-10min` and `<=2GB`.",
            "",
            "## Outputs",
            "",
            f"- summary_json: `{summary_json.resolve()}`",
            f"- entrypoints_csv: `{entrypoints_csv.resolve()}`",
            f"- artifacts_csv: `{artifacts_csv.resolve()}`",
        ]
    )
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"gate_pass": gate_pass, "summary_json": str(summary_json.resolve()), "panel_generated": panel_generated}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
