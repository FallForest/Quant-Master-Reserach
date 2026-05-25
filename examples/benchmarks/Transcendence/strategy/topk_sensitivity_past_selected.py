#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
DEFAULT_TOPK = 40
DEFAULT_N_DROP = 2


@dataclass(frozen=True)
class Candidate:
    topk: int
    n_drop: int
    rebalance_mode: str = "daily"
    rebalance_interval: int = 1

    @property
    def candidate_id(self) -> str:
        return f"tk{self.topk}_nd{self.n_drop}_{self.rebalance_mode}"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _candidate_grid() -> List[Candidate]:
    return [
        Candidate(topk=topk, n_drop=n_drop)
        for topk in (35, 40, 45)
        for n_drop in (2, 4, 6)
        if n_drop < topk
    ]


def _combo(candidate: Candidate) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": candidate.rebalance_mode,
        "rebalance_interval": int(candidate.rebalance_interval),
        "topk": int(candidate.topk),
        "n_drop": int(candidate.n_drop),
        "hold_topk": int(candidate.topk),
    }


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _score_for_selection(metrics: Dict[str, Any]) -> float:
    return (
        float(metrics["costed_ir"])
        + 0.35 * float(metrics["costed_annret"])
        - 0.25 * abs(float(metrics["max_drawdown"]))
    )


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_traceback_tail": tb,
    }


def _eval_candidate(
    *,
    candidate: Candidate,
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_time).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_time).date())
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )

    pred_slice = conv._slice_pred(pred_df, start_time, end_time)
    if pred_slice.empty:
        raise ValueError(f"empty base signal slice in {start_time.date()} ~ {end_time.date()}")

    strategy_obj = conv._build_strategy_object(
        combo=_combo(candidate),
        pred_df=pred_slice,
        base_strategy_kwargs=base_strategy_kwargs,
    )
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = (
        str(backtest_cfg["start_time"]),
        str(backtest_cfg["end_time"]),
        float(open_cost),
        float(close_cost),
        limit_threshold,
        deal_price,
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = conv.get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = conv.run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report_df = conv._get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = conv._calc_costed_metrics(report_df)
    excess = report_df["return"] - report_df["bench"] - report_df["cost"]
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "elapsed_sec": float(time.perf_counter() - t0),
        "report_df": report_df,
        "excess_series": excess,
    }


def _apply_slices() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    return [
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_predeclared_default"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]


def _selection_windows() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str, pd.Timestamp, pd.Timestamp]]:
    return [
        (
            "2024H1",
            TEST_START,
            pd.Timestamp("2024-06-30"),
            "2024H2",
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2024-12-31"),
        ),
        (
            "2024",
            TEST_START,
            pd.Timestamp("2024-12-31"),
            "2025",
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
        ),
        (
            "2024_2025",
            TEST_START,
            pd.Timestamp("2025-12-31"),
            "2026_ytd",
            pd.Timestamp("2026-01-01"),
            TEST_END,
        ),
    ]


def _split_rows(excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, start, end, selected_by in _apply_slices():
        part = excess.loc[(excess.index >= start) & (excess.index <= end)]
        if part.empty:
            continue
        rows.append(
            {
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "selection_rule": selected_by,
                **_metrics_from_excess(part),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Past-only topk/n_drop sensitivity on the base signal.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=conv.SOTA_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--output-prefix", default="topk_sensitivity_past_selected")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    trans_dir = Path(__file__).resolve().parent
    train_csv = trans_dir / f"{args.output_prefix}_candidate_train_metrics_{stamp}.csv"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    test_json = trans_dir / f"{args.output_prefix}_test_metrics_{stamp}.json"
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(base_cfg)
    base_port_cfg = conv._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)
    base_pred = conv._as_score_df(conv._load_pickle(base_dir / "artifacts" / "pred.pkl"))

    grid = _candidate_grid()
    default_candidate = Candidate(topk=DEFAULT_TOPK, n_drop=DEFAULT_N_DROP)
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    train_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selected_by_apply: Dict[str, Candidate] = {"2024H1_default": default_candidate}

    for train_tag, train_start, train_end, apply_tag, apply_start, apply_end in _selection_windows():
        scored: List[Dict[str, Any]] = []
        for candidate in grid:
            row: Dict[str, Any] = {
                "train_tag": train_tag,
                "apply_tag": apply_tag,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "ok": False,
                "candidate_id": candidate.candidate_id,
                **asdict(candidate),
            }
            try:
                metrics = _eval_candidate(
                    candidate=candidate,
                    pred_df=base_pred,
                    base_port_cfg=base_port_cfg,
                    base_strategy_kwargs=base_strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start_time=train_start,
                    end_time=train_end,
                    exchange_cache=exchange_cache,
                )
                selection_score = _score_for_selection(metrics)
                row.update(
                    {
                        "ok": True,
                        "selection_score": selection_score,
                        "costed_ir": float(metrics["costed_ir"]),
                        "costed_annret": float(metrics["costed_annret"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "turnover": float(metrics["turnover"]),
                        "elapsed_sec": float(metrics["elapsed_sec"]),
                        "error_type": "",
                        "error_message": "",
                    }
                )
                scored.append(row.copy())
            except Exception as exc:  # noqa: BLE001
                row.update(_error_fields(exc))
            train_rows.append(row)

        ranked = sorted(scored, key=lambda x: float(x["selection_score"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["train_rank"] = rank
        if not ranked:
            selection_rows.append(
                {
                    "apply_tag": apply_tag,
                    "selected": False,
                    "selection_train_tag": train_tag,
                    "error_type": "NoSelectableCandidate",
                    "error_message": f"no valid candidate in {train_tag}",
                }
            )
            continue

        best = ranked[0]
        selected = Candidate(topk=int(best["topk"]), n_drop=int(best["n_drop"]))
        selected_by_apply[apply_tag] = selected
        selection_rows.append(
            {
                "apply_tag": apply_tag,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "selected": True,
                "selection_train_tag": train_tag,
                "candidate_id": selected.candidate_id,
                "topk": selected.topk,
                "n_drop": selected.n_drop,
                "rebalance_mode": selected.rebalance_mode,
                "selection_score": float(best["selection_score"]),
                "train_ir": float(best["costed_ir"]),
                "train_annret": float(best["costed_annret"]),
                "train_mdd": float(best["max_drawdown"]),
                "train_turnover": float(best["turnover"]),
                "strict_non_test_selected": True,
                "candidate_family_predeclared": True,
            }
        )

    selection_rows.insert(
        0,
        {
            "apply_tag": "2024H1_default",
            "apply_start": str(TEST_START.date()),
            "apply_end": str(pd.Timestamp("2024-06-30").date()),
            "selected": True,
            "selection_train_tag": "predeclared_default_no_2024H1_selection",
            "candidate_id": default_candidate.candidate_id,
            "topk": default_candidate.topk,
            "n_drop": default_candidate.n_drop,
            "rebalance_mode": default_candidate.rebalance_mode,
            "selection_score": "",
            "train_ir": "",
            "train_annret": "",
            "train_mdd": "",
            "train_turnover": "",
            "strict_non_test_selected": True,
            "candidate_family_predeclared": True,
        },
    )

    apply_rows: List[Dict[str, Any]] = []
    excess_parts: List[pd.Series] = []
    reports: List[pd.DataFrame] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        candidate = selected_by_apply.get(apply_tag)
        row: Dict[str, Any] = {
            "split": apply_tag,
            "start": str(apply_start.date()),
            "end": str(apply_end.date()),
            "selection_rule": selection_rule,
            "ok": False,
        }
        if candidate is None:
            row.update({"error_type": "MissingSelection", "error_message": f"no selected candidate for {apply_tag}"})
            apply_rows.append(row)
            continue
        row.update({"candidate_id": candidate.candidate_id, **asdict(candidate)})
        try:
            metrics = _eval_candidate(
                candidate=candidate,
                pred_df=base_pred,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_time=apply_start,
                end_time=apply_end,
                exchange_cache=exchange_cache,
            )
            row.update(
                {
                    "ok": True,
                    "costed_ir": float(metrics["costed_ir"]),
                    "costed_annret": float(metrics["costed_annret"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "turnover": float(metrics["turnover"]),
                    "elapsed_sec": float(metrics["elapsed_sec"]),
                    "error_type": "",
                    "error_message": "",
                }
            )
            excess_parts.append(metrics["excess_series"])
            reports.append(metrics["report_df"])
        except Exception as exc:  # noqa: BLE001
            row.update(_error_fields(exc))
        apply_rows.append(row)

    evaluation_complete = len(excess_parts) == 4 and all(bool(row.get("ok")) for row in apply_rows)
    if excess_parts:
        stitched_excess = pd.concat(excess_parts).sort_index()
        full_metrics = _metrics_from_excess(stitched_excess)
        split_rows = _split_rows(stitched_excess)
    else:
        stitched_excess = pd.Series(dtype=float)
        full_metrics = {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan")}
        split_rows = []

    hard_gate_pass = bool(
        evaluation_complete
        and full_metrics["ir"] > HARD_GATE_IR
        and full_metrics["annret"] > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"
    strict_non_test_selected = bool(
        all(bool(row.get("strict_non_test_selected")) for row in selection_rows)
        and set(selected_by_apply).issuperset({"2024H1_default", "2024H2", "2025", "2026_ytd"})
    )

    test_metrics = {
        "full_window": {
            "start": str(TEST_START.date()),
            "end": str(TEST_END.date()),
            **full_metrics,
        },
        "splits": split_rows,
        "apply_rows": apply_rows,
    }
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "topk_sensitivity_past_selected",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "strict_non_test_selected": strict_non_test_selected,
        "protocol": (
            "Predeclared base-signal topk/n_drop daily family. 2024H1 uses fixed default "
            "tk40/nd2; 2024H2, 2025, and 2026_ytd are selected only from prior windows."
        ),
        "hard_gate": {
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "ir_gt": HARD_GATE_IR,
            "annret_gt": HARD_GATE_ANNRET,
        },
        "candidate_family": {
            "base_run_id": str(base_dir.name),
            "topk_grid": [35, 40, 45],
            "n_drop_grid": [2, 4, 6],
            "rebalance_mode": "daily",
            "default_for_2024H1": {"topk": DEFAULT_TOPK, "n_drop": DEFAULT_N_DROP},
            "candidate_count": len(grid),
        },
        "counts": {
            "train_rows": len(train_rows),
            "selection_rows": len(selection_rows),
            "apply_rows": len(apply_rows),
            "stitched_observations": int(len(stitched_excess)),
        },
        "stitched_metrics": full_metrics,
        "split_metrics": split_rows,
        "selections": selection_rows,
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "candidate_train_metrics_csv": str(train_csv),
            "selections_csv": str(selections_csv),
            "split_metrics_csv": str(splits_csv),
            "test_metrics_json": str(test_json),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }

    _write_csv(train_csv, train_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(splits_csv, split_rows)
    _write_json(test_json, test_metrics)
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Topk Sensitivity Past Selected {stamp}",
                "",
                f"Verdict: **{verdict}**",
                f"- Hard gate pass: `{hard_gate_pass}`",
                f"- Strict non-test-selected: `{strict_non_test_selected}`",
                f"- Full stitched metrics: `{json.dumps(full_metrics, ensure_ascii=False)}`",
                f"- Train/apply rows: `{len(train_rows)}/{len(apply_rows)}`",
                f"- Summary JSON: `{summary_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "verdict": verdict,
                "hard_gate_pass": hard_gate_pass,
                "strict_non_test_selected": strict_non_test_selected,
                "stitched_metrics": full_metrics,
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
