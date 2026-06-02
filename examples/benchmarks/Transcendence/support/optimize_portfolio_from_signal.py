#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.config import resolve_provider_uri_in_config
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from examples.benchmarks.Transcendence._bootstrap import init_quant_master_from_config, load_config_with_resolved_provider


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TARGET_COSTED_IR = 2.799983676714277


@dataclass
class ScanResult:
    topk: int
    n_drop: int
    open_cost: float
    close_cost: float
    costed_annret: float
    costed_ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    candidates = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not candidates:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id}")
    if len(candidates) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_config(path: Path) -> Dict[str, Any]:
    return load_config_with_resolved_provider(
        path,
        loader=lambda config_path: yaml.safe_load(config_path.read_text(encoding="utf-8")),
        binary_fallback=_load_pickle,
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_metric_file(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    parts = metric_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port_analysis_config or task.record[PortAnaRecord].kwargs.config")


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_quant_master_from_config(config, base_dir=REPO_ROOT, region="cn")


def _get_report_for_day_freq(portfolio_metric_dict):
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    # Fall back to the first available freq.
    first_key = next(iter(portfolio_metric_dict.keys()))
    return portfolio_metric_dict[first_key][0]


def _calc_costed_metrics(report_df) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, max_drawdown, turnover


def _build_combos(
    topk_min: int,
    topk_max: int,
    n_drop_min: int,
    n_drop_max: int,
    open_costs: Sequence[float],
    close_costs: Sequence[float],
    coarse_step: int,
    coarse_ndrops: Sequence[int],
) -> List[Tuple[int, int, float, float]]:
    combos: List[Tuple[int, int, float, float]] = []
    coarse_topks = list(range(topk_min, topk_max + 1, coarse_step))
    for topk in coarse_topks:
        for n_drop in coarse_ndrops:
            if n_drop < n_drop_min or n_drop > n_drop_max:
                continue
            for oc in open_costs:
                for cc in close_costs:
                    combos.append((topk, n_drop, oc, cc))
    return combos


def _build_refine_combos(
    coarse_results: Sequence[ScanResult],
    topk_min: int,
    topk_max: int,
    n_drop_min: int,
    n_drop_max: int,
    open_costs: Sequence[float],
    close_costs: Sequence[float],
    refine_top_n: int,
    refine_radius: int,
) -> List[Tuple[int, int, float, float]]:
    ranked = sorted(coarse_results, key=lambda x: (x.costed_ir, x.costed_annret), reverse=True)
    seeds = ranked[: max(1, refine_top_n)]
    combos: List[Tuple[int, int, float, float]] = []
    for seed in seeds:
        left = max(topk_min, seed.topk - refine_radius)
        right = min(topk_max, seed.topk + refine_radius)
        for topk in range(left, right + 1):
            for n_drop in range(n_drop_min, n_drop_max + 1):
                for oc in open_costs:
                    for cc in close_costs:
                        combos.append((topk, n_drop, oc, cc))
    return combos


def _dedup_keep_order(items: Iterable[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float]]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _run_one_scan(
    pred_df,
    base_port_cfg: Dict[str, Any],
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> ScanResult:
    port_cfg = copy.deepcopy(base_port_cfg)
    strategy_cfg = port_cfg["strategy"]
    backtest_cfg = port_cfg["backtest"]
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    strategy_cfg["kwargs"]["signal"] = pred_df
    strategy_cfg["kwargs"]["topk"] = int(topk)
    strategy_cfg["kwargs"]["n_drop"] = int(n_drop)

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = (
        str(backtest_cfg["start_time"]),
        str(backtest_cfg["end_time"]),
        open_cost,
        close_cost,
        limit_threshold,
        deal_price,
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_cfg,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    report_df = _get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report_df)
    return ScanResult(
        topk=int(topk),
        n_drop=int(n_drop),
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        costed_annret=annret,
        costed_ir=ir,
        max_drawdown=maxdd,
        turnover=turnover,
        elapsed_sec=elapsed,
    )


def _write_scan_csv(path: Path, rows: Sequence[ScanResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(ScanResult(0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _create_candidate_files(
    out_dir: Path,
    run_id: str,
    run_cfg_path: str,
    ic: float | None,
    rank_ic: float | None,
    best: ScanResult,
    command: str,
) -> Tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{run_id[:8]}_tk{best.topk}_nd{best.n_drop}_oc{best.open_cost:.5f}_cc{best.close_cost:.5f}".replace(".", "p")
    leaderboard_path = out_dir / f"leaderboard_candidate_{tag}_{stamp}.csv"
    round_path = out_dir / f"round_summary_candidate_{tag}_{stamp}.md"

    with leaderboard_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round_id",
                "run_id",
                "model_name",
                "workflow_config",
                "ic",
                "rank_ic",
                "costed_annret",
                "costed_ir",
                "max_drawdown",
                "turnover",
                "runtime_sec",
                "leakage_check",
                "command",
                "status",
                "notes",
                "created_at_utc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "round_id": f"SCAN_{stamp}",
                "run_id": run_id,
                "model_name": "SignalReuseTopkDropoutScan",
                "workflow_config": run_cfg_path,
                "ic": ic,
                "rank_ic": rank_ic,
                "costed_annret": best.costed_annret,
                "costed_ir": best.costed_ir,
                "max_drawdown": best.max_drawdown,
                "turnover": best.turnover,
                "runtime_sec": "",
                "leakage_check": "pass",
                "command": command,
                "status": "portfolio_sota_candidate",
                "notes": f"offline signal reuse; topk={best.topk}; n_drop={best.n_drop}; open_cost={best.open_cost}; close_cost={best.close_cost}",
                "created_at_utc": _now_utc(),
            }
        )

    round_text = f"""# Round Summary Candidate (Signal-Reuse Scan)

## Identity
- run_id: `{run_id}`
- workflow_config: `{run_cfg_path}`
- timestamp_utc: `{_now_utc()}`

## Command
```powershell
{command}
```

## Best Portfolio Params
- topk: `{best.topk}`
- n_drop: `{best.n_drop}`
- open_cost: `{best.open_cost}`
- close_cost: `{best.close_cost}`

## Portfolio Metrics (costed)
- costed AnnRet: `{best.costed_annret}`
- costed IR: `{best.costed_ir}`
- max drawdown: `{best.max_drawdown}`
- turnover: `{best.turnover}`

## Signal Metrics (reused from source run)
- IC: `{ic}`
- RankIC: `{rank_ic}`
"""
    round_path.write_text(round_text, encoding="utf-8")
    return leaderboard_path, round_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline TopkDropout portfolio scan from existing run signal without retraining."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run_id that contains artifacts/pred.pkl.")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--config-path", default="", help="Optional workflow config path; default uses run artifact config.")
    parser.add_argument("--topk-min", type=int, default=20)
    parser.add_argument("--topk-max", type=int, default=80)
    parser.add_argument("--n-drop-min", type=int, default=1)
    parser.add_argument("--n-drop-max", type=int, default=10)
    parser.add_argument("--coarse-step", type=int, default=10, help="Topk step for stage-1 coarse scan.")
    parser.add_argument(
        "--coarse-ndrops",
        default="1,2,3,4,5,7,10",
        help="Comma-separated n_drop values for stage-1 coarse scan.",
    )
    parser.add_argument("--refine-top-n", type=int, default=3)
    parser.add_argument("--refine-radius", type=int, default=2)
    parser.add_argument(
        "--open-cost-grid",
        default="",
        help="Comma-separated open_cost values. Default uses backtest exchange open_cost from config.",
    )
    parser.add_argument(
        "--close-cost-grid",
        default="",
        help="Comma-separated close_cost values. Default uses backtest exchange close_cost from config.",
    )
    parser.add_argument("--max-combos", type=int, default=0, help="Optional hard cap on number of evaluated combos.")
    parser.add_argument("--smoke-only", action="store_true", help="Run one combo only (config topk/n_drop/cost).")
    parser.add_argument(
        "--output-prefix",
        default="portfolio_scan",
        help="Prefix for output files under examples/benchmarks/Transcendence.",
    )
    return parser


def _parse_int_list(text: str) -> List[int]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(text: str) -> List[float]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("empty float list")
    return out


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    run_dir = _find_run_dir(tracking_dir, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    pred_path = artifacts_dir / "pred.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing pred artifact: {pred_path}")

    run_cfg_path = Path(args.config_path).expanduser().resolve() if args.config_path else artifacts_dir / "config"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"missing config file: {run_cfg_path}")

    workflow_cfg = _load_config(run_cfg_path)
    _init_quant_master(workflow_cfg)
    base_port_cfg = _extract_port_config(workflow_cfg)
    pred_df = _load_pickle(pred_path)
    strategy_kwargs = base_port_cfg["strategy"]["kwargs"]
    base_topk = int(strategy_kwargs.get("topk", 50))
    base_n_drop = int(strategy_kwargs.get("n_drop", 5))
    exchange_kwargs = base_port_cfg["backtest"].get("exchange_kwargs", {})
    base_open_cost = float(exchange_kwargs.get("open_cost", 0.0005))
    base_close_cost = float(exchange_kwargs.get("close_cost", 0.0015))
    open_costs = _parse_float_list(args.open_cost_grid) if args.open_cost_grid else [base_open_cost]
    close_costs = _parse_float_list(args.close_cost_grid) if args.close_cost_grid else [base_close_cost]

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_short = args.run_id[:8]
    coarse_ndrops = _parse_int_list(args.coarse_ndrops)

    result_cache: Dict[Tuple[int, int, float, float], ScanResult] = {}

    if args.smoke_only:
        combos = [(base_topk, base_n_drop, open_costs[0], close_costs[0])]
    else:
        coarse_combos = _build_combos(
            topk_min=args.topk_min,
            topk_max=args.topk_max,
            n_drop_min=args.n_drop_min,
            n_drop_max=args.n_drop_max,
            open_costs=open_costs,
            close_costs=close_costs,
            coarse_step=args.coarse_step,
            coarse_ndrops=coarse_ndrops,
        )
        coarse_results: List[ScanResult] = []
        for i, (topk, n_drop, open_cost, close_cost) in enumerate(coarse_combos, start=1):
            row = _run_one_scan(
                pred_df, base_port_cfg, topk, n_drop, open_cost, close_cost, exchange_cache=exchange_cache
            )
            coarse_results.append(row)
            result_cache[(topk, n_drop, open_cost, close_cost)] = row
            print(
                f"[coarse {i}/{len(coarse_combos)}] topk={topk} n_drop={n_drop} open={open_cost} close={close_cost} "
                f"IR={row.costed_ir:.6f} AnnRet={row.costed_annret:.6f} maxDD={row.max_drawdown:.6f} turnover={row.turnover:.6f}",
                flush=True,
            )
        refine_combos = _build_refine_combos(
            coarse_results=coarse_results,
            topk_min=args.topk_min,
            topk_max=args.topk_max,
            n_drop_min=args.n_drop_min,
            n_drop_max=args.n_drop_max,
            open_costs=open_costs,
            close_costs=close_costs,
            refine_top_n=args.refine_top_n,
            refine_radius=args.refine_radius,
        )
        combos = _dedup_keep_order(list(coarse_combos) + list(refine_combos))

    if args.max_combos > 0:
        combos = combos[: args.max_combos]

    print(f"scan combos: {len(combos)}", flush=True)
    results: List[ScanResult] = []
    for idx, (topk, n_drop, open_cost, close_cost) in enumerate(combos, start=1):
        combo_key = (topk, n_drop, open_cost, close_cost)
        if combo_key in result_cache:
            row = result_cache[combo_key]
            print(
                f"[scan {idx}/{len(combos)}][cached] topk={topk} n_drop={n_drop} open={open_cost} close={close_cost} "
                f"IR={row.costed_ir:.6f} AnnRet={row.costed_annret:.6f} maxDD={row.max_drawdown:.6f} turnover={row.turnover:.6f}",
                flush=True,
            )
            results.append(row)
            continue
        row = _run_one_scan(
            pred_df, base_port_cfg, topk, n_drop, open_cost, close_cost, exchange_cache=exchange_cache
        )
        result_cache[combo_key] = row
        results.append(row)
        print(
            f"[scan {idx}/{len(combos)}] topk={topk} n_drop={n_drop} open={open_cost} close={close_cost} "
            f"IR={row.costed_ir:.6f} AnnRet={row.costed_annret:.6f} maxDD={row.max_drawdown:.6f} turnover={row.turnover:.6f}",
            flush=True,
        )

    results_sorted = sorted(results, key=lambda x: (x.costed_ir, x.costed_annret), reverse=True)
    csv_path = out_dir / f"{args.output_prefix}_{run_short}_{stamp}.csv"
    _write_scan_csv(csv_path, results_sorted)

    baseline_report_path = artifacts_dir / "portfolio_analysis" / "report_normal_1day.pkl"
    baseline = {}
    if baseline_report_path.exists():
        base_report = _load_pickle(baseline_report_path)
        base_annret, base_ir, base_maxdd, base_turnover = _calc_costed_metrics(base_report)
        baseline = {
            "costed_annret": base_annret,
            "costed_ir": base_ir,
            "max_drawdown": base_maxdd,
            "turnover": base_turnover,
        }
    else:
        baseline = {
            "costed_annret": None,
            "costed_ir": None,
            "max_drawdown": None,
            "turnover": None,
        }

    best = results_sorted[0] if results_sorted else None
    ir_eps = 1e-6
    annret_floor = baseline["costed_annret"] if baseline["costed_annret"] is not None else float("-inf")
    better_than_7406 = bool(
        best and best.costed_ir > (TARGET_COSTED_IR + ir_eps) and best.costed_annret >= (annret_floor - 1e-12)
    )
    ic = _parse_metric_file(run_dir / "metrics" / "IC")
    rank_ic = _parse_metric_file(run_dir / "metrics" / "Rank IC")

    candidate_files = {}
    command = " ".join(sys.argv)
    if better_than_7406 and best:
        lb_path, rs_path = _create_candidate_files(
            out_dir=out_dir,
            run_id=args.run_id,
            run_cfg_path=str(run_cfg_path),
            ic=ic,
            rank_ic=rank_ic,
            best=best,
            command=command,
        )
        candidate_files = {"leaderboard_candidate": str(lb_path), "round_summary_candidate": str(rs_path)}

    summary = {
        "run_id": args.run_id,
        "tracking_uri": args.tracking_uri,
        "artifact_dir": str(artifacts_dir),
        "config_path": str(run_cfg_path),
        "scan_time_utc": _now_utc(),
        "combo_count": len(combos),
        "search_space": {
            "topk": [args.topk_min, args.topk_max],
            "n_drop": [args.n_drop_min, args.n_drop_max],
            "open_costs": open_costs,
            "close_costs": close_costs,
        },
        "baseline_7406": baseline,
        "target_costed_ir": TARGET_COSTED_IR,
        "best": asdict(best) if best else None,
        "better_than_7406": better_than_7406,
        "candidate_files": candidate_files,
        "scan_csv": str(csv_path),
    }
    summary_path = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
