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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.order_generator import OrderGenWInteract
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TARGET_COSTED_IR = 2.80638


class RebalanceMixin:
    def __init__(self, rebalance_mode: str = "daily", rebalance_interval: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.rebalance_mode = str(rebalance_mode).lower()
        self.rebalance_interval = int(max(1, rebalance_interval))
        self._last_rebalance_key: Optional[Tuple[int, int]] = None

    def _get_rebalance_key(self, ts: pd.Timestamp) -> Tuple[int, int]:
        if self.rebalance_mode == "daily":
            return int(ts.year), int(ts.dayofyear)
        if self.rebalance_mode == "weekly":
            iso = ts.isocalendar()
            return int(iso.year), int(iso.week)
        if self.rebalance_mode == "monthly":
            return int(ts.year), int(ts.month)
        return int(ts.year), int(ts.dayofyear)

    def should_rebalance(self, trade_step: int, trade_start_time: pd.Timestamp) -> bool:
        if self.rebalance_mode == "interval":
            return trade_step % self.rebalance_interval == 0
        if self.rebalance_mode in {"daily", "weekly", "monthly"}:
            key = self._get_rebalance_key(pd.Timestamp(trade_start_time))
            if self._last_rebalance_key != key:
                self._last_rebalance_key = key
                return True
            return False
        raise ValueError(f"unsupported rebalance_mode={self.rebalance_mode}")


class ScheduledTopkDropoutStrategy(RebalanceMixin, TopkDropoutStrategy):
    def __init__(
        self,
        *,
        n_drop_schedule: Optional[Sequence[int]] = None,
        rebalance_mode: str = "daily",
        rebalance_interval: int = 5,
        **kwargs,
    ):
        self.n_drop_schedule = [int(x) for x in (n_drop_schedule or [])]
        self._rebalance_count = 0
        super().__init__(rebalance_mode=rebalance_mode, rebalance_interval=rebalance_interval, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return TradeDecisionWO([], self)

        orig_n_drop = int(self.n_drop)
        if self.n_drop_schedule:
            self.n_drop = int(self.n_drop_schedule[self._rebalance_count % len(self.n_drop_schedule)])
        self._rebalance_count += 1
        try:
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.n_drop = orig_n_drop


class BufferedTopkWeightStrategy(RebalanceMixin, WeightStrategyBase):
    def __init__(
        self,
        *,
        topk: int,
        hold_topk: Optional[int] = None,
        weight_mode: str = "equal",
        score_power: float = 1.0,
        rebalance_mode: str = "daily",
        rebalance_interval: int = 5,
        **kwargs,
    ):
        self.topk = int(topk)
        self.hold_topk = int(hold_topk) if hold_topk is not None else int(topk)
        self.weight_mode = str(weight_mode).lower()
        self.score_power = float(score_power)
        super().__init__(
            rebalance_mode=rebalance_mode,
            rebalance_interval=rebalance_interval,
            order_generator_cls_or_obj=OrderGenWInteract,
            **kwargs,
        )

    def _to_series(self, score: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        return score.dropna()

    def _calc_weights(self, ranked_score: pd.Series, target: List[str]) -> Dict[str, float]:
        if not target:
            return {}
        if self.weight_mode == "equal":
            w = 1.0 / len(target)
            return {code: w for code in target}
        if self.weight_mode == "score":
            s = ranked_score.reindex(target).astype(float)
            shifted = s - s.min()
            if float(shifted.sum()) <= 0:
                raw = pd.Series(np.arange(len(target), 0, -1), index=target, dtype=float)
            else:
                raw = (shifted + 1e-12) ** max(1e-6, self.score_power)
            norm = float(raw.sum())
            if norm <= 0:
                w = 1.0 / len(target)
                return {code: w for code in target}
            return {code: float(raw.loc[code] / norm) for code in target}
        raise ValueError(f"unsupported weight_mode={self.weight_mode}")

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_step = self.trade_calendar.get_trade_step()
        if not self.should_rebalance(trade_step=trade_step, trade_start_time=trade_start_time):
            return None

        score_s = self._to_series(score)
        if score_s.empty:
            return {}
        ranked = score_s.sort_values(ascending=False)

        tradable_ranked_idx = []
        for code in ranked.index:
            try:
                ok = self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                tradable_ranked_idx.append(code)
        if not tradable_ranked_idx:
            return None
        ranked = ranked.reindex(tradable_ranked_idx).dropna()
        if ranked.empty:
            return None

        rank_pos = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)

        current_stocks = [s for s in current.get_stock_list() if s in ranked.index]
        keep = [s for s in current_stocks if int(rank_pos.get(s, 10**9)) <= self.hold_topk]
        keep = sorted(keep, key=lambda x: float(ranked.loc[x]), reverse=True)
        if len(keep) > self.topk:
            keep = keep[: self.topk]

        need = max(0, self.topk - len(keep))
        add_list = [s for s in ranked.index if s not in keep][:need]
        target = keep + add_list
        if not target:
            return {}

        return self._calc_weights(ranked_score=ranked, target=target)


@dataclass
class ScanResult:
    family: str
    tag: str
    topk: int
    hold_topk: int
    n_drop: int
    rebalance_mode: str
    rebalance_interval: int
    n_drop_schedule: str
    weight_mode: str
    score_power: float
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
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _load_pickle(path)


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
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _get_report_for_day_freq(portfolio_metric_dict):
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    first_key = next(iter(portfolio_metric_dict.keys()))
    return portfolio_metric_dict[first_key][0]


def _calc_costed_metrics(report_df) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, max_drawdown, turnover


def _parse_int_list(text: str) -> List[int]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_mode_list(text: str) -> List[str]:
    out = []
    for x in text.split(","):
        x = x.strip().lower()
        if x:
            out.append(x)
    if not out:
        raise ValueError("empty mode list")
    return out


def _parse_schedule_specs(text: str) -> List[List[int]]:
    # Use ';' between schedules to avoid ambiguity with comma.
    # Example: "none;1,2,3;2,2,3,3"
    specs = []
    for seg in text.split(";"):
        seg = seg.strip().lower()
        if not seg or seg == "none":
            specs.append([])
            continue
        specs.append([int(x.strip()) for x in seg.split(",") if x.strip()])
    if not specs:
        return [[]]
    return specs


def _write_scan_csv(path: Path, rows: Sequence[ScanResult]) -> None:
    fallback = ScanResult(
        family="",
        tag="",
        topk=0,
        hold_topk=0,
        n_drop=0,
        rebalance_mode="",
        rebalance_interval=0,
        n_drop_schedule="",
        weight_mode="",
        score_power=0.0,
        open_cost=0.0,
        close_cost=0.0,
        costed_annret=0.0,
        costed_ir=0.0,
        max_drawdown=0.0,
        turnover=0.0,
        elapsed_sec=0.0,
    )
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(fallback).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_strategy_object(
    combo: Dict[str, Any],
    pred_df,
    base_strategy_kwargs: Dict[str, Any],
):
    common_kwargs = {"signal": pred_df}
    if "risk_degree" in base_strategy_kwargs:
        common_kwargs["risk_degree"] = float(base_strategy_kwargs["risk_degree"])

    if combo["family"] == "topk_dropout_sched":
        kwargs = {
            "topk": int(combo["topk"]),
            "n_drop": int(combo["n_drop"]),
            "rebalance_mode": combo["rebalance_mode"],
            "rebalance_interval": int(combo["rebalance_interval"]),
            "n_drop_schedule": combo["n_drop_schedule"],
            "method_sell": base_strategy_kwargs.get("method_sell", "bottom"),
            "method_buy": base_strategy_kwargs.get("method_buy", "top"),
            "hold_thresh": int(base_strategy_kwargs.get("hold_thresh", 1)),
            "only_tradable": bool(base_strategy_kwargs.get("only_tradable", False)),
            "forbid_all_trade_at_limit": bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
        }
        kwargs.update(common_kwargs)
        return ScheduledTopkDropoutStrategy(**kwargs)

    if combo["family"] == "buffered_weight":
        kwargs = {
            "topk": int(combo["topk"]),
            "hold_topk": int(combo["hold_topk"]),
            "weight_mode": combo["weight_mode"],
            "score_power": float(combo["score_power"]),
            "rebalance_mode": combo["rebalance_mode"],
            "rebalance_interval": int(combo["rebalance_interval"]),
        }
        kwargs.update(common_kwargs)
        return BufferedTopkWeightStrategy(**kwargs)

    raise ValueError(f"unsupported family={combo['family']}")


def _run_one_scan(
    combo: Dict[str, Any],
    pred_df,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> ScanResult:
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )

    strategy_obj = _build_strategy_object(combo=combo, pred_df=pred_df, base_strategy_kwargs=base_strategy_kwargs)

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["benchmark"] = "SH000300"

    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
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
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark="SH000300",
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    report_df = _get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report_df)
    return ScanResult(
        family=str(combo["family"]),
        tag=str(combo["tag"]),
        topk=int(combo["topk"]),
        hold_topk=int(combo.get("hold_topk", combo["topk"])),
        n_drop=int(combo.get("n_drop", 0)),
        rebalance_mode=str(combo["rebalance_mode"]),
        rebalance_interval=int(combo["rebalance_interval"]),
        n_drop_schedule=",".join(str(x) for x in combo.get("n_drop_schedule", [])),
        weight_mode=str(combo.get("weight_mode", "")),
        score_power=float(combo.get("score_power", 1.0)),
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        costed_annret=annret,
        costed_ir=ir,
        max_drawdown=maxdd,
        turnover=turnover,
        elapsed_sec=elapsed,
    )


def _build_combos(args) -> List[Dict[str, Any]]:
    topks = _parse_int_list(args.topk_grid)
    n_drops = _parse_int_list(args.n_drop_grid)
    hold_gaps = _parse_int_list(args.hold_gap_grid)
    rebalance_modes = _parse_mode_list(args.rebalance_modes)
    schedules = _parse_schedule_specs(args.n_drop_schedules)

    combos: List[Dict[str, Any]] = []
    for topk in topks:
        for n_drop in n_drops:
            for mode in rebalance_modes:
                for sched in schedules:
                    combos.append(
                        {
                            "family": "topk_dropout_sched",
                            "tag": f"topkdrop_tk{topk}_nd{n_drop}_{mode}_sch{','.join(map(str, sched)) or 'none'}",
                            "topk": topk,
                            "n_drop": n_drop,
                            "rebalance_mode": mode,
                            "rebalance_interval": int(args.rebalance_interval),
                            "n_drop_schedule": sched,
                        }
                    )

    for topk in topks:
        for gap in hold_gaps:
            hold_topk = topk + max(0, gap)
            for mode in rebalance_modes:
                for weight_mode in ("equal", "score"):
                    combos.append(
                        {
                            "family": "buffered_weight",
                            "tag": f"buffered_tk{topk}_hk{hold_topk}_{weight_mode}_{mode}",
                            "topk": topk,
                            "hold_topk": hold_topk,
                            "weight_mode": weight_mode,
                            "score_power": float(args.score_power),
                            "rebalance_mode": mode,
                            "rebalance_interval": int(args.rebalance_interval),
                        }
                    )
    return combos


def _dedup_keep_order(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in combos:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline portfolio structure scan from existing signal run.")
    parser.add_argument("--run-id", default=TARGET_RUN_ID, help="MLflow run_id that contains artifacts/pred.pkl.")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--config-path", default="", help="Optional workflow config path; default uses run artifact config.")
    parser.add_argument("--topk-grid", default="35,40,45", help="Comma-separated topk values.")
    parser.add_argument("--n-drop-grid", default="1,2,3", help="Comma-separated n_drop values.")
    parser.add_argument("--hold-gap-grid", default="0,10,20", help="Comma-separated hold_topk-topk values.")
    parser.add_argument("--rebalance-modes", default="daily,weekly", help="daily,weekly,monthly,interval")
    parser.add_argument("--rebalance-interval", type=int, default=5, help="Used only when rebalance_mode=interval.")
    parser.add_argument(
        "--n-drop-schedules",
        default="none;1,2,3",
        help="';'-separated schedule specs. Example: none;1,2,3;2,2,3,3",
    )
    parser.add_argument("--score-power", type=float, default=1.0, help="Exponent for score weighting.")
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--max-combos", type=int, default=0, help="Optional hard cap on number of evaluated combos.")
    parser.add_argument("--output-prefix", default="portfolio_innov_scan", help="Output file prefix under Transcendence.")
    return parser


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
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    combos = _dedup_keep_order(_build_combos(args))
    if args.max_combos > 0:
        combos = combos[: int(args.max_combos)]

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    results: List[ScanResult] = []
    failed: List[Dict[str, Any]] = []
    print(f"scan combos: {len(combos)}", flush=True)
    for idx, combo in enumerate(combos, start=1):
        try:
            row = _run_one_scan(
                combo=combo,
                pred_df=pred_df,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                exchange_cache=exchange_cache,
            )
            results.append(row)
            print(
                f"[scan {idx}/{len(combos)}] {row.tag} IR={row.costed_ir:.6f} AnnRet={row.costed_annret:.6f} "
                f"maxDD={row.max_drawdown:.6f} turnover={row.turnover:.6f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            failed.append({"combo": combo, "error": str(e)})
            print(f"[scan {idx}/{len(combos)}][failed] {combo.get('tag', '')} error={e}", flush=True)

    if not results:
        raise RuntimeError("no result rows produced")

    results_sorted = sorted(results, key=lambda x: (x.costed_ir, x.costed_annret), reverse=True)
    best = results_sorted[0]
    best_by_family: Dict[str, Dict[str, Any]] = {}
    for family in sorted(set(r.family for r in results_sorted)):
        fam_rows = [r for r in results_sorted if r.family == family]
        if fam_rows:
            best_by_family[family] = asdict(fam_rows[0])

    baseline_report_path = artifacts_dir / "portfolio_analysis" / "report_normal_1day.pkl"
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
        baseline = {"costed_annret": None, "costed_ir": None, "max_drawdown": None, "turnover": None}

    run_short = args.run_id[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"{args.output_prefix}_{run_short}_{stamp}.csv"
    _write_scan_csv(csv_path, results_sorted)

    summary = {
        "run_id": args.run_id,
        "tracking_uri": args.tracking_uri,
        "artifact_dir": str(artifacts_dir),
        "config_path": str(run_cfg_path),
        "scan_time_utc": _now_utc(),
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "benchmark": "SH000300",
        "combo_count": len(combos),
        "success_count": len(results),
        "failed_count": len(failed),
        "baseline_7406": baseline,
        "target_costed_ir": TARGET_COSTED_IR,
        "best": asdict(best),
        "best_by_family": best_by_family,
        "better_than_7406": bool(best.costed_ir > TARGET_COSTED_IR + 1e-9),
        "better_than_3": bool(best.costed_ir > 3.0),
        "scan_csv": str(csv_path),
        "failed_examples": failed[:20],
        "ic": _parse_metric_file(run_dir / "metrics" / "IC"),
        "rank_ic": _parse_metric_file(run_dir / "metrics" / "Rank IC"),
    }
    summary_path = out_dir / f"{args.output_prefix}_summary_{run_short}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
