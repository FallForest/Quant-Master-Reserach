#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


TARGET_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

RUN_ALIAS = {
    "7406": "7406e47063e9479cb34d300b9ed03bad",
    "773": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "1a085": "1a085ff9b5a34f408a44ad74055fc5da",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    cands = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not cands:
        cands = [p for p in tracking_dir.glob(f"*/{run_id}*") if (p / "artifacts").exists()]
    if not cands:
        raise FileNotFoundError(f"run_id not found: {run_id}")
    if len(cands) > 1:
        exact = [p for p in cands if p.name == run_id]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"ambiguous run_id {run_id}: {[str(p) for p in cands]}")
    return cands[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _load_pickle(path)


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            cfg = rec.get("kwargs", {}).get("config")
            if isinstance(cfg, dict):
                return copy.deepcopy(cfg)
    raise KeyError("cannot find portfolio analysis config")


def _as_score_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.astype(float).to_frame("score")
    if isinstance(obj, pd.DataFrame):
        if "score" in obj.columns:
            return obj[["score"]].astype(float)
        if obj.shape[1] == 1:
            return obj.iloc[:, [0]].rename(columns={obj.columns[0]: "score"}).astype(float)
        return obj.iloc[:, [0]].rename(columns={obj.columns[0]: "score"}).astype(float)
    raise TypeError(f"unsupported pred type: {type(obj)}")


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(df.index.get_level_values(0))
    return df.loc[(dates >= start) & (dates <= end)]


def _rank_pct(score: pd.Series) -> pd.Series:
    if isinstance(score.index, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.rank(method="average", pct=True)


def _rank_ensemble(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        pred = _as_score_df(_load_pickle(_find_run_dir(tracking_dir, run_id) / "artifacts" / "pred.pkl"))
        pred = _slice_df(pred, start, end)
        s = _rank_pct(pred["score"])
        s.name = run_id
        cols.append(s)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


def _signal_dispersion(pred_df: pd.DataFrame) -> pd.Series:
    return pred_df["score"].astype(float).groupby(level=0).std().sort_index()


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, float]:
    excess = report["return"] - report["bench"] - report["cost"]
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].mean()),
    }


def _get_day_report(pm: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in pm:
        return pm["1day"][0]
    if "day" in pm:
        return pm["day"][0]
    return pm[next(iter(pm.keys()))][0]


def _eval_topk(
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = start
    backtest_cfg["end_time"] = end
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = open_cost
    exchange_kwargs["close_cost"] = close_cost
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs["exchange"] = get_exchange(
        freq=freq,
        start_time=start,
        end_time=end,
        deal_price=str(exchange_kwargs.get("deal_price", "close")),
        limit_threshold=float(exchange_kwargs.get("limit_threshold", 0.095)),
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=float(exchange_kwargs.get("min_cost", 5)),
    )
    strategy = TopkDropoutStrategy(
        signal=_slice_df(pred_df, pd.Timestamp(start), pd.Timestamp(end)),
        topk=int(topk),
        n_drop=int(n_drop),
        method_sell=base_strategy_kwargs.get("method_sell", "bottom"),
        method_buy=base_strategy_kwargs.get("method_buy", "top"),
        hold_thresh=int(base_strategy_kwargs.get("hold_thresh", 1)),
        only_tradable=bool(base_strategy_kwargs.get("only_tradable", False)),
        forbid_all_trade_at_limit=bool(base_strategy_kwargs.get("forbid_all_trade_at_limit", True)),
    )
    t0 = time.perf_counter()
    pm, _ = run_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = _get_day_report(pm)
    metrics = _metrics_from_report(report)
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    return {"metrics": metrics, "report": report, "excess": report["return"] - report["bench"] - report["cost"]}


def _safe_eval(*args, **kwargs) -> Dict[str, Any]:
    try:
        out = _eval_topk(*args, **kwargs)
        return {"ok": True, "metrics": out["metrics"], "report": out["report"], "excess": out["excess"], "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "metrics": None, "report": None, "excess": None, "error": {"type": type(exc).__name__, "message": str(exc)}}


def _safe_quantile(hist: pd.Series, q: float) -> float:
    s = hist.dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def _causal_select(
    reports: Dict[str, Dict[str, Any]],
    dispersions: Dict[str, pd.Series],
    warmup_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    idx = pd.DatetimeIndex(reports["base45"]["report"].index)
    bench = reports["base45"]["report"]["bench"].astype(float)
    bench_vol20 = bench.rolling(20).std().shift(1)
    selected = pd.Series(index=idx, dtype=object)
    rows: List[Dict[str, Any]] = []
    prev = "base45"
    for i, dt in enumerate(idx):
        if i < warmup_days:
            sid = prev
            reason = "warmup"
        else:
            vol_now = float(bench_vol20.iloc[i]) if pd.notna(bench_vol20.iloc[i]) else float("nan")
            vol_thr = _safe_quantile(bench_vol20.iloc[:i], 0.75)
            base_disp = dispersions["base45"].reindex(idx)
            disp_now = float(base_disp.iloc[i]) if pd.notna(base_disp.iloc[i]) else float("nan")
            disp_thr = _safe_quantile(base_disp.iloc[:i], 0.35)
            score_map = {}
            for sid0, rec in reports.items():
                ex = rec["excess"]
                rep = rec["report"]
                hist5 = ex.iloc[max(0, i - 5) : i]
                hist20 = ex.iloc[max(0, i - 20) : i]
                ret5 = float(hist5.mean() * 252.0) if len(hist5) else 0.0
                ret20 = float(hist20.mean() * 252.0) if len(hist20) else 0.0
                vol20 = float(hist20.std(ddof=0) * np.sqrt(252.0)) if len(hist20) > 1 else 0.0
                turn5 = float(rep["turnover"].iloc[max(0, i - 5) : i].mean()) if i > 0 else 0.0
                disp_s = dispersions[sid0].reindex(idx)
                disp_now_s = float(disp_s.iloc[i]) if pd.notna(disp_s.iloc[i]) else 0.0
                disp_med20 = float(disp_s.iloc[max(0, i - 20) : i].median()) if i > 0 else 0.0
                disp_ratio = disp_now_s / (disp_med20 + 1e-12) if np.isfinite(disp_med20) else 1.0
                score_map[sid0] = 0.8 * ret5 + 0.7 * ret20 - 0.55 * vol20 - 0.14 * turn5 + 0.06 * (disp_ratio - 1.0)
            high_vol = np.isfinite(vol_now) and np.isfinite(vol_thr) and vol_now > vol_thr
            low_disp = np.isfinite(disp_now) and np.isfinite(disp_thr) and disp_now < disp_thr
            candidate = "base40" if (high_vol or low_disp) else max(score_map, key=score_map.get)
            sid = prev if candidate != prev and score_map[candidate] < score_map[prev] + 0.02 else candidate
            reason = "hysteresis_hold" if sid == prev and candidate != prev else "rule_select"
            rows.append(
                {
                    "date": str(pd.Timestamp(dt).date()),
                    "selected_signal": sid,
                    "candidate_signal": candidate,
                    "prev_signal": prev,
                    "reason": reason,
                    "high_vol": bool(high_vol),
                    "low_disp": bool(low_disp),
                    "candidate_score": float(score_map.get(candidate, np.nan)),
                    "prev_score": float(score_map.get(prev, np.nan)),
                }
            )
        selected.loc[dt] = sid
        prev = sid
    return selected, rows


def _build_dynamic_signal(signals: Dict[str, pd.DataFrame], selected: pd.Series, sharpen: bool) -> pd.DataFrame:
    parts = []
    for dt, sid in selected.items():
        source = signals[str(sid)]
        try:
            day = source.xs(dt, level=0).copy()
        except KeyError:
            continue
        score = _rank_pct(day["score"])
        if sharpen and str(sid) == "base40":
            order = score.sort_values(ascending=False)
            mask_top = pd.Series(0.0, index=order.index)
            mask_top.iloc[:40] = 1.0
            score = score + 0.08 * mask_top.reindex(score.index).fillna(0.0)
        mi = pd.MultiIndex.from_product([[pd.Timestamp(dt)], score.index.astype(str)], names=["datetime", "instrument"])
        parts.append(pd.Series(score.to_numpy(dtype=float), index=mi, name="score"))
    if not parts:
        raise RuntimeError("dynamic signal is empty")
    return pd.concat(parts).sort_index().to_frame("score")


def _series_metrics(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal signal-side regime blend candidate.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=TARGET_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--warmup-days", type=int, default=25)
    p.add_argument("--output-prefix", default="causal_signal_regime_blend")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    trans_dir = Path(__file__).resolve().parent
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    stamp = _stamp()

    base_dir = _find_run_dir(tracking_dir, args.base_run_id)
    cfg = _load_config(base_dir / "artifacts" / "config")
    _init_quant_master(cfg)
    port_cfg = _extract_port_config(cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    base = _as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    dates = pd.to_datetime(base.index.get_level_values(0))
    start, end = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    rank_a = _rank_ensemble(tracking_dir, [RUN_ALIAS["7406"], RUN_ALIAS["773"], RUN_ALIAS["bc641"]], [0.6, 0.2, 0.2], start, end)
    rank_b = _rank_ensemble(tracking_dir, [RUN_ALIAS["7406"], RUN_ALIAS["1a085"], RUN_ALIAS["773"]], [0.4, 0.2, 0.4], start, end)
    signals = {
        "base45": base,
        "base40": base,
        "base50": base,
        "rank45": rank_a,
        "rank50": rank_a,
        "gru45": rank_b,
    }
    strategy_defs = {
        "base45": (base, 45, 4),
        "base40": (base, 40, 2),
        "base50": (base, 50, 5),
        "rank45": (rank_a, 45, 4),
        "rank50": (rank_a, 50, 5),
        "gru45": (rank_b, 45, 4),
    }

    reports: Dict[str, Dict[str, Any]] = {}
    slice_rows: List[Dict[str, Any]] = []
    for sid, (sig, topk, ndrop) in strategy_defs.items():
        ev = _safe_eval(sig, port_cfg, strategy_kwargs, str(start.date()), str(end.date()), topk, ndrop, args.open_cost, args.close_cost)
        if not ev["ok"]:
            raise RuntimeError(f"predeclared strategy failed: {sid}: {ev['error']}")
        reports[sid] = {"report": ev["report"], "excess": ev["excess"], "metrics": ev["metrics"]}
        rec = {"strategy_id": sid, **ev["metrics"]}
        slice_rows.append(rec)

    dispersions = {sid: _signal_dispersion(sig) for sid, sig in signals.items()}
    selected, selector_rows = _causal_select(reports, dispersions, warmup_days=args.warmup_days)
    dyn_plain = _build_dynamic_signal(signals, selected, sharpen=False)
    dyn_sharp = _build_dynamic_signal(signals, selected, sharpen=True)

    eval_defs = [
        ("dynamic_plain_tk45_nd4", dyn_plain, 45, 4),
        ("dynamic_plain_tk50_nd5", dyn_plain, 50, 5),
        ("dynamic_sharp_tk45_nd4", dyn_sharp, 45, 4),
        ("dynamic_sharp_tk50_nd5", dyn_sharp, 50, 5),
    ]
    candidate_rows = []
    best = None
    for sid, sig, topk, ndrop in eval_defs:
        ev = _safe_eval(sig, port_cfg, strategy_kwargs, str(start.date()), str(end.date()), topk, ndrop, args.open_cost, args.close_cost)
        row = {"candidate_id": sid, "topk": topk, "n_drop": ndrop, "ok": bool(ev["ok"]), "error": ev["error"]}
        if ev["ok"]:
            row.update(ev["metrics"])
            if best is None or float(ev["metrics"]["ir"]) > float(best["ir"]):
                best = {**row, "signal_df": sig}
        candidate_rows.append(row)

    hard_pass = bool(best and float(best["ir"]) > HARD_GATE_IR and float(best["annret"]) > HARD_GATE_ANNRET)
    verdict = "BREAKTHROUGH" if hard_pass else "NO_GO"

    pred_path = trans_dir / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    selector_csv = trans_dir / f"{args.output_prefix}_selector_{stamp}.csv"
    eval_csv = trans_dir / f"{args.output_prefix}_eval_{stamp}.csv"
    if best is not None:
        with pred_path.open("wb") as f:
            pickle.dump(best["signal_df"]["score"], f)
    _write_csv(selector_csv, selector_rows)
    _write_csv(eval_csv, [{k: v for k, v in row.items() if k != "error"} | {"error": json.dumps(row.get("error"), ensure_ascii=False)} for row in candidate_rows])

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "causal_signal_side_regime_blend",
        "verdict": verdict,
        "hard_gate_pass": hard_pass,
        "protocol": {
            "selection": "causal t-1 diagnostics select among predeclared signal sources; no test-period parameter search",
            "candidate_eval_grid": [x[0] for x in eval_defs],
            "warmup_days": args.warmup_days,
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        },
        "predeclared_strategy_metrics": slice_rows,
        "candidate_rows": candidate_rows,
        "best_candidate": {k: v for k, v in best.items() if k != "signal_df"} if best else None,
        "selector_counts": selected.value_counts().to_dict(),
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "selector_csv": str(selector_csv),
            "eval_csv": str(eval_csv),
            "candidate_pred_pkl": str(pred_path) if best else None,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        f"# Causal Signal Regime Blend {stamp}",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Best candidate: `{summary['best_candidate']['candidate_id'] if summary['best_candidate'] else 'none'}`",
        f"- Best metrics: `{json.dumps(summary['best_candidate'], ensure_ascii=False)}`",
        f"- Hard gate pass: `{hard_pass}`",
        f"- Selector counts: `{json.dumps(summary['selector_counts'], ensure_ascii=False)}`",
    ]
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "hard_gate_pass": hard_pass, "best": summary["best_candidate"], "summary": str(summary_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
