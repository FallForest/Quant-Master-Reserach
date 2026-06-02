#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
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


RUN_ALIAS = {
    "e2300230": "e2300230e0994a1a9ccbbd3bc4606d97",
    "7406e470": "7406e47063e9479cb34d300b9ed03bad",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "bcbecf55": "bcbecf55a3924357ba93fc55b1140e99",
    "d4526da": "d4526da7854245af954fc99cf02963f0",
    "1a085ff": "1a085ff9b5a34f408a44ad74055fc5da",
    "05ef8bd1": "05ef8bd12e0e407f9fdf0cad3ef72652",
    "0ed35c": "0ed35c572e104ddab555a8af6a7fe981",
    "2ac6": "2ac6ebc249bf42e5a9f83c6ca0725941",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "94a52": "94a52e5949104218ab2a3b4cd84dce08",
    "6feaa": "6feaa8c5b0fc437784592bb7b534d710",
    "ae098013": "ae0980136de44dc58a0b9d3f7d947363",
}


@dataclass
class RunSignal:
    key: str
    base_key: str
    run_id: str
    pred_path: str
    series: pd.Series
    coverage_ratio: float
    source_ir: float | None
    source_annret: float | None
    model_class: str
    dataset_class: str
    instruments: str
    invertible: bool


@dataclass
class BlendCandidate:
    stage: str
    members: Tuple[str, ...]
    weights: Tuple[float, ...]
    heuristic_score: float
    parent_members: Tuple[str, ...] | None = None
    parent_weights: Tuple[float, ...] | None = None


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return resolve_provider_uri_in_config(yaml.safe_load(path.read_text(encoding="utf-8")), base_dir=path.parent)
    except UnicodeDecodeError:
        return _load_pickle(path)


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
    init_cfg.setdefault("provider_uri", "~/.quant_master/quant_master_data/tdx_cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _as_score_series(pred_obj: Any) -> pd.Series:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float)
    if isinstance(pred_obj, pd.DataFrame):
        if pred_obj.shape[1] == 1:
            return pred_obj.iloc[:, 0].astype(float)
        if "score" in pred_obj.columns:
            return pred_obj["score"].astype(float)
        return pred_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _slice_period(series: pd.Series, start: str, end: str) -> pd.Series:
    idx = series.index.get_level_values(0)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return series[mask]


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(method="average", pct=True)


def _iter_weight_compositions(n_parts: int, step: float) -> Iterable[Tuple[float, ...]]:
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"step={step} does not divide 1 exactly")

    def rec(remain: int, k: int) -> Iterable[Tuple[int, ...]]:
        if k == 1:
            yield (remain,)
            return
        for x in range(1, remain - (k - 1) + 1):
            for tail in rec(remain - x, k - 1):
                yield (x,) + tail

    for comp in rec(units, n_parts):
        yield tuple(round(x * step, 10) for x in comp)


def _calc_costed_metrics(report_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    annret = float(risk_df.loc["annualized_return", "risk"])
    ir = float(risk_df.loc["information_ratio", "risk"])
    max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
    turnover = float(report_df["turnover"].mean())
    return annret, ir, max_drawdown, turnover


def _build_exchange_cache_key(
    start_time: str, end_time: str, open_cost: float, close_cost: float, limit_threshold: float, deal_price: str
) -> Tuple[str, str, float, float, float, str]:
    return (start_time, end_time, open_cost, close_cost, limit_threshold, deal_price)


def _blend_scores(
    ranked_map: Dict[str, pd.Series], members: Sequence[str], weights: Sequence[float], base_index: pd.Index
) -> pd.Series:
    cols = []
    for k in members:
        cols.append(ranked_map[k].reindex(base_index))
    panel = pd.concat(cols, axis=1)
    panel.columns = list(members)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0))
    blend.name = "score"
    return blend.dropna()


def _run_portfolio(
    signal: pd.Series,
    base_port_cfg: Dict[str, Any],
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
    start_time: str,
    end_time: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[float, float, float, float]:
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
    strategy_cfg["kwargs"]["signal"] = signal
    strategy_cfg["kwargs"]["topk"] = int(topk)
    strategy_cfg["kwargs"]["n_drop"] = int(n_drop)

    backtest_cfg["start_time"] = start_time
    backtest_cfg["end_time"] = end_time

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    cache_key = _build_exchange_cache_key(
        start_time=start_time,
        end_time=end_time,
        open_cost=open_cost,
        close_cost=close_cost,
        limit_threshold=limit_threshold,
        deal_price=deal_price,
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=start_time,
            end_time=end_time,
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    portfolio_metric_dict, _ = run_backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy_cfg,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    if "1day" in portfolio_metric_dict:
        report_df = portfolio_metric_dict["1day"][0]
    else:
        key = next(iter(portfolio_metric_dict.keys()))
        report_df = portfolio_metric_dict[key][0]
    return _calc_costed_metrics(report_df)


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_run_tokens(text: str) -> List[str]:
    out = []
    for tok in [x.strip() for x in text.split(",") if x.strip()]:
        out.append(RUN_ALIAS.get(tok, tok))
    return out


def _resolve_run_ids(tracking_dir: Path, run_tokens: Sequence[str]) -> List[str]:
    all_run_ids = {
        p.name for p in tracking_dir.glob("*/*") if p.is_dir() and (p / "artifacts").exists() and len(p.name) == 32
    }
    resolved: List[str] = []
    for token in run_tokens:
        mapped = RUN_ALIAS.get(token, token)
        if mapped in all_run_ids:
            resolved.append(mapped)
            continue
        cands = [rid for rid in all_run_ids if rid.startswith(mapped)]
        if len(cands) == 1:
            resolved.append(cands[0])
            continue
        if len(cands) == 0:
            raise FileNotFoundError(f"run token cannot be resolved: {token}")
        raise RuntimeError(f"run token matches multiple run_ids: token={token}, candidates={cands}")
    # keep order, deduplicate
    return list(dict.fromkeys(resolved))


def _load_run_meta(run_dir: Path) -> Tuple[str, str, str]:
    cfg_path = run_dir / "artifacts" / "config"
    if not cfg_path.exists():
        return "", "", ""
    try:
        cfg = _load_config(cfg_path)
    except Exception:
        return "", "", ""
    if not isinstance(cfg, dict):
        return "", "", ""
    task = cfg.get("task", {})
    if not isinstance(task, dict):
        return "", "", ""
    model_cfg = task.get("model", {})
    dataset_cfg = task.get("dataset", {})
    model_class = str(model_cfg.get("class", "")) if isinstance(model_cfg, dict) else ""
    dataset_class = ""
    instruments = ""
    if isinstance(dataset_cfg, dict):
        kwargs = dataset_cfg.get("kwargs", {})
        if isinstance(kwargs, dict):
            handler = kwargs.get("handler", {})
            if isinstance(handler, dict):
                dataset_class = str(handler.get("class", ""))
                hkwargs = handler.get("kwargs", {})
                if isinstance(hkwargs, dict):
                    instruments = str(hkwargs.get("instruments", ""))
    return model_class, dataset_class, instruments


def _discover_pred_run_ids(
    tracking_dir: Path, comparable_instruments: str | None, require_comparable: bool
) -> List[str]:
    out: List[str] = []
    for run_dir in tracking_dir.glob("*/*"):
        if not run_dir.is_dir():
            continue
        if len(run_dir.name) != 32:
            continue
        if not (run_dir / "artifacts" / "pred.pkl").exists():
            continue
        if comparable_instruments and require_comparable:
            _, _, instruments = _load_run_meta(run_dir)
            if instruments and instruments != comparable_instruments:
                continue
        out.append(run_dir.name)
    return sorted(out)


def _extract_source_metrics(run_dir: Path) -> Tuple[float | None, float | None]:
    metric_dir = run_dir / "metrics"
    ir = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.information_ratio")
    ann = _parse_metric_file(metric_dir / "1day.excess_return_with_cost.annualized_return")
    if ir is None:
        ir = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.information_ratio")
    if ann is None:
        ann = _parse_metric_file(metric_dir / "1day.excess_return_without_cost.annualized_return")
    return ir, ann


def _make_unique_key(base_key: str, used: Dict[str, int]) -> str:
    if base_key not in used:
        used[base_key] = 1
        return base_key
    n = used[base_key]
    used[base_key] = n + 1
    return f"{base_key}_{n + 1}"


def _audit_signals(
    tracking_dir: Path,
    run_ids: Sequence[str],
    start_date: str,
    end_date: str,
    invertible_keywords: Sequence[str],
    explicit_invert_ids: Sequence[str],
) -> Tuple[List[RunSignal], List[Dict[str, Any]], pd.Index]:
    loaded: List[RunSignal] = []
    audit_rows: List[Dict[str, Any]] = []
    longest_index = None
    max_len = -1
    key_counts: Dict[str, int] = {}
    explicit_invert_set = set(explicit_invert_ids)
    for run_id in run_ids:
        run_dir = _find_run_dir(tracking_dir, run_id)
        artifacts_dir = run_dir / "artifacts"
        pred_path = artifacts_dir / "pred.pkl"
        model_class, dataset_class, instruments = _load_run_meta(run_dir)
        source_ir, source_annret = _extract_source_metrics(run_dir)
        base_key = next((k for k, v in RUN_ALIAS.items() if v == run_id), run_id[:8])
        key = _make_unique_key(base_key, key_counts)
        model_text = model_class.lower()
        invertible = run_id in explicit_invert_set or any(k in model_text for k in invertible_keywords)
        row = {
            "key": key,
            "base_key": base_key,
            "run_id": run_id,
            "pred_pkl": pred_path.exists(),
            "pred_path": str(pred_path).replace("\\", "/"),
            "source_ir": source_ir,
            "source_annret": source_annret,
            "model_class": model_class,
            "dataset_class": dataset_class,
            "instruments": instruments,
            "invertible": invertible,
        }
        if pred_path.exists():
            s = _slice_period(_as_score_series(_load_pickle(pred_path)), start_date, end_date)
            if len(s) > max_len:
                max_len = len(s)
                longest_index = s.index
            row["rows_in_period"] = int(len(s))
            row["date_min"] = str(s.index.get_level_values(0).min()) if len(s) else None
            row["date_max"] = str(s.index.get_level_values(0).max()) if len(s) else None
            row["coverage_ratio_vs_longest"] = None
            loaded.append(
                RunSignal(
                    key=key,
                    base_key=base_key,
                    run_id=run_id,
                    pred_path=row["pred_path"],
                    series=s,
                    coverage_ratio=0.0,
                    source_ir=source_ir,
                    source_annret=source_annret,
                    model_class=model_class,
                    dataset_class=dataset_class,
                    instruments=instruments,
                    invertible=invertible,
                )
            )
        audit_rows.append(row)

    if longest_index is None:
        raise RuntimeError("no run has usable pred.pkl")

    for sig in loaded:
        inter = longest_index.intersection(sig.series.index)
        sig.coverage_ratio = len(inter) / max(1, len(longest_index))
        for row in audit_rows:
            if row["run_id"] == sig.run_id and row["key"] == sig.key:
                row["coverage_ratio_vs_longest"] = sig.coverage_ratio
                break
    return loaded, audit_rows, longest_index


def _entropy(weights: Sequence[float]) -> float:
    e = 0.0
    for w in weights:
        if w > 0:
            e -= w * math.log(w)
    return e


def _normalize_weights(weights: Sequence[float]) -> Tuple[float, ...]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights sum must be positive")
    return tuple(float(w) / total for w in weights)


def _sample_dirichlet(alpha: Sequence[float], rng: random.Random) -> Tuple[float, ...]:
    draws = [rng.gammavariate(a, 1.0) for a in alpha]
    return _normalize_weights(draws)


def _weight_sig(weights: Sequence[float], nd: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(w), nd) for w in weights)


def _l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def _build_weight_cache(combo_sizes: Sequence[int], step: float) -> Dict[int, List[Tuple[float, ...]]]:
    out: Dict[int, List[Tuple[float, ...]]] = {}
    for k in combo_sizes:
        out[k] = list(_iter_weight_compositions(k, step))
    return out


def _build_member_maps(usable: Sequence[RunSignal]) -> Tuple[Dict[str, pd.Series], Dict[str, float], Dict[str, List[str]]]:
    ranked_map: Dict[str, pd.Series] = {}
    source_ir_map: Dict[str, float] = {}
    by_base: Dict[str, List[str]] = {}
    for x in usable:
        pos_key = x.key
        ranked_map[pos_key] = _cs_rank(x.series)
        source_ir_map[pos_key] = x.source_ir if x.source_ir is not None else 0.0
        choices = [pos_key]
        if x.invertible:
            neg_key = f"{x.key}_inv"
            ranked_map[neg_key] = _cs_rank(-x.series)
            src = x.source_ir if x.source_ir is not None else 0.0
            source_ir_map[neg_key] = -src
            choices.append(neg_key)
        by_base[x.key] = choices
    return ranked_map, source_ir_map, by_base


def _sample_members(
    base_keys: Sequence[str], by_base: Dict[str, List[str]], combo_sizes: Sequence[int], rng: random.Random
) -> Tuple[str, ...]:
    k = int(rng.choice(combo_sizes))
    chosen_base = rng.sample(list(base_keys), k)
    members = [rng.choice(by_base[b]) for b in chosen_base]
    return tuple(members)


def _build_stage1_candidates(
    base_keys: Sequence[str],
    by_base: Dict[str, List[str]],
    source_ir_map: Dict[str, float],
    combo_sizes: Sequence[int],
    weight_cache: Dict[int, List[Tuple[float, ...]]],
    random_grid_n: int,
    random_dirichlet_n: int,
    seed: int,
) -> List[BlendCandidate]:
    rng = random.Random(seed)
    out: List[BlendCandidate] = []
    seen = set()
    target = random_grid_n + random_dirichlet_n
    attempts = 0
    while len(out) < target and attempts < target * 120:
        attempts += 1
        members = _sample_members(base_keys=base_keys, by_base=by_base, combo_sizes=combo_sizes, rng=rng)
        k = len(members)
        mode = "grid" if len(out) < random_grid_n else "dirichlet"
        if mode == "grid":
            weights = rng.choice(weight_cache[k])
        else:
            weights = _sample_dirichlet([1.0] * k, rng=rng)
        sig = (members, _weight_sig(weights, nd=5))
        if sig in seen:
            continue
        seen.add(sig)
        base = sum(source_ir_map[m] * w for m, w in zip(members, weights))
        score = base + 0.04 * _entropy(weights)
        out.append(
            BlendCandidate(
                stage="stage1",
                members=members,
                weights=tuple(float(x) for x in weights),
                heuristic_score=score,
            )
        )
    return out


def _refine_one_structure(
    members: Tuple[str, ...],
    best_weights: Tuple[float, ...],
    source_ir_map: Dict[str, float],
    weight_cache: Dict[int, List[Tuple[float, ...]]],
    local_grid_radius: float,
    dirichlet_n: int,
    rng: random.Random,
) -> List[BlendCandidate]:
    k = len(members)
    out: List[BlendCandidate] = []
    seen = set()
    best_sig = _weight_sig(best_weights, nd=5)
    seen.add(best_sig)
    base = sum(source_ir_map[m] * w for m, w in zip(members, best_weights))
    out.append(
        BlendCandidate(
            stage="stage2",
            members=members,
            weights=best_weights,
            heuristic_score=base + 0.04 * _entropy(best_weights),
            parent_members=members,
            parent_weights=best_weights,
        )
    )

    local = []
    for w in weight_cache[k]:
        if _l1_distance(w, best_weights) <= local_grid_radius:
            local.append(w)
    local.sort(key=lambda x: _l1_distance(x, best_weights))
    for w in local:
        sig = _weight_sig(w, nd=5)
        if sig in seen:
            continue
        seen.add(sig)
        score = sum(source_ir_map[m] * x for m, x in zip(members, w)) + 0.04 * _entropy(w)
        out.append(
            BlendCandidate(
                stage="stage2",
                members=members,
                weights=tuple(float(x) for x in w),
                heuristic_score=score,
                parent_members=members,
                parent_weights=best_weights,
            )
        )
        if len(out) >= max(4, dirichlet_n // 2):
            break

    alpha = [max(1e-3, w * 30.0) + 1.0 for w in best_weights]
    for _ in range(dirichlet_n):
        w = _sample_dirichlet(alpha, rng=rng)
        sig = _weight_sig(w, nd=5)
        if sig in seen:
            continue
        seen.add(sig)
        score = sum(source_ir_map[m] * x for m, x in zip(members, w)) + 0.04 * _entropy(w)
        out.append(
            BlendCandidate(
                stage="stage2",
                members=members,
                weights=tuple(float(x) for x in w),
                heuristic_score=score,
                parent_members=members,
                parent_weights=best_weights,
            )
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pick_top_structures(stage1_rows: Sequence[Dict[str, Any]], keep_n: int) -> List[Dict[str, Any]]:
    sorted_rows = sorted(stage1_rows, key=lambda x: (x["ir"], x["annret"]), reverse=True)
    out = []
    seen = set()
    for r in sorted_rows:
        members = r["members"]
        if members in seen:
            continue
        seen.add(members)
        out.append(r)
        if len(out) >= keep_n:
            break
    return out


def _evaluate_candidates(
    candidates: Sequence[BlendCandidate],
    stage_name: str,
    ranked_map: Dict[str, pd.Series],
    base_index: pd.Index,
    base_port_cfg: Dict[str, Any],
    topk_grid: Sequence[int],
    n_drop_grid: Sequence[int],
    open_cost: float,
    close_cost: float,
    start_date: str,
    end_date: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    all_rows: List[Dict[str, Any]] = []
    global_best = None
    total = len(candidates)
    for i, cand in enumerate(candidates, start=1):
        signal = _blend_scores(ranked_map, cand.members, cand.weights, base_index=base_index)
        members_text = "|".join(cand.members)
        weights_text = "|".join(f"{x:.4f}" for x in cand.weights)
        local_best = None
        for topk in topk_grid:
            for n_drop in n_drop_grid:
                annret, ir, maxdd, turnover = _run_portfolio(
                    signal=signal,
                    base_port_cfg=base_port_cfg,
                    topk=topk,
                    n_drop=n_drop,
                    open_cost=open_cost,
                    close_cost=close_cost,
                    start_time=start_date,
                    end_time=end_date,
                    exchange_cache=exchange_cache,
                )
                row = {
                    "stage": stage_name,
                    "candidate_idx": i,
                    "members": members_text,
                    "weights": weights_text,
                    "topk": int(topk),
                    "n_drop": int(n_drop),
                    "annret": float(annret),
                    "ir": float(ir),
                    "max_drawdown": float(maxdd),
                    "turnover": float(turnover),
                    "heuristic_score": float(cand.heuristic_score),
                }
                if cand.parent_members is not None:
                    row["parent_members"] = "|".join(cand.parent_members)
                if cand.parent_weights is not None:
                    row["parent_weights"] = "|".join(f"{x:.4f}" for x in cand.parent_weights)
                all_rows.append(row)
                if local_best is None or (row["ir"], row["annret"]) > (local_best["ir"], local_best["annret"]):
                    local_best = row
                if global_best is None or (row["ir"], row["annret"]) > (global_best["ir"], global_best["annret"]):
                    global_best = row
        print(
            f"[{stage_name} {i}/{total}] members={members_text} weights={weights_text} "
            f"bestIR={local_best['ir']:.6f} bestAnnRet={local_best['annret']:.6f}",
            flush=True,
        )
    return all_rows, global_best


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Broad two-stage cross-sectional rank ensemble search from mlruns.")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument(
        "--run-ids",
        default="7406e470,773bd6d,e2300230,1a085ff,05ef8bd1,d4526da,bcbecf55,0ed35c,2ac6,bc641,94a52,6feaa,ae098013",
        help="Comma separated run ids or aliases.",
    )
    parser.add_argument(
        "--invert-run-ids",
        default="",
        help="Comma separated run ids/aliases/prefixes to force inversion support on those runs.",
    )
    parser.add_argument("--base-run-id", default="7406e47063e9479cb34d300b9ed03bad")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--open-cost", type=float, default=0.0001)
    parser.add_argument("--close-cost", type=float, default=0.0006)
    parser.add_argument("--topk-grid", default="25,30,35,40,45,50,55,60,65,70")
    parser.add_argument("--n-drop-grid", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--stage1-topk-grid", default="45")
    parser.add_argument("--stage1-n-drop-grid", default="4")
    parser.add_argument("--combo-sizes", default="2,3,4,5")
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--stage1-random-grid-blends", type=int, default=90)
    parser.add_argument("--stage1-random-dirichlet-blends", type=int, default=170)
    parser.add_argument("--stage1-seed", type=int, default=20260520)
    parser.add_argument("--stage2-keep-structures", type=int, default=3)
    parser.add_argument("--stage2-max-blends", type=int, default=24)
    parser.add_argument("--stage2-local-grid-radius", type=float, default=0.5)
    parser.add_argument("--stage2-dirichlet-per-structure", type=int, default=24)
    parser.add_argument("--min-coverage", type=float, default=0.35)
    parser.add_argument(
        "--include-all-pred-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto include all pred.pkl runs in mlruns (filtered by comparability setting).",
    )
    parser.add_argument(
        "--require-comparable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When auto-discovering runs, keep only comparable instruments.",
    )
    parser.add_argument("--comparable-instruments", default="csi300")
    parser.add_argument("--invertible-model-keywords", default="gru,metalabel,topkmetalabel")
    parser.add_argument("--output-prefix", default="rank_ensemble_scan_broad")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)

    requested = _resolve_run_ids(tracking_dir, _parse_run_tokens(args.run_ids))
    discovered: List[str] = []
    if args.include_all_pred_runs:
        discovered = _discover_pred_run_ids(
            tracking_dir=tracking_dir,
            comparable_instruments=args.comparable_instruments,
            require_comparable=bool(args.require_comparable),
        )

    run_ids = list(dict.fromkeys(requested + discovered))
    invert_run_ids = _resolve_run_ids(tracking_dir, _parse_run_tokens(args.invert_run_ids)) if args.invert_run_ids else []
    invertible_keywords = tuple(x.lower() for x in _parse_str_list(args.invertible_model_keywords))
    combo_sizes = _parse_int_list(args.combo_sizes)
    topk_grid = _parse_int_list(args.topk_grid)
    n_drop_grid = _parse_int_list(args.n_drop_grid)
    stage1_topk_grid = _parse_int_list(args.stage1_topk_grid)
    stage1_n_drop_grid = _parse_int_list(args.stage1_n_drop_grid)

    loaded, audit_rows, base_index = _audit_signals(
        tracking_dir=tracking_dir,
        run_ids=run_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        invertible_keywords=invertible_keywords,
        explicit_invert_ids=invert_run_ids,
    )

    usable = [x for x in loaded if x.coverage_ratio >= args.min_coverage]
    if len(usable) < 2:
        raise RuntimeError("usable signals < 2 after coverage filter")

    run_cfg_dir = _find_run_dir(tracking_dir, args.base_run_id)
    workflow_cfg = _load_config(run_cfg_dir / "artifacts" / "config")
    _init_quant_master(workflow_cfg)
    base_port_cfg = _extract_port_config(workflow_cfg)

    ranked_map, source_ir_map, by_base = _build_member_maps(usable)
    base_keys = list(by_base.keys())
    if len(base_keys) < 2:
        raise RuntimeError("usable base signals < 2")

    weight_cache = _build_weight_cache(combo_sizes=combo_sizes, step=args.weight_step)
    stage1_candidates = _build_stage1_candidates(
        base_keys=base_keys,
        by_base=by_base,
        source_ir_map=source_ir_map,
        combo_sizes=combo_sizes,
        weight_cache=weight_cache,
        random_grid_n=int(args.stage1_random_grid_blends),
        random_dirichlet_n=int(args.stage1_random_dirichlet_blends),
        seed=int(args.stage1_seed),
    )
    if not stage1_candidates:
        raise RuntimeError("stage1 has no candidates")

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = out_dir / f"{args.output_prefix}_artifact_audit_{stamp}.json"
    results_path = out_dir / f"{args.output_prefix}_results_{stamp}.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    candidate_path = out_dir / f"{args.output_prefix}_candidate_{stamp}.json"

    audit_obj = {
        "timestamp_utc": _now_utc(),
        "tracking_dir": str(tracking_dir),
        "requested_run_ids": requested,
        "auto_discovered_run_ids": discovered,
        "final_run_ids": run_ids,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": audit_rows,
    }
    audit_path.write_text(json.dumps(audit_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    stage1_rows, stage1_best = _evaluate_candidates(
        candidates=stage1_candidates,
        stage_name="stage1",
        ranked_map=ranked_map,
        base_index=base_index,
        base_port_cfg=base_port_cfg,
        topk_grid=stage1_topk_grid,
        n_drop_grid=stage1_n_drop_grid,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        start_date=args.start_date,
        end_date=args.end_date,
        exchange_cache=exchange_cache,
    )

    top_structures = _pick_top_structures(stage1_rows, keep_n=int(args.stage2_keep_structures))
    rng_stage2 = random.Random(int(args.stage1_seed) + 911)
    stage2_candidates: List[BlendCandidate] = []
    for r in top_structures:
        members = tuple(r["members"].split("|"))
        best_weights = tuple(float(x) for x in r["weights"].split("|"))
        refined = _refine_one_structure(
            members=members,
            best_weights=best_weights,
            source_ir_map=source_ir_map,
            weight_cache=weight_cache,
            local_grid_radius=float(args.stage2_local_grid_radius),
            dirichlet_n=int(args.stage2_dirichlet_per_structure),
            rng=rng_stage2,
        )
        stage2_candidates.extend(refined)

    # Deduplicate and cap stage2 blends by heuristic score.
    uniq_stage2: List[BlendCandidate] = []
    seen_stage2 = set()
    for c in sorted(stage2_candidates, key=lambda x: x.heuristic_score, reverse=True):
        sig = (c.members, _weight_sig(c.weights, nd=5))
        if sig in seen_stage2:
            continue
        seen_stage2.add(sig)
        uniq_stage2.append(c)
        if len(uniq_stage2) >= int(args.stage2_max_blends):
            break
    if not uniq_stage2:
        raise RuntimeError("stage2 has no candidates")

    stage2_rows, stage2_best = _evaluate_candidates(
        candidates=uniq_stage2,
        stage_name="stage2",
        ranked_map=ranked_map,
        base_index=base_index,
        base_port_cfg=base_port_cfg,
        topk_grid=topk_grid,
        n_drop_grid=n_drop_grid,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        start_date=args.start_date,
        end_date=args.end_date,
        exchange_cache=exchange_cache,
    )

    all_rows = stage1_rows + stage2_rows
    all_rows.sort(key=lambda x: (x["ir"], x["annret"]), reverse=True)
    _write_csv(results_path, all_rows)

    best_row = stage2_best or stage1_best
    meaningful = bool(best_row and best_row["ir"] > 2.90 and best_row["annret"] > 0.27)
    ideal = bool(best_row and best_row["ir"] > 3.0 and best_row["annret"] > 0.30)
    if best_row:
        candidate_path.write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "timestamp_utc": _now_utc(),
        "tracking_uri": args.tracking_uri,
        "selection_notice": (
            f"Exploratory scan: weights/topk/n_drop selected directly on test period "
            f"{args.start_date}..{args.end_date}."
        ),
        "run_pool": {
            "requested_run_ids": requested,
            "auto_discovered_run_ids": discovered,
            "final_run_ids": run_ids,
            "usable_signal_count": len(usable),
            "usable_signals": [
                {
                    "key": x.key,
                    "base_key": x.base_key,
                    "run_id": x.run_id,
                    "coverage_ratio": x.coverage_ratio,
                    "source_ir": x.source_ir,
                    "source_annret": x.source_annret,
                    "model_class": x.model_class,
                    "dataset_class": x.dataset_class,
                    "instruments": x.instruments,
                    "invertible": x.invertible,
                    "pred_path": x.pred_path,
                }
                for x in usable
            ],
        },
        "search_space": {
            "combo_sizes": combo_sizes,
            "weight_step": args.weight_step,
            "stage1_random_grid_blends": args.stage1_random_grid_blends,
            "stage1_random_dirichlet_blends": args.stage1_random_dirichlet_blends,
            "stage1_topk_grid": stage1_topk_grid,
            "stage1_n_drop_grid": stage1_n_drop_grid,
            "stage2_keep_structures": args.stage2_keep_structures,
            "stage2_max_blends": args.stage2_max_blends,
            "stage2_dirichlet_per_structure": args.stage2_dirichlet_per_structure,
            "stage2_local_grid_radius": args.stage2_local_grid_radius,
            "topk_grid": topk_grid,
            "n_drop_grid": n_drop_grid,
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "stage_best": {"stage1": stage1_best, "stage2": stage2_best},
        "best": best_row,
        "meaningful_breakthrough": meaningful,
        "ideal_breakthrough": ideal,
        "thresholds": {
            "minimum_ir": 2.90,
            "minimum_annret": 0.27,
            "ideal_ir": 3.0,
            "ideal_annret": 0.30,
        },
        "benchmarks": {
            "portfolio_sota_ir": 2.79998,
            "portfolio_sota_annret": 0.24466,
        },
        "comparison": {
            "beat_portfolio_sota_ir": bool(best_row and best_row["ir"] > 2.79998),
            "beat_portfolio_sota_annret": bool(best_row and best_row["annret"] > 0.24466),
            "beat_minimum_breakthrough_threshold": meaningful,
            "beat_ideal_breakthrough_threshold": ideal,
        },
        "artifacts": {
            "audit_json": str(audit_path),
            "results_csv": str(results_path),
            "summary_json": str(summary_path),
            "candidate_json": str(candidate_path) if best_row else None,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

