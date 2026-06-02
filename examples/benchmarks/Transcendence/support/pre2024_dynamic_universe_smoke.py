#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
from quant_master.config import resolve_provider_uri

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]

YEARS = (2020, 2021, 2022, 2023)
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2023-12-31")
MARKETS = ("csi300", "csi500", "csi800", "csi1000", "csiall", "all")
FIELDS = ("amount", "volume", "change", "factor")
DEFAULT_PROVIDER_URI = Path("~/.quant_master/quant_master_data/tdx_cn_data")

GATE_MIN_POOL_SIZE = 800
GATE_MAX_MISSING_RATE = 0.10
DEFAULT_DYNAMIC_TOPK = 1000
DEFAULT_MIN_AMOUNT_QUANTILE = 0.60


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(_json_sanitize(list(rows)))


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "universe_csv": THIS_DIR / f"{output_prefix}_universe_coverage_{stamp}.csv",
        "year_metrics_csv": THIS_DIR / f"{output_prefix}_year_metrics_{stamp}.csv",
        "rule_csv": THIS_DIR / f"{output_prefix}_rule_candidates_{stamp}.csv",
    }


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    cal = pd.to_datetime(pd.Index(vals))
    if (cal > END_DATE).any():
        cal = cal[cal <= END_DATE]
    return cal


def _parse_instrument_intervals(inst_path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if not inst_path.exists():
        return out
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        inst = parts[0].lower()
        st = pd.Timestamp(parts[1])
        ed = min(pd.Timestamp(parts[2]), END_DATE)
        if ed < START_DATE:
            continue
        out.setdefault(inst, []).append((st, ed))
    return out


def _active_mask(dates: pd.DatetimeIndex, intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for st, ed in intervals:
        mask |= (dates >= st) & (dates <= ed)
    return mask


def _read_feature_bin(path: Path, n_cal: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype="<f4")
    if raw.size <= 1:
        return None
    start_idx = int(raw[0])
    vals = raw[1:]
    arr = np.full(n_cal, np.nan, dtype=np.float32)
    s = max(0, start_idx)
    e = min(n_cal, start_idx + vals.size)
    if s < e:
        arr[s:e] = vals[(s - start_idx) : (e - start_idx)]
    return arr


def _summarize_universes(provider_uri: Path, cal_2020_2023: pd.DatetimeIndex) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    parsed: Dict[str, Dict[str, Any]] = {}
    for market in MARKETS:
        intervals = _parse_instrument_intervals(provider_uri / "instruments" / f"{market}.txt")
        instruments = sorted(intervals)
        per_year_counts: Dict[int, int] = {}
        for year in YEARS:
            year_dates = cal_2020_2023[cal_2020_2023.year == year]
            covered = 0
            for inst in instruments:
                if _active_mask(year_dates, intervals.get(inst, [])).any():
                    covered += 1
            per_year_counts[year] = covered
        row = {
            "market": market,
            "instrument_count_file": len(instruments),
            "covered_2020_2023_any": sum(1 for inst in instruments if _active_mask(cal_2020_2023, intervals.get(inst, [])).any()),
            "covered_2020": per_year_counts[2020],
            "covered_2021": per_year_counts[2021],
            "covered_2022": per_year_counts[2022],
            "covered_2023": per_year_counts[2023],
            "min_year_covered": min(per_year_counts.values()) if per_year_counts else 0,
            "uses_2024_plus": False,
        }
        rows.append(row)
        parsed[market] = {"intervals": intervals, "row": row}
    return rows, parsed


def _choose_diagnostic_market(universe_rows: Sequence[Dict[str, Any]]) -> str:
    eligible = [r for r in universe_rows if int(r.get("min_year_covered", 0)) >= GATE_MIN_POOL_SIZE]
    if eligible:
        preferred = {"csiall": 0, "all": 1, "csi1000": 2, "csi800": 3, "csi500": 4, "csi300": 5}
        eligible.sort(key=lambda r: (preferred.get(str(r["market"]), 99), -int(r["min_year_covered"])))
        return str(eligible[0]["market"])
    return max(universe_rows, key=lambda r: int(r.get("min_year_covered", 0)))["market"]


def _quantiles(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    qs = np.nanquantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]), "p75": float(qs[3]), "p90": float(qs[4])}


def _diagnose_dynamic_pool(
    provider_uri: Path,
    cal: pd.DatetimeIndex,
    market: str,
    intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    dynamic_topk: int,
    min_amount_quantile: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    mask = (cal >= START_DATE) & (cal <= END_DATE)
    dates = cal[mask]
    start_pos = int(np.flatnonzero(mask)[0])
    end_pos = int(np.flatnonzero(mask)[-1])
    n_days = len(dates)
    instruments = sorted(intervals)

    counts_by_year = {year: np.zeros(int((dates.year == year).sum()), dtype=np.int32) for year in YEARS}
    missing_by_year = {year: np.zeros(int((dates.year == year).sum()), dtype=np.int32) for year in YEARS}
    tradable_by_year = {year: np.zeros(int((dates.year == year).sum()), dtype=np.int32) for year in YEARS}
    zero_volume_by_year = {year: np.zeros(int((dates.year == year).sum()), dtype=np.int32) for year in YEARS}
    bad_change_by_year = {year: np.zeros(int((dates.year == year).sum()), dtype=np.int32) for year in YEARS}
    amount_values_by_year: Dict[int, List[float]] = {year: [] for year in YEARS}
    mean_amount: Dict[str, float] = {}
    per_inst_years: Dict[str, Dict[int, Dict[str, float]]] = {}

    year_local_idx = {year: np.flatnonzero(dates.year == year) for year in YEARS}
    global_to_year_pos: Dict[int, Tuple[int, int]] = {}
    for year, idxs in year_local_idx.items():
        for pos, global_idx in enumerate(idxs):
            global_to_year_pos[int(global_idx)] = (year, pos)

    for inst in instruments:
        feat_dir = provider_uri / "features" / inst
        active = _active_mask(dates, intervals.get(inst, []))
        if not active.any():
            continue
        amount = _read_feature_bin(feat_dir / "amount.day.bin", len(cal))
        volume = _read_feature_bin(feat_dir / "volume.day.bin", len(cal))
        change = _read_feature_bin(feat_dir / "change.day.bin", len(cal))
        factor = _read_feature_bin(feat_dir / "factor.day.bin", len(cal))
        if amount is None:
            amount_sub = np.full(n_days, np.nan, dtype=np.float32)
        else:
            amount_sub = amount[start_pos : end_pos + 1]
        volume_sub = np.full(n_days, np.nan, dtype=np.float32) if volume is None else volume[start_pos : end_pos + 1]
        change_sub = np.full(n_days, np.nan, dtype=np.float32) if change is None else change[start_pos : end_pos + 1]
        factor_sub = np.full(n_days, np.nan, dtype=np.float32) if factor is None else factor[start_pos : end_pos + 1]

        finite_amount = np.isfinite(amount_sub) & (amount_sub > 0)
        finite_volume = np.isfinite(volume_sub) & (volume_sub > 0)
        finite_change = np.isfinite(change_sub)
        finite_factor = np.isfinite(factor_sub) & (factor_sub > 0)
        tradable = active & finite_amount & finite_volume & finite_change & finite_factor
        missing = active & ~tradable
        zero_volume = active & (~finite_volume)
        bad_change = active & finite_change & (np.abs(change_sub) >= 0.095)

        finite_amount_vals = amount_sub[active & finite_amount]
        if finite_amount_vals.size:
            mean_amount[inst] = float(np.nanmean(finite_amount_vals))

        for year in YEARS:
            idx = year_local_idx[year]
            if idx.size == 0:
                continue
            y_active = active[idx]
            y_tradable = tradable[idx]
            y_missing = missing[idx]
            counts_by_year[year] += y_active.astype(np.int32)
            missing_by_year[year] += y_missing.astype(np.int32)
            tradable_by_year[year] += y_tradable.astype(np.int32)
            zero_volume_by_year[year] += zero_volume[idx].astype(np.int32)
            bad_change_by_year[year] += bad_change[idx].astype(np.int32)
            vals = amount_sub[idx][y_tradable]
            if vals.size:
                amount_values_by_year[year].extend(vals.astype(float).tolist())
            per_inst_years.setdefault(inst, {})[year] = {
                "active_days": float(y_active.sum()),
                "tradable_days": float(y_tradable.sum()),
                "mean_amount": float(np.nanmean(vals)) if vals.size else float("nan"),
            }

    dynamic_topk = min(dynamic_topk, len(mean_amount))
    top_inst = {inst for inst, _ in sorted(mean_amount.items(), key=lambda kv: kv[1], reverse=True)[:dynamic_topk]}
    prev_pool: Optional[set[str]] = None
    turnover_by_year: Dict[int, List[float]] = {year: [] for year in YEARS}
    pool_size_by_year: Dict[int, List[int]] = {year: [] for year in YEARS}
    threshold_by_year: Dict[int, List[float]] = {year: [] for year in YEARS}

    for year in YEARS:
        candidates: List[Tuple[str, float]] = []
        for inst in instruments:
            stats = per_inst_years.get(inst, {}).get(year)
            if not stats or stats["active_days"] <= 0:
                continue
            if stats["tradable_days"] / max(stats["active_days"], 1.0) < 0.90:
                continue
            amt = stats["mean_amount"]
            if np.isfinite(amt):
                candidates.append((inst, float(amt)))
        if not candidates:
            continue
        vals = np.asarray([x[1] for x in candidates], dtype=np.float64)
        threshold = float(np.nanquantile(vals, min_amount_quantile))
        pool = {inst for inst, amt in candidates if amt >= threshold}
        pool &= top_inst
        if prev_pool is not None:
            denom = max(len(pool | prev_pool), 1)
            turnover_by_year[year].append(1.0 - (len(pool & prev_pool) / denom))
        prev_pool = set(pool)
        pool_size_by_year[year].append(len(pool))
        threshold_by_year[year].append(threshold)

    year_rows: List[Dict[str, Any]] = []
    for year in YEARS:
        active_counts = counts_by_year[year]
        tradable_counts = tradable_by_year[year]
        missing_counts = missing_by_year[year]
        active_sum = int(active_counts.sum())
        missing_sum = int(missing_counts.sum())
        q = _quantiles(amount_values_by_year[year])
        row = {
            "market": market,
            "year": year,
            "trade_days": int(active_counts.size),
            "avg_active_instruments_per_day": float(active_counts.mean()) if active_counts.size else None,
            "max_active_instruments_per_day": int(active_counts.max()) if active_counts.size else 0,
            "avg_tradable_instruments_per_day": float(tradable_counts.mean()) if tradable_counts.size else None,
            "max_tradable_instruments_per_day": int(tradable_counts.max()) if tradable_counts.size else 0,
            "missing_rate_active_fieldset": float(missing_sum / active_sum) if active_sum else None,
            "zero_or_missing_volume_rate": float(zero_volume_by_year[year].sum() / active_sum) if active_sum else None,
            "limit_or_extreme_change_proxy_rate": float(bad_change_by_year[year].sum() / active_sum) if active_sum else None,
            "amount_p10": q["p10"],
            "amount_p25": q["p25"],
            "amount_p50": q["p50"],
            "amount_p75": q["p75"],
            "amount_p90": q["p90"],
            "dynamic_pool_size": int(np.mean(pool_size_by_year[year])) if pool_size_by_year[year] else 0,
            "dynamic_pool_turnover_proxy": float(np.mean(turnover_by_year[year])) if turnover_by_year[year] else None,
            "liquidity_threshold_amount": float(np.mean(threshold_by_year[year])) if threshold_by_year[year] else None,
        }
        year_rows.append(row)

    rule_rows = [
        {
            "rule_id": "pre2024_amount_volume_change_factor_liquid_topk",
            "market": market,
            "data_end": str(END_DATE.date()),
            "definition": (
                f"For each pre-2024 year, require >=90% tradable days on amount>0, volume>0, finite change, factor>0; "
                f"keep instruments above yearly amount q{min_amount_quantile:.2f}, capped by 2020-2023 mean amount top {dynamic_topk}."
            ),
            "uses_2024_plus": False,
            "trains_model": False,
            "runs_backtest": False,
        }
    ]
    diag_meta = {"diagnostic_market": market, "instrument_count": len(instruments), "dynamic_topk": dynamic_topk}
    return year_rows, rule_rows, diag_meta


def _gate(universe_rows: Sequence[Dict[str, Any]], year_rows: Sequence[Dict[str, Any]], diagnostic_market: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    market_row = next((r for r in universe_rows if r["market"] == diagnostic_market), None)
    if not market_row or int(market_row.get("min_year_covered", 0)) < GATE_MIN_POOL_SIZE:
        reasons.append(f"min_year_covered<{GATE_MIN_POOL_SIZE} for diagnostic_market={diagnostic_market}")
    if diagnostic_market == "csi300":
        reasons.append("diagnostic_market=csi300 is not a new broad universe")
    for row in year_rows:
        year = row["year"]
        pool_size = int(row.get("dynamic_pool_size") or 0)
        missing_rate = row.get("missing_rate_active_fieldset")
        if pool_size < GATE_MIN_POOL_SIZE:
            reasons.append(f"{year}_dynamic_pool_size={pool_size}<{GATE_MIN_POOL_SIZE}")
        if missing_rate is None or float(missing_rate) >= GATE_MAX_MISSING_RATE:
            reasons.append(f"{year}_missing_rate={missing_rate}>=0.10")
    return not reasons, reasons


def _write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# pre-2024 dynamic universe smoke",
        "",
        f"- task_id: {summary['task_id']}",
        f"- status: {summary['status']}",
        f"- gate_pass: {summary['gate_pass']}",
        f"- diagnostic_market: {summary['diagnostic_market']}",
        f"- data_window: {summary['data_window']}",
        f"- uses_2024_plus: {summary['uses_2024_plus']}",
        f"- trains_model: {summary['trains_model']}",
        f"- runs_backtest: {summary['runs_backtest']}",
        "",
        "## Gate reasons",
    ]
    reasons = summary.get("gate_reasons") or ["PASS"]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(["", "## Key yearly metrics"])
    for row in summary["key_year_counts"]:
        lines.append(
            "- "
            f"{row['year']}: active_avg={row['avg_active_instruments_per_day']:.1f}, "
            f"tradable_avg={row['avg_tradable_instruments_per_day']:.1f}, "
            f"pool={row['dynamic_pool_size']}, missing={row['missing_rate_active_fieldset']:.4f}, "
            f"amount_p50={row['amount_p50']:.2f}"
        )
    lines.extend(["", "## Artifacts"])
    for name, artifact_path in summary["artifacts"].items():
        lines.append(f"- {name}: {artifact_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pre-2024 dynamic universe/liquidity feasibility smoke. No model, no 2024+.")
    p.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
    p.add_argument("--output-prefix", default="pre2024_dynamic_universe_smoke")
    p.add_argument("--dynamic-topk", type=int, default=DEFAULT_DYNAMIC_TOPK)
    p.add_argument("--min-amount-quantile", type=float, default=DEFAULT_MIN_AMOUNT_QUANTILE)
    args = p.parse_args(argv)

    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)

    cal = _read_calendar(provider_uri)
    cal_2020_2023 = cal[(cal >= START_DATE) & (cal <= END_DATE)]
    if cal_2020_2023.empty:
        raise RuntimeError("No 2020-2023 calendar overlap; refusing to inspect any 2024+ dates.")

    universe_rows, parsed = _summarize_universes(provider_uri, cal_2020_2023)
    diagnostic_market = _choose_diagnostic_market(universe_rows)
    year_rows, rule_rows, diag_meta = _diagnose_dynamic_pool(
        provider_uri=provider_uri,
        cal=cal,
        market=diagnostic_market,
        intervals=parsed[diagnostic_market]["intervals"],
        dynamic_topk=int(args.dynamic_topk),
        min_amount_quantile=float(args.min_amount_quantile),
    )
    gate_pass, gate_reasons = _gate(universe_rows, year_rows, diagnostic_market)

    _write_csv(paths["universe_csv"], universe_rows)
    _write_csv(paths["year_metrics_csv"], year_rows)
    _write_csv(paths["rule_csv"], rule_rows)

    summary = {
        "task_id": "Q-DYNAMIC-UNIVERSE-PRE2024-SMOKE",
        "created_at": _now_utc(),
        "status": "completed",
        "provider_uri": str(provider_uri),
        "data_window": f"{START_DATE.date()}..{END_DATE.date()}",
        "uses_2024_plus": False,
        "trains_model": False,
        "runs_backtest": False,
        "diagnostic_market": diagnostic_market,
        "diagnostic_meta": diag_meta,
        "gate_pass": gate_pass,
        "gate_reasons": gate_reasons,
        "gate_thresholds": {
            "min_pool_size_per_year": GATE_MIN_POOL_SIZE,
            "max_missing_rate": GATE_MAX_MISSING_RATE,
        },
        "universe_coverage": universe_rows,
        "key_year_counts": year_rows,
        "rule_candidates": rule_rows,
        "risks": [
            "Dynamic pool turnover is a year-to-year membership proxy, not an executed portfolio turnover.",
            "Extreme change proxy uses abs(change)>=9.5%; this approximates limit/abnormal days without exchange status flags.",
            "amount units follow local qlib bins and are not normalized across data vendors.",
        ],
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    _write_json(paths["summary_json"], summary)
    _write_summary_md(paths["summary_md"], summary)

    print(f"summary_json={paths['summary_json']}")
    print(f"summary_md={paths['summary_md']}")
    print(f"gate_pass={gate_pass}")
    print(f"diagnostic_market={diagnostic_market}")
    for row in year_rows:
        print(
            f"year={row['year']} pool={row['dynamic_pool_size']} "
            f"missing={row['missing_rate_active_fieldset']:.6f} "
            f"active_avg={row['avg_active_instruments_per_day']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
