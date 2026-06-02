#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

TASK_ID = "Q-DYNAMIC-UNIVERSE-PRE2024-RULE-SMOKE"
YEARS = (2020, 2021, 2022, 2023)
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2023-12-31")
DEFAULT_PROVIDER_URI = Path("~/.quant_master/quant_master_data/tdx_cn_data")

DEFAULT_MARKET = "csiall"
DEFAULT_DYNAMIC_TOPK = 1000
DEFAULT_MIN_AMOUNT_QUANTILE = 0.60
DEFAULT_TOPK = 50
DEFAULT_MIN_NAMES = 25
LOOKBACK_DAYS = 80

OPEN_COST = 0.0001
CLOSE_COST = 0.0006

GATE_MIN_COMBINED_IR = 1.8
GATE_MIN_YEAR_IR = 0.0
GATE_MIN_IR_GT_ONE_YEARS = 3
GATE_MIN_COMBINED_MDD = -0.12


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


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
        "rules_csv": THIS_DIR / f"{output_prefix}_rules_{stamp}.csv",
        "year_metrics_csv": THIS_DIR / f"{output_prefix}_year_metrics_{stamp}.csv",
        "universe_csv": THIS_DIR / f"{output_prefix}_universe_{stamp}.csv",
    }


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    cal = pd.to_datetime(pd.Index(vals))
    return cal[cal <= END_DATE]


def _parse_instrument_intervals(inst_path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
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


def _pct_change(mat: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(mat.shape, np.nan, dtype=np.float32)
    if periods <= 0 or mat.shape[0] <= periods:
        return out
    prev = mat[:-periods]
    curr = mat[periods:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0.0)
    vals = np.full(curr.shape, np.nan, dtype=np.float32)
    vals[valid] = curr[valid] / prev[valid] - 1.0
    out[periods:] = vals
    return out


def _rolling_mean(mat: np.ndarray, window: int) -> np.ndarray:
    df = pd.DataFrame(mat)
    return df.rolling(window, min_periods=max(2, window // 2)).mean().to_numpy(dtype=np.float32)


def _rolling_std(mat: np.ndarray, window: int) -> np.ndarray:
    df = pd.DataFrame(mat)
    return df.rolling(window, min_periods=max(2, window // 2)).std(ddof=0).to_numpy(dtype=np.float32)


def _daily_quantile_floor(vals: np.ndarray, tradable: np.ndarray, q: float) -> np.ndarray:
    floor = np.full(vals.shape[0], np.nan, dtype=np.float32)
    for i in range(vals.shape[0]):
        row = vals[i]
        ok = tradable[i] & np.isfinite(row)
        if ok.any():
            floor[i] = float(np.nanquantile(row[ok], q))
    return floor[:, None]


def _rank_pct_rows(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(score.shape, np.nan, dtype=np.float32)
    for i in range(score.shape[0]):
        ok = valid[i] & np.isfinite(score[i])
        n = int(ok.sum())
        if n <= 1:
            continue
        vals = score[i, ok]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[i, ok] = ranks
    return out


def _max_drawdown(net_returns: np.ndarray) -> float:
    if net_returns.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + net_returns.astype(np.float64))
    high = np.maximum.accumulate(equity)
    dd = equity / np.maximum(high, 1e-12) - 1.0
    return float(np.nanmin(dd))


def _calc_metrics(records: Sequence[Dict[str, Any]], open_cost: float, close_cost: float) -> Dict[str, Any]:
    if not records:
        return {
            "days": 0,
            "annret": float("nan"),
            "ir": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            "coverage": float("nan"),
            "avg_holdings": float("nan"),
            "finite": False,
        }

    prev: Dict[str, float] = {}
    net_returns: List[float] = []
    turnovers: List[float] = []
    coverages: List[float] = []
    holdings: List[float] = []
    costs: List[float] = []
    for rec in sorted(records, key=lambda r: r["date"]):
        names = list(rec["names"])
        gross = float(rec["gross_return"])
        w = 1.0 / len(names) if names else 0.0
        cur = {name: w for name in names}
        all_names = set(prev) | set(cur)
        buy = 0.0
        sell = 0.0
        abs_turn = 0.0
        for name in all_names:
            diff = cur.get(name, 0.0) - prev.get(name, 0.0)
            if diff > 0.0:
                buy += diff
            elif diff < 0.0:
                sell += -diff
            abs_turn += abs(diff)
        cost = buy * open_cost + sell * close_cost
        net_returns.append(gross - cost)
        turnovers.append(0.5 * abs_turn)
        coverages.append(float(rec["coverage"]))
        holdings.append(float(len(names)))
        costs.append(cost)
        prev = cur

    arr = np.asarray(net_returns, dtype=np.float64)
    finite = bool(np.isfinite(arr).all())
    if arr.size > 1 and np.nanstd(arr, ddof=1) > 0:
        ir = float(np.nanmean(arr) / np.nanstd(arr, ddof=1) * np.sqrt(252.0))
    else:
        ir = float("nan")
    compounded = float(np.prod(1.0 + arr))
    annret = float(compounded ** (252.0 / arr.size) - 1.0) if arr.size and compounded > 0 else float("nan")
    return {
        "days": int(arr.size),
        "annret": annret,
        "ir": ir,
        "max_drawdown": _max_drawdown(arr),
        "turnover": float(np.nanmean(turnovers)) if turnovers else float("nan"),
        "coverage": float(np.nanmean(coverages)) if coverages else float("nan"),
        "avg_holdings": float(np.nanmean(holdings)) if holdings else float("nan"),
        "avg_cost": float(np.nanmean(costs)) if costs else float("nan"),
        "finite": finite and all(np.isfinite(x) for x in (annret, ir, _max_drawdown(arr), np.nanmean(turnovers), np.nanmean(coverages))),
    }


def _build_dynamic_pools(
    provider_uri: Path,
    cal: pd.DatetimeIndex,
    intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    dynamic_topk: int,
    min_amount_quantile: float,
) -> Tuple[Dict[int, List[str]], List[Dict[str, Any]]]:
    dates = cal[(cal >= START_DATE) & (cal <= END_DATE)]
    if dates.empty:
        raise RuntimeError("No 2020-2023 calendar overlap; refusing to inspect any 2024+ dates.")
    start_pos = int(np.searchsorted(cal.values, dates[0].to_datetime64()))
    end_pos = int(np.searchsorted(cal.values, dates[-1].to_datetime64()))
    year_idx = {year: np.flatnonzero(dates.year == year) for year in YEARS}
    instruments = sorted(intervals)
    mean_amount_all: Dict[str, float] = {}
    per_inst_year: Dict[str, Dict[int, Dict[str, float]]] = {}

    for inst in instruments:
        feat_dir = provider_uri / "features" / inst
        active = _active_mask(dates, intervals.get(inst, []))
        if not active.any():
            continue
        amount = _read_feature_bin(feat_dir / "amount.day.bin", len(cal))
        volume = _read_feature_bin(feat_dir / "volume.day.bin", len(cal))
        change = _read_feature_bin(feat_dir / "change.day.bin", len(cal))
        factor = _read_feature_bin(feat_dir / "factor.day.bin", len(cal))
        amount_sub = np.full(len(dates), np.nan, dtype=np.float32) if amount is None else amount[start_pos : end_pos + 1]
        volume_sub = np.full(len(dates), np.nan, dtype=np.float32) if volume is None else volume[start_pos : end_pos + 1]
        change_sub = np.full(len(dates), np.nan, dtype=np.float32) if change is None else change[start_pos : end_pos + 1]
        factor_sub = np.full(len(dates), np.nan, dtype=np.float32) if factor is None else factor[start_pos : end_pos + 1]
        tradable = (
            active
            & np.isfinite(amount_sub)
            & (amount_sub > 0.0)
            & np.isfinite(volume_sub)
            & (volume_sub > 0.0)
            & np.isfinite(change_sub)
            & np.isfinite(factor_sub)
            & (factor_sub > 0.0)
        )
        vals_all = amount_sub[tradable]
        if vals_all.size:
            mean_amount_all[inst] = float(np.nanmean(vals_all))
        for year, idx in year_idx.items():
            y_active = active[idx]
            y_tradable = tradable[idx]
            vals = amount_sub[idx][y_tradable]
            per_inst_year.setdefault(inst, {})[year] = {
                "active_days": float(y_active.sum()),
                "tradable_days": float(y_tradable.sum()),
                "mean_amount": float(np.nanmean(vals)) if vals.size else float("nan"),
            }

    top_inst = {
        inst
        for inst, _ in sorted(mean_amount_all.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(dynamic_topk))]
    }
    pools: Dict[int, List[str]] = {}
    rows: List[Dict[str, Any]] = []
    prev_pool: Optional[set[str]] = None
    for year in YEARS:
        candidates: List[Tuple[str, float, float]] = []
        for inst in instruments:
            stats = per_inst_year.get(inst, {}).get(year)
            if not stats or stats["active_days"] <= 0:
                continue
            tradable_rate = stats["tradable_days"] / max(stats["active_days"], 1.0)
            amt = stats["mean_amount"]
            if tradable_rate >= 0.90 and np.isfinite(amt):
                candidates.append((inst, float(amt), float(tradable_rate)))
        threshold = float("nan")
        if candidates:
            threshold = float(np.nanquantile([x[1] for x in candidates], min_amount_quantile))
        pool = sorted({inst for inst, amt, _ in candidates if np.isfinite(threshold) and amt >= threshold} & top_inst)
        pool_set = set(pool)
        turnover = float("nan")
        if prev_pool is not None:
            turnover = 1.0 - (len(pool_set & prev_pool) / max(len(pool_set | prev_pool), 1))
        prev_pool = pool_set
        pools[year] = pool
        rows.append(
            {
                "year": year,
                "market": DEFAULT_MARKET,
                "active_candidates": len(candidates),
                "dynamic_pool_size": len(pool),
                "liquidity_threshold_amount": threshold,
                "year_to_year_pool_turnover_proxy": turnover,
                "uses_2024_plus": False,
            }
        )
    return pools, rows


def _load_year_panel(
    provider_uri: Path,
    cal: pd.DatetimeIndex,
    intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    instruments: Sequence[str],
    year: int,
) -> Dict[str, Any]:
    year_positions = np.flatnonzero(cal.year == year)
    if year_positions.size == 0:
        raise RuntimeError(f"No calendar dates for year={year}")
    win_start = max(0, int(year_positions[0]) - LOOKBACK_DAYS)
    win_end = int(year_positions[-1])
    if win_end + 1 < len(cal):
        win_end += 1
    dates = cal[win_start : win_end + 1]
    n_days = len(dates)
    n_names = len(instruments)

    mats = {
        "close": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "amount": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "volume": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "vwap": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "change": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "factor": np.full((n_days, n_names), np.nan, dtype=np.float32),
        "active": np.zeros((n_days, n_names), dtype=bool),
    }

    for j, inst in enumerate(instruments):
        feat_dir = provider_uri / "features" / inst
        mats["active"][:, j] = _active_mask(dates, intervals.get(inst, []))
        for field in ("close", "amount", "volume", "vwap", "change", "factor"):
            arr = _read_feature_bin(feat_dir / f"{field}.day.bin", len(cal))
            if arr is not None:
                mats[field][:, j] = arr[win_start : win_end + 1]

    fallback_vwap = mats["amount"] / np.maximum(mats["volume"], 1e-12) * mats["factor"]
    missing_vwap = ~np.isfinite(mats["vwap"])
    mats["vwap"][missing_vwap] = fallback_vwap[missing_vwap]
    tradable = (
        mats["active"]
        & np.isfinite(mats["close"])
        & (mats["close"] > 0.0)
        & np.isfinite(mats["amount"])
        & (mats["amount"] > 0.0)
        & np.isfinite(mats["volume"])
        & (mats["volume"] > 0.0)
        & np.isfinite(mats["change"])
        & np.isfinite(mats["factor"])
        & (mats["factor"] > 0.0)
    )
    mats["tradable"] = tradable
    mats["dates"] = dates
    mats["instruments"] = list(instruments)
    return mats


def _rule_scores(panel: Dict[str, Any], amount_quantile: float) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    close = panel["close"]
    amount = panel["amount"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    tradable = panel["tradable"]

    ret20 = _pct_change(close, 20)
    ret10 = _pct_change(close, 10)
    ret5 = _pct_change(close, 5)
    log_amount = np.log1p(np.where(np.isfinite(amount) & (amount > 0.0), amount, np.nan)).astype(np.float32)
    log_volume = np.log1p(np.where(np.isfinite(volume) & (volume > 0.0), volume, np.nan)).astype(np.float32)
    amount_mean20 = _rolling_mean(log_amount, 20)
    amount_std20 = _rolling_std(log_amount, 20)
    volume_std20 = _rolling_std(log_volume, 20)
    amount_z20 = (log_amount - amount_mean20) / np.maximum(amount_std20, 1e-6)
    amount_trend20 = log_amount - np.roll(log_amount, 20, axis=0)
    amount_trend20[:20] = np.nan
    vwap_gap = np.abs(vwap / close - 1.0)

    floor60 = _daily_quantile_floor(amount, tradable, amount_quantile)
    floor65 = _daily_quantile_floor(amount, tradable, max(amount_quantile, 0.65))
    stable_floor = _daily_quantile_floor(-volume_std20, tradable, 0.50)

    rules: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
    valid1 = tradable & (amount >= floor60) & np.isfinite(ret20)
    rules["liq_adj_mom20_amt60"] = (
        ret20 + 0.05 * np.nan_to_num(amount_z20, nan=0.0),
        valid1,
        "20d close momentum within dynamic liquid pool, amount daily q60 floor, small liquidity-z tilt.",
    )

    valid2 = tradable & (amount >= floor60) & np.isfinite(ret5) & np.isfinite(volume_std20) & (-volume_std20 >= stable_floor)
    rules["short_reversal5_liq_stable"] = (
        -ret5 - 0.02 * np.nan_to_num(volume_std20, nan=0.0),
        valid2,
        "5d short reversal with amount floor and below-median 20d volume instability.",
    )

    valid3 = tradable & (amount >= floor65) & np.isfinite(ret10) & np.isfinite(vwap_gap) & (vwap_gap <= 0.02)
    rules["vwap_quality_mom10"] = (
        ret10 - 2.0 * np.nan_to_num(vwap_gap, nan=0.0),
        valid3,
        "10d momentum with amount q65 floor and vwap/close execution-quality gap <= 2%.",
    )

    valid4 = tradable & (amount >= floor60) & np.isfinite(amount_trend20) & np.isfinite(volume_std20) & (-volume_std20 >= stable_floor)
    rules["amount_trend_volume_stability"] = (
        amount_trend20 - 0.10 * np.nan_to_num(volume_std20, nan=0.0),
        valid4,
        "20d amount trend signal gated by liquid names and stable 20d volume profile.",
    )
    return rules


def _evaluate_rule_year(
    panel: Dict[str, Any],
    rule_id: str,
    score: np.ndarray,
    valid: np.ndarray,
    topk: int,
    min_names: int,
) -> List[Dict[str, Any]]:
    dates: pd.DatetimeIndex = panel["dates"]
    instruments: List[str] = panel["instruments"]
    tradable = panel["tradable"]
    change = panel["change"]
    records: List[Dict[str, Any]] = []
    for i in range(len(dates) - 1):
        date = pd.Timestamp(dates[i])
        next_date = pd.Timestamp(dates[i + 1])
        if date < START_DATE or date > END_DATE or next_date > END_DATE:
            continue
        next_ret = change[i + 1]
        ok = valid[i] & tradable[i + 1] & np.isfinite(score[i]) & np.isfinite(next_ret)
        coverage = int(ok.sum())
        if coverage < int(min_names):
            records.append(
                {
                    "rule_id": rule_id,
                    "date": date,
                    "next_date": next_date,
                    "gross_return": 0.0,
                    "names": [],
                    "coverage": coverage,
                }
            )
            continue
        k = min(int(topk), coverage)
        ok_idx = np.flatnonzero(ok)
        top_local = ok_idx[np.argsort(score[i, ok_idx], kind="mergesort")[-k:]]
        gross = float(np.nanmean(next_ret[top_local]))
        records.append(
            {
                "rule_id": rule_id,
                "date": date,
                "next_date": next_date,
                "gross_return": gross,
                "names": [instruments[j] for j in top_local],
                "coverage": coverage,
            }
        )
    return records


def _gate_reasons(per_year: Dict[int, Dict[str, Any]], combined: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    irs = [_safe_float(per_year[y].get("ir")) for y in YEARS]
    bad_years = [str(y) for y, ir in zip(YEARS, irs) if not (np.isfinite(ir) and ir > GATE_MIN_YEAR_IR)]
    if bad_years:
        reasons.append(f"year_ir_not_positive={','.join(bad_years)}")
    ir_gt_one = sum(1 for ir in irs if np.isfinite(ir) and ir > 1.0)
    if ir_gt_one < GATE_MIN_IR_GT_ONE_YEARS:
        reasons.append(f"year_ir_gt_1_count={ir_gt_one}<3")
    combined_ir = _safe_float(combined.get("ir"))
    if not (np.isfinite(combined_ir) and combined_ir >= GATE_MIN_COMBINED_IR):
        reasons.append(f"combined_ir={combined_ir:.6g}<1.8" if np.isfinite(combined_ir) else "combined_ir_nonfinite")
    combined_mdd = _safe_float(combined.get("max_drawdown"))
    if not (np.isfinite(combined_mdd) and combined_mdd >= GATE_MIN_COMBINED_MDD):
        reasons.append(f"combined_mdd={combined_mdd:.6g}<-0.12" if np.isfinite(combined_mdd) else "combined_mdd_nonfinite")
    finite_checks: Iterable[Tuple[str, Dict[str, Any]]] = [("combined", combined)] + [(str(y), per_year[y]) for y in YEARS]
    for label, row in finite_checks:
        if not bool(row.get("finite")):
            reasons.append(f"{label}_turnover_or_coverage_or_metric_nonfinite")
    return reasons


def _write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# pre-2024 dynamic universe rule smoke",
        "",
        f"- task_id: {summary['task_id']}",
        f"- status: {summary['status']}",
        f"- verdict: {summary['verdict']}",
        f"- is_proxy: `{summary['is_proxy']}`",
        f"- gate_pass_count: `{summary['gate_pass_count']}` / `{summary['rule_count']}`",
        f"- data_window: `{summary['data_window']}`",
        f"- uses_2024_plus: `{summary['uses_2024_plus']}`",
        f"- costs: open `{summary['costs']['open_cost']}`, close `{summary['costs']['close_cost']}`",
        "",
        "## Top rules",
    ]
    for row in summary.get("top_rules", []):
        raw_reasons = row.get("fail_reasons") or "PASS"
        reasons = raw_reasons if isinstance(raw_reasons, str) else "; ".join(raw_reasons)
        lines.append(
            f"- {row['rule_id']}: IR={_safe_float(row['combined_ir']):.4f}, "
            f"AnnRet={_safe_float(row['combined_annret']):.4f}, MDD={_safe_float(row['combined_mdd']):.4f}, "
            f"TO={_safe_float(row['combined_turnover']):.4f}, gate={row['gate_pass']}, reasons={reasons}"
        )
    lines.extend(["", "## Artifacts"])
    for name, artifact_path in summary["artifacts"].items():
        lines.append(f"- {name}: {artifact_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Low-memory 2020-2023 dynamic-universe fixed-rule smoke. Proxy only; never reads/evaluates 2024+."
    )
    p.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
    p.add_argument("--market", default=DEFAULT_MARKET)
    p.add_argument("--output-prefix", default="pre2024_dynamic_universe_rule_smoke")
    p.add_argument("--dynamic-topk", type=int, default=DEFAULT_DYNAMIC_TOPK)
    p.add_argument("--min-amount-quantile", type=float, default=DEFAULT_MIN_AMOUNT_QUANTILE)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--min-names", type=int, default=DEFAULT_MIN_NAMES)
    p.add_argument("--years", default="2020,2021,2022,2023")
    p.add_argument("--open-cost", type=float, default=OPEN_COST)
    p.add_argument("--close-cost", type=float, default=CLOSE_COST)
    p.add_argument("--low-memory", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
    years = tuple(int(x.strip()) for x in str(args.years).split(",") if x.strip())

    if years != YEARS:
        raise ValueError(f"This smoke is hard-wired to exactly 2020,2021,2022,2023; got {years}")
    if str(args.market) != DEFAULT_MARKET:
        raise ValueError(f"This task targets market={DEFAULT_MARKET}; got {args.market}")

    cal = _read_calendar(provider_uri)
    if cal.empty or cal.max() > END_DATE:
        raise RuntimeError("Calendar guard failed; refusing to read/evaluate 2024+.")
    intervals = _parse_instrument_intervals(provider_uri / "instruments" / f"{args.market}.txt")
    pools, universe_rows = _build_dynamic_pools(
        provider_uri=provider_uri,
        cal=cal,
        intervals=intervals,
        dynamic_topk=int(args.dynamic_topk),
        min_amount_quantile=float(args.min_amount_quantile),
    )

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    rule_defs: Dict[str, str] = {}
    for year in YEARS:
        panel = _load_year_panel(provider_uri, cal, intervals, pools[year], year)
        scores = _rule_scores(panel, float(args.min_amount_quantile))
        for rule_id, (score, valid, definition) in scores.items():
            rule_defs[rule_id] = definition
            all_records.setdefault(rule_id, []).extend(
                _evaluate_rule_year(
                    panel=panel,
                    rule_id=rule_id,
                    score=score,
                    valid=valid,
                    topk=int(args.topk),
                    min_names=int(args.min_names),
                )
            )
        del panel

    rule_rows: List[Dict[str, Any]] = []
    year_rows: List[Dict[str, Any]] = []
    for rule_id, records in all_records.items():
        per_year: Dict[int, Dict[str, Any]] = {}
        for year in YEARS:
            y_records = [
                rec
                for rec in records
                if pd.Timestamp(rec["date"]).year == year and pd.Timestamp(rec["next_date"]).year == year
            ]
            metric = _calc_metrics(y_records, float(args.open_cost), float(args.close_cost))
            per_year[year] = metric
            year_rows.append(
                {
                    "rule_id": rule_id,
                    "year": year,
                    "annret": metric["annret"],
                    "ir": metric["ir"],
                    "max_drawdown": metric["max_drawdown"],
                    "turnover": metric["turnover"],
                    "coverage": metric["coverage"],
                    "avg_holdings": metric["avg_holdings"],
                    "days": metric["days"],
                    "finite": metric["finite"],
                }
            )
        combined = _calc_metrics(records, float(args.open_cost), float(args.close_cost))
        reasons = _gate_reasons(per_year, combined)
        year_ir_gt_one = int(sum(1 for year in YEARS if _safe_float(per_year[year].get("ir")) > 1.0))
        rule_rows.append(
            {
                "rule_id": rule_id,
                "definition": rule_defs.get(rule_id, ""),
                "topk": int(args.topk),
                "min_names": int(args.min_names),
                "combined_annret": combined["annret"],
                "combined_ir": combined["ir"],
                "combined_mdd": combined["max_drawdown"],
                "combined_turnover": combined["turnover"],
                "combined_coverage": combined["coverage"],
                "combined_avg_holdings": combined["avg_holdings"],
                "combined_days": combined["days"],
                "year_ir_gt_1_count": year_ir_gt_one,
                "gate_pass": not reasons,
                "fail_reasons": ";".join(reasons),
                "is_proxy": True,
                "uses_2024_plus": False,
            }
        )

    rule_rows.sort(
        key=lambda r: (
            _safe_float(r.get("combined_ir")) if np.isfinite(_safe_float(r.get("combined_ir"))) else -1e9,
            _safe_float(r.get("combined_annret")) if np.isfinite(_safe_float(r.get("combined_annret"))) else -1e9,
        ),
        reverse=True,
    )
    gate_pass_count = int(sum(1 for row in rule_rows if bool(row.get("gate_pass"))))

    _write_csv(paths["rules_csv"], rule_rows)
    _write_csv(paths["year_metrics_csv"], year_rows)
    _write_csv(paths["universe_csv"], universe_rows)

    summary = {
        "task_id": TASK_ID,
        "created_at": _now_utc(),
        "status": "completed",
        "verdict": "REPORT_LEAD_DO_NOT_RUN_2024_PLUS" if gate_pass_count else "NO_GO",
        "provider_uri": str(provider_uri),
        "market": str(args.market),
        "data_window": f"{START_DATE.date()}..{END_DATE.date()}",
        "uses_2024_plus": False,
        "trains_model": False,
        "runs_full_backtest": False,
        "is_proxy": True,
        "proxy_note": (
            "Lightweight rank long-topK daily return proxy using t-day/historical signals and next-trading-day "
            "change for ex-post evaluation; it is not a formal QuantMaster hard-gate backtest."
        ),
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "config": {
            "dynamic_pool": {
                "source": "pre2024_dynamic_universe_smoke style: yearly tradable>=90%, yearly amount quantile floor, capped by 2020-2023 mean-amount topK.",
                "dynamic_topk": int(args.dynamic_topk),
                "min_amount_quantile": float(args.min_amount_quantile),
            },
            "topk": int(args.topk),
            "min_names": int(args.min_names),
            "low_memory": bool(args.low_memory),
            "years": list(YEARS),
        },
        "gate_thresholds": {
            "combined_pre2024_ir": f">= {GATE_MIN_COMBINED_IR}",
            "each_year_ir": f"> {GATE_MIN_YEAR_IR}",
            "year_ir_gt_1_count": f">= {GATE_MIN_IR_GT_ONE_YEARS}/4",
            "combined_mdd": f">= {GATE_MIN_COMBINED_MDD}",
            "turnover_and_coverage": "finite and reported",
        },
        "rule_count": len(rule_rows),
        "gate_pass_count": gate_pass_count,
        "top_rules": rule_rows[: min(5, len(rule_rows))],
        "gate_pass_rules": [row for row in rule_rows if bool(row.get("gate_pass"))],
        "universe": universe_rows,
        "memory_risks": [
            "No full csiall factor matrix is loaded; dynamic-pool discovery streams one instrument at a time.",
            "Rule evaluation loads only one year and that year's dynamic pool at a time, with about 80 trading days of lookback.",
            "CSV daily positions are intentionally not written to avoid large artifacts; only aggregate rule/year metrics are persisted.",
        ],
        "artifacts": {k: str(v) for k, v in paths.items()},
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    _write_json(paths["summary_json"], summary)
    _write_summary_md(paths["summary_md"], summary)

    print(json.dumps(_json_sanitize(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

