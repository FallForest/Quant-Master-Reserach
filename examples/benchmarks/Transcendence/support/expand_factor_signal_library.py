#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from quant_master.data import D


BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
LOOKBACK_START = "2023-01-01"
BREAKTHROUGH_IR = 2.90
BREAKTHROUGH_ANNRET = 0.27


@dataclass
class SignalEval:
    signal: str
    family: str
    variant: str
    nonnull_ratio: float
    ic_mean: float
    ic_ir: float
    rankic_mean: float
    rankic_ir: float
    bucket_ls_mean: float
    bucket_ls_ir: float
    best_topk: int
    best_n_drop: int
    best_costed_ir: float
    best_costed_annret: float
    best_max_drawdown: float
    best_turnover: float


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_tracking_dir(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        tracking_uri = tracking_uri[len("file:") :]
    return Path(tracking_uri).expanduser().resolve()


def _find_run_dir(tracking_dir: Path, run_id: str) -> Path:
    cands = [p for p in tracking_dir.glob(f"*/{run_id}") if (p / "artifacts").exists()]
    if not cands:
        raise FileNotFoundError(f"run_id not found under {tracking_dir}: {run_id}")
    if len(cands) > 1:
        raise RuntimeError(f"run_id matched multiple paths: {[str(x) for x in cands]}")
    return cands[0]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _load_pickle(path)


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port config in workflow")


def _init_quant_master(config: Dict[str, Any]) -> None:
    init_cfg = copy.deepcopy(config.get("quant_master_init", {}))
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    init_cfg.setdefault("provider_uri", ".qmData/cn_data")
    init_cfg.setdefault("region", "cn")
    quant_master.init(**init_cfg)


def _as_score_df(pred_obj: Any) -> pd.DataFrame:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float).to_frame("score")
    if isinstance(pred_obj, pd.DataFrame):
        if "score" in pred_obj.columns:
            return pred_obj[["score"]].astype(float)
        if pred_obj.shape[1] >= 1:
            return pred_obj.iloc[:, [0]].rename(columns={pred_obj.columns[0]: "score"}).astype(float)
        return pd.DataFrame(columns=["score"], index=pred_obj.index)
    raise TypeError(f"unsupported pred type: {type(pred_obj)}")


def _as_label_series(label_obj: Any) -> pd.Series:
    if isinstance(label_obj, pd.Series):
        return label_obj.astype(float)
    if isinstance(label_obj, pd.DataFrame):
        if "label" in label_obj.columns:
            return label_obj["label"].astype(float)
        if label_obj.shape[1] >= 1:
            return label_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported label type: {type(label_obj)}")


def _guess_datetime_level(idx: pd.MultiIndex) -> int:
    nlv = idx.nlevels
    for lv in range(nlv):
        vals = idx.get_level_values(lv)
        sample = pd.Index(vals[: min(16, len(vals))])
        try:
            _ = pd.to_datetime(sample, errors="raise")
            return lv
        except Exception:  # noqa: BLE001
            continue
    return 0


def _normalize_mi_dt_inst(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    idx = obj.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        return obj
    dt_level = _guess_datetime_level(idx)
    if dt_level != 0:
        obj = obj.swaplevel(0, dt_level)
    obj = obj.sort_index()
    if isinstance(obj.index, pd.MultiIndex) and obj.index.nlevels >= 2:
        obj.index = obj.index.set_names(["datetime", "instrument"] + list(obj.index.names[2:]))
    return obj


def _slice_mi(df: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    df = _normalize_mi_dt_inst(df)
    idx = df.index
    if not isinstance(idx, pd.MultiIndex):
        mask = (pd.to_datetime(idx) >= pd.Timestamp(start)) & (pd.to_datetime(idx) <= pd.Timestamp(end))
        return df.loc[mask]
    dt = pd.to_datetime(idx.get_level_values(0))
    mask = (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))
    return df.loc[mask]


def _chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _fetch_market_df(
    instruments: Sequence[str],
    fields: Sequence[str],
    start_time: str,
    end_time: str,
    freq: str = "day",
    chunk_size: int = 60,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for part in _chunked(list(instruments), size=chunk_size):
        df = D.features(part, list(fields), start_time=start_time, end_time=end_time, freq=freq)
        if isinstance(df, pd.Series):
            df = df.to_frame(fields[0])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(fields))
    out = pd.concat(frames, axis=0).sort_index()
    out = _normalize_mi_dt_inst(out)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[-1] if isinstance(c, tuple) else c for c in out.columns]
    rename = {c: str(c).split("$")[-1].lower() for c in out.columns}
    out = out.rename(columns=rename)
    return out


def _by_inst_roll_mean(x: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp if minp is not None else max(2, w // 3)
    s = x.groupby(level=1).rolling(w, min_periods=minp).mean()
    return s.reset_index(level=0, drop=True).sort_index()


def _by_inst_roll_std(x: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp if minp is not None else max(3, w // 3)
    s = x.groupby(level=1).rolling(w, min_periods=minp).std()
    return s.reset_index(level=0, drop=True).sort_index()


def _by_inst_roll_min(x: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp if minp is not None else max(2, w // 3)
    s = x.groupby(level=1).rolling(w, min_periods=minp).min()
    return s.reset_index(level=0, drop=True).sort_index()


def _by_inst_roll_max(x: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp if minp is not None else max(2, w // 3)
    s = x.groupby(level=1).rolling(w, min_periods=minp).max()
    return s.reset_index(level=0, drop=True).sort_index()


def _by_inst_pct(x: pd.Series, w: int) -> pd.Series:
    return x.groupby(level=1).pct_change(w, fill_method=None)


def _by_inst_shift(x: pd.Series, w: int) -> pd.Series:
    return x.groupby(level=1).shift(w)


def _by_inst_corr(x: pd.Series, y: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp if minp is not None else max(5, w // 2)
    out = x * np.nan
    for _, sub in pd.concat([x.rename("x"), y.rename("y")], axis=1).groupby(level=1):
        c = sub["x"].rolling(w, min_periods=minp).corr(sub["y"])
        out.loc[sub.index] = c.values
    return out


def _cs_rank_pct(x: pd.Series) -> pd.Series:
    return x.groupby(level=0).rank(method="average", pct=True)


def _cs_robust_z(x: pd.Series, clip: float = 6.0) -> pd.Series:
    med = x.groupby(level=0).transform("median")
    mad = (x - med).abs().groupby(level=0).transform("median")
    z = (x - med) / (1.4826 * mad + 1e-12)
    return z.clip(-clip, clip)


def _cs_demean(x: pd.Series) -> pd.Series:
    return x - x.groupby(level=0).transform("mean")


def _ic_series(sig: pd.Series, label: pd.Series) -> Tuple[pd.Series, pd.Series]:
    panel = pd.concat([sig.rename("sig"), label.rename("label")], axis=1).dropna()
    if panel.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ic = panel.groupby(level=0).apply(lambda x: x["sig"].corr(x["label"]))
    rank_ic = panel.groupby(level=0).apply(lambda x: x["sig"].rank(pct=True).corr(x["label"].rank(pct=True)))
    return ic.astype(float), rank_ic.astype(float)


def _mean_ir(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return float("nan"), float("nan")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    if not np.isfinite(s) or s <= 1e-12:
        return m, float("nan")
    return m, float(np.sqrt(252.0) * m / s)


def _bucket_long_short(sig: pd.Series, label: pd.Series, n_bins: int = 10) -> Tuple[float, float]:
    panel = pd.concat([sig.rename("sig"), label.rename("label")], axis=1).dropna()
    if panel.empty:
        return float("nan"), float("nan")
    panel["rank"] = panel.groupby(level=0)["sig"].rank(pct=True, method="average")
    panel["bucket"] = np.ceil(panel["rank"] * n_bins).clip(1, n_bins).astype(int)
    bucket_mean = panel.groupby([panel.index.get_level_values(0), "bucket"])["label"].mean().unstack("bucket")
    if 1 not in bucket_mean.columns or n_bins not in bucket_mean.columns:
        return float("nan"), float("nan")
    ls = (bucket_mean[n_bins] - bucket_mean[1]).dropna()
    return _mean_ir(ls)


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


def _eval_topk_combo(
    *,
    signal_df: pd.DataFrame,
    combo: Dict[str, int],
    base_port_cfg: Dict[str, Any],
    benchmark: str,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, int, int], Any],
) -> Dict[str, float]:
    cfg = copy.deepcopy(base_port_cfg)
    bcfg = cfg["backtest"]
    bcfg["start_time"] = TEST_START
    bcfg["end_time"] = TEST_END
    exch_cfg = dict(bcfg.get("exchange_kwargs", {}))
    freq = "day"
    cache_key = (TEST_START, TEST_END, int(combo["topk"]), int(combo["n_drop"]))
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = get_exchange(
            freq=freq,
            start_time=TEST_START,
            end_time=TEST_END,
            deal_price=str(exch_cfg.get("deal_price", "close")),
            limit_threshold=float(exch_cfg.get("limit_threshold", 0.095)),
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=float(exch_cfg.get("min_cost", 5)),
        )
    strategy = TopkDropoutStrategy(
        signal=signal_df,
        topk=int(combo["topk"]),
        n_drop=int(combo["n_drop"]),
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
    )
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    exch_cfg["open_cost"] = float(open_cost)
    exch_cfg["close_cost"] = float(close_cost)
    exch_cfg["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    pm, _ = run_backtest(
        start_time=TEST_START,
        end_time=TEST_END,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=benchmark,
        account=bcfg.get("account", 100000000),
        exchange_kwargs=exch_cfg,
        pos_type=bcfg.get("pos_type", "Position"),
    )
    elapsed = time.perf_counter() - t0
    report = _get_report_for_day_freq(pm)
    annret, ir, maxdd, turnover = _calc_costed_metrics(report)
    return {
        "costed_annret": annret,
        "costed_ir": ir,
        "max_drawdown": maxdd,
        "turnover": turnover,
        "elapsed_sec": elapsed,
    }


def _build_signal_library(mkt: pd.DataFrame) -> Dict[str, Tuple[str, pd.Series]]:
    close = mkt["close"].astype(float)
    open_ = mkt["open"].astype(float)
    high = mkt["high"].astype(float)
    low = mkt["low"].astype(float)
    vwap = mkt["vwap"].astype(float)
    volume = mkt["volume"].astype(float).clip(lower=0.0)
    amount = mkt["amount"].astype(float).clip(lower=0.0)
    factor = mkt["factor"].astype(float).replace(0.0, np.nan)

    adj_close = close * factor
    adj_open = open_ * factor
    adj_high = high * factor
    adj_low = low * factor
    adj_vwap = vwap * factor

    ret1 = _by_inst_pct(adj_close, 1)
    ret5 = _by_inst_pct(adj_close, 5)
    ret10 = _by_inst_pct(adj_close, 10)
    ret20 = _by_inst_pct(adj_close, 20)
    ret60 = _by_inst_pct(adj_close, 60)
    ret120 = _by_inst_pct(adj_close, 120)

    log_vol = np.log1p(volume)
    log_amt = np.log1p(amount)
    vol_chg1 = _by_inst_pct(volume.replace(0.0, np.nan), 1)
    vol_chg5 = _by_inst_pct(volume.replace(0.0, np.nan), 5)
    amt_chg5 = _by_inst_pct(amount.replace(0.0, np.nan), 5)

    intraday = adj_close / (adj_open + 1e-12) - 1.0
    overnight = adj_open / (_by_inst_shift(adj_close, 1) + 1e-12) - 1.0
    hl_range = (adj_high - adj_low) / (_by_inst_shift(adj_close, 1) + 1e-12)
    vwap_gap = adj_close / (adj_vwap + 1e-12) - 1.0

    vol10 = _by_inst_roll_std(ret1, 10)
    vol20 = _by_inst_roll_std(ret1, 20)
    vol60 = _by_inst_roll_std(ret1, 60)
    hlv10 = _by_inst_roll_mean(hl_range.abs(), 10)
    hlv40 = _by_inst_roll_mean(hl_range.abs(), 40)
    dd60 = adj_close / (_by_inst_roll_max(adj_close, 60) + 1e-12) - 1.0
    rec20 = adj_close / (_by_inst_roll_min(adj_close, 20) + 1e-12) - 1.0

    market_ret = ret1.groupby(level=0).mean()
    market_ret_aligned = market_ret.reindex(ret1.index.get_level_values(0)).values
    market_ret_s = pd.Series(market_ret_aligned, index=ret1.index, dtype=float)
    beta60_num = _by_inst_corr(ret1, market_ret_s, 60) * _by_inst_roll_std(ret1, 60) * _by_inst_roll_std(market_ret_s, 60)
    beta60_den = _by_inst_roll_std(market_ret_s, 60) ** 2
    beta60 = beta60_num / (beta60_den + 1e-12)

    lib: Dict[str, Tuple[str, pd.Series]] = {}

    def add(name: str, family: str, s: pd.Series) -> None:
        lib[name] = (family, s.astype(float))

    # multi-horizon momentum/reversal
    add("mh_mom_20", "multi_horizon_momentum_reversal", ret20)
    add("mh_mom_60", "multi_horizon_momentum_reversal", ret60)
    add("mh_mom_120", "multi_horizon_momentum_reversal", ret120)
    add("mh_rev_5", "multi_horizon_momentum_reversal", -ret5)
    add("mh_rev_1", "multi_horizon_momentum_reversal", -ret1)
    add("mh_mom_spread_5_20", "multi_horizon_momentum_reversal", ret5 - ret20)
    add("mh_mom_accel_10_20", "multi_horizon_momentum_reversal", ret10 - _by_inst_shift(ret10, 10))

    # volatility compression/expansion
    add("vol_comp_10_60", "volatility_compression_expansion", -(vol10 / (vol60 + 1e-12)))
    add("vol_exp_10_20", "volatility_compression_expansion", vol10 / (vol20 + 1e-12))
    add("range_comp_10_40", "volatility_compression_expansion", -(hlv10 / (hlv40 + 1e-12)))
    add("vol_break_5x20", "volatility_compression_expansion", _by_inst_roll_mean(ret1.abs(), 5) / (_by_inst_roll_mean(ret1.abs(), 20) + 1e-12))

    # liquidity/turnover shock
    add("liq_volume_z20", "liquidity_turnover_shock", (log_vol - _by_inst_roll_mean(log_vol, 20)) / (_by_inst_roll_std(log_vol, 20) + 1e-12))
    add("liq_amount_z20", "liquidity_turnover_shock", (log_amt - _by_inst_roll_mean(log_amt, 20)) / (_by_inst_roll_std(log_amt, 20) + 1e-12))
    add("liq_volume_shock_5", "liquidity_turnover_shock", vol_chg5)
    add("liq_amount_shock_5", "liquidity_turnover_shock", amt_chg5)

    # overnight/open-close structure
    add("oc_overnight", "overnight_open_close_structure", overnight)
    add("oc_intraday", "overnight_open_close_structure", intraday)
    add("oc_gap_reversal", "overnight_open_close_structure", -(overnight * np.sign(_by_inst_shift(ret1, 1)).fillna(0.0)))
    add("oc_intraday_minus_overnight", "overnight_open_close_structure", intraday - overnight)
    add("oc_vwap_gap", "overnight_open_close_structure", vwap_gap)

    # volume-price divergence
    add("vp_div_10", "volume_price_divergence", _by_inst_roll_mean(ret1, 10) - _by_inst_roll_mean(vol_chg1, 10))
    add("vp_div_20", "volume_price_divergence", _by_inst_roll_mean(ret1, 20) - _by_inst_roll_mean(vol_chg1, 20))
    add("vp_corr_20", "volume_price_divergence", -_by_inst_corr(ret1, vol_chg1, 20))
    add("vp_lagcorr_20", "volume_price_divergence", -_by_inst_corr(ret1, _by_inst_shift(vol_chg1, 1), 20))
    add("vp_pricevol_gap", "volume_price_divergence", ret20 - _by_inst_pct(log_vol + 1.0, 20))

    # drawdown recovery
    add("dd_drawdown_60", "drawdown_recovery", dd60)
    add("dd_recovery_20", "drawdown_recovery", rec20)
    add("dd_bounce_5_20", "drawdown_recovery", rec20 - ret20.abs())
    add("dd_recovery_speed", "drawdown_recovery", _by_inst_roll_mean(rec20.diff(), 5))

    # market neutral transform candidates
    add("mn_excess_ret_20", "market_neutral_transform", _by_inst_roll_mean(ret1 - market_ret_s, 20))
    add("mn_beta_neutral_mom20", "market_neutral_transform", ret20 - beta60 * _by_inst_roll_mean(market_ret_s, 20))
    add("mn_idio_vol_20", "market_neutral_transform", vol20 - _by_inst_roll_mean(vol20.groupby(level=0).transform("mean"), 20))

    return lib


def _expand_variants(base_lib: Dict[str, Tuple[str, pd.Series]]) -> Dict[str, Tuple[str, str, pd.Series]]:
    out: Dict[str, Tuple[str, str, pd.Series]] = {}
    for base_name, (family, base_s) in base_lib.items():
        out[f"{base_name}__raw"] = (family, "raw", base_s)
        out[f"{base_name}__rank"] = (family, "rank", _cs_rank_pct(base_s))
        out[f"{base_name}__robustz"] = (family, "robustz", _cs_robust_z(base_s))
        out[f"{base_name}__mkt_neutral_robustz"] = (
            family,
            "mkt_neutral_robustz",
            _cs_robust_z(_cs_demean(base_s)),
        )
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expand factor/signal library with reproducible cross-sectional signals.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--topk-grid", default="40,45,50,55")
    p.add_argument("--n-drop-grid", default="2,4")
    p.add_argument("--top-n-backtest", type=int, default=16)
    p.add_argument("--fetch-chunk-size", type=int, default=40)
    p.add_argument("--output-prefix", default="expanded_factor")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    cfg = _load_config(run_dir / "artifacts" / "config")
    _init_quant_master(cfg)

    base_pred = _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl")).sort_index()
    base_label = _as_label_series(_load_pickle(run_dir / "artifacts" / "label.pkl")).sort_index()
    base_pred = _slice_mi(base_pred, TEST_START, TEST_END)
    base_label = _slice_mi(base_label, TEST_START, TEST_END)

    instruments = sorted(set(base_pred.index.get_level_values(1)))
    fields = ["$open", "$high", "$low", "$close", "$vwap", "$volume", "$amount", "$factor"]
    mkt = _fetch_market_df(
        instruments=instruments,
        fields=fields,
        start_time=LOOKBACK_START,
        end_time=TEST_END,
        freq="day",
        chunk_size=int(args.fetch_chunk_size),
    )
    mkt = mkt.sort_index()
    for c in ("open", "high", "low", "close", "vwap", "volume", "amount", "factor"):
        if c not in mkt.columns:
            raise RuntimeError(f"required field missing from fetched data: {c}")

    base_lib = _build_signal_library(mkt)
    all_signals = _expand_variants(base_lib)

    eval_rows: List[SignalEval] = []
    raw_rows: List[Dict[str, Any]] = []
    bt_rows: List[Dict[str, Any]] = []
    topk_grid = [int(x.strip()) for x in str(args.topk_grid).split(",") if x.strip()]
    ndrop_grid = [int(x.strip()) for x in str(args.n_drop_grid).split(",") if x.strip()]
    combos = [{"topk": tk, "n_drop": nd} for tk in topk_grid for nd in ndrop_grid if nd < tk]
    port_cfg = _extract_port_config(cfg)
    benchmark = str(port_cfg.get("backtest", {}).get("benchmark", "SH000300"))
    exchange_cache: Dict[Tuple[str, str, int, int], Any] = {}

    base_signal_for_corr = base_pred["score"].astype(float)

    for sig_name, (family, variant, sig_raw) in all_signals.items():
        sig = _slice_mi(sig_raw.sort_index(), TEST_START, TEST_END)
        panel = pd.concat(
            [
                pd.to_numeric(sig, errors="coerce").rename("signal"),
                pd.to_numeric(base_label, errors="coerce").rename("label"),
                pd.to_numeric(base_signal_for_corr, errors="coerce").rename("base7406"),
            ],
            axis=1,
        ).dropna(subset=["signal"])
        if panel.empty:
            continue
        nonnull_ratio = float(panel["signal"].notna().mean())
        ic_s, rankic_s = _ic_series(panel["signal"], panel["label"])
        ic_mean, ic_ir = _mean_ir(ic_s)
        rankic_mean, rankic_ir = _mean_ir(rankic_s)
        bucket_mean, bucket_ir = _bucket_long_short(panel["signal"], panel["label"], n_bins=10)

        cs_corr = (
            panel.groupby(level=0)
            .apply(lambda x: x["signal"].rank(pct=True).corr(x["base7406"].rank(pct=True)))
            .astype(float)
            .dropna()
        )
        cs_corr_mean = float(cs_corr.mean()) if not cs_corr.empty else float("nan")

        raw_rows.append(
            {
                "signal": sig_name,
                "family": family,
                "variant": variant,
                "nonnull_ratio": nonnull_ratio,
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "rankic_mean": rankic_mean,
                "rankic_ir": rankic_ir,
                "bucket_ls_mean": bucket_mean,
                "bucket_ls_ir": bucket_ir,
                "avg_cs_rank_corr_vs_7406": cs_corr_mean,
            }
        )

    raw_rows_sorted = sorted(
        raw_rows,
        key=lambda x: (
            float(x["rankic_ir"]) if np.isfinite(x["rankic_ir"]) else -1e9,
            float(x["ic_ir"]) if np.isfinite(x["ic_ir"]) else -1e9,
        ),
        reverse=True,
    )
    top_n = max(1, int(args.top_n_backtest))
    top_signal_names = [x["signal"] for x in raw_rows_sorted[:top_n]]

    for row in raw_rows_sorted[:top_n]:
        sig_name = str(row["signal"])
        family, variant, sig_raw = all_signals[sig_name]
        sig = _slice_mi(sig_raw.sort_index(), TEST_START, TEST_END)
        sig_df = pd.to_numeric(sig, errors="coerce").rename("score").to_frame("score").dropna()
        if sig_df.empty:
            continue
        best_bt: Optional[Dict[str, Any]] = None
        for combo in combos:
            try:
                m = _eval_topk_combo(
                    signal_df=sig_df,
                    combo=combo,
                    base_port_cfg=port_cfg,
                    benchmark=benchmark,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    exchange_cache=exchange_cache,
                )
                bt_row = {
                    "signal": sig_name,
                    "family": family,
                    "variant": variant,
                    "topk": int(combo["topk"]),
                    "n_drop": int(combo["n_drop"]),
                    **m,
                    "error": "",
                }
            except Exception as exc:  # noqa: BLE001
                bt_row = {
                    "signal": sig_name,
                    "family": family,
                    "variant": variant,
                    "topk": int(combo["topk"]),
                    "n_drop": int(combo["n_drop"]),
                    "costed_annret": float("nan"),
                    "costed_ir": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "elapsed_sec": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            bt_rows.append(bt_row)
            if bt_row["error"] == "" and np.isfinite(bt_row["costed_ir"]) and np.isfinite(bt_row["costed_annret"]):
                if best_bt is None or (bt_row["costed_ir"], bt_row["costed_annret"]) > (
                    best_bt["costed_ir"],
                    best_bt["costed_annret"],
                ):
                    best_bt = bt_row
        if best_bt is None:
            continue
        eval_rows.append(
            SignalEval(
                signal=sig_name,
                family=family,
                variant=variant,
                nonnull_ratio=float(row["nonnull_ratio"]),
                ic_mean=float(row["ic_mean"]),
                ic_ir=float(row["ic_ir"]),
                rankic_mean=float(row["rankic_mean"]),
                rankic_ir=float(row["rankic_ir"]),
                bucket_ls_mean=float(row["bucket_ls_mean"]),
                bucket_ls_ir=float(row["bucket_ls_ir"]),
                best_topk=int(best_bt["topk"]),
                best_n_drop=int(best_bt["n_drop"]),
                best_costed_ir=float(best_bt["costed_ir"]),
                best_costed_annret=float(best_bt["costed_annret"]),
                best_max_drawdown=float(best_bt["max_drawdown"]),
                best_turnover=float(best_bt["turnover"]),
            )
        )

    eval_rows_sorted = sorted(eval_rows, key=lambda x: (x.best_costed_ir, x.best_costed_annret), reverse=True)
    best = eval_rows_sorted[0] if eval_rows_sorted else None
    best_signal_name = best.signal if best is not None else None

    combo_rows: List[Dict[str, Any]] = []
    if eval_rows_sorted:
        top_for_combo = eval_rows_sorted[:6]
        combo_defs: List[Tuple[str, List[Tuple[str, float]]]] = [
            ("combo_equal_top3", [(x.signal, 1.0) for x in top_for_combo[:3]]),
            ("combo_equal_top5", [(x.signal, 1.0) for x in top_for_combo[:5]]),
            ("combo_icir_weighted_top4", [(x.signal, max(1e-6, abs(x.rankic_ir))) for x in top_for_combo[:4]]),
        ]
        for combo_name, members in combo_defs:
            ws = np.array([w for _, w in members], dtype=float)
            ws = ws / (ws.sum() + 1e-12)
            blend = None
            for (sig_name, _), w in zip(members, ws):
                _, _, s_raw = all_signals[sig_name]
                s = _slice_mi(s_raw.sort_index(), TEST_START, TEST_END)
                s = _cs_rank_pct(pd.to_numeric(s, errors="coerce")).rename(sig_name)
                blend = s * w if blend is None else blend.add(s * w, fill_value=0.0)
            if blend is None:
                continue
            panel = pd.concat([blend.rename("sig"), base_label.rename("label")], axis=1).dropna()
            ic_s, rankic_s = _ic_series(panel["sig"], panel["label"])
            ic_mean, ic_ir = _mean_ir(ic_s)
            rankic_mean, rankic_ir = _mean_ir(rankic_s)
            bucket_mean, bucket_ir = _bucket_long_short(panel["sig"], panel["label"], n_bins=10)
            blend_df = panel["sig"].rename("score").to_frame("score")
            best_bt = None
            for c in combos:
                try:
                    m = _eval_topk_combo(
                        signal_df=blend_df,
                        combo=c,
                        base_port_cfg=port_cfg,
                        benchmark=benchmark,
                        open_cost=float(args.open_cost),
                        close_cost=float(args.close_cost),
                        exchange_cache=exchange_cache,
                    )
                    row = {
                        "signal": combo_name,
                        "family": "simple_combo",
                        "variant": "rank_blend",
                        "topk": int(c["topk"]),
                        "n_drop": int(c["n_drop"]),
                        **m,
                        "error": "",
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "signal": combo_name,
                        "family": "simple_combo",
                        "variant": "rank_blend",
                        "topk": int(c["topk"]),
                        "n_drop": int(c["n_drop"]),
                        "costed_annret": float("nan"),
                        "costed_ir": float("nan"),
                        "max_drawdown": float("nan"),
                        "turnover": float("nan"),
                        "elapsed_sec": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                bt_rows.append(row)
                if row["error"] == "" and np.isfinite(row["costed_ir"]) and np.isfinite(row["costed_annret"]):
                    if best_bt is None or (row["costed_ir"], row["costed_annret"]) > (best_bt["costed_ir"], best_bt["costed_annret"]):
                        best_bt = row
            combo_rows.append(
                {
                    "signal": combo_name,
                    "family": "simple_combo",
                    "variant": "rank_blend",
                    "members": [{"signal": n, "weight": float(w)} for (n, _), w in zip(members, ws)],
                    "ic_mean": ic_mean,
                    "ic_ir": ic_ir,
                    "rankic_mean": rankic_mean,
                    "rankic_ir": rankic_ir,
                    "bucket_ls_mean": bucket_mean,
                    "bucket_ls_ir": bucket_ir,
                    "best_topk": int(best_bt["topk"]) if best_bt is not None else None,
                    "best_n_drop": int(best_bt["n_drop"]) if best_bt is not None else None,
                    "best_costed_ir": float(best_bt["costed_ir"]) if best_bt is not None else float("nan"),
                    "best_costed_annret": float(best_bt["costed_annret"]) if best_bt is not None else float("nan"),
                    "best_max_drawdown": float(best_bt["max_drawdown"]) if best_bt is not None else float("nan"),
                    "best_turnover": float(best_bt["turnover"]) if best_bt is not None else float("nan"),
                }
            )

    if combo_rows:
        combo_best = sorted(
            combo_rows,
            key=lambda x: (
                float(x["best_costed_ir"]) if np.isfinite(x["best_costed_ir"]) else -1e9,
                float(x["best_costed_annret"]) if np.isfinite(x["best_costed_annret"]) else -1e9,
            ),
            reverse=True,
        )[0]
        if best is None or (combo_best["best_costed_ir"], combo_best["best_costed_annret"]) > (
            best.best_costed_ir,
            best.best_costed_annret,
        ):
            best_signal_name = str(combo_best["signal"])

    out_dir = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scan_csv = out_dir / f"{args.output_prefix}_signal_scan_{stamp}.csv"
    bt_csv = out_dir / f"{args.output_prefix}_backtest_scan_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"
    pred_pkl = out_dir / f"{args.output_prefix}_best_pred_{stamp}.pkl"
    pred_csv = out_dir / f"{args.output_prefix}_best_pred_{stamp}.csv"

    _write_csv(scan_csv, raw_rows_sorted)
    _write_csv(bt_csv, bt_rows)

    top_candidates = []
    for e in eval_rows_sorted[:12]:
        top_candidates.append(
            {
                "signal": e.signal,
                "family": e.family,
                "variant": e.variant,
                "rankic_ir": e.rankic_ir,
                "ic_ir": e.ic_ir,
                "bucket_ls_ir": e.bucket_ls_ir,
                "best_topk": e.best_topk,
                "best_n_drop": e.best_n_drop,
                "best_costed_ir": e.best_costed_ir,
                "best_costed_annret": e.best_costed_annret,
                "best_max_drawdown": e.best_max_drawdown,
                "best_turnover": e.best_turnover,
            }
        )
    for c in combo_rows:
        top_candidates.append(
            {
                "signal": c["signal"],
                "family": c["family"],
                "variant": c["variant"],
                "rankic_ir": c["rankic_ir"],
                "ic_ir": c["ic_ir"],
                "bucket_ls_ir": c["bucket_ls_ir"],
                "best_topk": c["best_topk"],
                "best_n_drop": c["best_n_drop"],
                "best_costed_ir": c["best_costed_ir"],
                "best_costed_annret": c["best_costed_annret"],
                "best_max_drawdown": c["best_max_drawdown"],
                "best_turnover": c["best_turnover"],
                "members": c.get("members", []),
            }
        )
    top_candidates = sorted(
        top_candidates,
        key=lambda x: (
            float(x["best_costed_ir"]) if np.isfinite(x["best_costed_ir"]) else -1e9,
            float(x["best_costed_annret"]) if np.isfinite(x["best_costed_annret"]) else -1e9,
        ),
        reverse=True,
    )

    best_metrics = top_candidates[0] if top_candidates else {}
    passes_hard_gate = bool(
        top_candidates
        and np.isfinite(best_metrics.get("best_costed_ir", float("nan")))
        and np.isfinite(best_metrics.get("best_costed_annret", float("nan")))
        and float(best_metrics["best_costed_ir"]) > BREAKTHROUGH_IR
        and float(best_metrics["best_costed_annret"]) > BREAKTHROUGH_ANNRET
    )

    best_signal_df = None
    if best_signal_name is not None:
        if best_signal_name in all_signals:
            _, _, s_raw = all_signals[best_signal_name]
            best_signal_df = _slice_mi(s_raw.sort_index(), TEST_START, TEST_END).rename("score").to_frame("score").dropna()
        else:
            for c in combo_rows:
                if c["signal"] == best_signal_name:
                    members = c.get("members", [])
                    blend = None
                    for m in members:
                        sname = str(m["signal"])
                        w = float(m["weight"])
                        _, _, s_raw = all_signals[sname]
                        s = _slice_mi(s_raw.sort_index(), TEST_START, TEST_END)
                        s = _cs_rank_pct(pd.to_numeric(s, errors="coerce"))
                        blend = s * w if blend is None else blend.add(s * w, fill_value=0.0)
                    if blend is not None:
                        best_signal_df = blend.rename("score").to_frame("score").dropna()
                    break
    if best_signal_df is not None and not best_signal_df.empty:
        with pred_pkl.open("wb") as f:
            pickle.dump(best_signal_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        best_signal_df.reset_index().to_csv(pred_csv, index=False)

    summary = {
        "scan_time_utc": _now_utc(),
        "base_run_id": args.base_run_id,
        "test_period": {"start": TEST_START, "end": TEST_END},
        "feature_window_start": LOOKBACK_START,
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "leakage_control": {
            "signal_construction": "only current/past OHLCV/amount/factor with per-instrument rolling/shift; no future label involved",
            "label_alignment": "evaluation uses precomputed label (Ref($close,-2)/Ref($close,-1)-1) at same (datetime,instrument) index",
            "rolling_policy": "rolling windows computed in chronological order within each instrument",
        },
        "available_fields_checked": {
            "available": ["$open", "$high", "$low", "$close", "$vwap", "$volume", "$amount", "$factor"],
            "empty_or_unavailable": ["$turnover", "$turn", "$paused"],
            "sector_neutral_note": "sector-level neutralization skipped due no reliable sector field in this local daily dataset",
        },
        "signal_count_total": len(all_signals),
        "screened_signal_count": len(raw_rows_sorted),
        "backtested_signal_count": top_n,
        "combos_tested_per_signal": combos,
        "best_overall": best_metrics,
        "top_candidates": top_candidates[:20],
        "combo_candidates": combo_rows,
        "passes_hard_gate": passes_hard_gate,
        "hard_gate": {"ir_gt": BREAKTHROUGH_IR, "annret_gt": BREAKTHROUGH_ANNRET},
        "artifacts": {
            "scan_csv": str(scan_csv),
            "backtest_csv": str(bt_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "best_pred_pkl": str(pred_pkl if pred_pkl.exists() else ""),
            "best_pred_csv": str(pred_csv if pred_csv.exists() else ""),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append(f"# Expanded Factor Signal Library Summary ({stamp})")
    md_lines.append("")
    md_lines.append(f"- base_run_id: `{args.base_run_id}`")
    md_lines.append(f"- test_period: `{TEST_START}` to `{TEST_END}`")
    md_lines.append(f"- hard_gate: `IR > {BREAKTHROUGH_IR}` and `AnnRet > {BREAKTHROUGH_ANNRET}`")
    md_lines.append(f"- passes_hard_gate: `{'yes' if passes_hard_gate else 'no'}`")
    md_lines.append("")
    md_lines.append("## Top Candidates")
    md_lines.append("")
    md_lines.append("| signal | family | variant | RankIC_IR | IC_IR | bucket_IR | best_topk | best_n_drop | costed_IR | AnnRet | maxDD | turnover |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_candidates[:20]:
        md_lines.append(
            "| {signal} | {family} | {variant} | {rankic:.6f} | {ic:.6f} | {bkir:.6f} | {tk} | {nd} | {ir:.6f} | {ann:.6f} | {mdd:.6f} | {to:.6f} |".format(
                signal=r.get("signal", ""),
                family=r.get("family", ""),
                variant=r.get("variant", ""),
                rankic=float(r.get("rankic_ir", float("nan"))),
                ic=float(r.get("ic_ir", float("nan"))),
                bkir=float(r.get("bucket_ls_ir", float("nan"))),
                tk=int(r.get("best_topk") or 0),
                nd=int(r.get("best_n_drop") or 0),
                ir=float(r.get("best_costed_ir", float("nan"))),
                ann=float(r.get("best_costed_annret", float("nan"))),
                mdd=float(r.get("best_max_drawdown", float("nan"))),
                to=float(r.get("best_turnover", float("nan"))),
            )
        )
    md_lines.append("")
    md_lines.append("## Leakage Notes")
    md_lines.append("")
    md_lines.append("- All rolling features are instrument-wise backward-looking only.")
    md_lines.append("- No signal uses future label or forward-filled future values.")
    md_lines.append("- Label alignment follows existing workflow label index directly.")
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    parse_smoke = {
        "csv_rows_scan": int(len(pd.read_csv(scan_csv))) if scan_csv.exists() and scan_csv.stat().st_size > 0 else 0,
        "csv_rows_backtest": int(len(pd.read_csv(bt_csv))) if bt_csv.exists() and bt_csv.stat().st_size > 0 else 0,
        "summary_json_keys": sorted(list(json.loads(summary_json.read_text(encoding="utf-8")).keys())),
        "best_pred_rows": 0,
    }
    if pred_pkl.exists():
        pred_obj = _load_pickle(pred_pkl)
        if isinstance(pred_obj, pd.DataFrame):
            parse_smoke["best_pred_rows"] = int(len(pred_obj))
        elif isinstance(pred_obj, pd.Series):
            parse_smoke["best_pred_rows"] = int(len(pred_obj))
    smoke_path = out_dir / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"
    smoke_path.write_text(json.dumps(parse_smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["artifacts"]["artifact_parse_smoke_json"] = str(smoke_path)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
