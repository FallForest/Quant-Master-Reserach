#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


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
    "3e594f3f": "3e594f3ffafb47ffa36491c3f04f2afa",
    "587bba62": "587bba6200be43e68cf02f59d1b7f890",
    "29864d9c": "29864d9c5d00463b9fdbc065c10b0093",
    "4a98f99b": "4a98f99bdb6848bab789ff6c46d0a1ff",
}


@dataclass
class FoldStat:
    fold: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    pred_start: str
    pred_end: str
    train_rows: int
    valid_rows: int
    pred_rows: int
    best_epoch: int
    best_valid_ic: float
    best_valid_loss: float
    elapsed_sec: float


class DeepStackRanker(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        n_regimes: int,
        emb_dim: int = 6,
        hidden_dim: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.regime_emb = nn.Embedding(n_regimes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_numeric + emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_regime: torch.Tensor) -> torch.Tensor:
        e = self.regime_emb(x_regime)
        x = torch.cat([x_num, e], dim=1)
        return self.net(x).squeeze(-1)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _resolve_run_ids(tracking_dir: Path, text: str) -> List[str]:
    all_run_ids = {
        p.name for p in tracking_dir.glob("*/*") if p.is_dir() and (p / "artifacts").exists() and len(p.name) == 32
    }
    out: List[str] = []
    for raw in [x.strip() for x in text.split(",") if x.strip()]:
        mapped = RUN_ALIAS.get(raw, raw)
        if mapped in all_run_ids:
            out.append(mapped)
            continue
        cands = [rid for rid in all_run_ids if rid.startswith(mapped)]
        if len(cands) == 1:
            out.append(cands[0])
            continue
        if not cands:
            raise FileNotFoundError(f"run token cannot be resolved: {raw}")
        raise RuntimeError(f"run token matches multiple run_ids: token={raw}, candidates={cands}")
    return list(dict.fromkeys(out))


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
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port_analysis_config or task.record[PortAnaRecord].kwargs.config")


def _as_series(pred_obj: Any, preferred: str = "score") -> pd.Series:
    if isinstance(pred_obj, pd.Series):
        return pred_obj.astype(float)
    if isinstance(pred_obj, pd.DataFrame):
        if preferred in pred_obj.columns:
            return pred_obj[preferred].astype(float)
        if pred_obj.shape[1] == 1:
            return pred_obj.iloc[:, 0].astype(float)
        return pred_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported object type: {type(pred_obj)}")


def _slice_period(series: pd.Series, start: str, end: str) -> pd.Series:
    idx = pd.to_datetime(series.index.get_level_values(0))
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return series[mask]


def _cross_section_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(method="average", pct=True)


def _cross_section_zscore(series: pd.Series) -> pd.Series:
    g = series.groupby(level=0)
    mu = g.transform("mean")
    sd = g.transform("std").replace(0.0, np.nan)
    z = (series - mu) / sd
    return z.fillna(0.0)


def _daily_rank_ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    df = pd.DataFrame({"date": dates, "pred": pred, "y": y})
    out = []
    for _, g in df.groupby("date"):
        if len(g) < 5:
            continue
        corr = g["pred"].corr(g["y"], method="spearman")
        if pd.notna(corr):
            out.append(float(corr))
    if not out:
        return float("nan")
    return float(np.mean(out))


def _weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    loss = 0.5 * quad * quad / delta + lin
    return (loss * weight).mean()


def _evaluate_valid(
    model: DeepStackRanker,
    x_num: np.ndarray,
    x_reg: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    dates: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    model.eval()
    preds: List[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            j = min(len(x_num), i + batch_size)
            bx = torch.from_numpy(x_num[i:j]).to(device)
            br = torch.from_numpy(x_reg[i:j]).to(device)
            by = torch.from_numpy(y[i:j]).to(device)
            bw = torch.from_numpy(w[i:j]).to(device)
            out = model(bx, br)
            loss = _weighted_huber_loss(out, by, bw)
            loss_sum += float(loss.item()) * (j - i)
            count += j - i
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    valid_loss = loss_sum / max(1, count)
    valid_ic = _daily_rank_ic(pred=pred, y=y, dates=dates)
    return valid_loss, valid_ic


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _train_one_fold(
    *,
    x_train: np.ndarray,
    r_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    r_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    valid_dates: np.ndarray,
    x_pred: np.ndarray,
    r_pred: np.ndarray,
    n_regimes: int,
    device: torch.device,
    seed: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    emb_dim: int,
    hidden_dim: int,
    dropout: float,
) -> Tuple[np.ndarray, int, float, float]:
    _set_seed(seed)
    model = DeepStackRanker(
        n_numeric=x_train.shape[1],
        n_regimes=n_regimes,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(r_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_ic = -np.inf
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        for bx, br, by, bw in loader:
            bx = bx.to(device)
            br = br.to(device)
            by = by.to(device)
            bw = bw.to(device)
            out = model(bx, br)
            loss = _weighted_huber_loss(out, by, bw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        v_loss, v_ic = _evaluate_valid(
            model=model,
            x_num=x_valid,
            x_reg=r_valid,
            y=y_valid,
            w=w_valid,
            dates=valid_dates,
            device=device,
            batch_size=batch_size,
        )
        improved = False
        if np.isfinite(v_ic):
            if v_ic > best_ic + 1e-6:
                improved = True
        else:
            if v_loss < best_loss:
                improved = True
        if improved:
            best_ic = float(v_ic)
            best_loss = float(v_loss)
            best_epoch = ep
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_out = []
        for i in range(0, len(x_pred), batch_size):
            j = min(len(x_pred), i + batch_size)
            bx = torch.from_numpy(x_pred[i:j]).to(device)
            br = torch.from_numpy(r_pred[i:j]).to(device)
            yhat = model(bx, br).detach().cpu().numpy()
            pred_out.append(yhat)
    pred_arr = np.concatenate(pred_out) if pred_out else np.zeros(0, dtype=np.float32)
    return pred_arr, best_epoch, float(best_ic), float(best_loss)


def _build_base_dataframe(
    *,
    tracking_dir: Path,
    run_ids: Sequence[str],
    base_run_id: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
    rows = []
    pred_map: Dict[str, pd.Series] = {}
    label_series = None
    for run_id in run_ids:
        run_dir = _find_run_dir(tracking_dir, run_id)
        art = run_dir / "artifacts"
        pred = _slice_period(_as_series(_load_pickle(art / "pred.pkl"), preferred="score"), start_date, end_date)
        pred_map[run_id] = pred
        rec = {"run_id": run_id, "pred_rows": int(len(pred))}
        lp = art / "label.pkl"
        if lp.exists():
            lbl = _slice_period(_as_series(_load_pickle(lp), preferred="label"), start_date, end_date)
            rec["label_rows"] = int(len(lbl))
            if run_id == base_run_id:
                label_series = lbl
        rows.append(rec)
    if label_series is None:
        base_dir = _find_run_dir(tracking_dir, base_run_id)
        label_series = _slice_period(
            _as_series(_load_pickle(base_dir / "artifacts" / "label.pkl"), preferred="label"), start_date, end_date
        )
    panel = pd.concat([pred_map[rid].rename(f"pred_{rid[:8]}") for rid in run_ids], axis=1, join="inner")
    panel = panel.sort_index()
    label_series = label_series.reindex(panel.index).astype(float)
    valid = label_series.notna()
    panel = panel[valid]
    label_series = label_series[valid]
    return panel, label_series, rows


def _build_features(panel_pred: pd.DataFrame, label_series: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    feat = pd.DataFrame(index=panel_pred.index)
    rank_cols = []
    for col in panel_pred.columns:
        s = panel_pred[col].astype(float)
        rk = _cross_section_rank(s)
        z = _cross_section_zscore(s)
        feat[f"{col}_rank"] = rk
        feat[f"{col}_z"] = z
        feat[f"{col}_inv_rank"] = 1.0 - rk
        rank_cols.append(f"{col}_rank")

    rank_panel = feat[rank_cols]
    feat["rank_mean"] = rank_panel.mean(axis=1)
    feat["rank_std"] = rank_panel.std(axis=1).fillna(0.0)
    feat["rank_spread"] = rank_panel.max(axis=1) - rank_panel.min(axis=1)

    base_rank_col = rank_cols[0]
    base_rank = feat[base_rank_col]
    feat["base_rank_lag1"] = base_rank.groupby(level=1, group_keys=False).shift(1)
    feat["base_rank_lag3"] = base_rank.groupby(level=1, group_keys=False).shift(3)
    feat["base_rank_mom3"] = feat["base_rank_lag1"] - feat["base_rank_lag3"]
    feat["rank_mean_lag1"] = feat["rank_mean"].groupby(level=1, group_keys=False).shift(1)
    feat["rank_mean_lag5"] = feat["rank_mean"].groupby(level=1, group_keys=False).shift(5)
    feat["rank_mean_mom5"] = feat["rank_mean_lag1"] - feat["rank_mean_lag5"]

    y_rank = _cross_section_rank(label_series) - 0.5
    feat["target"] = y_rank.astype(float)

    daily = pd.DataFrame(index=sorted(set(feat.index.get_level_values(0))))
    daily["base_disp"] = base_rank.groupby(level=0).std().astype(float)
    daily["ens_disp"] = feat["rank_std"].groupby(level=0).mean().astype(float)
    daily["mkt_proxy"] = label_series.groupby(level=0).mean().astype(float).shift(1).fillna(0.0)
    daily["base_disp_q"] = pd.qcut(daily["base_disp"], q=3, labels=False, duplicates="drop").astype(int)
    daily["ens_disp_q"] = pd.qcut(daily["ens_disp"], q=3, labels=False, duplicates="drop").astype(int)
    daily["regime_id"] = (daily["base_disp_q"] * 3 + daily["ens_disp_q"]).astype(int)

    idx_dates = feat.index.get_level_values(0)
    feat["regime_id"] = idx_dates.map(daily["regime_id"]).astype(int)
    feat["regime_strength"] = idx_dates.map(daily["ens_disp"]).astype(float)
    feat["mkt_proxy"] = idx_dates.map(daily["mkt_proxy"]).astype(float)

    feat = feat.dropna().sort_index()
    return feat, rank_cols


def _sample_rows(idx: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=max_rows, replace=False)
    pick.sort()
    return pick


def _run_walkforward_training(
    *,
    feat_df: pd.DataFrame,
    min_train_days: int,
    train_window_days: int,
    valid_days: int,
    block_days: int,
    max_folds: int,
    max_train_rows: int,
    max_valid_rows: int,
    seed: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    emb_dim: int,
    hidden_dim: int,
    dropout: float,
) -> Tuple[pd.Series, List[FoldStat]]:
    dates = pd.Index(sorted(feat_df.index.get_level_values(0).unique()))
    date_vals = feat_df.index.get_level_values(0).to_numpy()

    numeric_cols = [c for c in feat_df.columns if c not in {"target", "regime_id"}]
    x_all = feat_df[numeric_cols].to_numpy(np.float32)
    r_all = feat_df["regime_id"].to_numpy(np.int64)
    y_all = feat_df["target"].to_numpy(np.float32)
    w_all = (1.0 + np.clip(np.abs(y_all) * 6.0, 0.0, 4.0)).astype(np.float32)
    pred_out = np.full(len(feat_df), np.nan, dtype=np.float32)

    fold_stats: List[FoldStat] = []
    n_regimes = int(feat_df["regime_id"].max()) + 1

    fold_idx = 0
    for start_i in range(min_train_days, len(dates), block_days):
        if max_folds > 0 and fold_idx >= max_folds:
            break
        pred_dates = dates[start_i : min(len(dates), start_i + block_days)]
        if len(pred_dates) == 0:
            continue

        tr_end = start_i
        tr_start = max(0, tr_end - train_window_days)
        train_dates_all = dates[tr_start:tr_end]
        if len(train_dates_all) <= valid_days + 20:
            continue
        valid_dates = train_dates_all[-valid_days:]
        train_dates = train_dates_all[:-valid_days]

        tr_mask = np.isin(date_vals, train_dates.to_numpy())
        va_mask = np.isin(date_vals, valid_dates.to_numpy())
        pr_mask = np.isin(date_vals, pred_dates.to_numpy())
        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)
        pr_idx = np.flatnonzero(pr_mask)
        if len(tr_idx) < 1000 or len(va_idx) < 500 or len(pr_idx) < 500:
            continue

        tr_idx = _sample_rows(tr_idx, max_train_rows, seed + fold_idx)
        va_idx = _sample_rows(va_idx, max_valid_rows, seed + 1000 + fold_idx)

        mu, sd = _standardize_fit(x_all[tr_idx])
        x_train = _standardize_apply(x_all[tr_idx], mu, sd)
        x_valid = _standardize_apply(x_all[va_idx], mu, sd)
        x_pred = _standardize_apply(x_all[pr_idx], mu, sd)
        r_train = r_all[tr_idx]
        r_valid = r_all[va_idx]
        r_pred = r_all[pr_idx]
        y_train = y_all[tr_idx]
        y_valid = y_all[va_idx]
        w_train = w_all[tr_idx]
        w_valid = w_all[va_idx]
        valid_day_arr = pd.to_datetime(date_vals[va_idx]).astype(str).to_numpy()

        t0 = time.perf_counter()
        pred_arr, best_epoch, best_ic, best_loss = _train_one_fold(
            x_train=x_train,
            r_train=r_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            r_valid=r_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            valid_dates=valid_day_arr,
            x_pred=x_pred,
            r_pred=r_pred,
            n_regimes=n_regimes,
            device=device,
            seed=seed + fold_idx,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        elapsed = float(time.perf_counter() - t0)
        pred_out[pr_idx] = pred_arr.astype(np.float32)
        fold_stats.append(
            FoldStat(
                fold=fold_idx + 1,
                train_start=str(pd.Timestamp(train_dates.min()).date()),
                train_end=str(pd.Timestamp(train_dates.max()).date()),
                valid_start=str(pd.Timestamp(valid_dates.min()).date()),
                valid_end=str(pd.Timestamp(valid_dates.max()).date()),
                pred_start=str(pd.Timestamp(pred_dates.min()).date()),
                pred_end=str(pd.Timestamp(pred_dates.max()).date()),
                train_rows=int(len(tr_idx)),
                valid_rows=int(len(va_idx)),
                pred_rows=int(len(pr_idx)),
                best_epoch=int(best_epoch),
                best_valid_ic=float(best_ic),
                best_valid_loss=float(best_loss),
                elapsed_sec=elapsed,
            )
        )
        fold_idx += 1
        print(
            f"[fold {fold_idx}] pred={fold_stats[-1].pred_start}..{fold_stats[-1].pred_end} "
            f"valid_ic={best_ic:.6f} rows(train/valid/pred)={len(tr_idx)}/{len(va_idx)}/{len(pr_idx)} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    pred_series = pd.Series(pred_out, index=feat_df.index, name="deep_score")
    return pred_series, fold_stats


def _build_exchange_cache_key(
    start_time: str, end_time: str, open_cost: float, close_cost: float, limit_threshold: float, deal_price: str
) -> Tuple[str, str, float, float, float, str]:
    return (start_time, end_time, open_cost, close_cost, limit_threshold, deal_price)


def _calc_costed_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    risk_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq="1day")
    return {
        "costed_annret": float(risk_df.loc["annualized_return", "risk"]),
        "costed_ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report_df["turnover"].mean()),
    }


def _eval_portfolio(
    *,
    signal: pd.Series,
    base_port_cfg: Dict[str, Any],
    topk: int,
    n_drop: int,
    open_cost: float,
    close_cost: float,
    start_time: str,
    end_time: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, float]:
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
    pred_df = signal.to_frame("score")
    pred_idx = pred_df.index.get_level_values(0)
    m = (pred_idx >= pd.Timestamp(start_time)) & (pred_idx <= pd.Timestamp(end_time))
    pred_df = pred_df[m]

    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)
    strategy = TopkDropoutStrategy(
        signal=pred_df,
        topk=int(topk),
        n_drop=int(n_drop),
        method_sell=strategy_kwargs.get("method_sell", "bottom"),
        method_buy=strategy_kwargs.get("method_buy", "top"),
        hold_thresh=int(strategy_kwargs.get("hold_thresh", 1)),
        only_tradable=bool(strategy_kwargs.get("only_tradable", False)),
        forbid_all_trade_at_limit=bool(strategy_kwargs.get("forbid_all_trade_at_limit", True)),
        risk_degree=float(strategy_kwargs.get("risk_degree", 0.95)),
    )

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
        open_cost=float(open_cost),
        close_cost=float(close_cost),
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
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = run_backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    elapsed = float(time.perf_counter() - t0)
    if "1day" in portfolio_metric_dict:
        report_df = portfolio_metric_dict["1day"][0]
    else:
        k = next(iter(portfolio_metric_dict.keys()))
        report_df = portfolio_metric_dict[k][0]
    met = _calc_costed_metrics(report_df)
    met["elapsed_sec"] = elapsed
    return met


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _candidate_signals(base_rank: pd.Series, deep_score: pd.Series) -> Dict[str, pd.Series]:
    deep_rank = _cross_section_rank(deep_score.fillna(deep_score.groupby(level=0).transform("median")).fillna(0.5))
    out = {
        "deep_raw_rank": deep_rank,
        "blend_base85_deep15": 0.85 * base_rank + 0.15 * deep_rank,
        "blend_base70_deep30": 0.70 * base_rank + 0.30 * deep_rank,
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU deep stacking model for Transcendence signals.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default="7406e47063e9479cb34d300b9ed03bad")
    p.add_argument(
        "--run-ids",
        default="7406e470,587bba62,3e594f3f,e2300230,773bd6d,5ae326c0,d4526da,1a085ff,29864d9c,4a98f99b,bcbecf55",
    )
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-04-30")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--topk", type=int, default=45)
    p.add_argument("--n-drop", type=int, default=4)
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--output-prefix", default="gpu_deep_stack")
    return p


def main() -> int:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))

    tracking_dir = _parse_tracking_dir(args.tracking_uri)
    run_ids = _resolve_run_ids(tracking_dir, args.run_ids)
    if args.base_run_id not in run_ids:
        run_ids = [args.base_run_id] + run_ids
    run_ids = list(dict.fromkeys(run_ids))

    mode_cfg = {
        "smoke": {
            "train_window_days": 180,
            "valid_days": 30,
            "min_train_days": 140,
            "block_days": 21,
            "max_folds": 4,
            "max_train_rows": 45000,
            "max_valid_rows": 9000,
            "epochs": 3,
            "patience": 2,
            "batch_size": 4096,
            "lr": 2e-3,
            "weight_decay": 5e-4,
            "hidden_dim": 72,
            "dropout": 0.1,
            "emb_dim": 4,
        },
        "full": {
            "train_window_days": 300,
            "valid_days": 50,
            "min_train_days": 180,
            "block_days": 21,
            "max_folds": 0,  # 0 means all
            "max_train_rows": 90000,
            "max_valid_rows": 18000,
            "epochs": 8,
            "patience": 3,
            "batch_size": 8192,
            "lr": 1.5e-3,
            "weight_decay": 8e-4,
            "hidden_dim": 96,
            "dropout": 0.15,
            "emb_dim": 6,
        },
    }[args.mode]

    base_run_dir = _find_run_dir(tracking_dir, args.base_run_id)
    workflow_cfg = _load_config(base_run_dir / "artifacts" / "config")
    _init_quant_master(workflow_cfg)
    base_port_cfg = _extract_port_config(workflow_cfg)

    panel_pred, label_series, signal_rows = _build_base_dataframe(
        tracking_dir=tracking_dir,
        run_ids=run_ids,
        base_run_id=args.base_run_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    feat_df, rank_cols = _build_features(panel_pred=panel_pred, label_series=label_series)
    if feat_df.empty:
        raise RuntimeError("feature frame is empty after preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}",
        flush=True,
    )

    deep_score, fold_stats = _run_walkforward_training(
        feat_df=feat_df,
        min_train_days=int(mode_cfg["min_train_days"]),
        train_window_days=int(mode_cfg["train_window_days"]),
        valid_days=int(mode_cfg["valid_days"]),
        block_days=int(mode_cfg["block_days"]),
        max_folds=int(mode_cfg["max_folds"]),
        max_train_rows=int(mode_cfg["max_train_rows"]),
        max_valid_rows=int(mode_cfg["max_valid_rows"]),
        seed=int(args.seed),
        device=device,
        lr=float(mode_cfg["lr"]),
        weight_decay=float(mode_cfg["weight_decay"]),
        batch_size=int(mode_cfg["batch_size"]),
        epochs=int(mode_cfg["epochs"]),
        patience=int(mode_cfg["patience"]),
        emb_dim=int(mode_cfg["emb_dim"]),
        hidden_dim=int(mode_cfg["hidden_dim"]),
        dropout=float(mode_cfg["dropout"]),
    )
    base_rank = feat_df[rank_cols[0]].astype(float)
    # For warmup gaps not covered by walk-forward blocks, keep base rank.
    deep_score_filled = deep_score.copy()
    miss = deep_score_filled.isna()
    deep_score_filled[miss] = base_rank[miss]
    cands = _candidate_signals(base_rank=base_rank, deep_score=deep_score_filled)

    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    cand_rows: List[Dict[str, Any]] = []
    for name, sig in cands.items():
        full_met = _eval_portfolio(
            signal=sig,
            base_port_cfg=base_port_cfg,
            topk=int(args.topk),
            n_drop=int(args.n_drop),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_time=args.start_date,
            end_time=args.end_date,
            exchange_cache=exchange_cache,
        )
        ytd_met = _eval_portfolio(
            signal=sig,
            base_port_cfg=base_port_cfg,
            topk=int(args.topk),
            n_drop=int(args.n_drop),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start_time="2026-01-01",
            end_time=args.end_date,
            exchange_cache=exchange_cache,
        )
        row = {
            "candidate": name,
            "topk": int(args.topk),
            "n_drop": int(args.n_drop),
            "full_costed_ir": float(full_met["costed_ir"]),
            "full_costed_annret": float(full_met["costed_annret"]),
            "full_max_drawdown": float(full_met["max_drawdown"]),
            "full_turnover": float(full_met["turnover"]),
            "full_eval_sec": float(full_met["elapsed_sec"]),
            "ytd2026_costed_ir": float(ytd_met["costed_ir"]),
            "ytd2026_costed_annret": float(ytd_met["costed_annret"]),
            "passes_hard_gate": bool(full_met["costed_ir"] > 2.90 and full_met["costed_annret"] > 0.27),
        }
        cand_rows.append(row)
        print(
            f"[cand {name}] IR={row['full_costed_ir']:.6f} AnnRet={row['full_costed_annret']:.6f} "
            f"2026YTD_IR={row['ytd2026_costed_ir']:.6f}",
            flush=True,
        )

    cand_rows.sort(key=lambda x: (x["full_costed_ir"], x["full_costed_annret"]), reverse=True)
    best = cand_rows[0] if cand_rows else None

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{args.output_prefix}_{args.mode}"
    summary_json = out_dir / f"{prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{prefix}_summary_{stamp}.md"
    fold_csv = out_dir / f"{prefix}_folds_{stamp}.csv"
    cand_csv = out_dir / f"{prefix}_candidates_{stamp}.csv"
    pred_pkl = out_dir / f"{prefix}_pred_{stamp}.pkl"
    dataset_csv = out_dir / f"{prefix}_dataset_{stamp}.csv"

    _write_csv(
        fold_csv,
        [
            {
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "valid_start": f.valid_start,
                "valid_end": f.valid_end,
                "pred_start": f.pred_start,
                "pred_end": f.pred_end,
                "train_rows": f.train_rows,
                "valid_rows": f.valid_rows,
                "pred_rows": f.pred_rows,
                "best_epoch": f.best_epoch,
                "best_valid_ic": f.best_valid_ic,
                "best_valid_loss": f.best_valid_loss,
                "elapsed_sec": f.elapsed_sec,
            }
            for f in fold_stats
        ],
    )
    _write_csv(cand_csv, cand_rows)

    pred_df = pd.DataFrame(
        {
            "base_rank": base_rank.astype(float),
            "deep_score_raw": deep_score.astype(float),
            "deep_score_filled": deep_score_filled.astype(float),
            "deep_rank": _cross_section_rank(deep_score_filled).astype(float),
        },
        index=feat_df.index,
    )
    with pred_pkl.open("wb") as f:
        pickle.dump(pred_df, f)

    dataset_profile = pd.DataFrame(
        [
            {
                "row_count": int(len(feat_df)),
                "trade_days": int(feat_df.index.get_level_values(0).nunique()),
                "instruments": int(feat_df.index.get_level_values(1).nunique()),
                "feature_count": int(len([c for c in feat_df.columns if c not in {"target", "regime_id"}])),
                "target_mean": float(feat_df["target"].mean()),
                "target_std": float(feat_df["target"].std()),
            }
        ]
    )
    dataset_profile.to_csv(dataset_csv, index=False)

    leakage_risk = {
        "assessment": "medium",
        "notes": [
            "Walk-forward uses only historical labels per fold; no future-label fitting inside each fold.",
            "Candidate family is predefined (deep raw + fixed blends), but best-of-candidates is still selected on full 2024-01-01..2026-04-30 report.",
            "Warmup gaps are filled with base rank signal; this reduces instability but weakens pure deep attribution.",
        ],
    }
    summary = {
        "timestamp_utc": _now_utc(),
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "environment": {
            "python": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
        },
        "signals": signal_rows,
        "run_ids": run_ids,
        "split": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "walkforward": {
                "train_window_days": int(mode_cfg["train_window_days"]),
                "valid_days": int(mode_cfg["valid_days"]),
                "min_train_days": int(mode_cfg["min_train_days"]),
                "block_days": int(mode_cfg["block_days"]),
                "fold_count": int(len(fold_stats)),
            },
            "portfolio_eval": {
                "full_period": [args.start_date, args.end_date],
                "ytd_2026": ["2026-01-01", args.end_date],
                "topk": int(args.topk),
                "n_drop": int(args.n_drop),
                "open_cost": float(args.open_cost),
                "close_cost": float(args.close_cost),
            },
        },
        "metrics": {
            "candidates": cand_rows,
            "best": best,
            "hard_gate": {"ir_gt": 2.90, "annret_gt": 0.27},
            "hard_gate_passed": bool(best and best["passes_hard_gate"]),
        },
        "leakage_risk": leakage_risk,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "fold_csv": str(fold_csv),
            "candidate_csv": str(cand_csv),
            "pred_pkl": str(pred_pkl),
            "dataset_csv": str(dataset_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# GPU Deep Stack Summary ({args.mode}, {stamp})",
        "",
        f"- command: `{summary['command']}`",
        f"- torch: `{torch.__version__}`; cuda_available: `{torch.cuda.is_available()}`; device: `{device}`",
        f"- test_period: `{args.start_date}..{args.end_date}`",
        f"- costs: `open={args.open_cost}`, `close={args.close_cost}`",
        "",
        "## Best Candidate",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- name: `{best['candidate']}`",
                f"- full IR: `{best['full_costed_ir']:.6f}`",
                f"- full AnnRet: `{best['full_costed_annret']:.6f}`",
                f"- full MDD: `{best['full_max_drawdown']:.6f}`",
                f"- turnover: `{best['full_turnover']:.6f}`",
                f"- 2026YTD IR: `{best['ytd2026_costed_ir']:.6f}`",
                f"- 2026YTD AnnRet: `{best['ytd2026_costed_annret']:.6f}`",
                f"- hard_gate_passed: `{best['passes_hard_gate']}`",
            ]
        )
    else:
        lines.append("- no candidate generated")
    lines.extend(
        [
            "",
            "## Leakage Risk",
            "",
            f"- level: `{leakage_risk['assessment']}`",
        ]
    )
    for note in leakage_risk["notes"]:
        lines.append(f"- {note}")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
