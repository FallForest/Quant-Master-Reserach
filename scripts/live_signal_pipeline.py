#!/usr/bin/env python
"""Live signal pipeline for regime-switching model.

Generates daily stock picks using the causal regime-switching meta-strategy
and submits orders to a configured broker (default: paper).

Usage:
    python scripts/live_signal_pipeline.py --date 2026-05-30
    python scripts/live_signal_pipeline.py --date 2026-05-30 --broker paper --dry-run
    python scripts/live_signal_pipeline.py --date 2026-05-30 --signal-only
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("live_signal_pipeline")

# --- Constants matching backtest config ---
DEFAULT_TOPK = 45
DEFAULT_N_DROP = 4
DEFAULT_OPEN_COST = 0.0001
DEFAULT_CLOSE_COST = 0.0006
DEFAULT_INIT_CASH = 1000000.0
DEFAULT_BENCHMARK = "SH000300"
WARMUP_DAYS = 25

REGIME_WEIGHTS = {
    "seed57_de": 0.7,
    "seed42_liquidity": 0.2,
    "seed7_de": 0.1,
}

SCORE_COEFF = {
    "ret5": 0.8,
    "ret20": 0.7,
    "vol20": -0.55,
    "turn5": -0.14,
    "disp_ratio": 0.06,
}

HYSTERESIS_THRESHOLD = 0.02


def _ensure_quant_master(provider_uri: str = None, region: str = "cn"):
    """Initialize quant_master with data directory."""
    import quant_master
    if provider_uri is None:
        provider_uri = str(Path.home() / ".quant_master" / "quant_master_data" / "tdx_cn_data")
    quant_master.init(provider_uri=provider_uri, region=region)
    return provider_uri


# ============================================================
# Signal generation
# ============================================================

def generate_model_signal(
    config_path: str,
    pred_date: pd.Timestamp,
    provider_uri: str = None,
) -> pd.Series:
    """Run model prediction for a single date.

    Loads a trained model from its workflow config and generates
    cross-sectional predictions for all active stocks.

    Returns: pd.Series indexed by instrument, values = prediction scores.
    """
    from quant_master.model.trainer import task_train
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe", pure=True)
    with open(config_path, encoding="utf-8") as f:
        wf_config = yaml.load(f)

    task = wf_config.get("task", {})
    dataset_cfg = task.get("dataset", {})
    ds_kwargs = dataset_cfg.get("kwargs", {})

    pred_date_str = str(pd.Timestamp(pred_date).date())
    segments = ds_kwargs.get("segments", {})
    segments["test"] = [pred_date_str, pred_date_str]
    ds_kwargs["segments"] = segments

    task["record"] = [
        {
            "class": "SignalRecord",
            "module_path": "quant_master.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        },
    ]
    wf_config["task"] = task

    if provider_uri:
        init_cfg = wf_config.setdefault("quant_master_init", {})
        init_cfg["provider_uri"] = provider_uri
        init_cfg.setdefault("region", "cn")

    recorder = task_train(task, experiment_name=f"live_pred_{pred_date_str}")
    pred = recorder.load_object("pred.pkl")

    if isinstance(pred, pd.DataFrame):
        if "score" in pred.columns:
            pred = pred["score"]
        else:
            pred = pred.iloc[:, 0]

    return pred


def generate_weighted_ensemble_signal(
    pred_date: pd.Timestamp,
    model_configs: Dict[str, Dict[str, Any]],
    provider_uri: str = None,
) -> pd.DataFrame:
    """Generate weighted ensemble prediction from multiple models.

    model_configs: {
        "seed57_de": {"config_path": "...", "weight": 0.7},
        "seed42_liquidity": {"config_path": "...", "weight": 0.2},
    }

    Returns: pd.DataFrame with columns [score] indexed by instrument.
    """
    series_list = []
    weights = []
    names = []

    for name, cfg in model_configs.items():
        pred = generate_model_signal(cfg["config_path"], pred_date, provider_uri)
        rank_s = pred.groupby(pred.index).rank(method="average", pct=True)
        rank_s.name = name
        series_list.append(rank_s)
        weights.append(cfg["weight"])
        names.append(name)

    panel = pd.concat(series_list, axis=1)
    w = pd.Series(weights, index=names, dtype=float)
    weighted = panel.mul(w, axis=1)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = weighted.fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


# ============================================================
# Regime switching
# ============================================================

def calc_strategy_excess_metrics(
    excess_series: pd.Series,
    report_df: pd.DataFrame,
    date_idx: int,
) -> Dict[str, float]:
    """Calculate regime features using only data up to date_idx-1 (no leakage)."""
    hist5 = excess_series.iloc[max(0, date_idx - 5):date_idx]
    hist20 = excess_series.iloc[max(0, date_idx - 20):date_idx]

    ret5 = float(hist5.mean() * 252.0) if len(hist5) > 0 else 0.0
    ret20 = float(hist20.mean() * 252.0) if len(hist20) > 0 else 0.0
    vol20 = float(hist20.std(ddof=0) * np.sqrt(252.0)) if len(hist20) > 1 else 0.0
    turn5 = float(report_df["turnover"].iloc[max(0, date_idx - 5):date_idx].mean()) if date_idx > 0 else 0.0

    return {"ret5": ret5, "ret20": ret20, "vol20": vol20, "turn5": turn5}


def calc_regime_score(features: Dict[str, float], disp_ratio: float = 1.0) -> float:
    """Calculate regime score from features."""
    return (
        SCORE_COEFF["ret5"] * features["ret5"]
        + SCORE_COEFF["ret20"] * features["ret20"]
        + SCORE_COEFF["vol20"] * features["vol20"]
        + SCORE_COEFF["turn5"] * features["turn5"]
        + SCORE_COEFF["disp_ratio"] * (disp_ratio - 1.0)
    )


def select_regime_strategy(
    date: pd.Timestamp,
    strategy_scores: Dict[str, float],
    bench_vol20: float,
    bench_vol20_q75: float,
    base_disp: float,
    base_disp_q35: float,
    prev_strategy: str,
    prev_score: float,
) -> Tuple[str, str]:
    """Select strategy for today using causal regime switch rules.

    Returns: (selected_strategy_id, reason)
    """
    CONSERVATIVE_STRATEGY = "fixed_topk40_nd2_daily"

    high_vol = np.isfinite(bench_vol20) and np.isfinite(bench_vol20_q75) and bench_vol20 > bench_vol20_q75
    low_disp = np.isfinite(base_disp) and np.isfinite(base_disp_q35) and base_disp < base_disp_q35

    if high_vol or low_disp:
        candidate = CONSERVATIVE_STRATEGY
    else:
        candidate = max(strategy_scores, key=strategy_scores.get)

    if (
        candidate != prev_strategy
        and strategy_scores.get(candidate, 0) < strategy_scores.get(prev_strategy, 0) + HYSTERESIS_THRESHOLD
    ):
        return prev_strategy, "hysteresis_hold"

    return candidate, "rule_select"


# ============================================================
# Signal to orders
# ============================================================

def signals_to_orders(
    target_stocks: List[str],
    current_positions: Dict[str, int],
    available_cash: float,
    stock_prices: Dict[str, float],
    total_assets: float,
) -> List[Dict[str, Any]]:
    """Convert target stock list to buy/sell orders.

    Equal-weight target: each stock targets total_assets / len(target_stocks).
    Sells first to free cash, then buys.
    Round to 100-share lots.
    """
    orders = []
    n = len(target_stocks)
    if n == 0:
        for stock_id, shares in current_positions.items():
            if shares > 0 and stock_id in stock_prices:
                orders.append({
                    "stock_id": stock_id,
                    "direction": "sell",
                    "price": stock_prices[stock_id],
                    "amount": int(shares),
                })
        return orders

    target_weight = 1.0 / n
    target_value = {s: total_assets * target_weight for s in target_stocks}

    # Sell first
    for stock_id, shares in list(current_positions.items()):
        if shares <= 0:
            continue
        if stock_id not in target_stocks:
            if stock_id in stock_prices:
                orders.append({
                    "stock_id": stock_id,
                    "direction": "sell",
                    "price": stock_prices[stock_id],
                    "amount": int(shares),
                })
        else:
            price = stock_prices.get(stock_id, 0)
            if price <= 0:
                continue
            target_shares = int(target_value[stock_id] / price / 100) * 100
            if shares > target_shares:
                orders.append({
                    "stock_id": stock_id,
                    "direction": "sell",
                    "price": price,
                    "amount": int(shares - target_shares),
                })

    # Buy
    for stock_id in target_stocks:
        price = stock_prices.get(stock_id, 0)
        if price <= 0:
            continue
        target_shares = int(target_value[stock_id] / price / 100) * 100
        current_shares = current_positions.get(stock_id, 0)
        buy_amount = target_shares - current_shares
        if buy_amount > 0:
            orders.append({
                "stock_id": stock_id,
                "direction": "buy",
                "price": price,
                "amount": int(buy_amount),
            })

    return orders


def get_stock_prices(instruments: List[str], date: pd.Timestamp) -> Dict[str, float]:
    """Get latest close prices for instruments on a given date."""
    from quant_master.data import D
    try:
        df = D.features(instruments, ["$close"], start_time=date, end_time=date)
        if df.empty:
            return {}
        prices = {}
        for inst in instruments:
            try:
                p = df.loc[inst, "$close"]
                if isinstance(p, pd.Series):
                    p = p.iloc[0]
                prices[inst] = float(p)
            except (KeyError, IndexError):
                pass
        return prices
    except Exception as e:
        logger.warning(f"Failed to get prices: {e}")
        return {}


# ============================================================
# Persistence
# ============================================================

LIVE_DATA_DIR = ROOT / "live_data"


def _ensure_live_dirs():
    for subdir in ["signals", "orders", "positions", "regime"]:
        (LIVE_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def save_signal(date: pd.Timestamp, signal_data: Dict[str, Any]):
    _ensure_live_dirs()
    date_str = str(pd.Timestamp(date).date())
    path = LIVE_DATA_DIR / "signals" / f"{date_str}.json"
    path.write_text(json.dumps(signal_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Signal saved to {path}")


def save_orders(date: pd.Timestamp, orders: List[Dict[str, Any]]):
    _ensure_live_dirs()
    date_str = str(pd.Timestamp(date).date())
    path = LIVE_DATA_DIR / "orders" / f"{date_str}.json"
    path.write_text(json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Orders saved to {path}")


def save_positions(positions: Dict[str, Any]):
    _ensure_live_dirs()
    path = LIVE_DATA_DIR / "positions" / "latest.json"
    path.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")


def save_regime(date: pd.Timestamp, regime_data: Dict[str, Any]):
    _ensure_live_dirs()
    date_str = str(pd.Timestamp(date).date())
    path = LIVE_DATA_DIR / "regime" / f"{date_str}.json"
    path.write_text(json.dumps(regime_data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_signal(date: pd.Timestamp) -> Optional[Dict[str, Any]]:
    date_str = str(pd.Timestamp(date).date())
    path = LIVE_DATA_DIR / "signals" / f"{date_str}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_positions() -> Dict[str, Any]:
    path = LIVE_DATA_DIR / "positions" / "latest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cash": DEFAULT_INIT_CASH, "positions": {}}


def load_order_history(limit: int = 30) -> List[Dict[str, Any]]:
    orders_dir = LIVE_DATA_DIR / "orders"
    if not orders_dir.exists():
        return []
    files = sorted(orders_dir.glob("*.json"), reverse=True)[:limit]
    history = []
    for f in files:
        date_str = f.stem
        day_orders = json.loads(f.read_text(encoding="utf-8"))
        for o in day_orders:
            o["date"] = date_str
        history.extend(day_orders)
    return history


# ============================================================
# Main pipeline
# ============================================================

def run_daily(
    trade_date: str,
    broker_kind: str = "paper",
    dry_run: bool = False,
    signal_only: bool = False,
    provider_uri: str = None,
) -> Dict[str, Any]:
    """End-to-end daily pipeline: signal 鈫?regime switch 鈫?order execution.

    Returns summary dict with regime info, selected stocks, orders, execution.
    """
    from quant_master.contrib.broker import BrokerOrderDir, LiveOrderExecutor, LiveOrderRequest, create_broker

    trade_date = pd.Timestamp(trade_date)
    date_str = str(trade_date.date())
    logger.info(f"=== Live pipeline for {date_str} ===")

    # 1. Initialize quant_master
    provider_uri = _ensure_quant_master(provider_uri)

    # 2. Load current positions
    pos_data = load_positions()
    current_positions = {k: v["shares"] for k, v in pos_data.get("positions", {}).items()}
    available_cash = pos_data.get("cash", DEFAULT_INIT_CASH)
    total_assets = pos_data.get("total_assets", DEFAULT_INIT_CASH)

    # 3. Generate signals
    logger.info("Generating model predictions...")

    model_configs = {
        "seed57_de": {
            "config_path": str(ROOT / "examples/benchmarks/Transcendence/configs/workflows/regime_horizon/variants/workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_seed57_Alpha158_2026_csi300.yaml"),
            "weight": 0.7,
        },
        "seed42_liquidity": {
            "config_path": str(ROOT / "examples/benchmarks/Transcendence/configs/workflows/regime_horizon/variants/workflow_config_regime_horizon_alpha158_liquidity_state_Alpha158_2026_csi300.yaml"),
            "weight": 0.2,
        },
    }

    base_pred = generate_weighted_ensemble_signal(trade_date, model_configs, provider_uri)

    # 4. Regime switch
    selected_strategy = "baseline_7406_default_tk45_nd4_daily"
    strategy_topk = DEFAULT_TOPK
    strategy_n_drop = DEFAULT_N_DROP
    regime_reason = "default_baseline"

    regime_history_path = LIVE_DATA_DIR / "regime" / "history.json"
    if regime_history_path.exists():
        try:
            regime_history = json.loads(regime_history_path.read_text(encoding="utf-8"))
            prev_strategy = regime_history.get("last_strategy", selected_strategy)
            prev_score = regime_history.get("last_score", 0.0)

            from quant_master.data import D
            bench_data = D.features(
                ["SH000300"],
                ["$close/Ref($close,1)-1"],
                start_time=trade_date - pd.Timedelta(days=40),
                end_time=trade_date - pd.Timedelta(days=1),
            )
            if not bench_data.empty:
                bench_ret = bench_data.iloc[:, 0]
                bench_vol20 = float(bench_ret.tail(20).std() * np.sqrt(252))
                bench_vol20_q75 = (
                    float(bench_ret.expanding(20).std().quantile(0.75) * np.sqrt(252))
                    if len(bench_ret) > 20
                    else float("nan")
                )

                strategy_scores = {
                    "baseline_7406_default_tk45_nd4_daily": 0.0,
                    "fixed_topk40_nd2_daily": 0.0,
                    "fixed_topk50_nd5_daily": 0.0,
                }
                for sid in strategy_scores:
                    features = {"ret5": 0.0, "ret20": 0.0, "vol20": bench_vol20, "turn5": 0.0}
                    strategy_scores[sid] = calc_regime_score(features)

                base_disp = float(base_pred["score"].std()) if len(base_pred) > 0 else 0.0
                base_disp_q35 = base_disp * 0.9

                selected_strategy, regime_reason = select_regime_strategy(
                    trade_date, strategy_scores, bench_vol20, bench_vol20_q75,
                    base_disp, base_disp_q35, prev_strategy, prev_score,
                )

                if "topk40" in selected_strategy:
                    strategy_topk = 40
                    strategy_n_drop = 2
                elif "topk50" in selected_strategy:
                    strategy_topk = 50
                    strategy_n_drop = 5

            regime_history["last_strategy"] = selected_strategy
            regime_history["last_score"] = strategy_scores.get(selected_strategy, 0.0)
            regime_history["last_date"] = date_str
            regime_history_path.write_text(
                json.dumps(regime_history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Regime switch failed, using default: {e}")
    else:
        regime_history_path.parent.mkdir(parents=True, exist_ok=True)
        regime_history = {"last_strategy": selected_strategy, "last_score": 0.0, "last_date": date_str}
        regime_history_path.write_text(
            json.dumps(regime_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(f"Regime: {selected_strategy} (reason: {regime_reason}, topk={strategy_topk}, n_drop={strategy_n_drop})")

    # 5. Select top-k stocks
    ranked = base_pred.sort_values("score", ascending=False)
    target_stocks = ranked.index[:strategy_topk].tolist()
    if target_stocks and isinstance(target_stocks[0], tuple):
        target_stocks = [s[1] if isinstance(s, tuple) else s for s in target_stocks]

    logger.info(f"Selected {len(target_stocks)} stocks: {target_stocks[:5]}...")

    # 6. Save signal
    signal_data = {
        "date": date_str,
        "strategy": selected_strategy,
        "regime_reason": regime_reason,
        "topk": strategy_topk,
        "n_drop": strategy_n_drop,
        "stocks": target_stocks,
        "scores": {},
    }
    for s in target_stocks:
        try:
            val = ranked.loc[s, "score"]
            signal_data["scores"][str(s)] = float(val.iloc[0] if isinstance(val, pd.Series) else val)
        except Exception:
            pass

    save_signal(trade_date, signal_data)
    save_regime(trade_date, {
        "selected_strategy": selected_strategy,
        "reason": regime_reason,
        "topk": strategy_topk,
        "n_drop": strategy_n_drop,
    })

    if signal_only:
        logger.info("Signal-only mode, skipping order execution.")
        return signal_data

    # 7. Get prices and generate orders
    stock_prices = get_stock_prices(target_stocks, trade_date)
    orders = signals_to_orders(target_stocks, current_positions, available_cash, stock_prices, total_assets)

    logger.info(f"Generated {len(orders)} orders")
    save_orders(trade_date, orders)

    if not orders:
        logger.info("No orders to execute.")
        return {**signal_data, "orders": [], "execution": []}

    # 8. Execute via broker
    broker = create_broker(broker_kind)
    if hasattr(broker, "connect"):
        broker.connect()

    executor = LiveOrderExecutor(broker)
    order_requests = [
        LiveOrderRequest(
            stock_id=o["stock_id"],
            direction=BrokerOrderDir.BUY if o["direction"] == "buy" else BrokerOrderDir.SELL,
            price=o["price"],
            amount=o["amount"],
        )
        for o in orders
    ]

    results = executor.submit_many(order_requests, dry_run=dry_run)
    execution = []
    for r in results:
        execution.append({
            "stock_id": r.request.stock_id,
            "direction": r.request.direction.name,
            "price": r.request.price,
            "amount": r.request.amount,
            "accepted": r.accepted,
            "rejection_reason": r.rejection_reason,
            "order_id": None if r.broker_order is None else r.broker_order.order_id,
        })
        if r.accepted:
            logger.info(f"  OK  {r.request.direction.name} {r.request.stock_id} x{r.request.amount} @ {r.request.price}")
        else:
            logger.warning(f"  FAIL {r.request.direction.name} {r.request.stock_id}: {r.rejection_reason}")

    # 9. Update positions
    for r in results:
        if not r.accepted:
            continue
        sid = r.request.stock_id
        if r.request.direction == BrokerOrderDir.BUY:
            current_positions[sid] = current_positions.get(sid, 0) + r.request.amount
            available_cash -= r.request.price * r.request.amount
        else:
            current_positions[sid] = current_positions.get(sid, 0) - r.request.amount
            available_cash += r.request.price * r.request.amount
            if current_positions[sid] <= 0:
                del current_positions[sid]

    pos_snapshot = {
        "date": date_str,
        "cash": round(available_cash, 2),
        "total_assets": round(total_assets, 2),
        "positions": {
            sid: {"shares": shares, "price": stock_prices.get(sid, 0)}
            for sid, shares in current_positions.items()
            if shares > 0
        },
    }
    save_positions(pos_snapshot)

    result = {
        **signal_data,
        "orders": orders,
        "execution": execution,
        "accepted": sum(1 for e in execution if e["accepted"]),
        "rejected": sum(1 for e in execution if not e["accepted"]),
    }
    logger.info(f"Pipeline complete: {result['accepted']} accepted, {result['rejected']} rejected")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live signal pipeline with regime switching")
    p.add_argument("--date", required=True, help="Trade date (YYYY-MM-DD)")
    p.add_argument("--broker", default="paper", help="Broker type: paper, tcdll, tdx")
    p.add_argument("--dry-run", action="store_true", help="Generate signals/orders but don't execute")
    p.add_argument("--signal-only", action="store_true", help="Only generate signals, skip order creation")
    p.add_argument("--provider-uri", default=None, help="Quant Master data directory")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    result = run_daily(
        trade_date=args.date,
        broker_kind=args.broker,
        dry_run=args.dry_run,
        signal_only=args.signal_only,
        provider_uri=args.provider_uri,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

