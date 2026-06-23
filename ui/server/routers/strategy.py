"""策略路由：调仓解释视图。"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query

from .. import app
from ..helpers import calc_trade_fee, fee_settings_from_raw
from ..position_service import enrich_positions, load_positions_file

router = APIRouter(tags=["strategy"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_previous_holdings(enriched_positions: dict) -> dict:
    return {item["instrument"]: float(item.get("weight", 0.0)) / 100.0 for item in enriched_positions.get("positions", [])}


def _pick_default_alias():
    svc = getattr(app, "model_service", None)
    if svc is None:
        return None
    models = svc.list_models()
    return models[0]["alias"] if models else None


def _error_payload(msg: str):
    return {"error": msg, "stocks": [], "date": None}


def _load_prediction(alias: str | None, date: str | None, top_k: int):
    svc = getattr(app, "model_service", None)
    if svc is None:
        return _error_payload("模型服务未初始化"), None
    chosen_alias = alias or _pick_default_alias()
    if not chosen_alias:
        return _error_payload("无可用模型，请先在模型选股中训练或加载模型"), None
    if date:
        data = svc.get_predictions(chosen_alias, date=date, top_k=top_k)
    else:
        data = svc.get_predictions(chosen_alias, top_k=top_k)
    return data, chosen_alias


def _weights_from_scores(selected_stocks: list, weight_mode: str) -> dict:
    if not selected_stocks:
        return {}
    if weight_mode == "equal":
        w = 1.0 / len(selected_stocks)
        return {item["instrument"]: w for item in selected_stocks}
    scores = [float(item.get("score", 0.0)) for item in selected_stocks]
    min_score = min(scores)
    shifted = [score - min_score + 1e-12 for score in scores]
    total = sum(shifted)
    if total <= 0:
        w = 1.0 / len(selected_stocks)
        return {item["instrument"]: w for item in selected_stocks}
    return {item["instrument"]: shifted[idx] / total for idx, item in enumerate(selected_stocks)}


def _strip_prefix(code: str) -> str:
    """Strip SH/SZ prefix from instrument codes to match holdings format."""
    for prefix in ("SH", "SZ", "sh", "sz"):
        if code.startswith(prefix):
            return code[len(prefix):]
    return code


def _normalize_prediction_stocks(prediction: dict) -> list:
    """Return prediction stocks with SH/SZ prefix stripped from instrument codes."""
    out = []
    for item in prediction.get("stocks", []):
        item = dict(item)
        item["instrument"] = _strip_prefix(item["instrument"])
        out.append(item)
    return out


def _round_lot_shares(target_value: float, price: float, board_lot: int = 100) -> int:
    if price <= 0 or target_value <= 0:
        return 0
    raw_shares = int(target_value / price)
    return max(0, (raw_shares // board_lot) * board_lot)


def _next_rebalance_date(today_str: str, mode: str) -> str:
    """Calculate next rebalance date based on mode."""
    if mode != "weekly":
        return today_str
    today = datetime.strptime(today_str, "%Y-%m-%d")
    days_ahead = (7 - today.weekday()) % 7  # Monday=0, days until next Mon
    if days_ahead == 0:
        days_ahead = 7  # today is Monday → next Monday
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


STRATEGY_REFERENCES = {
    "buffered_weight": {
        "label": "BufferedWeightStrategy",
        "desc": "缓冲保留机制，减少不必要换手",
        "proven_params": "tk=55, hk=85, equal, 每周",
        "proven_ir": 3.023,
        "proven_annret": "38.79%",
        "source": "Transcendence SOTA leaderboard",
    },
    "topk_dropout": {
        "label": "TopkDropoutStrategy",
        "desc": "经典 Top-K 轮出策略，每日换手固定数量",
        "proven_params": "tk=45, nd=4, 每日",
        "proven_ir": 2.800,
        "proven_annret": "24.47%",
        "source": "Transcendence R7406 基准",
    },
}


def _strategy_reference(strategy_type: str, top_k: int) -> dict:
    ref = STRATEGY_REFERENCES.get(strategy_type, {})
    return {
        "strategyType": strategy_type,
        "strategyLabel": ref.get("label", ""),
        "strategyDesc": ref.get("desc", ""),
        "provenParams": ref.get("proven_params", ""),
        "provenIR": ref.get("proven_ir"),
        "provenAnnRet": ref.get("proven_annret"),
        "source": ref.get("source", ""),
    }


def _build_trade_rows(
    all_instruments: list,
    position_map: dict,
    target_weights: dict,
    stock_map: dict,
    total_assets: float,
    previous_holdings: dict,
    raw_positions: dict,
) -> tuple:
    """Shared trade row builder for all strategies."""
    target_positions = []
    trades = []
    total_buy = 0.0
    total_sell = 0.0
    estimated_fees = 0.0

    for instrument in sorted(set(position_map.keys()) | set(target_weights.keys())):
        current = position_map.get(instrument, {})
        current_shares = int(current.get("shares") or 0)
        current_weight = float(current.get("weight") or 0.0) / 100.0
        current_price = float(current.get("currentPrice") or 0.0)
        score_row = stock_map.get(instrument, {})
        target_weight = float(target_weights.get(instrument, 0.0))
        trade_price = current_price or 10.0
        target_value = total_assets * target_weight
        target_shares = _round_lot_shares(target_value, trade_price)
        # Buffer retention: if this stock is deliberately kept by the strategy
        # (in previous_holdings AND in target_weights), ensure target shares
        # never drop below current — the buffer exists to protect these positions.
        buffer_kept = instrument in previous_holdings and instrument in target_weights
        if buffer_kept and target_shares == 0 and current_shares > 0:
            target_shares = current_shares
            target_value = current_shares * trade_price
        delta_shares = target_shares - current_shares
        trade_side = "hold"
        if delta_shares > 0:
            trade_side = "buy"
        elif delta_shares < 0:
            trade_side = "sell"
        trade_amount = abs(delta_shares) * trade_price
        fee_breakdown = calc_trade_fee(instrument, trade_amount, trade_side, fee_settings_from_raw(raw_positions))
        estimated_fees += fee_breakdown["total"]
        if trade_side == "buy":
            total_buy += trade_amount
        elif trade_side == "sell":
            total_sell += trade_amount

        is_new = instrument not in previous_holdings and instrument in target_weights

        if trade_side == "buy":
            if is_new:
                reason = "模型排名靠前，新入选目标组合"
            else:
                reason = "缓冲保留，增持至目标权重"
        elif trade_side == "sell":
            if instrument not in target_weights:
                reason = "排名跌出目标池，卖出清仓"
            elif target_shares == 0 and current_shares > 0:
                reason = "模型选入但可买不足1手，清仓"
            else:
                reason = "权重超配，减持至目标比例"
        else:
            reason = "权重已达标，无需调整"

        row = {
            "instrument": instrument,
            "name": current.get("name") or score_row.get("name") or instrument,
            "score": score_row.get("score"),
            "currentShares": current_shares,
            "targetShares": target_shares,
            "deltaShares": delta_shares,
            "currentWeight": round(current_weight * 100, 2),
            "targetWeight": round(target_weight * 100, 2),
            "currentPrice": round(trade_price, 4),
            "targetValue": round(target_value, 2),
            "tradeAmount": round(trade_amount, 2),
            "side": trade_side,
            "bufferKept": buffer_kept,
            "isNew": is_new,
            "reason": reason,
            "fee": fee_breakdown,
        }
        target_positions.append(row)
        if trade_side != "hold":
            trades.append(row)

    target_positions.sort(key=lambda item: item["targetWeight"], reverse=True)
    trades.sort(key=lambda item: (item["side"] != "sell", -abs(item["tradeAmount"])))
    return target_positions, trades, total_buy, total_sell, estimated_fees


def _build_summary(target_positions: list, trades: list, total_buy: float, total_sell: float, estimated_fees: float, total_assets: float, cash: float) -> dict:
    turnover = (total_buy + total_sell) / total_assets if total_assets > 0 else 0.0
    return {
        "keptCount": sum(1 for item in target_positions if item["bufferKept"]),
        "newCount": sum(1 for item in target_positions if item["isNew"]),
        "sellCount": sum(1 for item in trades if item["side"] == "sell"),
        "buyCount": sum(1 for item in trades if item["side"] == "buy"),
        "estimatedBuyAmount": round(total_buy, 2),
        "estimatedSellAmount": round(total_sell, 2),
        "estimatedFees": round(estimated_fees, 2),
        "turnoverPct": round(turnover * 100, 2),
        "cashAfterTrades": round(float(cash) + total_sell - total_buy - estimated_fees, 2),
    }


def _auto_cap_topk(total_assets: float, risk_degree: float, position_map: dict, requested_topk: int, board_lot: int = 100) -> int:
    """Cap topK to what the portfolio can actually afford (each position >= 1 board lot).

    Uses a conservative price estimate (75th percentile of current holdings, min 10 yuan)
    so the cap works even for stocks pricier than the median.
    """
    if total_assets <= 0 or risk_degree <= 0:
        return requested_topk
    prices = [float(p.get("currentPrice", 0)) for p in position_map.values() if float(p.get("currentPrice", 0)) > 0]
    if not prices:
        return requested_topk  # no positions yet, trust the user's request
    prices.sort()
    # Use 75th percentile with a floor of 10 yuan (typical A-share)
    idx = min(len(prices) * 3 // 4, len(prices) - 1)
    ref_price = max(prices[idx], 10.0)
    min_per_position = ref_price * board_lot
    feasible = int((total_assets * risk_degree) / min_per_position)
    return max(1, min(requested_topk, feasible))


def _build_buffered_preview(params: dict) -> dict:
    """BufferedWeightStrategy: project-proven top-55/hold-85/equal/weekly."""
    from quant_master.contrib.strategy.topk_cost_aware import select_buffered_topk

    alias = params.get("alias")
    date = params.get("date")
    requested_topk = max(1, int(params.get("top_k", 55)))
    requested_hold_topk = max(requested_topk, int(params.get("hold_topk") or max(requested_topk, 85)))
    weight_mode = str(params.get("weight_mode", "equal")).lower()
    risk_degree = float(params.get("risk_degree", 0.95))
    rebalance_mode = str(params.get("rebalance_mode", "weekly")).lower()

    prediction, resolved_alias = _load_prediction(alias, date, requested_hold_topk)
    if "error" in prediction:
        return prediction

    raw_positions = load_positions_file()
    enriched = enrich_positions(raw_positions, data=app.data)
    previous_holdings = _normalize_previous_holdings(enriched)

    # Cap topK to what the portfolio can actually afford
    total_assets = float(enriched.get("totalAssets") or 0.0)
    position_map = {item["instrument"]: item for item in enriched.get("positions", [])}
    top_k = _auto_cap_topk(total_assets, risk_degree, position_map, requested_topk)
    hold_topk = max(top_k, requested_hold_topk)

    # Normalize SH/SZ prefix to match holdings
    prediction["stocks"] = _normalize_prediction_stocks(prediction)
    stock_map = {item["instrument"]: item for item in prediction["stocks"]}

    selected_index = select_buffered_topk(
        scores=pd.Series({item["instrument"]: float(item["score"]) for item in prediction["stocks"]}),
        topk=top_k,
        previous_holdings=previous_holdings,
        rank_buffer=max(0, hold_topk - top_k),
    )

    selected_stocks = [stock_map[code] for code in selected_index if code in stock_map]
    target_weights_raw = _weights_from_scores(selected_stocks, weight_mode=weight_mode)
    target_weights = {k: v * risk_degree for k, v in target_weights_raw.items()}

    target_positions, trades, total_buy, total_sell, estimated_fees = _build_trade_rows(
        list(set(position_map.keys()) | set(target_weights.keys())),
        position_map, target_weights, stock_map, total_assets, previous_holdings, raw_positions,
    )

    # Only mark stocks that actually end up with target shares as "selected"
    actual_target = {r["instrument"] for r in target_positions if r["targetShares"] > 0}
    selected_stocks = [s for s in selected_stocks if s["instrument"] in actual_target]

    trade_date = prediction.get("date") or date

    return {
        "alias": resolved_alias,
        "tradeDate": trade_date,
        "strategyRef": _strategy_reference("buffered_weight", top_k),
        "rebalanceMode": rebalance_mode,
        "nextRebalanceDate": _next_rebalance_date(trade_date, rebalance_mode) if rebalance_mode == "weekly" else trade_date,
        "config": {
            "topK": top_k,
            "holdTopk": hold_topk,
            "rankBuffer": max(0, hold_topk - top_k),
            "weightMode": weight_mode,
            "riskDegree": risk_degree,
            "requestedTopK": requested_topk if requested_topk != top_k else None,
        },
        "holdings": enriched,
        "prediction": prediction,
        "selected": selected_stocks,
        "targetPositions": target_positions,
        "trades": trades,
        "summary": _build_summary(target_positions, trades, total_buy, total_sell, estimated_fees, total_assets, enriched.get("cash") or 0.0),
        "explanation": {
            "title": f"BufferedWeightStrategy 调仓预览（tk={top_k}, hk={hold_topk}）",
            "why": "先保留当前已持有且排名仍落在 topk + buffer 内的股票，再用最新模型高分股票补齐 topk，减少不必要换手。",
            "how": [
                "读取当前持仓权重作为 previous holdings",
                "用模型选股分数生成当日候选排名",
                "保留仍在 hold_topk 范围内的旧仓位",
                "再补足 topk，并按 equal / score 模式分配目标权重",
                "最后对比当前持仓与目标持仓，生成买卖建议",
            ],
            "note": (f"账户总资产 ¥{total_assets:,.0f}，实际可持有约 {top_k} 只（每只≥1手），已从请求的 {requested_topk} 只自动下调。"
                     if requested_topk != top_k else None),
        },
    }


def _build_dropout_preview(params: dict) -> dict:
    """TopkDropoutStrategy: baseline top-45/n-drop-4/daily."""
    alias = params.get("alias")
    date = params.get("date")
    top_k = max(1, int(params.get("top_k", 45)))
    hold_topk = None
    n_drop = int(params.get("n_drop", 4))
    weight_mode = "equal"
    risk_degree = float(params.get("risk_degree", 0.95))
    rebalance_mode = "daily"

    prediction, resolved_alias = _load_prediction(alias, date, top_k + n_drop)
    if "error" in prediction:
        return prediction

    raw_positions = load_positions_file()
    enriched = enrich_positions(raw_positions, data=app.data)

    # Normalize SH/SZ prefix to match holdings
    prediction["stocks"] = _normalize_prediction_stocks(prediction)
    stock_map = {item["instrument"]: item for item in prediction["stocks"]}

    # Sort by score, take top-k
    ranked = sorted(prediction["stocks"], key=lambda x: float(x.get("score", 0)), reverse=True)
    selected_stocks = ranked[:top_k]
    target_weight = risk_degree / top_k
    target_weights = {item["instrument"]: target_weight for item in selected_stocks}

    total_assets = float(enriched.get("totalAssets") or 0.0)
    position_map = {item["instrument"]: item for item in enriched.get("positions", [])}
    previous_holdings = _normalize_previous_holdings(enriched)
    target_positions, trades, total_buy, total_sell, estimated_fees = _build_trade_rows(
        list(set(position_map.keys()) | set(target_weights.keys())),
        position_map, target_weights, stock_map, total_assets, previous_holdings, raw_positions,
    )

    trade_date = prediction.get("date") or date

    return {
        "alias": resolved_alias,
        "tradeDate": trade_date,
        "strategyRef": _strategy_reference("topk_dropout", top_k),
        "rebalanceMode": rebalance_mode,
        "nextRebalanceDate": trade_date,
        "config": {
            "topK": top_k,
            "nDrop": n_drop,
            "riskDegree": risk_degree,
            "weightMode": "equal",
        },
        "holdings": enriched,
        "prediction": prediction,
        "selected": selected_stocks,
        "targetPositions": target_positions,
        "trades": trades,
        "summary": _build_summary(target_positions, trades, total_buy, total_sell, estimated_fees, total_assets, enriched.get("cash") or 0.0),
        "explanation": {
            "title": f"TopkDropoutStrategy 调仓预览（tk={top_k}, nd={n_drop}）",
            "why": "每日按模型分数排名，选 top-k 只股票等权持有，非入选股票全部卖出。",
            "how": [
                "读取模型对全市场股票的当日打分",
                "按分数从高到低排序，取前 top_k 只",
                f"前次持仓中未进入前 {top_k} 的股票全部卖出",
                "目标股票等权配置（每只 1/{top_k}）",
                "匹配当前持仓，生成买卖建议",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/strategy-buffered-rebalance")
def buffered_rebalance_preview(
    alias: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    top_k: int = Query(55),
    hold_topk: Optional[int] = Query(85),
    weight_mode: str = Query("equal"),
    risk_degree: float = Query(0.95),
    strategy_type: str = Query("buffered_weight"),
    rebalance_mode: str = Query("weekly"),
    n_drop: int = Query(4),
):
    params = {
        "alias": alias,
        "date": date,
        "top_k": top_k,
        "hold_topk": hold_topk,
        "weight_mode": weight_mode,
        "risk_degree": risk_degree,
        "strategy_type": strategy_type,
        "rebalance_mode": rebalance_mode,
        "n_drop": n_drop,
    }
    strategy_type = str(strategy_type).lower()
    if strategy_type == "buffered_weight":
        return _build_buffered_preview(params)
    elif strategy_type == "topk_dropout":
        return _build_dropout_preview(params)
    else:
        return {"error": f"不支持策略类型: {strategy_type}"}
