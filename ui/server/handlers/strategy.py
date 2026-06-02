"""策略研究与调仓解释视图。"""

from __future__ import annotations

import pandas as pd

from quant_master.contrib.strategy.topk_cost_aware import select_buffered_topk

from .. import app
from . import position


def _normalize_previous_holdings(enriched_positions):
    return {item["instrument"]: float(item.get("weight", 0.0)) / 100.0 for item in enriched_positions.get("positions", [])}


def _pick_default_alias():
    svc = getattr(app, "model_service", None)
    if svc is None:
        return None
    models = svc.list_models()
    return models[0]["alias"] if models else None


def _sample_prediction(date: str, top_k: int):
    stocks = [
        {"rank": 1, "instrument": "SH600001", "name": "Test Stock A", "score": 0.9123},
        {"rank": 2, "instrument": "SH600002", "name": "Test Stock B", "score": 0.8751},
        {"rank": 3, "instrument": "SH600003", "name": "Test Stock C", "score": 0.8012},
    ][: max(1, top_k)]
    return {
        "date": date,
        "topK": len(stocks),
        "stocks": stocks,
        "totalStocks": len(stocks),
        "scoreStats": {"mean": 0.8629, "std": 0.056, "min": 0.8012, "max": 0.9123},
        "source": "sample",
    }


def _load_prediction(alias: str | None, date: str | None, top_k: int):
    svc = getattr(app, "model_service", None)
    if svc is None:
        return _sample_prediction(date or "2025-02-20", top_k), None
    chosen_alias = alias or _pick_default_alias()
    if not chosen_alias:
        return _sample_prediction(date or "2025-02-20", top_k), None
    if date:
        data = svc.get_predictions(chosen_alias, date=date, top_k=top_k)
    else:
        data = svc.get_predictions(chosen_alias, top_k=top_k)
    return data, chosen_alias


def _weights_from_scores(selected_stocks, weight_mode: str):
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


def _round_lot_shares(target_value: float, price: float, board_lot: int = 100):
    if price <= 0 or target_value <= 0:
        return 0
    raw_shares = int(target_value / price)
    return max(0, (raw_shares // board_lot) * board_lot)


def build_buffered_rebalance_preview(params: dict):
    alias = params.get("alias")
    date = params.get("date")
    top_k = max(1, int(params.get("top_k", 5)))
    hold_topk = max(top_k, int(params.get("hold_topk", max(top_k, 8))))
    weight_mode = str(params.get("weight_mode", "equal")).lower()
    risk_degree = float(params.get("risk_degree", 0.95))

    prediction, resolved_alias = _load_prediction(alias, date, hold_topk)
    raw_positions = position._load_positions_file()
    enriched = position._enrich_positions(raw_positions)

    previous_holdings = _normalize_previous_holdings(enriched)
    selected_index = select_buffered_topk(
        scores=pd.Series({item["instrument"]: float(item["score"]) for item in prediction.get("stocks", [])}),
        topk=top_k,
        previous_holdings=previous_holdings,
        rank_buffer=max(0, hold_topk - top_k),
    )

    stock_map = {item["instrument"]: item for item in prediction.get("stocks", [])}
    selected_stocks = [stock_map[code] for code in selected_index if code in stock_map]
    target_weights_raw = _weights_from_scores(selected_stocks, weight_mode=weight_mode)
    target_weights = {k: v * risk_degree for k, v in target_weights_raw.items()}

    total_assets = float(enriched.get("totalAssets") or 0.0)
    position_map = {item["instrument"]: item for item in enriched.get("positions", [])}
    target_positions = []
    trades = []
    total_buy = 0.0
    total_sell = 0.0
    estimated_fees = 0.0

    all_instruments = sorted(set(position_map.keys()) | set(target_weights.keys()))
    for instrument in all_instruments:
        current = position_map.get(instrument, {})
        current_shares = int(current.get("shares") or 0)
        current_weight = float(current.get("weight") or 0.0) / 100.0
        current_price = float(current.get("currentPrice") or 0.0)
        score_row = stock_map.get(instrument, {})
        target_weight = float(target_weights.get(instrument, 0.0))
        trade_price = current_price or 10.0
        target_value = total_assets * target_weight
        target_shares = _round_lot_shares(target_value, trade_price)
        delta_shares = target_shares - current_shares
        trade_side = "hold"
        if delta_shares > 0:
            trade_side = "buy"
        elif delta_shares < 0:
            trade_side = "sell"
        trade_amount = abs(delta_shares) * trade_price
        fee_breakdown = position._calc_trade_fee(instrument, trade_amount, trade_side, position._fee_settings_from_raw(raw_positions))
        estimated_fees += fee_breakdown["total"]
        if trade_side == "buy":
            total_buy += trade_amount
        elif trade_side == "sell":
            total_sell += trade_amount

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
            "bufferKept": instrument in previous_holdings and instrument in target_weights,
            "isNew": instrument not in previous_holdings and instrument in target_weights,
            "fee": fee_breakdown,
        }
        target_positions.append(row)
        if trade_side != "hold":
            trades.append(row)

    target_positions.sort(key=lambda item: item["targetWeight"], reverse=True)
    trades.sort(key=lambda item: (item["side"] != "sell", -abs(item["tradeAmount"])))

    turnover = (total_buy + total_sell) / total_assets if total_assets > 0 else 0.0
    kept_count = sum(1 for item in target_positions if item["bufferKept"])
    new_count = sum(1 for item in target_positions if item["isNew"])

    return {
        "alias": resolved_alias,
        "tradeDate": prediction.get("date") or date,
        "config": {
            "topK": top_k,
            "holdTopk": hold_topk,
            "rankBuffer": max(0, hold_topk - top_k),
            "weightMode": weight_mode,
            "riskDegree": risk_degree,
        },
        "holdings": enriched,
        "prediction": prediction,
        "selected": selected_stocks,
        "targetPositions": target_positions,
        "trades": trades,
        "summary": {
            "keptCount": kept_count,
            "newCount": new_count,
            "sellCount": sum(1 for item in trades if item["side"] == "sell"),
            "buyCount": sum(1 for item in trades if item["side"] == "buy"),
            "estimatedBuyAmount": round(total_buy, 2),
            "estimatedSellAmount": round(total_sell, 2),
            "estimatedFees": round(estimated_fees, 2),
            "turnoverPct": round(turnover * 100, 2),
            "cashAfterTrades": round(float(enriched.get("cash") or 0.0) + total_sell - total_buy - estimated_fees, 2),
        },
        "explanation": {
            "title": "BufferedWeightStrategy 调仓预览",
            "why": "先保留当前已持有且排名仍落在 topk + buffer 内的股票，再用最新模型高分股票补齐 topk，减少不必要换手。",
            "how": [
                "读取当前持仓权重作为 previous holdings",
                "用模型选股分数生成当日候选排名",
                "保留仍在 hold_topk 范围内的旧仓位",
                "再补足 topk，并按 equal / score 模式分配目标权重",
                "最后对比当前持仓与目标持仓，生成买卖建议",
            ],
        },
    }


def buffered_rebalance_preview(rh):
    rh._json_response(build_buffered_rebalance_preview(rh._query_params()))
