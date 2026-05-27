# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""TQL response parsers and request parameter builders.

Parameter format confirmed from AddinFlatJy.dll analysis:
  Buy:  zqdm=%s%s|mmbz=1|wtsl=%d|zdbl=%.2f|cllx=%d|wtfssh=0|wtfssz=0|zjzh=%s|gddm=%s|zhlb=%d|szid=%s|
  Sell: zqdm=%s%s|mmbz=0|wtsl=%d|zdbl=%.2f|cllx=%d|wtfssh=0|wtfssz=0|zjzh=%s|gddm=%s|zhlb=%d|szid=%s|

For HTTP POST to /TQLEX?Entry=Stock.Buy, these are sent as form-encoded fields.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .consts import MARKET_MAP, MARKET_PREFIX, MARKET_SZ


def infer_market(stock_id: str) -> int:
    """Infer TDX market code from stock code prefix."""
    if not stock_id:
        return MARKET_SZ
    prefix = stock_id[:3]
    if prefix in MARKET_MAP:
        return MARKET_MAP[prefix]
    # Fallback: codes starting with 6 are Shanghai
    return 1 if stock_id[0] == "6" else 0


def encode_zqdm(stock_id: str, market: int) -> str:
    """Encode security code with market prefix for TQL parameter.

    DLL format: zqdm=%s%s (market_prefix + stock_code)
    Example: "1sh600036" for Shanghai 600036, "0sz000001" for Shenzhen 000001
    """
    prefix = MARKET_PREFIX.get(market, "0sz")
    return f"{prefix}{stock_id}"


def parse_tql_response(text: str) -> pd.DataFrame:
    """Parse a TQL response string into a DataFrame.

    TDX TQL responses are typically tab-separated text:
    - First line: column headers (tab-separated)
    - Remaining lines: data rows (tab-separated)

    Some responses may be JSON arrays.
    """
    text = text.strip()
    if not text:
        return pd.DataFrame()

    # Try JSON first
    if text[0] in ("[", "{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                for key in ("data", "result", "list"):
                    if key in data:
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
        except (json.JSONDecodeError, ValueError):
            pass

    # TSV parsing
    lines = text.split("\n")
    if not lines:
        return pd.DataFrame()

    headers = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    return pd.DataFrame(rows, columns=headers)


def build_buy_params(
    stock_id: str,
    market: int,
    price: float,
    amount: int,
    **kwargs,
) -> Dict[str, str]:
    """Build form-encoded parameters for a buy order.

    Confirmed from AddinFlatJy.dll parameter format:
      zqdm=<market_prefix><code>|mmbz=1|wtsl=<qty>|

    For HTTP POST, sent as individual form fields.
    """
    zjzh = kwargs.get("account", "")
    gddm = kwargs.get("shareholder_code", "")

    return {
        "zqdm": encode_zqdm(stock_id, market),
        "mmbz": "1",
        "wtsl": str(amount),
        "wtjg": f"{price:.3f}",
        "zdbl": "0.00",
        "cllx": "0",
        "wtfssh": "0",
        "wtfssz": "0",
        "zjzh": zjzh,
        "gddm": gddm,
        "zhlb": "0",
        "szid": str(market),
    }


def build_sell_params(
    stock_id: str,
    market: int,
    price: float,
    amount: int,
    **kwargs,
) -> Dict[str, str]:
    """Build form-encoded parameters for a sell order.

    Same as buy but mmbz=0.
    """
    zjzh = kwargs.get("account", "")
    gddm = kwargs.get("shareholder_code", "")

    return {
        "zqdm": encode_zqdm(stock_id, market),
        "mmbz": "0",
        "wtsl": str(amount),
        "wtjg": f"{price:.3f}",
        "zdbl": "0.00",
        "cllx": "0",
        "wtfssh": "0",
        "wtfssz": "0",
        "zjzh": zjzh,
        "gddm": gddm,
        "zhlb": "0",
        "szid": str(market),
    }


def build_cancel_params(order_id: str) -> Dict[str, str]:
    """Build parameters for cancelling an order.

    Entry: Stock.ktqx (from AddinFlatJy.dll)
    """
    return {"Wth": order_id}


def parse_positions_response(text: str) -> List[Dict[str, Any]]:
    """Parse query_positions response into structured records."""
    df = parse_tql_response(text)
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "stock_id": str(row.iloc[0]).strip(),
                "volume": int(float(row.iloc[2])) if len(row) > 2 else 0,
                "available_volume": int(float(row.iloc[3])) if len(row) > 3 else 0,
                "cost_price": float(row.iloc[4]) if len(row) > 4 else 0.0,
                "current_price": float(row.iloc[5]) if len(row) > 5 else 0.0,
                "market_value": float(row.iloc[6]) if len(row) > 6 else 0.0,
            }
        )
    return records


def parse_account_response(text: str) -> Dict[str, float]:
    """Parse query_account response into account info dict."""
    df = parse_tql_response(text)
    if df.empty:
        return {"total_assets": 0.0, "available_cash": 0.0, "market_value": 0.0, "frozen_amount": 0.0}
    row = df.iloc[0]
    return {
        "total_assets": _safe_float(row, 0),
        "available_cash": _safe_float(row, 1),
        "market_value": _safe_float(row, 2),
        "frozen_amount": _safe_float(row, 3),
    }


def _safe_float(row, idx: int) -> float:
    try:
        return float(row.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return 0.0
