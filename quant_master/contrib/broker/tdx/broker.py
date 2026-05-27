# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""High-level TDX broker adapter implementing BaseBroker interface.

Uses /TQLEX endpoint for trading operations (confirmed from web JS source).
Entry names confirmed from AddinFlatJy.dll:
  - Stock.Buy, Stock.Sell, Stock.ktqx
"""

import threading
from typing import List, Optional

from quant_master.log import get_module_logger

from ..base import (
    AccountInfo,
    BaseBroker,
    BrokerOrder,
    BrokerOrderDir,
    OrderStatus,
    Position,
)
from .client import TDXClient
from .consts import ORDER_STATUS_MAP, TDX_ENTRIES
from .exceptions import TDXLoginError, TDXTradeError
from .protocol import (
    infer_market,
    parse_account_response,
    parse_positions_response,
    parse_tql_response,
    build_buy_params,
    build_cancel_params,
    build_sell_params,
)

logger = get_module_logger("TDXBroker")


class TDXBroker(BaseBroker):
    """High-level TDX broker.

    Usage::

        broker = TDXBroker(host="61.135.173.138", port=7708)
        broker.connect()
        broker.login("account", "password")
        account = broker.query_account()
        broker.buy("600036", price=35.50, amount=100)
        broker.logout()
    """

    def __init__(
        self,
        host: str,
        port: int = 7708,
        use_https: bool = False,
        heartbeat_interval: int = 1200,
    ):
        self.client = TDXClient(host=host, port=port, use_https=use_https)
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_thread: Optional[threading.Timer] = None
        self._logged_in = False
        self._entries = dict(TDX_ENTRIES)
        # Account context populated after login/query_account
        self._account: Optional[str] = None
        self._shareholder_sh: Optional[str] = None  # Shanghai shareholder code
        self._shareholder_sz: Optional[str] = None  # Shenzhen shareholder code

    def set_entry(self, operation: str, entry_name: str):
        """Override the TQL Entry name for an operation."""
        if operation not in TDX_ENTRIES:
            raise ValueError(f"Unknown operation '{operation}'. Valid: {list(TDX_ENTRIES.keys())}")
        self._entries[operation] = entry_name

    def connect(self) -> bool:
        ok = self.client.connect()
        if ok:
            self._start_heartbeat()
        return ok

    def login(self, account: str, password: str, **kwargs) -> bool:
        data = {
            "UserName": account,
            "PassWord": password,
            "AuthCode": kwargs.get("auth_code", ""),
            "DeptID": kwargs.get("department_id", ""),
        }
        try:
            text = self.client.call_tql(self._entries["login"], data)
            if "error" in text.lower() or "fail" in text.lower() or text.strip() == "":
                raise TDXLoginError(f"Login failed: {text[:200]}")
            self._logged_in = True
            self._account = account
            logger.info(f"Logged in as {account}")
            return True
        except TDXLoginError:
            raise
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise TDXLoginError(f"Login failed: {e}")

    def logout(self) -> None:
        self._stop_heartbeat()
        self.client.quit()
        self._logged_in = False

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        if kwargs.get("dry_run", False):
            return BrokerOrder(
                stock_id=stock_id,
                price=price,
                amount=amount,
                direction=BrokerOrderDir.BUY,
                order_id="DRY-RUN",
                status=OrderStatus.PENDING,
            )
        market = infer_market(stock_id)
        data = build_buy_params(
            stock_id, market, price, amount,
            account=self._account or "",
            shareholder_code=self._get_shareholder(market),
        )
        text = self.client.call_tqlex(self._entries["buy"], data)
        return self._parse_order_result(text, stock_id, price, amount, BrokerOrderDir.BUY)

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        if kwargs.get("dry_run", False):
            return BrokerOrder(
                stock_id=stock_id,
                price=price,
                amount=amount,
                direction=BrokerOrderDir.SELL,
                order_id="DRY-RUN",
                status=OrderStatus.PENDING,
            )
        market = infer_market(stock_id)
        data = build_sell_params(
            stock_id, market, price, amount,
            account=self._account or "",
            shareholder_code=self._get_shareholder(market),
        )
        text = self.client.call_tqlex(self._entries["sell"], data)
        return self._parse_order_result(text, stock_id, price, amount, BrokerOrderDir.SELL)

    def cancel_order(self, order_id: str) -> bool:
        data = build_cancel_params(order_id)
        text = self.client.call_tqlex(self._entries["cancel_order"], data)
        return "error" not in text.lower() and text.strip() != ""

    def query_orders(self) -> List[BrokerOrder]:
        text = self.client.call_tqlex(self._entries["query_orders"])
        df = parse_tql_response(text)
        orders = []
        for _, row in df.iterrows():
            direction = BrokerOrderDir.BUY if "买" in str(row.iloc[3] if len(row) > 3 else "") else BrokerOrderDir.SELL
            status_str = str(row.iloc[8] if len(row) > 8 else "")
            orders.append(
                BrokerOrder(
                    stock_id=str(row.iloc[1] if len(row) > 1 else ""),
                    price=float(row.iloc[4]) if len(row) > 4 else 0.0,
                    amount=int(float(row.iloc[5])) if len(row) > 5 else 0,
                    direction=direction,
                    order_id=str(row.iloc[0] if len(row) > 0 else ""),
                    deal_amount=int(float(row.iloc[6])) if len(row) > 6 else 0,
                    status=self._parse_status(status_str),
                )
            )
        return orders

    def query_deals(self) -> List[dict]:
        text = self.client.call_tqlex(self._entries["query_deals"])
        df = parse_tql_response(text)
        return df.to_dict(orient="records")

    def query_positions(self) -> List[Position]:
        text = self.client.call_tqlex(self._entries["query_positions"])
        records = parse_positions_response(text)
        return [Position(**r) for r in records]

    def query_account(self) -> AccountInfo:
        text = self.client.call_tqlex(self._entries["query_account"])
        return AccountInfo(**parse_account_response(text))

    def is_connected(self) -> bool:
        return self.client.is_connected()

    # --- Internal helpers ---

    def _get_shareholder(self, market: int) -> str:
        """Get shareholder code for a given market."""
        if market == 1:  # Shanghai
            return self._shareholder_sh or ""
        return self._shareholder_sz or ""

    def _parse_order_result(self, text, stock_id, price, amount, direction) -> BrokerOrder:
        df = parse_tql_response(text)
        order_id = None
        if not df.empty:
            order_id = str(df.iloc[0].iloc[0]) if len(df.columns) > 0 else None
        return BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=direction,
            order_id=order_id,
            status=OrderStatus.PENDING,
        )

    def _parse_status(self, status_str: str) -> OrderStatus:
        mapped = ORDER_STATUS_MAP.get(status_str, "pending")
        return OrderStatus(mapped)

    def _start_heartbeat(self):
        self._heartbeat_thread = threading.Timer(self._heartbeat_interval, self._heartbeat_loop)
        self._heartbeat_thread.daemon = True
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        if self.client.alive():
            logger.debug("Heartbeat OK")
        else:
            logger.warning("Heartbeat failed")
        self._start_heartbeat()

    def _stop_heartbeat(self):
        if self._heartbeat_thread:
            self._heartbeat_thread.cancel()
