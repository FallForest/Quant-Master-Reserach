# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Easytrader-based broker implementation for Chinese A-share brokers.

Wraps the easytrader library (GUI automation via pywinauto) to provide
a BaseBroker-compatible interface. Supports YinHe/Galaxy Securities (银河证券)
via xiadan.exe and other brokers supported by easytrader.

Two connection modes:
  1. ``login()`` — starts xiadan.exe, enters credentials, saves CAPTCHA
     image for the user to solve, then completes login.
  2. ``connect()`` — attaches to an already-logged-in trading client window.
"""

import os
import tempfile
import time
from typing import Dict, List, Optional

from quant_master.log import get_module_logger

from .base import (
    AccountInfo,
    BaseBroker,
    BrokerOrder,
    BrokerOrderDir,
    OrderStatus,
    Position,
)

logger = get_module_logger("EasytraderBroker")

# easytrader column name mappings (Chinese -> English)
_POSITION_COLS = {
    "证券代码": "stock_id",
    "股票代码": "stock_id",
    "证券名称": "name",
    "股票名称": "name",
    "股票余额": "volume",
    "可用余额": "available_volume",
    "当前持仓": "volume",
    "成本价": "cost_price",
    "参考成本价": "cost_price",
    "市价": "current_price",
    "最新价": "current_price",
    "参考市价": "current_price",
    "市值": "market_value",
    "参考市值": "market_value",
}

_ORDER_COLS = {
    "委托编号": "order_id",
    "合同编号": "order_id",
    "委托时间": "order_time",
    "证券代码": "stock_id",
    "股票代码": "stock_id",
    "委托价格": "price",
    "委托数量": "amount",
    "成交数量": "deal_amount",
    "买卖方向": "direction",
    "操作": "direction",
    "委托状态": "status",
    "状态": "status",
}


class EasytraderBroker(BaseBroker):
    """Broker implementation using easytrader (GUI automation via pywinauto).

    Wraps ``easytrader.YHClientTrader`` for YinHe Securities (银河证券).
    Works with ``xiadan.exe`` (standalone trading panel).

    Usage — login from scratch::

        broker = EasytraderBroker()
        broker.login("account", "password")
        account = broker.query_account()
        broker.buy("600036", price=35.50, amount=100)

    Usage — connect to already-logged-in xiadan::

        broker = EasytraderBroker()
        broker.connect()  # uses default C:\\silkriver\\xiadan\\xiadan.exe
        positions = broker.query_positions()
    """

    # Default xiadan path for YinHe Securities
    DEFAULT_EXE_PATH = r"C:\silkriver\xiadan\xiadan.exe"

    def __init__(self, broker_type: str = "yh_client"):
        import easytrader

        self._broker_type = broker_type
        self._user = easytrader.use(broker_type)
        self._connected = False

    def connect(self, exe_path: Optional[str] = None, **kwargs) -> bool:
        """Connect to an already-logged-in trading client.

        Parameters
        ----------
        exe_path : str, optional
            Path to the trading client executable.
            Default: ``C:\\silkriver\\xiadan\\xiadan.exe``
        """
        path = exe_path or self.DEFAULT_EXE_PATH
        try:
            self._user.connect(exe_path=path)
            self._connected = True
            logger.info(f"Connected to {self._broker_type} client")
            return True
        except Exception as e:
            logger.error(f"Connect failed: {e}")
            self._connected = False
            return False

    def login(
        self,
        account: str,
        password: str,
        captcha_code: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Login to xiadan.exe.

        Starts xiadan.exe (or finds a running instance), enters account
        and password, then clicks the login button.

        If CAPTCHA is enabled on the login dialog, the first call to
        ``login()`` (without *captcha_code*) will save the CAPTCHA image
        to a temp file and return ``False``.  Read the image, then call
        ``login()`` again with *captcha_code* set.

        Parameters
        ----------
        account : str
            Trading account number (客户号).
        password : str
            Trading password (交易密码).
        captcha_code : str, optional
            CAPTCHA code, if required.
        exe_path : str, optional
            Path to xiadan.exe (default: ``C:\\silkriver\\xiadan\\xiadan.exe``).

        Returns
        -------
        bool
            ``True`` on successful login, ``False`` otherwise.
        """
        import pywinauto
        import pywinauto.findwindows as fw

        exe_path = kwargs.get("exe_path") or self.DEFAULT_EXE_PATH

        try:
            # Start or find xiadan
            try:
                fw.find_windows(title="用户登录")
            except fw.ElementNotFoundError:
                pywinauto.Application().start(exe_path)
                time.sleep(3)

            # Connect to the login dialog by title
            app = pywinauto.Application().connect(
                title="用户登录", timeout=10
            )
            win = app.top_window()

            # Find the visible Edit controls (sorted by Y position)
            edits = []
            for ctrl in win.descendants():
                try:
                    if ctrl.friendly_class_name() == "Edit" and ctrl.is_visible():
                        edits.append((ctrl.rectangle().top, ctrl))
                except Exception:
                    pass
            edits.sort(key=lambda x: x[0])

            if len(edits) < 2:
                logger.error("Could not find account/password fields")
                return False

            # First visible Edit = account field
            account_field = edits[0][1]
            account_field.set_text("")
            account_field.type_keys(account)

            # Second visible Edit = password field
            pwd_field = edits[1][1]
            pwd_field.set_text("")
            pwd_field.type_keys(password)

            # Check if CAPTCHA field exists (CID 1005 or 1003)
            captcha_field = None
            for cid in (1005, 1003):
                try:
                    ctrl = win.child_window(control_id=cid)
                    if ctrl.is_visible():
                        captcha_field = ctrl
                        break
                except Exception:
                    pass

            if captcha_field and not captcha_code:
                # Save CAPTCHA image for user
                try:
                    captcha_path = os.path.join(
                        tempfile.gettempdir(), "xiadan_captcha.jpg"
                    )
                    for img_cid in (1499,):
                        try:
                            img_ctrl = win.child_window(
                                control_id=img_cid, class_name="Static"
                            )
                            img_ctrl.capture_as_image().save(
                                captcha_path, "jpeg"
                            )
                            logger.warning(
                                f"CAPTCHA saved to {captcha_path} — "
                                "read it and call login() again with "
                                "captcha_code=..."
                            )
                            break
                        except Exception:
                            pass
                except Exception:
                    pass
                return False

            if captcha_field and captcha_code:
                captcha_field.set_text(captcha_code)

            # Click login button (CID 1006)
            login_btn = win.child_window(control_id=1006, class_name="Button")
            login_btn.click()

            # Wait for login dialog to close
            try:
                win.wait_not("exists visible", timeout=15)
            except Exception:
                pass

            # Connect to the trading window
            time.sleep(2)
            try:
                app2 = pywinauto.Application().connect(
                    path=exe_path, timeout=10
                )
                self._user._app = app2
                self._user._close_prompt_windows()
                self._user._main = app2.window(title="网上股票交易系统5.0")
                self._connected = True
                logger.info(f"Logged in as {account}")
                return True
            except Exception as e:
                logger.error(f"Post-login connection failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def logout(self) -> None:
        self._connected = False
        logger.info("Logged out")

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        result = self._user.buy(security=stock_id, price=price, amount=amount)
        return self._parse_trade_result(result, stock_id, price, amount, BrokerOrderDir.BUY)

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        result = self._user.sell(security=stock_id, price=price, amount=amount)
        return self._parse_trade_result(result, stock_id, price, amount, BrokerOrderDir.SELL)

    def cancel_order(self, order_id: str) -> bool:
        try:
            result = self._user.cancel_entrust(entrust_no=order_id)
            return "成功" in str(result) or result is not None
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    def query_orders(self) -> List[BrokerOrder]:
        raw = self._user.today_entrusts
        return [self._dict_to_order(r) for r in raw]

    def query_deals(self) -> List[dict]:
        return self._user.today_trades

    def query_positions(self) -> List[Position]:
        raw = self._user.position
        return [self._dict_to_position(r) for r in raw]

    def query_account(self) -> AccountInfo:
        balance = self._user.balance
        return AccountInfo(
            total_assets=float(balance.get("总资产", balance.get("total_assets", 0))),
            available_cash=float(balance.get("可用金额", balance.get("available_cash", 0))),
            market_value=float(balance.get("股票市值", balance.get("market_value", 0))),
            frozen_amount=float(balance.get("冻结金额", balance.get("frozen_amount", 0))),
        )

    def is_connected(self) -> bool:
        return self._connected

    # --- Internal helpers ---

    def _parse_trade_result(self, result, stock_id, price, amount, direction) -> BrokerOrder:
        order_id = None
        if isinstance(result, dict):
            order_id = result.get("entrust_no") or result.get("order_id")
        return BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=direction,
            order_id=str(order_id) if order_id else None,
            status=OrderStatus.PENDING,
        )

    def _dict_to_order(self, d: dict) -> BrokerOrder:
        # Map Chinese keys to English
        mapped = {}
        for k, v in d.items():
            eng = _ORDER_COLS.get(k)
            if eng:
                mapped[eng] = v

        direction = BrokerOrderDir.BUY
        dir_str = str(mapped.get("direction", ""))
        if "卖" in dir_str:
            direction = BrokerOrderDir.SELL

        status = OrderStatus.PENDING
        status_str = str(mapped.get("status", ""))
        if "已成" in status_str or "全部成交" in status_str:
            status = OrderStatus.FILLED
        elif "部成" in status_str:
            status = OrderStatus.PARTIAL_FILLED
        elif "已撤" in status_str or "废单" in status_str:
            status = OrderStatus.CANCELLED

        return BrokerOrder(
            stock_id=str(mapped.get("stock_id", "")),
            price=float(mapped.get("price", 0)),
            amount=int(float(mapped.get("amount", 0))),
            direction=direction,
            order_id=str(mapped.get("order_id", "")),
            deal_amount=int(float(mapped.get("deal_amount", 0))),
            status=status,
        )

    def _dict_to_position(self, d: dict) -> Position:
        mapped = {}
        for k, v in d.items():
            eng = _POSITION_COLS.get(k)
            if eng:
                mapped[eng] = v

        return Position(
            stock_id=str(mapped.get("stock_id", "")),
            volume=int(float(mapped.get("volume", 0))),
            available_volume=int(float(mapped.get("available_volume", 0))),
            cost_price=float(mapped.get("cost_price", 0)),
            current_price=float(mapped.get("current_price", 0)),
            market_value=float(mapped.get("market_value", 0)),
        )
