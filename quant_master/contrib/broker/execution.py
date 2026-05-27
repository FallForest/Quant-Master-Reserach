# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Live-order execution helpers with lightweight risk checks.

This module turns broker adapters into a reusable execution pipeline:

1. Validate order parameters before submission.
2. Submit to the selected broker implementation.
3. Best-effort query broker state after submission for confirmation.

The goal is to keep strategy code away from broker-specific edge cases and
provide a safer default path than ad hoc direct broker calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from quant_master.log import get_module_logger

from .base import BaseBroker, BrokerOrder, BrokerOrderDir, OrderStatus

logger = get_module_logger("LiveOrderExecutor")

A_SHARE_TRADE_UNIT = 100


@dataclass
class LiveOrderRequest:
    stock_id: str
    price: float
    amount: int
    direction: BrokerOrderDir
    note: str = ""


@dataclass
class ExecutionResult:
    request: LiveOrderRequest
    accepted: bool
    broker_order: Optional[BrokerOrder] = None
    rejection_reason: str = ""
    post_check_status: str = "not_checked"


class LiveOrderExecutor:
    """Unified order executor with simple pre-trade validation."""

    def __init__(
        self,
        broker: BaseBroker,
        *,
        max_order_value: Optional[float] = None,
        max_position_ratio: float = 1.0,
        trade_unit: int = A_SHARE_TRADE_UNIT,
        allow_sell_without_position_check: bool = False,
        validate_account_state: bool = True,
    ):
        self.broker = broker
        self.max_order_value = max_order_value
        self.max_position_ratio = float(max(0.0, min(max_position_ratio, 1.0)))
        self.trade_unit = int(trade_unit)
        self.allow_sell_without_position_check = allow_sell_without_position_check
        self.validate_account_state = validate_account_state

    def submit(self, request: LiveOrderRequest, *, dry_run: bool = False) -> ExecutionResult:
        rejection = self._validate_request(request)
        if rejection:
            return ExecutionResult(
                request=request,
                accepted=False,
                rejection_reason=rejection,
            )

        if self.validate_account_state:
            position_rejection = self._validate_with_account_state(request)
            if position_rejection:
                return ExecutionResult(
                    request=request,
                    accepted=False,
                    rejection_reason=position_rejection,
                )

        try:
            if request.direction == BrokerOrderDir.BUY:
                broker_order = self.broker.buy(
                    request.stock_id,
                    request.price,
                    request.amount,
                    dry_run=dry_run,
                )
            else:
                broker_order = self.broker.sell(
                    request.stock_id,
                    request.price,
                    request.amount,
                    dry_run=dry_run,
                )
        except Exception as exc:
            logger.error(f"broker submit failed for {request.stock_id}: {exc}")
            return ExecutionResult(
                request=request,
                accepted=False,
                rejection_reason=f"{type(exc).__name__}: {exc}",
                post_check_status="submit_failed",
            )

        post_check_status = self._post_check(broker_order)
        return ExecutionResult(
            request=request,
            accepted=True,
            broker_order=broker_order,
            post_check_status=post_check_status,
        )

    def submit_many(self, requests: Iterable[LiveOrderRequest], *, dry_run: bool = False) -> List[ExecutionResult]:
        results: List[ExecutionResult] = []
        halt_reason = ""
        for req in requests:
            if halt_reason:
                results.append(
                    ExecutionResult(
                        request=req,
                        accepted=False,
                        rejection_reason=halt_reason,
                        post_check_status="skipped",
                    )
                )
                continue

            result = self.submit(req, dry_run=dry_run)
            results.append(result)
            if result.post_check_status == "submit_failed":
                halt_reason = "skipped_after_submit_failure"
        return results

    def _validate_request(self, request: LiveOrderRequest) -> str:
        if not request.stock_id or len(str(request.stock_id).strip()) < 6:
            return "invalid_stock_id"
        if request.price <= 0:
            return "invalid_price"
        if request.amount <= 0:
            return "invalid_amount"
        if self.trade_unit > 0 and request.amount % self.trade_unit != 0:
            return f"amount_must_be_multiple_of_{self.trade_unit}"
        if self.max_order_value is not None and request.price * request.amount > self.max_order_value:
            return "order_value_exceeds_limit"
        return ""

    def _validate_with_account_state(self, request: LiveOrderRequest) -> str:
        try:
            account = self.broker.query_account()
        except Exception as exc:  # pragma: no cover - defensive for live adapters
            logger.warning(f"query_account failed before order submit: {exc}")
            account = None

        order_value = request.price * request.amount

        if request.direction == BrokerOrderDir.BUY and account is not None:
            budget = float(account.available_cash) * self.max_position_ratio
            if order_value > budget:
                return "insufficient_available_cash"

        if request.direction == BrokerOrderDir.SELL:
            try:
                positions = self.broker.query_positions()
            except Exception as exc:  # pragma: no cover - defensive for live adapters
                logger.warning(f"query_positions failed before sell submit: {exc}")
                positions = []

            if positions:
                available = 0
                for pos in positions:
                    if pos.stock_id == request.stock_id:
                        available = int(pos.available_volume)
                        break
                if request.amount > available:
                    return "insufficient_available_position"
            elif not self.allow_sell_without_position_check:
                return "position_check_unavailable"

        return ""

    def _post_check(self, broker_order: BrokerOrder) -> str:
        if broker_order is None:
            return "missing_broker_order"
        if broker_order.order_id:
            return "order_id_received"
        try:
            orders = self.broker.query_orders()
        except Exception as exc:  # pragma: no cover - defensive for live adapters
            logger.warning(f"query_orders failed after submit: {exc}")
            return "query_orders_failed"

        for order in orders:
            if (
                order.stock_id == broker_order.stock_id
                and order.amount == broker_order.amount
                and order.direction == broker_order.direction
            ):
                return f"matched_in_query_orders:{order.status.value}"
        return "submitted_but_not_observed"
