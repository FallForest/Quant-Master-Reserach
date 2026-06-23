# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import pandas as pd

from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase
from .topk_cost_aware import rerank_topk_with_turnover_limit, select_buffered_topk, transform_scores_with_rank_buffer


class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model=None,
        dataset=None,
        topk=None,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        trade_impact_limit=None,
        selection_rank_buffer=0,
        selection_max_new_names: Optional[int] = None,
        risk_degree=0.95,
        dynamic_risk_gate=None,
        buy_method="first_fill",
        **kwargs,
    ):
        """
        Refactored SoftTopkStrategy with a budget-constrained rebalancing engine.

        Parameters
        ----------
        topk : int
            The number of top-N stocks to be held in the portfolio.
        trade_impact_limit : float
            Maximum weight change for each stock in one trade. If None, fallback to max_sold_weight.
        max_sold_weight : float
            Backward-compatible alias for trade_impact_limit. Use 1.0 to effectively disable the limit.
        selection_rank_buffer : int
            Past-only top-k buffer. Previously held names are retained while
            their current rank stays within topk + selection_rank_buffer.
            Defaults to 0, which preserves the original score top-k behavior.
        selection_max_new_names : Optional[int]
            Optional cap on newly-entered names versus current holdings after
            applying selection_rank_buffer. None preserves the existing
            selection behavior. A finite cap is strict even on the first
            rebalance, so it may intentionally leave cash idle.
        risk_degree : float
            The target percentage of total value to be invested.
        dynamic_risk_gate : dict
            Optional drawdown-based risk gate. Disabled by default.
        """
        if selection_max_new_names is not None and selection_max_new_names < 0:
            raise ValueError("selection_max_new_names must be non-negative")
        super(SoftTopkStrategy, self).__init__(
            model=model, dataset=dataset, order_generator_cls_or_obj=order_generator_cls_or_obj, **kwargs
        )

        self.topk = topk
        self.trade_impact_limit = trade_impact_limit if trade_impact_limit is not None else max_sold_weight
        self.selection_rank_buffer = selection_rank_buffer
        self.selection_max_new_names = selection_max_new_names
        self.risk_degree = risk_degree
        self.dynamic_risk_gate = dynamic_risk_gate
        self._dynamic_risk_scale = 1.0
        self._dynamic_equity_history = []
        self._last_effective_risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        if self._dynamic_gate_enabled():
            return getattr(self, "_last_effective_risk_degree", self.risk_degree)
        return self.risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        """
        Generates target position using Proportional Budget Allocation.
        Ensures deterministic sells and synchronized buys under impact limits.
        """

        if self.topk is None or self.topk <= 0:
            return {}
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        effective_risk_degree = self._update_effective_risk_degree(current)

        def apply_impact_limit(weight):
            return weight if self.trade_impact_limit is None else min(weight, self.trade_impact_limit)

        cur_weights = current.get_stock_weight_dict(only_stock=True)
        rank_buffer = getattr(self, "selection_rank_buffer", 0)
        max_new_names = getattr(self, "selection_max_new_names", None)
        ideal_per_stock = effective_risk_degree / self.topk
        if max_new_names is not None:
            if max_new_names < 0:
                raise ValueError("selection_max_new_names must be non-negative")
            # Preserve rank-buffer selection first, then cap entrants on that deterministic ranking.
            adjusted_score = transform_scores_with_rank_buffer(
                score,
                topk=self.topk,
                previous_holdings=cur_weights,
                rank_buffer=rank_buffer,
            )
            ideal_list = rerank_topk_with_turnover_limit(
                adjusted_scores=adjusted_score,
                topk=self.topk,
                previous_holdings=cur_weights,
                max_new_names=max_new_names,
            )
            ideal_list = _trim_new_names_to_cap(ideal_list, cur_weights, max_new_names)
        else:
            ideal_list = select_buffered_topk(
                score,
                topk=self.topk,
                previous_holdings=cur_weights,
                rank_buffer=rank_buffer,
            )
        ideal_list = ideal_list.tolist()
        initial_total_weight = sum(cur_weights.values())

        # --- Case A: Cold Start ---
        if not cur_weights:
            fill = apply_impact_limit(ideal_per_stock)
            return {code: fill for code in ideal_list}

        # --- Case B: Rebalancing ---
        all_tickers = set(cur_weights.keys()) | set(ideal_list)
        next_weights = {t: cur_weights.get(t, 0.0) for t in all_tickers}

        # Phase 1: Deterministic Sell Phase
        released_cash = 0.0
        for t in list(next_weights.keys()):
            cur = next_weights[t]
            if cur <= 1e-8:
                continue

            if t not in ideal_list:
                sell = apply_impact_limit(cur)
                next_weights[t] -= sell
                released_cash += sell
            elif cur > ideal_per_stock + 1e-8:
                excess = cur - ideal_per_stock
                sell = apply_impact_limit(excess)
                next_weights[t] -= sell
                released_cash += sell

        # Phase 2: Budget Calculation
        # Budget = Cash from sells + Available space from target risk degree
        total_budget = released_cash + (effective_risk_degree - initial_total_weight)

        # Phase 3: Proportional Buy Allocation
        if total_budget > 1e-8:
            shortfalls = {
                t: (ideal_per_stock - next_weights.get(t, 0.0))
                for t in ideal_list
                if next_weights.get(t, 0.0) < ideal_per_stock - 1e-8
            }

            if shortfalls:
                total_shortfall = sum(shortfalls.values())
                # Normalize total_budget to not exceed total_shortfall
                available_to_spend = min(total_budget, total_shortfall)

                for t, shortfall in shortfalls.items():
                    # Every stock gets its fair share based on its distance to target
                    share_of_budget = (shortfall / total_shortfall) * available_to_spend

                    # Capped by impact limit
                    max_buy_cap = apply_impact_limit(shortfall)

                    next_weights[t] += min(share_of_budget, max_buy_cap)

        return {k: v for k, v in next_weights.items() if v > 1e-8}

    def _dynamic_gate_enabled(self):
        gate = getattr(self, "dynamic_risk_gate", None) or {}
        return bool(gate.get("enabled", False)) and gate.get("mode", "drawdown") == "drawdown"

    def _update_effective_risk_degree(self, current):
        base_risk_degree = float(getattr(self, "risk_degree", 0.0))
        if not self._dynamic_gate_enabled():
            self._last_effective_risk_degree = base_risk_degree
            return base_risk_degree

        equity = self._extract_equity(current)
        if equity is not None and equity > 0:
            history = getattr(self, "_dynamic_equity_history", [])
            history.append(float(equity))
            self._dynamic_equity_history = history
            self._dynamic_risk_scale = self._compute_dynamic_risk_scale(history, base_risk_degree)

        min_risk_degree = min(
            float((getattr(self, "dynamic_risk_gate", None) or {}).get("min_risk_degree", base_risk_degree)),
            base_risk_degree,
        )
        scale = float(getattr(self, "_dynamic_risk_scale", 1.0))
        effective = base_risk_degree * scale
        effective = min(base_risk_degree, max(min_risk_degree, effective))
        self._last_effective_risk_degree = effective
        return effective

    def _compute_dynamic_risk_scale(self, history, base_risk_degree):
        gate = getattr(self, "dynamic_risk_gate", None) or {}
        if base_risk_degree <= 0:
            return 1.0

        lookback = int(gate.get("lookback", 0) or 0)
        window = history[-lookback:] if lookback > 0 else history
        peak = max(window) if window else None
        if not peak or peak <= 0:
            return 1.0

        drawdown = max(0.0, (peak - history[-1]) / peak)
        threshold = float(gate.get("drawdown_threshold", 0.08))
        full_clamp = float(gate.get("full_clamp_threshold", 0.16))
        min_risk_degree = min(float(gate.get("min_risk_degree", base_risk_degree)), base_risk_degree)
        min_scale = min(1.0, max(0.0, min_risk_degree / base_risk_degree))
        current_scale = min(1.0, max(min_scale, float(getattr(self, "_dynamic_risk_scale", 1.0))))

        if drawdown <= threshold:
            recovery_rate = float(gate.get("recovery_rate", 0.10))
            return min(1.0, current_scale + (1.0 - current_scale) * recovery_rate)

        if full_clamp <= threshold:
            target_scale = min_scale
        elif drawdown >= full_clamp:
            target_scale = min_scale
        else:
            ratio = (drawdown - threshold) / (full_clamp - threshold)
            target_scale = 1.0 - ratio * (1.0 - min_scale)

        decay_rate = float(gate.get("decay_rate", 1.0))
        next_scale = current_scale + (target_scale - current_scale) * decay_rate
        return min(1.0, max(min_scale, next_scale))

    @staticmethod
    def _extract_equity(current):
        position = getattr(current, "position", None)
        if isinstance(position, dict):
            value = position.get("now_account_value")
            if value is not None:
                return float(value)
        calculate_value = getattr(current, "calculate_value", None)
        if callable(calculate_value):
            try:
                return float(calculate_value())
            except (KeyError, ZeroDivisionError):
                return None
        return None


def _trim_new_names_to_cap(selected: pd.Index, previous_holdings, max_new_names: int) -> pd.Index:
    previous = set(previous_holdings)
    kept = []
    new_count = 0
    for instrument in selected:
        if instrument in previous:
            kept.append(instrument)
            continue
        if new_count >= max_new_names:
            continue
        kept.append(instrument)
        new_count += 1
    return pd.Index(kept)
