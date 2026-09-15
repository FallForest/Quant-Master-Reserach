# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AdaptiveTopkStrategy — dynamically adjusts portfolio concentration based on
recent prediction volatility.

When the cross-sectional prediction distribution is stable (low volatility),
the model has high conviction and the portfolio concentrates into fewer names.
When predictions are volatile, the model is uncertain and the portfolio
diversifies into more names.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy
from quant_master.log import get_module_logger


class AdaptiveTopkStrategy(TopkDropoutStrategy):
    """Top-k dropout strategy with volatility-driven dynamic sizing.

    At each trading step the cross-sectional standard deviation of predictions
    is recorded.  A 20-day rolling window tracks this volatility, and the
    latest value is normalised to [0, 1] relative to the historical range.
    The effective topk is then set by linear interpolation:

        topk_dynamic = min_topk + alpha * normalized_vol * (max_topk - min_topk)

    In calm markets (normalized_vol near 0) the portfolio concentrates into
    ``min_topk`` names (high conviction).  In volatile markets (normalized_vol
    near 1) the portfolio diversifies into ``max_topk`` names.  The ``n_drop``
    parameter scales proportionally so turnover stays consistent.

    Parameters
    ----------
    topk : int
        Reference portfolio size used for ``n_drop`` scaling.  The actual
        number of held stocks floats between ``min_topk`` and ``max_topk``.
    n_drop : int
        Reference number of stocks to replace each period.  Scaled
        proportionally when topk adapts.
    min_topk : int
        Lower bound on dynamic topk (high-conviction, calm regime).
    max_topk : int
        Upper bound on dynamic topk (low-conviction, volatile regime).
    volatility_window : int
        Number of recent trading steps used to compute rolling volatility.
    alpha : float
        Sensitivity of topk adaptation to normalised volatility.
        ``alpha=1.0`` maps the full [0, 1] normalised-vol range onto the
        full [min_topk, max_topk] interval.  Values < 1 dampen the
        adaptation so topk stays closer to the midpoint.
    """

    def __init__(
        self,
        *,
        topk: int = 60,
        n_drop: int = 6,
        min_topk: int = 30,
        max_topk: int = 80,
        volatility_window: int = 20,
        alpha: float = 1.0,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        super().__init__(
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
            hold_thresh=hold_thresh,
            only_tradable=only_tradable,
            forbid_all_trade_at_limit=forbid_all_trade_at_limit,
            **kwargs,
        )
        self.logger = get_module_logger("AdaptiveTopkStrategy")

        self.base_topk = topk
        self.base_n_drop = n_drop
        self.min_topk = int(min_topk)
        self.max_topk = int(max_topk)
        self.volatility_window = int(volatility_window)
        self.alpha = float(alpha)

        # Rolling buffer of cross-sectional prediction std (one value per step)
        self._vol_history: deque = deque(maxlen=self.volatility_window)
        # Track step count for logging
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_sectional_volatility(pred_score: pd.Series) -> float:
        """Return the standard deviation of the prediction cross-section."""
        clean = pred_score.dropna()
        if len(clean) < 2:
            return 0.0
        return float(clean.std())

    def _normalized_volatility(self) -> float:
        """Normalise the most recent volatility observation to [0, 1].

        Uses the full historical range seen so far.  When the range is
        degenerate (all values identical), returns 0.0 (assume calm).
        """
        if len(self._vol_history) < 2:
            return 0.0
        arr = np.array(self._vol_history)
        vol_min = arr.min()
        vol_max = arr.max()
        if vol_max - vol_min < 1e-12:
            return 0.0
        return float((arr[-1] - vol_min) / (vol_max - vol_min))

    # ------------------------------------------------------------------
    # Dynamic sizing
    # ------------------------------------------------------------------

    def _compute_dynamic_topk(self) -> int:
        """Compute the volatility-adjusted topk value.

        Uses linear interpolation across [min_topk, max_topk] scaled by the
        alpha parameter:

            raw = min_topk + (max_topk - min_topk) * alpha * normalized_vol
            topk_dynamic = clamp(raw, min_topk, max_topk)

        When normalized_vol == 0 (calm):  topk_dynamic == min_topk
        When normalized_vol == 1 (storm): topk_dynamic == max_topk
        When alpha < 1:  adaptation is dampened toward base_topk
        When alpha == 1: full range is used
        """
        norm_vol = self._normalized_volatility()
        raw = self.min_topk + (self.max_topk - self.min_topk) * self.alpha * norm_vol
        dynamic = int(round(raw))
        # Clamp to [min_topk, max_topk]
        return max(self.min_topk, min(self.max_topk, dynamic))

    def _compute_dynamic_n_drop(self, dynamic_topk: int) -> int:
        """Keep n_drop proportional to the current topk."""
        ratio = self.base_n_drop / max(self.base_topk, 1)
        dynamic = max(1, int(round(ratio * dynamic_topk)))
        # Never drop more than the portfolio can hold
        return min(dynamic, dynamic_topk)

    # ------------------------------------------------------------------
    # Decision generation
    # ------------------------------------------------------------------

    def generate_trade_decision(self, execute_result=None):
        # --- Obtain prediction scores (same logic as parent) ---
        trade_step = self.trade_calendar.get_trade_step()
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]

        if pred_score is not None:
            # Update volatility history
            vol = self._cross_sectional_volatility(pred_score)
            self._vol_history.append(vol)
            self._step_count += 1

            # Compute dynamic parameters
            dynamic_topk = self._compute_dynamic_topk()
            dynamic_n_drop = self._compute_dynamic_n_drop(dynamic_topk)

            # Temporarily override parent parameters
            saved_topk = self.topk
            saved_n_drop = self.n_drop
            self.topk = dynamic_topk
            self.n_drop = dynamic_n_drop

            if self._step_count <= 5 or self._step_count % 20 == 0:
                norm_vol = self._normalized_volatility()
                self.logger.info(
                    f"[step={self._step_count}] "
                    f"vol={vol:.6f} norm_vol={norm_vol:.3f} "
                    f"topk={dynamic_topk} n_drop={dynamic_n_drop}"
                )

            # Delegate to parent implementation
            decision = super().generate_trade_decision(execute_result=execute_result)

            # Restore base parameters
            self.topk = saved_topk
            self.n_drop = saved_n_drop

            return decision

        return TradeDecisionWO([], self)
