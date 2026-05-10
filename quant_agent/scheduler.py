from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Metrics:
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    arr: float = 0.0
    ir: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.ic,
                self.icir,
                self.rank_ic,
                self.rank_icir,
                self.arr,
                self.ir,
                -self.mdd,
                self.sharpe,
            ],
            dtype=float,
        )

    @classmethod
    def from_feedback(cls, feedback: dict[str, Any] | None) -> "Metrics | None":
        if not isinstance(feedback, dict):
            return None
        metrics = feedback.get("metrics", {})
        if not isinstance(metrics, dict):
            return None
        m = cls(
            ic=_safe_float(metrics.get("IC"), 0.0),
            icir=_safe_float(metrics.get("ICIR"), 0.0),
            rank_ic=_safe_float(metrics.get("Rank IC"), 0.0),
            rank_icir=_safe_float(metrics.get("Rank ICIR"), 0.0),
            arr=_safe_float(
                metrics.get("1day.excess_return_with_cost.annualized_return"),
                _safe_float(metrics.get("1day.excess_return_with_cost.annualized_return "), 0.0),
            ),
            ir=_safe_float(metrics.get("1day.excess_return_with_cost.information_ratio"), 0.0),
            mdd=_safe_float(metrics.get("1day.excess_return_with_cost.max_drawdown"), 0.0),
        )
        if m.mdd < 0:
            m.sharpe = m.arr / -m.mdd if m.mdd != 0 else 0.0
        elif m.mdd > 0:
            m.sharpe = m.arr / m.mdd
        else:
            m.sharpe = 0.0
        return m


class LinearThompsonTwoArm:
    def __init__(self, dim: int, prior_var: float = 10.0, noise_var: float = 0.5):
        self.dim = dim
        self.noise_var = noise_var
        self.mean = {
            "factor": np.zeros(dim, dtype=float),
            "model": np.zeros(dim, dtype=float),
        }
        self.precision = {
            "factor": np.eye(dim, dtype=float) / prior_var,
            "model": np.eye(dim, dtype=float) / prior_var,
        }

    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        p = self.precision[arm]
        p = 0.5 * (p + p.T)
        eps = 1e-6
        try:
            cov = np.linalg.inv(p + eps * np.eye(self.dim))
            l = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + l @ z
        except np.linalg.LinAlgError:
            w_sample = self.mean[arm]
        return float(np.dot(w_sample, x))

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        p = self.precision[arm]
        p += np.outer(x, x) / self.noise_var
        self.precision[arm] = p
        self.mean[arm] = np.linalg.solve(p, p @ self.mean[arm] + (r / self.noise_var) * x)

    def next_arm(self, x: np.ndarray) -> str:
        scores = {arm: self.sample_reward(arm, x) for arm in ("factor", "model")}
        return max(scores, key=scores.get)


class EnvController:
    def __init__(self, weights: tuple[float, ...] | None = None):
        self.weights = np.asarray(weights or (0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2), dtype=float)
        self.bandit = LinearThompsonTwoArm(dim=8, prior_var=10.0, noise_var=0.5)

    def reward(self, m: Metrics) -> float:
        return float(np.dot(self.weights, m.as_vector()))

    def decide(self, m: Metrics) -> str:
        return self.bandit.next_arm(m.as_vector())

    def record(self, m: Metrics, arm: str) -> None:
        self.bandit.update(arm, m.as_vector(), self.reward(m))


def decide_bandit_action(trace_hist: list[dict[str, Any]], seed: int | None = None) -> str:
    if not trace_hist:
        return "factor"

    controller = EnvController()
    latest_metrics = Metrics()
    has_metrics = False
    for item in trace_hist:
        if not isinstance(item, dict):
            continue
        experiment = item.get("experiment", {})
        hypothesis = experiment.get("hypothesis", {}) if isinstance(experiment, dict) else {}
        action = hypothesis.get("action")
        if action not in {"factor", "model"}:
            continue
        metrics = Metrics.from_feedback(item.get("feedback"))
        if metrics is None:
            continue
        controller.record(metrics, action)
        latest_metrics = metrics
        has_metrics = True

    if not has_metrics:
        return "factor"
    if seed is not None:
        np.random.seed(seed)
    return controller.decide(latest_metrics)


def decide_random_action(seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(["factor", "model"])


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
