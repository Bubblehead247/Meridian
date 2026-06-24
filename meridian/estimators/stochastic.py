"""Stochastic / time-series-model estimators.

These fit a small statistical model of how price evolves and read fair value
from it — either the model's one-step-ahead expected price, or, for the
Ornstein-Uhlenbeck model, the long-run level the process reverts to. This makes
them the most directly "mean-reversion aware" estimators in the library.
"""

from __future__ import annotations

import numpy as np

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import register


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """OLS coefficients for ``y ~ X`` (X includes intercept).

    Returns None if the solve fails on a degenerate window (e.g. constant
    prices), letting callers fall back to a robust default.
    """
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return coeffs


@register("ar1")
class AR1(RollingEstimator):
    """AR(1) one-step fair value: fit ``x_t = c + phi*x_{t-1}`` and predict the
    next expected price from the latest one."""

    def _compute_mean(self) -> float:
        a = self._last(self.window + 1)
        if len(a) < 3:
            return float(a.mean())
        y = a[1:]
        X = np.column_stack([np.ones(len(y)), a[:-1]])
        coeffs = _ols(X, y)
        if coeffs is None:
            return float(a.mean())
        c, phi = coeffs
        return float(c + phi * a[-1])


@register("ar2")
class AR2(RollingEstimator):
    """AR(2) one-step fair value: like AR(1) but using the two latest prices."""

    def _compute_mean(self) -> float:
        a = self._last(self.window + 2)
        if len(a) < 5:
            return float(a.mean())
        y = a[2:]
        X = np.column_stack([np.ones(len(y)), a[1:-1], a[:-2]])
        coeffs = _ols(X, y)
        if coeffs is None:
            return float(a.mean())
        c, p1, p2 = coeffs
        return float(c + p1 * a[-1] + p2 * a[-2])


@register("ou")
class OrnsteinUhlenbeck(RollingEstimator):
    """Ornstein-Uhlenbeck reversion level: fit AR(1) and report the long-run
    mean ``c/(1-phi)`` the process is pulled toward (not the next-step value)."""

    def _compute_mean(self) -> float:
        a = self._last(self.window + 1)
        if len(a) < 3:
            return float(a.mean())
        y = a[1:]
        X = np.column_stack([np.ones(len(y)), a[:-1]])
        coeffs = _ols(X, y)
        if coeffs is None:
            return float(a.mean())
        c, phi = coeffs
        if phi >= 1.0 or phi < -1.0:  # non-reverting fit -> fall back
            return float(a.mean())
        return float(c / (1.0 - phi))


@register("exp_reg")
class ExponentialRegression(RollingEstimator):
    """Log-linear regression endpoint: fit a line to log-prices (constant growth
    rate) and report its current value — the geometric trend's fair value."""

    def _compute_mean(self) -> float:
        a = self._last()
        if len(a) < 3 or np.any(a <= 0):
            return float(a.mean())
        t = np.arange(len(a), dtype=float)
        slope, intercept = np.polyfit(t, np.log(a), 1)
        return float(np.exp(intercept + slope * t[-1]))
