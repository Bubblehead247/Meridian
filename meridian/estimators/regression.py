"""Regression-based estimators.

Each fits a curve (line, polynomial, robust line, local smooth) to the recent
window and reads off its value at the current bar. Fair value is "where the
fitted trend says the price should be right now."
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import theilslopes
from statsmodels.nonparametric.smoothers_lowess import lowess

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import register


def _poly_endpoint(a: np.ndarray, degree: int) -> float:
    """Fit a degree-``d`` polynomial in time and evaluate at the last point."""
    k = len(a)
    if k < degree + 1:
        return float(a.mean())
    t = np.arange(k, dtype=float)
    coeffs = np.polyfit(t, a, degree)
    return float(np.polyval(coeffs, t[-1]))


@register("lsma")
class LinearRegressionMA(RollingEstimator):
    """Least-squares moving average: endpoint of a fitted line (a.k.a. LSMA)."""

    def _compute_mean(self) -> float:
        return _poly_endpoint(self._last(), 1)


@register("quadratic_reg")
class QuadraticRegression(RollingEstimator):
    """Endpoint of a fitted quadratic (captures gentle curvature in the trend)."""

    def _compute_mean(self) -> float:
        return _poly_endpoint(self._last(), 2)


@register("cubic_reg")
class CubicRegression(RollingEstimator):
    """Endpoint of a fitted cubic (captures S-shaped local trends)."""

    def _compute_mean(self) -> float:
        return _poly_endpoint(self._last(), 3)


@register("theil_sen")
class TheilSen(RollingEstimator):
    """Theil-Sen robust line: slope is the median of pairwise slopes, so the
    endpoint resists outliers far better than ordinary least squares."""

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        if k < 3:
            return float(a.mean())
        t = np.arange(k, dtype=float)
        slope, intercept, _, _ = theilslopes(a, t)
        return float(intercept + slope * t[-1])


@register("savgol")
class SavitzkyGolay(RollingEstimator):
    """Savitzky-Golay endpoint: a polynomial smoother that preserves shape."""

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        wl = self.window if self.window % 2 == 1 else self.window - 1
        if k < wl or wl < 3:
            return float(a.mean())
        y = savgol_filter(a[-wl:], window_length=wl, polyorder=2)
        return float(y[-1])


@register("lowess")
class Lowess(RollingEstimator):
    """LOWESS endpoint: locally-weighted regression, robust to outliers."""

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        if k < 5:
            return float(a.mean())
        t = np.arange(k, dtype=float)
        fitted = lowess(a, t, frac=0.5, return_sorted=False)
        return float(fitted[-1])
