"""Robust central-tendency estimators.

These resist outliers — a single spiky price barely moves the fair value. Useful
when prices have jumps or fat tails that would distort a plain average.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import register


@register("median")
class RollingMedian(RollingEstimator):
    """Rolling median: the middle price, fully insensitive to outliers."""

    def _compute_mean(self) -> float:
        return float(np.median(self._last()))


@register("trimmed_mean")
class TrimmedMean(RollingEstimator):
    """Mean after discarding the top and bottom 10% of the window."""

    def _compute_mean(self) -> float:
        a = self._last()
        if len(a) < 5:
            return float(a.mean())
        return float(trim_mean(a, proportiontocut=0.1))


@register("winsorized_mean")
class WinsorizedMean(RollingEstimator):
    """Mean after clamping the most extreme 10% on each side to the 10th/90th pct."""

    def _compute_mean(self) -> float:
        a = self._last()
        if len(a) < 5:
            return float(a.mean())
        return float(np.asarray(winsorize(a, limits=0.1)).mean())


@register("huber")
class HuberMean(RollingEstimator):
    """Huber M-estimator of location: averages like a mean near the center but
    down-weights far points like a median, via iterative reweighting."""

    def _compute_mean(self) -> float:
        a = self._last()
        mu = float(np.median(a))
        s = 1.4826 * float(np.median(np.abs(a - mu)))
        if s <= 0:
            return mu
        c = 1.345
        for _ in range(25):
            absr = np.abs(a - mu) / s
            w = np.ones_like(absr)
            far = absr > c
            w[far] = c / absr[far]  # down-weight far points (no divide-by-zero)
            new = float((w * a).sum() / w.sum())
            if abs(new - mu) <= 1e-10 * max(1.0, abs(mu)):
                mu = new
                break
            mu = new
        return mu


@register("midrange")
class Midrange(RollingEstimator):
    """Midrange (Donchian center): halfway between the window's high and low."""

    def _compute_mean(self) -> float:
        a = self._last()
        return float((a.max() + a.min()) / 2.0)
