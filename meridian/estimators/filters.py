"""Signal-processing filters as fair-value estimators.

These treat the price as a noisy signal and extract its smooth, low-frequency
component — the "true" level under the noise. Includes a Kalman filter, the
Hodrick-Prescott trend, and several of John Ehlers' trading filters.
"""

from __future__ import annotations

import math

import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import register


@register("kalman")
class KalmanLocalLevel(RollingEstimator):
    """Kalman local-level filter: models price as a slow random walk plus noise
    and optimally blends the prediction with each new observation."""

    def _init_state(self) -> None:
        self.x = float("nan")  # estimated level
        self.P = 1.0           # estimate variance
        self.R = 1.0           # measurement noise
        self.Q = 1.0 / self.window  # process noise (smaller window -> smoother)

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        if not math.isfinite(self.x):
            self.x = p
            return self.x
        self.P += self.Q
        k = self.P / (self.P + self.R)
        self.x = self.x + k * (p - self.x)
        self.P = (1 - k) * self.P
        return self.x

    def _extra_state(self) -> dict:
        return {"x": self.x, "P": self.P}


@register("hp_filter")
class HodrickPrescott(RollingEstimator):
    """Hodrick-Prescott trend: separates a smooth trend from cyclical noise over
    the window and reports the trend's current value."""

    def _compute_mean(self) -> float:
        a = self._last(self.window)
        if len(a) < 4:
            return float(a.mean())
        _, trend = hpfilter(a, lamb=129600 / max(1, self.window))
        return float(trend[-1])


@register("super_smoother")
class SuperSmoother(RollingEstimator):
    """Ehlers super smoother: a 2-pole filter that removes high-frequency noise
    with minimal lag."""

    def _init_state(self) -> None:
        w = self.window
        a1 = math.exp(-1.414 * math.pi / w)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / w)
        self.c2 = b1
        self.c3 = -a1 * a1
        self.c1 = 1 - self.c2 - self.c3
        self.s1 = float("nan")
        self.s2 = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        p1 = self._buf[-2] if len(self._buf) > 1 else p
        s1 = self.s1 if math.isfinite(self.s1) else p
        s2 = self.s2 if math.isfinite(self.s2) else p
        out = self.c1 * (p + p1) / 2 + self.c2 * s1 + self.c3 * s2
        self.s2 = s1
        self.s1 = out
        return out

    def _extra_state(self) -> dict:
        return {"s1": self.s1, "s2": self.s2}


@register("gaussian")
class GaussianFilter(RollingEstimator):
    """Ehlers 2-pole Gaussian filter: smooth, symmetric low-pass response."""

    def _init_state(self) -> None:
        beta = (1 - math.cos(2 * math.pi / self.window)) / (math.sqrt(2) - 1)
        self.alpha = -beta + math.sqrt(beta * beta + 2 * beta)
        self.g1 = float("nan")
        self.g2 = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        a = self.alpha
        g1 = self.g1 if math.isfinite(self.g1) else p
        g2 = self.g2 if math.isfinite(self.g2) else p
        # alpha**2 on the input keeps DC gain at 1 (alpha^2 + 2(1-a) - (1-a)^2 = 1).
        out = a * a * p + 2 * (1 - a) * g1 - (1 - a) ** 2 * g2
        self.g2 = g1
        self.g1 = out
        return out

    def _extra_state(self) -> dict:
        return {"g1": self.g1, "g2": self.g2}


@register("butterworth")
class Butterworth(RollingEstimator):
    """Ehlers 2-pole Butterworth filter: maximally flat passband low-pass."""

    def _init_state(self) -> None:
        w = self.window
        a1 = math.exp(-1.414 * math.pi / w)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / w)
        self.c2 = b1
        self.c3 = -a1 * a1
        self.c1 = (1 - b1 + a1 * a1) / 4
        self.f1 = float("nan")
        self.f2 = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        p1 = self._buf[-2] if len(self._buf) > 1 else p
        p2 = self._buf[-3] if len(self._buf) > 2 else p1
        f1 = self.f1 if math.isfinite(self.f1) else p
        f2 = self.f2 if math.isfinite(self.f2) else p
        out = self.c1 * (p + 2 * p1 + p2) + self.c2 * f1 + self.c3 * f2
        self.f2 = f1
        self.f1 = out
        return out

    def _extra_state(self) -> dict:
        return {"f1": self.f1, "f2": self.f2}


@register("fourier")
class FourierLowpass(RollingEstimator):
    """Fourier low-pass: keep only the lowest-frequency components of the window
    and reconstruct, reporting the smoothed value at the current bar."""

    def _compute_mean(self) -> float:
        a = self._last(self.window)
        k = len(a)
        if k < 4:
            return float(a.mean())
        spectrum = np.fft.rfft(a)
        cutoff = max(1, k // 4)
        spectrum[cutoff:] = 0
        recon = np.fft.irfft(spectrum, n=k)
        return float(recon[-1])
