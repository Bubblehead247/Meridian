"""Moving-average family of fair-value estimators.

Each defines fair value as some weighted/adaptive average of recent prices.
They differ in how much weight recent vs. older prices get, and whether the
weighting adapts to market conditions. All share the contract via
``RollingEstimator``.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import register


@register("sma")
class SMA(RollingEstimator):
    """Simple moving average: unweighted mean of the last ``window`` prices."""

    def _compute_mean(self) -> float:
        return float(self._last().mean())


@register("ema")
class EMA(RollingEstimator):
    """Exponential moving average: weights decay geometrically into the past."""

    def _init_state(self) -> None:
        self.alpha = 2.0 / (self.window + 1.0)

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        if not math.isfinite(self._mean):
            return p
        return self.alpha * p + (1 - self.alpha) * self._mean


@register("wma")
class WMA(RollingEstimator):
    """Weighted moving average with linearly increasing weights toward now."""

    def _compute_mean(self) -> float:
        a = self._last()
        w = np.arange(1, len(a) + 1, dtype=float)
        return float((a * w).sum() / w.sum())


@register("trima")
class TRIMA(RollingEstimator):
    """Triangular moving average: weights rise then fall (a smoothed SMA)."""

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        w = np.minimum(np.arange(1, k + 1), np.arange(k, 0, -1)).astype(float)
        return float((a * w).sum() / w.sum())


@register("dema")
class DEMA(RollingEstimator):
    """Double EMA: ``2*EMA - EMA(EMA)``, reduces the lag of a plain EMA."""

    def _init_state(self) -> None:
        self.alpha = 2.0 / (self.window + 1.0)
        self.e1 = float("nan")
        self.e2 = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        a = self.alpha
        self.e1 = p if not math.isfinite(self.e1) else a * p + (1 - a) * self.e1
        self.e2 = self.e1 if not math.isfinite(self.e2) else a * self.e1 + (1 - a) * self.e2
        return 2 * self.e1 - self.e2

    def _extra_state(self) -> dict:
        return {"e1": self.e1, "e2": self.e2}


@register("tema")
class TEMA(RollingEstimator):
    """Triple EMA: ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))``, even less lag."""

    def _init_state(self) -> None:
        self.alpha = 2.0 / (self.window + 1.0)
        self.e1 = self.e2 = self.e3 = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        a = self.alpha
        self.e1 = p if not math.isfinite(self.e1) else a * p + (1 - a) * self.e1
        self.e2 = self.e1 if not math.isfinite(self.e2) else a * self.e1 + (1 - a) * self.e2
        self.e3 = self.e2 if not math.isfinite(self.e3) else a * self.e2 + (1 - a) * self.e3
        return 3 * self.e1 - 3 * self.e2 + self.e3

    def _extra_state(self) -> dict:
        return {"e1": self.e1, "e2": self.e2, "e3": self.e3}


@register("zlema")
class ZLEMA(RollingEstimator):
    """Zero-lag EMA: EMA of a de-lagged price (price plus its recent momentum)."""

    def _init_state(self) -> None:
        self.alpha = 2.0 / (self.window + 1.0)
        self.lag = max(1, (self.window - 1) // 2)
        self.z = float("nan")

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        idx = -1 - self.lag
        p_lag = self._buf[idx] if len(self._buf) > self.lag else self._buf[0]
        de_lagged = 2 * p - p_lag
        self.z = de_lagged if not math.isfinite(self.z) else (
            self.alpha * de_lagged + (1 - self.alpha) * self.z
        )
        return self.z

    def _extra_state(self) -> dict:
        return {"z": self.z}


@register("kama")
class KAMA(RollingEstimator):
    """Kaufman adaptive MA: smooths faster in trends, slower in noise.

    Speed is set by the efficiency ratio — net move divided by total path
    length over the window.
    """

    def _init_state(self) -> None:
        self._fast = 2.0 / (2 + 1)
        self._slow = 2.0 / (30 + 1)

    def _compute_mean(self) -> float:
        a = self._last(self.window + 1)
        if len(a) < 2:
            return float(a[-1])
        change = abs(a[-1] - a[0])
        vol = float(np.abs(np.diff(a)).sum())
        er = change / vol if vol > 0 else 0.0
        sc = (er * (self._fast - self._slow) + self._slow) ** 2
        prev = self._mean if math.isfinite(self._mean) else float(a[0])
        return prev + sc * (a[-1] - prev)


@register("t3")
class T3(RollingEstimator):
    """Tillson T3: six cascaded EMAs blended for a smooth, low-lag average."""

    def _init_state(self) -> None:
        self.alpha = 2.0 / (self.window + 1.0)
        self.e = [float("nan")] * 6
        v = 0.7
        self.c1 = -(v**3)
        self.c2 = 3 * v**2 + 3 * v**3
        self.c3 = -6 * v**2 - 3 * v - 3 * v**3
        self.c4 = 1 + 3 * v + v**3 + 3 * v**2

    def _compute_mean(self) -> float:
        a = self.alpha
        x = self._buf[-1]
        for i in range(6):
            prev = self.e[i]
            self.e[i] = x if not math.isfinite(prev) else a * x + (1 - a) * prev
            x = self.e[i]
        return self.c1 * self.e[5] + self.c2 * self.e[4] + self.c3 * self.e[3] + self.c4 * self.e[2]

    def _extra_state(self) -> dict:
        return {f"e{i+1}": self.e[i] for i in range(6)}


@register("alma")
class ALMA(RollingEstimator):
    """Arnaud Legoux MA: Gaussian window weights, offset toward recent prices."""

    def _init_state(self) -> None:
        self._offset = 0.85
        self._sigma = 6.0

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        if k == 1:
            return float(a[-1])
        m = self._offset * (k - 1)
        s = k / self._sigma
        i = np.arange(k)
        w = np.exp(-((i - m) ** 2) / (2 * s * s))
        return float((a * w).sum() / w.sum())


@register("frama")
class FRAMA(RollingEstimator):
    """Fractal adaptive MA: adapts smoothing to the price's fractal dimension."""

    def _compute_mean(self) -> float:
        a = self._last(self.window)
        k = len(a)
        if k < 4:
            return float(a.mean())
        half = k // 2
        first, second = a[:half], a[half : 2 * half]
        n1 = (first.max() - first.min()) / half
        n2 = (second.max() - second.min()) / half
        n3 = (a.max() - a.min()) / (2 * half)
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
        else:
            d = 1.0
        alpha = float(np.clip(math.exp(-4.6 * (d - 1)), 0.01, 1.0))
        prev = self._mean if math.isfinite(self._mean) else float(a[0])
        return prev + alpha * (a[-1] - prev)


@register("vidya")
class VIDYA(RollingEstimator):
    """Chande's variable index dynamic average: speed scales with momentum."""

    def _compute_mean(self) -> float:
        a = self._last(self.window + 1)
        if len(a) < 2:
            return float(a[-1])
        diffs = np.diff(a)
        up = diffs[diffs > 0].sum()
        down = -diffs[diffs < 0].sum()
        cmo = abs((up - down) / (up + down)) if (up + down) > 0 else 0.0
        alpha = (2.0 / (self.window + 1)) * cmo
        prev = self._mean if math.isfinite(self._mean) else float(a[0])
        return prev + alpha * (a[-1] - prev)


@register("mcginley")
class McGinley(RollingEstimator):
    """McGinley dynamic: a self-adjusting average that tracks faster moves."""

    def _compute_mean(self) -> float:
        p = self._buf[-1]
        if not math.isfinite(self._mean) or self._mean == 0:
            return p
        prev = self._mean
        denom = self.window * (p / prev) ** 4
        return prev + (p - prev) / denom if denom != 0 else p


@register("sine_wma")
class SineWMA(RollingEstimator):
    """Sine-weighted MA: weights follow a sine curve, peaking mid-window."""

    def _compute_mean(self) -> float:
        a = self._last()
        k = len(a)
        w = np.sin(np.pi * np.arange(1, k + 1) / (k + 1))
        return float((a * w).sum() / w.sum())


@register("geometric")
class GeometricMA(RollingEstimator):
    """Geometric mean of recent prices (falls back to arithmetic if any <= 0)."""

    def _compute_mean(self) -> float:
        a = self._last()
        if np.any(a <= 0):
            return float(a.mean())
        return float(np.exp(np.log(a).mean()))


@register("harmonic")
class HarmonicMA(RollingEstimator):
    """Harmonic mean of recent prices (falls back to arithmetic if any <= 0)."""

    def _compute_mean(self) -> float:
        a = self._last()
        if np.any(a <= 0):
            return float(a.mean())
        return float(len(a) / (1.0 / a).sum())


@register("hull")
class HullMA(RollingEstimator):
    """Hull moving average: a near-lagless, smooth weighted average.

    HMA = WMA over sqrt(n) of (2*WMA(n/2) - WMA(n)). The inner term is tracked
    in a small buffer so only the endpoint is needed.
    """

    def _init_state(self) -> None:
        self._sqrtn = max(1, int(round(math.sqrt(self.window))))
        self._raw: deque[float] = deque(maxlen=self._sqrtn)

    @staticmethod
    def _wma(a: np.ndarray) -> float:
        w = np.arange(1, len(a) + 1, dtype=float)
        return float((a * w).sum() / w.sum())

    def _compute_mean(self) -> float:
        a = self._last(self.window)
        half = max(1, self.window // 2)
        raw = 2 * self._wma(a[-half:]) - self._wma(a)
        self._raw.append(raw)
        return self._wma(np.fromiter(self._raw, dtype=float))
