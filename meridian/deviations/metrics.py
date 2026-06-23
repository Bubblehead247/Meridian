"""Concrete deviation metrics.

Each standardizes a residual (price - fair value) differently:

- ``zscore``      — classic: (r - mean) / std of recent residuals.
- ``mad_z``       — robust scale: r / (1.4826 * MAD); resists outliers.
- ``modified_z``  — Iglewicz-Hoaglin: median-centered robust z-score.
- ``percentile``  — rank of r in recent residuals, mapped to [-1, 1].
- ``minmax``      — r placed in the recent residual range, mapped to [-1, 1].
- ``atr_norm``    — r / ATR; distance in units of typical price range (needs OHLC).

The constant 1.4826 makes the MAD a consistent estimator of the standard
deviation for normally distributed data, so robust and classic z-scores are on
the same scale.
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from meridian.deviations.base import RollingDeviation
from meridian.deviations.registry import register

_MAD_TO_STD = 1.4826


@register("zscore")
class ZScore(RollingDeviation):
    """Standard score: (r - mean) / std over the residual window."""

    def _value(self, r: float) -> float:
        a = self._arr
        s = float(a.std(ddof=0))
        if s <= 0:
            return float("nan")
        return (r - float(a.mean())) / s


@register("mad_z")
class MADZScore(RollingDeviation):
    """Robust z-score by scale only: r / (1.4826 * MAD).

    MAD (median absolute deviation) makes the scale insensitive to a few extreme
    residuals, unlike the standard deviation.
    """

    def _value(self, r: float) -> float:
        a = self._arr
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = _MAD_TO_STD * mad
        if scale <= 0:
            return float("nan")
        return r / scale


@register("modified_z")
class ModifiedZScore(RollingDeviation):
    """Iglewicz-Hoaglin modified z-score: (r - median) / (1.4826 * MAD).

    Like ``mad_z`` but also re-centers on the median, correcting for residuals
    that are not centered on zero (e.g. from a biased smoother).
    """

    def _value(self, r: float) -> float:
        a = self._arr
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = _MAD_TO_STD * mad
        if scale <= 0:
            return float("nan")
        return (r - med) / scale


@register("percentile")
class PercentileRank(RollingDeviation):
    """Empirical percentile rank of r among recent residuals, mapped to [-1, 1].

    -1 means "more negative than anything recent", +1 "more positive than
    anything recent", 0 the median. Distribution-free: no scale assumption.
    """

    def _value(self, r: float) -> float:
        a = self._arr
        frac = float(np.mean(a <= r))
        return 2.0 * frac - 1.0


@register("minmax")
class MinMaxNorm(RollingDeviation):
    """Residual placed in its recent [min, max] range, mapped to [-1, 1]."""

    def _value(self, r: float) -> float:
        a = self._arr
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return float("nan")
        v = 2.0 * (r - lo) / (hi - lo) - 1.0
        return float(np.clip(v, -1.0, 1.0))


@register("atr_norm")
class ATRNormalized(RollingDeviation):
    """Distance normalized by Average True Range: r / ATR.

    ATR measures the typical bar-to-bar price range, so this expresses the
    deviation in units a trader feels directly. **Requires OHLC bars**: pass
    ``bars`` to ``fit`` / ``bar`` to ``update`` (dict-like with high/low/close).
    Without bars the ATR is undefined and ``value`` returns NaN.
    """

    def _init_state(self) -> None:
        self._tr: deque[float] = deque(maxlen=self.window)
        self._prev_close: float = float("nan")

    def _post_update(self, bar: Mapping | None) -> None:
        if bar is None:
            return
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        if np.isfinite(self._prev_close):
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr.append(tr)
        self._prev_close = close

    def _value(self, r: float) -> float:
        if not self._tr:
            return float("nan")
        atr = float(np.mean(self._tr))
        if atr <= 0:
            return float("nan")
        return r / atr

    def _extra_state(self) -> dict:
        atr = float(np.mean(self._tr)) if self._tr else float("nan")
        return {"atr": atr, "prev_close": self._prev_close}
