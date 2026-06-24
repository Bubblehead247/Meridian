"""Concrete regime classifiers.

- ``trend``      — trending vs ranging, via the Kaufman efficiency ratio.
- ``volatility`` — low / normal / high, current vol vs a longer baseline.
- ``direction``  — bull vs bear, price above/below a long average.
- ``hurst``      — mean_reverting / random / trending, via the Hurst exponent.

The ``hurst`` classifier is the most directly relevant to this project: a Hurst
exponent below 0.5 indicates a mean-reverting series (the regime where these
strategies should work), 0.5 a random walk, above 0.5 a trending series.
"""

from __future__ import annotations

import numpy as np

from meridian.regimes.base import RollingRegime
from meridian.regimes.registry import register


@register("trend")
class TrendRegime(RollingRegime):
    """Trending vs ranging via the Kaufman efficiency ratio (ER).

    ER = |net move| / |total path length| over the window. Near 1 the market
    moves in a straight line (trend); near 0 it churns sideways (range).
    """

    labels = ("trend", "range")

    def __init__(self, window: int = 20, threshold: float = 0.3):
        self.threshold = float(threshold)
        super().__init__(window)

    def _label(self) -> str:
        a = self._prices(self.window + 1)
        change = abs(a[-1] - a[0])
        path = float(np.abs(np.diff(a)).sum())
        er = change / path if path > 0 else 0.0
        return "trend" if er >= self.threshold else "range"

    def _extra_state(self) -> dict:
        return {"threshold": self.threshold}


@register("volatility")
class VolatilityRegime(RollingRegime):
    """Low / normal / high volatility: recent realized vol vs a longer baseline.

    Compares the std of returns over ``window`` against the std over a longer
    ``ref_window``. Above ``high_mult``x baseline is "high"; below ``low_mult``x
    is "low"; otherwise "normal".
    """

    labels = ("low", "normal", "high")

    def __init__(
        self,
        window: int = 20,
        ref_window: int = 100,
        high_mult: float = 1.5,
        low_mult: float = 0.5,
    ):
        self.ref_window = int(ref_window)
        self.high_mult = float(high_mult)
        self.low_mult = float(low_mult)
        super().__init__(window)

    def _buffer_len(self) -> int:
        return self.ref_window + 1

    def _min_obs(self) -> int:
        return self.window + 1

    def _label(self) -> str:
        rets = self._returns()
        recent = float(np.std(rets[-self.window:], ddof=0))
        baseline = float(np.std(rets, ddof=0))
        if baseline <= 0:
            return "normal"
        ratio = recent / baseline
        if ratio >= self.high_mult:
            return "high"
        if ratio <= self.low_mult:
            return "low"
        return "normal"

    def _extra_state(self) -> dict:
        return {"ref_window": self.ref_window}


@register("direction")
class DirectionRegime(RollingRegime):
    """Bull vs bear: price above or below its long moving average."""

    labels = ("bull", "bear")

    def __init__(self, window: int = 50):
        super().__init__(window)

    def _label(self) -> str:
        a = self._prices(self.window)
        return "bull" if a[-1] >= float(a.mean()) else "bear"


@register("hurst")
class HurstRegime(RollingRegime):
    """Mean-reverting / random / trending via the Hurst exponent.

    The Hurst exponent H summarizes how a series' dispersion grows with the time
    lag: H < 0.5 means past moves tend to reverse (mean reversion), H ~ 0.5 is a
    random walk, H > 0.5 means moves tend to persist (trend).
    """

    labels = ("mean_reverting", "random", "trending")

    def __init__(
        self,
        window: int = 100,
        mr_threshold: float = 0.45,
        trend_threshold: float = 0.55,
    ):
        self.mr_threshold = float(mr_threshold)
        self.trend_threshold = float(trend_threshold)
        super().__init__(window)

    def _min_obs(self) -> int:
        return max(20, self.window)

    def _hurst(self, ts: np.ndarray) -> float:
        max_lag = min(20, len(ts) // 2)
        lags = np.arange(2, max_lag)
        # Dispersion of lag-differences; slope of log-log fit gives H.
        tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
        tau = np.asarray(tau)
        if np.any(tau <= 0):
            return 0.5
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return float(slope * 2.0)

    def _label(self) -> str:
        h = self._hurst(self._prices(self.window))
        if h < self.mr_threshold:
            return "mean_reverting"
        if h > self.trend_threshold:
            return "trending"
        return "random"

    def _extra_state(self) -> dict:
        ready = self.n >= self._min_obs()
        return {"hurst": self._hurst(self._prices(self.window)) if ready else float("nan")}
