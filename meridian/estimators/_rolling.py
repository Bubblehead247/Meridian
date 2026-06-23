"""Shared template base for fair-value estimators.

Every estimator answers the same question — "given the price history, what is
the current fair value the price reverts to?" — and differs only in *how* it
computes that central value. `RollingEstimator` implements the whole
`BaseEstimator` contract once (the fit/update loop, residual, z-score, state)
so each concrete estimator only fills in `_compute_mean()`.

How the scaffolding works:

- A bounded buffer of recent prices is kept. On each new price we compute the
  fair value and record the residual (price - fair value).
- `predict_scale()` is the standard deviation of recent residuals — the typical
  size of a deviation — and the z-score is residual / scale. This makes
  deviations comparable across symbols and across estimators, exactly as the
  contract intends.
- Because `update(price)` runs the identical step `fit` runs per bar, an
  incremental update is provably equivalent to having included that price in
  `fit` — a property the tests check.

Subclasses that are *recursive* (e.g. an exponential average) read the previous
fair value from `self._mean` and the latest price from `self._buf[-1]` inside
`_compute_mean()`. Subclasses that are *windowed* read `self._last()`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from meridian.estimators.base import BaseEstimator

# Below this many residuals the scale (and thus z-score) is undefined.
_MIN_RESID = 2


class RollingEstimator(BaseEstimator):
    """Template base implementing the full estimator contract.

    Args:
        window: Lookback length. Windowed estimators average over it; recursive
            estimators derive their smoothing constant from it.
    """

    #: Registry name, set by the @register decorator.
    name: str | None = None

    def __init__(self, window: int = 20):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = int(window)
        self._buf: deque[float] = deque(maxlen=self._buffer_len())
        self._resid: deque[float] = deque(maxlen=self.window)
        self._mean: float = float("nan")
        self.n: int = 0
        self._init_state()

    # --- hooks for subclasses ---------------------------------------------

    def _buffer_len(self) -> int:
        """Number of recent prices to retain. Override if more are needed."""
        return self.window + 1

    def _init_state(self) -> None:
        """Initialize any recursive state. Called on construction and `fit`."""

    def _compute_mean(self) -> float:
        """Return the current fair value from the buffer / recursive state."""
        raise NotImplementedError

    def _extra_state(self) -> dict:
        """Extra fields to expose in `state()`."""
        return {}

    # --- contract ---------------------------------------------------------

    def fit(self, prices: pd.Series) -> None:
        arr = np.asarray(getattr(prices, "to_numpy", lambda: prices)(), dtype=float)
        self._buf.clear()
        self._resid.clear()
        self._mean = float("nan")
        self.n = 0
        self._init_state()
        for p in arr:
            self._ingest(float(p))

    def update(self, price: float) -> None:
        self._ingest(float(price))

    def _ingest(self, price: float) -> None:
        self._buf.append(price)
        self.n += 1
        m = self._compute_mean()
        self._mean = float(m)
        if np.isfinite(m):
            self._resid.append(price - m)

    def predict_mean(self) -> float:
        return self._mean

    def residual(self, price: float) -> float:
        return float(price) - self._mean

    def predict_scale(self) -> float:
        """Typical deviation size: std of recent residuals (NaN until warmed)."""
        if len(self._resid) < _MIN_RESID:
            return float("nan")
        s = float(np.std(self._resid, ddof=0))
        return s if s > 0 else float("nan")

    def zscore(self, price: float) -> float:
        s = self.predict_scale()
        if not np.isfinite(s):
            return float("nan")
        return (float(price) - self._mean) / s

    def state(self) -> dict:
        st = {
            "estimator": self.name or type(self).__name__,
            "window": self.window,
            "n": self.n,
            "mean": self._mean,
            "scale": self.predict_scale(),
        }
        st.update(self._extra_state())
        return st

    # --- helpers for subclasses -------------------------------------------

    def _last(self, k: int | None = None) -> np.ndarray:
        """The most recent ``k`` prices (default: ``window``) as an array."""
        a = np.fromiter(self._buf, dtype=float)
        k = self.window if k is None else k
        return a[-k:]

    @property
    def _prev_mean(self) -> float:
        """Fair value from the previous bar (NaN on the first bar)."""
        return self._mean
