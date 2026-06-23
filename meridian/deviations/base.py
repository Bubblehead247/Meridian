"""Deviation-metric contract and shared rolling template.

A *deviation metric* answers: "how far is the current price from fair value, in
normalized units?" The raw distance is the estimator's residual
(``price - predict_mean``). A deviation metric standardizes that residual so it
is comparable across symbols, estimators, and time — e.g. a z-score, a robust
(median/MAD) z-score, a percentile rank, or an ATR-normalized distance.

Design principle (from CLAUDE.md): deviation metrics are **independent of the
estimator**. A metric consumes only a stream of residuals (plus, for
ATR-style metrics, the raw OHLC bars). Any metric therefore pairs with any
estimator. The estimator's own ``zscore`` is just the simplest case; this module
generalizes it and adds robust and rank-based alternatives.

As with estimators, one template (`RollingDeviation`) implements the whole
contract once; each concrete metric supplies only ``_value()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Mapping

import numpy as np
import pandas as pd

# Below this many residuals a metric's scale/rank is undefined.
_MIN_OBS = 2


class BaseDeviation(ABC):
    """Contract every deviation metric implements."""

    name: str | None = None

    @abstractmethod
    def fit(self, residuals: pd.Series, bars: pd.DataFrame | None = None) -> None:
        """Warm up on a history of residuals (and optional OHLC bars)."""

    @abstractmethod
    def update(self, residual: float, bar: Mapping | None = None) -> None:
        """Incrementally ingest one new residual (and optional OHLC bar)."""

    @abstractmethod
    def value(self, residual: float) -> float:
        """Return the normalized deviation for ``residual`` given current state."""

    @abstractmethod
    def state(self) -> dict:
        """Return full internal state for reproducibility."""


class RollingDeviation(BaseDeviation):
    """Template base: maintains a rolling window of residuals.

    Subclasses implement ``_value(r)`` using ``self._arr`` (the residual window).
    ATR-style metrics also override ``_post_update`` to track price bars.

    Args:
        window: Lookback length for the normalization statistics.
    """

    def __init__(self, window: int = 20):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = int(window)
        self._res: deque[float] = deque(maxlen=self.window)
        self.n = 0
        self._init_state()

    # --- hooks ------------------------------------------------------------

    def _init_state(self) -> None:
        """Initialize extra state (e.g. ATR buffers). Called on construct/fit."""

    def _post_update(self, bar: Mapping | None) -> None:
        """Hook after each residual is ingested; ATR metrics use the bar."""

    def _value(self, r: float) -> float:
        raise NotImplementedError

    def _extra_state(self) -> dict:
        return {}

    # --- contract ---------------------------------------------------------

    def fit(self, residuals: pd.Series, bars: pd.DataFrame | None = None) -> None:
        arr = np.asarray(getattr(residuals, "to_numpy", lambda: residuals)(), dtype=float)
        self._res.clear()
        self.n = 0
        self._init_state()
        for i, r in enumerate(arr):
            bar = bars.iloc[i] if bars is not None else None
            self.update(float(r), bar=bar)

    def update(self, residual: float, bar: Mapping | None = None) -> None:
        self._res.append(float(residual))
        self.n += 1
        self._post_update(bar)

    def value(self, residual: float) -> float:
        if len(self._res) < _MIN_OBS:
            return float("nan")
        return self._value(float(residual))

    def state(self) -> dict:
        st = {"deviation": self.name or type(self).__name__, "window": self.window, "n": self.n}
        st.update(self._extra_state())
        return st

    # --- helpers ----------------------------------------------------------

    @property
    def _arr(self) -> np.ndarray:
        return np.fromiter(self._res, dtype=float)
