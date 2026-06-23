"""Regime-classifier contract and rolling template.

A *regime classifier* labels the current market state — trending vs ranging,
high vs low volatility, bull vs bear, mean-reverting vs random. Mean-reversion
strategies behave very differently across these states, so regime labels are
used **downstream** to gate signals (e.g. only trade mean reversion when the
market is ranging / mean-reverting).

Per CLAUDE.md, regime logic is deliberately **separate from the estimator
interface**: classifiers do not implement `BaseEstimator`, and the signal engine
never sees them. A regime is applied as a filter on positions after the fact
(see `meridian.regimes.filter`).

As elsewhere, one template (`RollingRegime`) implements the contract once; each
classifier supplies only `_label()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Mapping

import numpy as np
import pandas as pd

#: Label returned before enough data has accumulated to classify.
UNKNOWN = "unknown"


class BaseRegime(ABC):
    """Contract every regime classifier implements."""

    name: str | None = None
    #: The full set of labels this classifier can emit (excluding UNKNOWN).
    labels: tuple[str, ...] = ()

    @abstractmethod
    def fit(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> None:
        """Warm up on price history (and optional OHLC bars)."""

    @abstractmethod
    def update(self, price: float, bar: Mapping | None = None) -> None:
        """Incrementally ingest one new price (and optional OHLC bar)."""

    @abstractmethod
    def label(self) -> str:
        """Return the current regime label (or ``UNKNOWN`` during warmup)."""

    @abstractmethod
    def state(self) -> dict:
        """Return full internal state for reproducibility."""


class RollingRegime(BaseRegime):
    """Template base: maintains a rolling window of prices.

    Subclasses implement ``_label()`` using ``self._prices()`` / ``self._returns()``.

    Args:
        window: Lookback length for the classification statistic.
    """

    def __init__(self, window: int = 50):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = int(window)
        self._buf: deque[float] = deque(maxlen=self._buffer_len())
        self.n = 0
        self._init_state()

    # --- hooks ------------------------------------------------------------

    def _buffer_len(self) -> int:
        return self.window + 1

    def _min_obs(self) -> int:
        """Observations required before a non-UNKNOWN label is emitted."""
        return self.window

    def _init_state(self) -> None:
        """Initialize extra state. Called on construction and `fit`."""

    def _post_update(self, bar: Mapping | None) -> None:
        """Hook after each price is ingested (for classifiers that use bars)."""

    def _label(self) -> str:
        raise NotImplementedError

    def _extra_state(self) -> dict:
        return {}

    # --- contract ---------------------------------------------------------

    def fit(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> None:
        arr = np.asarray(getattr(prices, "to_numpy", lambda: prices)(), dtype=float)
        self._buf.clear()
        self.n = 0
        self._init_state()
        for i, p in enumerate(arr):
            bar = bars.iloc[i] if bars is not None else None
            self.update(float(p), bar=bar)

    def update(self, price: float, bar: Mapping | None = None) -> None:
        self._buf.append(float(price))
        self.n += 1
        self._post_update(bar)

    def label(self) -> str:
        if self.n < self._min_obs():
            return UNKNOWN
        return self._label()

    def state(self) -> dict:
        st = {
            "regime": self.name or type(self).__name__,
            "window": self.window,
            "n": self.n,
            "label": self.label(),
        }
        st.update(self._extra_state())
        return st

    # --- helpers ----------------------------------------------------------

    def _prices(self, k: int | None = None) -> np.ndarray:
        a = np.fromiter(self._buf, dtype=float)
        return a if k is None else a[-k:]

    def _returns(self) -> np.ndarray:
        a = self._prices()
        if len(a) < 2:
            return np.array([])
        return np.diff(a) / a[:-1]
