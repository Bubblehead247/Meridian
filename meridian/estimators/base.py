"""Base estimator contract for all fair-value estimators.

Every estimator in Meridian — all 37+ of them — implements this exact API.
The signal engine, deviation metrics, and backtester all depend only on this
interface, never on a concrete estimator. This keeps the comparison fair: only
the estimator varies, never the machinery around it.

Regime classification is intentionally NOT part of this contract. Regime logic
lives in `meridian.regimes` and is applied downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseEstimator(ABC):
    """Abstract contract every fair-value estimator must satisfy.

    A "fair-value estimator" is anything that, given a price history, produces
    a current estimate of where the price *should* be (the mean it reverts to).
    Subclasses fill in the math; the surrounding pipeline only ever calls the
    methods defined here.
    """

    @abstractmethod
    def fit(self, prices: pd.Series) -> None:
        """Fit the estimator to a full history of prices.

        Args:
            prices: Time-indexed series of historical prices.
        """

    @abstractmethod
    def update(self, price: float) -> None:
        """Incrementally update internal state with one new price.

        This must be equivalent to having included the price in `fit`, so the
        estimator can run online (bar by bar) without refitting from scratch.
        """

    @abstractmethod
    def predict_mean(self) -> float:
        """Return the current fair-value estimate."""

    @abstractmethod
    def residual(self, price: float) -> float:
        """Raw difference between an observed price and the fair value.

        Defined as ``price - predict_mean()``.
        """

    @abstractmethod
    def zscore(self, price: float) -> float:
        """Standardized deviation of a price from fair value.

        The z-score expresses the residual in units of its own typical size
        (standard deviations), so deviations are comparable across symbols and
        across estimators.
        """

    @abstractmethod
    def state(self) -> dict:
        """Return the full internal state as a plain dict.

        Used for reproducibility: two runs from the same config and data must
        produce identical state.
        """
