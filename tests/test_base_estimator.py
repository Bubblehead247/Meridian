"""Tests for the BaseEstimator contract itself.

These do not test any real estimator (that is Phase 2). They confirm the
abstract contract is enforced: you cannot instantiate it directly, and a
subclass that implements every method works as expected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.estimators import BaseEstimator


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        BaseEstimator()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_instantiate():
    class Incomplete(BaseEstimator):
        def fit(self, prices):  # missing the rest of the contract
            pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


class _ConstantMean(BaseEstimator):
    """Minimal reference implementation: fair value is the mean of all prices."""

    def __init__(self):
        self._mean = 0.0
        self._std = 1.0
        self._n = 0

    def fit(self, prices: pd.Series) -> None:
        self._mean = float(prices.mean())
        self._std = float(prices.std(ddof=0)) or 1.0
        self._n = len(prices)

    def update(self, price: float) -> None:
        self._n += 1
        self._mean += (price - self._mean) / self._n

    def predict_mean(self) -> float:
        return self._mean

    def residual(self, price: float) -> float:
        return price - self.predict_mean()

    def zscore(self, price: float) -> float:
        return self.residual(price) / self._std

    def state(self) -> dict:
        return {"mean": self._mean, "std": self._std, "n": self._n}


def test_complete_subclass_satisfies_contract():
    est = _ConstantMean()
    est.fit(pd.Series([10.0, 12.0, 14.0]))

    assert est.predict_mean() == pytest.approx(12.0)
    assert est.residual(15.0) == pytest.approx(3.0)
    assert est.zscore(12.0) == pytest.approx(0.0)
    assert est.state()["n"] == 3

    est.update(18.0)  # online update must move the mean toward the new price
    assert est.predict_mean() > 12.0
