"""Phase 2 tests for the estimator library.

The parametrized sweep enforces the BaseEstimator contract for *every*
registered estimator, satisfying the standard that each estimator is tested on
fit/update/predict_mean/residual/zscore/state. Targeted tests pin exact math
for a few, and hypothesis checks general properties.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.estimators import all_estimators, create, list_estimators

warnings.filterwarnings("ignore")

ALL_NAMES = list_estimators()


def _series(n: int = 300, seed: int = 0) -> pd.Series:
    """A realistic positive, mean-reverting-ish price path."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, n)
    level = 100 + np.cumsum(steps) * 0.5
    noise = rng.normal(0, 1, n)
    return pd.Series(np.maximum(level + noise, 1.0))


# --- library-wide sanity --------------------------------------------------

def test_library_has_at_least_37_estimators():
    assert len(ALL_NAMES) >= 37


def test_names_are_unique_and_lowercase():
    assert ALL_NAMES == sorted(set(ALL_NAMES))
    assert all(name == name.lower() for name in ALL_NAMES)


# --- full contract sweep over every estimator -----------------------------

@pytest.mark.parametrize("name", ALL_NAMES)
def test_contract(name):
    prices = _series()
    est = create(name, window=20)

    est.fit(prices)

    # predict_mean is finite and within a sane band of recent prices.
    mean = est.predict_mean()
    assert np.isfinite(mean)
    recent = prices.iloc[-20:]
    assert recent.min() - 5 * recent.std() <= mean <= recent.max() + 5 * recent.std()

    # residual is exactly price - predict_mean.
    assert est.residual(123.0) == pytest.approx(123.0 - mean)

    # zscore is consistent with residual / scale when scale is defined.
    scale = est.predict_scale()
    z = est.zscore(123.0)
    if np.isfinite(scale):
        assert z == pytest.approx((123.0 - mean) / scale)
    else:
        assert np.isnan(z)

    # state round-trips the essentials.
    state = est.state()
    assert state["estimator"] == name
    assert state["window"] == 20
    assert state["n"] == len(prices)
    assert state["mean"] == pytest.approx(mean, nan_ok=True)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_update_equivalent_to_fit(name):
    """update(last) after fit(all[:-1]) must match fit(all)."""
    prices = _series()

    batch = create(name, window=20)
    batch.fit(prices)

    incr = create(name, window=20)
    incr.fit(prices.iloc[:-1])
    incr.update(float(prices.iloc[-1]))

    assert incr.predict_mean() == pytest.approx(batch.predict_mean(), rel=1e-9, abs=1e-9)
    assert incr.n == batch.n


@pytest.mark.parametrize("name", ALL_NAMES)
def test_window_validation(name):
    with pytest.raises(ValueError):
        create(name, window=1)


# --- exact-value targeted tests -------------------------------------------

def test_sma_exact():
    est = create("sma", window=3)
    est.fit(pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]))
    assert est.predict_mean() == pytest.approx((30 + 40 + 50) / 3)


def test_ema_recurrence():
    est = create("ema", window=4)  # alpha = 2/5 = 0.4
    est.fit(pd.Series([10.0, 20.0, 30.0]))
    # seed=10; then 0.4*20+0.6*10=14; then 0.4*30+0.6*14=20.4
    assert est.predict_mean() == pytest.approx(20.4)


def test_lsma_on_perfect_line_is_exact():
    # On a straight line, the least-squares endpoint equals the line's value.
    est = create("lsma", window=10)
    est.fit(pd.Series(np.arange(50, dtype=float) * 2.0 + 5.0))
    assert est.predict_mean() == pytest.approx(49 * 2.0 + 5.0)


def test_median_resists_outlier():
    est = create("median", window=5)
    est.fit(pd.Series([10.0, 11.0, 12.0, 13.0, 1000.0]))
    assert est.predict_mean() == pytest.approx(12.0)  # spike ignored


def test_midrange_is_high_low_center():
    est = create("midrange", window=4)
    est.fit(pd.Series([10.0, 5.0, 20.0, 15.0]))
    assert est.predict_mean() == pytest.approx((20 + 5) / 2)


# --- property-based tests -------------------------------------------------

@given(
    name=st.sampled_from(ALL_NAMES),
    price=st.floats(min_value=1.0, max_value=1000.0),
)
@settings(max_examples=60, deadline=None)
def test_residual_identity_property(name, price):
    """residual(p) == p - predict_mean() for any estimator and price."""
    est = create(name, window=15)
    est.fit(_series(120, seed=1))
    assert est.residual(price) == pytest.approx(price - est.predict_mean())


@given(price=st.floats(min_value=1.0, max_value=500.0))
@settings(max_examples=50, deadline=None)
def test_zscore_sign_matches_residual(price):
    est = create("sma", window=20)
    est.fit(_series(200, seed=2))
    z = est.zscore(price)
    r = est.residual(price)
    if np.isfinite(z) and abs(r) > 1e-9:
        assert np.sign(z) == np.sign(r)


def test_zscore_mean_near_zero_on_stationary_noise():
    """Over white noise, in-sample z-scores should average near zero."""
    rng = np.random.default_rng(7)
    prices = pd.Series(100 + rng.normal(0, 1, 2000))
    est = create("sma", window=20)
    zs = []
    est.fit(prices.iloc[:50])
    for p in prices.iloc[50:]:
        z = est.zscore(float(p))
        if np.isfinite(z):
            zs.append(z)
        est.update(float(p))
    assert abs(np.mean(zs)) < 0.15
