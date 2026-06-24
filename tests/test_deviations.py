"""Phase 3 tests for deviation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.deviations import create, list_deviations

ALL = list_deviations()
RESIDUAL_ONLY = [n for n in ALL if n != "atr_norm"]


def _residuals(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 2, n))


def _bars(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 1, n))
    low = close - np.abs(rng.normal(0, 1, n))
    return pd.DataFrame({"high": high, "low": low, "close": close})


# --- library-wide ---------------------------------------------------------

def test_expected_metrics_registered():
    assert set(ALL) == {"zscore", "mad_z", "modified_z", "percentile", "minmax", "atr_norm"}


# --- contract sweep -------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_contract(name):
    res = _residuals()
    bars = _bars()
    dev = create(name, window=20)
    dev.fit(res, bars=bars)

    v = dev.value(float(res.iloc[-1]))
    assert np.isfinite(v)

    state = dev.state()
    assert state["deviation"] == name
    assert state["window"] == 20
    assert state["n"] == len(res)


@pytest.mark.parametrize("name", ALL)
def test_warmup_returns_nan(name):
    dev = create(name, window=20)
    # Single observation: scale/rank undefined -> NaN.
    dev.update(1.0, bar={"high": 2.0, "low": 1.0, "close": 1.5})
    assert np.isnan(dev.value(1.0))


@pytest.mark.parametrize("name", ALL)
def test_update_equivalent_to_fit(name):
    res = _residuals()
    bars = _bars()

    batch = create(name, window=20)
    batch.fit(res, bars=bars)

    incr = create(name, window=20)
    incr.fit(res.iloc[:-1], bars=bars.iloc[:-1])
    incr.update(float(res.iloc[-1]), bar=bars.iloc[-1])

    test_r = 3.14
    assert incr.value(test_r) == pytest.approx(batch.value(test_r), nan_ok=True)


@pytest.mark.parametrize("name", ALL)
def test_window_validation(name):
    with pytest.raises(ValueError):
        create(name, window=1)


# --- exact-math targeted --------------------------------------------------

def test_zscore_exact():
    dev = create("zscore", window=5)
    dev.fit(pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0]))  # mean 0, std sqrt(2)
    assert dev.value(2.0) == pytest.approx(2.0 / np.sqrt(2.0))


def test_percentile_bounds_and_center():
    dev = create("percentile", window=5)
    dev.fit(pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert dev.value(2.0) == pytest.approx(1.0)   # >= all -> +1
    assert dev.value(-2.0) == pytest.approx(2 * 0.2 - 1)  # only itself <= -2


def test_minmax_endpoints():
    dev = create("minmax", window=4)
    dev.fit(pd.Series([0.0, 5.0, 10.0, 2.0]))  # min 0, max 10
    assert dev.value(10.0) == pytest.approx(1.0)
    assert dev.value(0.0) == pytest.approx(-1.0)
    assert dev.value(5.0) == pytest.approx(0.0)


def test_mad_z_resists_outlier_vs_zscore():
    # One huge residual inflates std but barely moves MAD.
    res = pd.Series([-1.0, 0.0, 1.0, 0.5, -0.5, 100.0])
    z = create("zscore", window=6)
    m = create("mad_z", window=6)
    z.fit(res)
    m.fit(res)
    # For a moderate deviation, the robust metric reports a larger (truer) score.
    assert abs(m.value(2.0)) > abs(z.value(2.0))


def test_atr_norm_equals_residual_over_atr():
    bars = pd.DataFrame(
        {"high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0], "close": [10.0, 11.0, 12.0]}
    )
    res = pd.Series([0.0, 0.0, 0.0])
    dev = create("atr_norm", window=3)
    dev.fit(res, bars=bars)
    atr = dev.state()["atr"]
    assert dev.value(4.0) == pytest.approx(4.0 / atr)


def test_atr_norm_without_bars_is_nan():
    dev = create("atr_norm", window=5)
    dev.fit(_residuals())  # no bars
    assert np.isnan(dev.value(3.0))


# --- integration with an estimator ----------------------------------------

def test_pairs_with_any_estimator():
    """Deviation metrics consume residuals from any estimator (decoupling)."""
    from meridian.estimators import create as make_est

    rng = np.random.default_rng(3)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))

    for est_name in ["sma", "ema", "kalman", "ou"]:
        est = make_est(est_name, window=20)
        est.fit(prices)
        residuals = pd.Series([est.residual(p) for p in prices])  # crude but valid
        dev = create("modified_z", window=20)
        dev.fit(residuals)
        assert np.isfinite(dev.value(float(residuals.iloc[-1])))


# --- property-based -------------------------------------------------------

@given(name=st.sampled_from(["percentile", "minmax"]), r=st.floats(-50, 50))
@settings(max_examples=80, deadline=None)
def test_bounded_metrics_stay_in_range(name, r):
    dev = create(name, window=20)
    dev.fit(_residuals(seed=5))
    v = dev.value(r)
    assert -1.0 <= v <= 1.0


@given(r=st.floats(min_value=-50, max_value=50))
@settings(max_examples=60, deadline=None)
def test_zscore_sign_matches_centered_residual(r):
    dev = create("zscore", window=20)
    res = _residuals(seed=6)
    dev.fit(res)
    v = dev.value(r)
    centered = r - float(res.iloc[-20:].mean())
    if np.isfinite(v) and abs(centered) > 1e-9:
        assert np.sign(v) == np.sign(centered)
