"""Phase 5 tests: regime classifiers and the downstream gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.regimes import (
    UNKNOWN,
    classify,
    create,
    gate_positions,
    list_regimes,
    run_gated_backtest,
)
from meridian.signals import SignalConfig

ALL = list_regimes()


def _series(n=400, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))


# --- library-wide ---------------------------------------------------------

def test_expected_regimes_registered():
    assert set(ALL) == {"trend", "volatility", "direction", "hurst"}


# --- contract sweep -------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_contract(name):
    clf = create(name)
    clf.fit(_series())
    lbl = clf.label()
    assert lbl in clf.labels  # a real label after warmup
    st = clf.state()
    assert st["regime"] == name
    assert st["label"] == lbl


@pytest.mark.parametrize("name", ALL)
def test_warmup_is_unknown(name):
    clf = create(name)
    clf.update(100.0)
    assert clf.label() == UNKNOWN


@pytest.mark.parametrize("name", ALL)
def test_update_equivalent_to_fit(name):
    prices = _series()
    batch = create(name)
    batch.fit(prices)
    incr = create(name)
    incr.fit(prices.iloc[:-1])
    incr.update(float(prices.iloc[-1]))
    assert incr.label() == batch.label()


@pytest.mark.parametrize("name", ALL)
def test_window_validation(name):
    with pytest.raises(ValueError):
        create(name, window=1)


# --- direction ------------------------------------------------------------

def test_direction_bull_and_bear():
    up = pd.Series(np.arange(100, 200, dtype=float))
    down = pd.Series(np.arange(200, 100, -1, dtype=float))
    assert create("direction", window=50).__class__ and _final_label("direction", up) == "bull"
    assert _final_label("direction", down) == "bear"


# --- trend ----------------------------------------------------------------

def test_trend_detects_strong_trend_and_range():
    rng = np.random.default_rng(2)
    strong = pd.Series(100 + np.arange(200) * 1.0 + rng.normal(0, 0.1, 200))
    choppy = pd.Series(100 + rng.normal(0, 1, 200))  # no net drift
    assert _final_label("trend", strong, window=20) == "trend"
    assert _final_label("trend", choppy, window=20) == "range"


# --- volatility -----------------------------------------------------------

def test_volatility_high_and_low():
    rng = np.random.default_rng(3)
    # calm baseline then a 10x volatility burst at the end. The burst is kept
    # short vs ref_window (100) so the baseline stays calm and the ratio spikes.
    calm = rng.normal(0, 0.005, 300)
    burst = rng.normal(0, 0.05, 30)
    rets = np.concatenate([calm, burst])
    prices = pd.Series(100 * np.cumprod(1 + rets))
    assert _final_label("volatility", prices) == "high"

    # a quiet tail after a noisier history -> low
    rets2 = np.concatenate([rng.normal(0, 0.05, 300), rng.normal(0, 0.004, 60)])
    prices2 = pd.Series(100 * np.cumprod(1 + rets2))
    assert _final_label("volatility", prices2) == "low"


# --- hurst ----------------------------------------------------------------

def test_hurst_orders_mean_reversion_below_trend():
    rng = np.random.default_rng(4)
    mr = np.zeros(400)
    for i in range(1, 400):
        mr[i] = -0.4 * mr[i - 1] + rng.normal(0, 1)
    mr = pd.Series(100 + mr * 3)
    persist = pd.Series(100 + np.cumsum(np.cumsum(rng.normal(0, 1, 400))) * 0.01)

    h_mr = create("hurst", window=150)
    h_mr.fit(mr)
    h_tr = create("hurst", window=150)
    h_tr.fit(persist)

    assert h_mr.state()["hurst"] < h_tr.state()["hurst"]
    assert h_mr.label() == "mean_reverting"
    assert h_tr.label() == "trending"


# --- downstream gate ------------------------------------------------------

def test_classify_is_causal_and_aligned():
    prices = _series()
    labels = classify(prices, "trend", window=20)
    assert len(labels) == len(prices)
    assert labels.index.equals(prices.index)
    assert labels.iloc[0] == UNKNOWN  # warmup


def test_gate_positions_flattens_disallowed_regimes():
    positions = pd.Series([1, 1, -1, 1, -1])
    labels = pd.Series(["range", "trend", "range", "trend", "range"])
    gated = gate_positions(positions, labels, allowed=["range"])
    assert list(gated) == [1, 0, -1, 0, -1]


def test_gate_never_increases_exposure():
    prices = _series(500, seed=5)
    sig = SignalConfig(entry_threshold=1.0)
    ungated = run_gated_backtest(
        prices, "sma", "zscore", "trend", ["trend", "range"], sig, window=20
    )  # allow everything -> equivalent to ungated
    gated = run_gated_backtest(
        prices, "sma", "zscore", "trend", ["range"], sig, window=20
    )
    assert gated.summary()["exposure"] <= ungated.summary()["exposure"] + 1e-9
    assert gated.summary()["regime"] == "trend"
    assert gated.summary()["allowed"] == ["range"]


# --- helpers --------------------------------------------------------------

def _final_label(name: str, prices: pd.Series, **kw) -> str:
    clf = create(name, **kw)
    clf.fit(prices)
    return clf.label()
