"""Phase 8 tests: ensemble / adaptive meta-model estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.estimators import create, list_estimators, make_ensemble
from meridian.signals import SignalConfig, run_backtest
from meridian.validation import WalkForwardSpec, validate


def _prices(n=400, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))


def _mean_reverting(n=600, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(100 + x)


# --- registration ---------------------------------------------------------

def test_named_ensembles_registered():
    names = set(list_estimators())
    assert {"ens_mean", "ens_median", "ens_invvar", "ens_skill"} <= names


def test_unknown_combine_and_empty_members_raise():
    with pytest.raises(ValueError):
        make_ensemble(["sma"], combine="bogus")
    with pytest.raises(ValueError):
        make_ensemble([], combine="mean")


def test_window_validation():
    with pytest.raises(ValueError):
        make_ensemble(["sma", "ema"], combine="mean", window=1)


# --- combiner correctness -------------------------------------------------

def test_mean_combine_equals_average_of_members():
    px = _prices()
    ens = make_ensemble(["sma", "ema"], combine="mean", window=20)
    ens.fit(px)
    m1, m2 = create("sma", window=20), create("ema", window=20)
    m1.fit(px)
    m2.fit(px)
    assert ens.predict_mean() == pytest.approx((m1.predict_mean() + m2.predict_mean()) / 2)


def test_median_combine_equals_member_median():
    px = _prices()
    members = ["sma", "ema", "kalman", "lsma", "median"]
    ens = make_ensemble(members, combine="median", window=20)
    ens.fit(px)
    member_means = []
    for name in members:
        m = create(name, window=20)
        m.fit(px)
        member_means.append(m.predict_mean())
    assert ens.predict_mean() == pytest.approx(float(np.median(member_means)))


# --- adaptive weights -----------------------------------------------------

@pytest.mark.parametrize("combine", ["inverse_variance", "skill"])
def test_adaptive_weights_are_valid(combine):
    ens = make_ensemble(["sma", "ema", "kalman", "lsma", "median", "ou"], combine=combine, window=20)
    ens.fit(_mean_reverting())
    w = np.array(ens.state()["weights"])
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0)


def test_inverse_variance_differentiates_members():
    # The minimum-variance blend reliably tilts away from equal weights toward
    # members that track price more tightly.
    ens = make_ensemble(["sma", "ema", "kalman", "lsma", "median", "ou"],
                        combine="inverse_variance", window=20)
    ens.fit(_prices(seed=0))
    w = np.array(ens.state()["weights"])
    assert not np.allclose(w, 1 / 6)


def test_skill_can_differentiate_across_data():
    # The skill blend defaults to equal weights when no member shows positive
    # reversion skill (honest behavior), but *can* differentiate — verify it
    # does on at least one of several series.
    differentiated = False
    for seed in range(6):
        ens = create("ens_skill", window=20)
        ens.fit(_prices(500, seed=seed))
        if not np.allclose(np.array(ens.state()["weights"]), 1 / 6):
            differentiated = True
            break
    assert differentiated


# --- contract / integration ----------------------------------------------

def test_ensemble_is_an_estimator_in_the_pipeline():
    """An ensemble instance flows through the Phase 4 pipeline unchanged."""
    px = _mean_reverting()
    ens = make_ensemble(["sma", "ema", "ou"], combine="median", window=20)
    res = run_backtest(px, ens, "zscore", SignalConfig(entry_threshold=1.0), window=20)
    assert res.summary()["n_trades"] >= 0
    assert np.isfinite(res.equity.iloc[-1])


def test_update_equivalent_to_fit_for_adaptive_ensemble():
    px = _mean_reverting()
    batch = make_ensemble(["sma", "ema", "ou"], combine="skill", window=20)
    batch.fit(px)
    incr = make_ensemble(["sma", "ema", "ou"], combine="skill", window=20)
    incr.fit(px.iloc[:-1])
    incr.update(float(px.iloc[-1]))
    assert incr.predict_mean() == pytest.approx(batch.predict_mean(), rel=1e-9, abs=1e-9)


def test_registered_ensemble_runs_through_validation():
    px = _mean_reverting()
    spec = WalkForwardSpec(mode="anchored", min_train=200, test_span=100, step=100)
    df = validate(
        px, ["ens_median", "ens_skill", "sma"], "zscore",
        SignalConfig(entry_threshold=1.0), spec=spec, window=20,
        n_boot=150, n_mc=150, block=10, seed=0,
    )
    assert set(df["estimator"]) == {"ens_median", "ens_skill", "sma"}
    assert df["significant"].dtype == bool
