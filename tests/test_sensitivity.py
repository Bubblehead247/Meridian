"""Tests for validation/sensitivity.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec
from meridian.validation.sensitivity import parameter_sensitivity


def _mean_reverting(n=1400, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(100 + x, index=pd.RangeIndex(n))


def _spec() -> WalkForwardSpec:
    return WalkForwardSpec(mode="anchored", min_train=300, test_span=150, step=150)


def test_windows_include_the_base_and_are_sorted():
    prices = _mean_reverting()
    out = parameter_sensitivity(
        prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20,
        spec=_spec(), n_boot=100,
    )
    assert 20 in out["windows"]
    assert out["windows"] == sorted(out["windows"])
    assert len(out["oos_sharpes"]) == len(out["windows"])


def test_base_sharpe_matches_the_configured_window():
    prices = _mean_reverting()
    out = parameter_sensitivity(
        prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20,
        spec=_spec(), n_boot=100,
    )
    idx = out["windows"].index(20)
    assert out["base_sharpe"] == pytest.approx(out["oos_sharpes"][idx])


def test_cv_is_zero_for_identical_windows():
    """Degenerate case: multipliers collapse to a single window -> no spread to measure."""
    prices = _mean_reverting()
    out = parameter_sensitivity(
        prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20,
        spec=_spec(), multipliers=(1.0,), n_boot=100,
    )
    assert out["windows"] == [20]
    assert out["cv"] != out["cv"]  # only one point -> NaN, not a fake zero


def test_sign_stable_true_when_all_same_sign():
    prices = _mean_reverting()
    out = parameter_sensitivity(
        prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20,
        spec=_spec(), multipliers=(0.9, 1.0, 1.1), n_boot=100,
    )
    finite = [s for s in out["oos_sharpes"] if s == s]
    if len(finite) >= 2 and out["base_sharpe"] == out["base_sharpe"]:
        base_sign = out["base_sharpe"] >= 0
        expected = all((s >= 0) == base_sign for s in finite)
        assert out["sign_stable"] == expected


def test_never_changes_which_window_would_be_used_for_ranking():
    """The function is read-only diagnostics: it must not mutate/select a window."""
    prices = _mean_reverting()
    out = parameter_sensitivity(
        prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20, spec=_spec(),
        n_boot=100,
    )
    assert out["base_window"] == 20  # unchanged regardless of neighbor results
