"""Tests for validation/effective_tests.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.validation.effective_tests import effective_num_tests


def test_perfectly_collinear_trials_give_n_eff_near_one():
    base = pd.Series(np.linspace(0, 1, 300)) + pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    trials = {f"t{i}": base for i in range(5)}  # identical series -> correlation 1.0
    n_eff = effective_num_tests(trials)
    assert n_eff == pytest.approx(1.0, abs=1e-6)


def test_independent_trials_give_n_eff_near_m():
    rng = np.random.default_rng(1)
    trials = {f"t{i}": pd.Series(rng.normal(0, 1, 2000)) for i in range(6)}
    n_eff = effective_num_tests(trials)
    assert n_eff == pytest.approx(6.0, abs=0.5)  # near-independent, small-sample noise


def test_partial_correlation_is_between_one_and_m():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 1, 2000)
    trials = {f"t{i}": pd.Series(common + rng.normal(0, 1, 2000)) for i in range(5)}
    n_eff = effective_num_tests(trials)
    assert 1.0 < n_eff < 5.0


def test_fewer_than_two_trials_returns_raw_count():
    assert effective_num_tests({}) == 0.0
    assert effective_num_tests({"a": pd.Series([1.0, 2.0, 3.0])}) == 1.0


def test_degenerate_zero_variance_trials_fall_back_to_raw_count():
    trials = {"a": pd.Series([1.0] * 50), "b": pd.Series([1.0] * 50)}  # undefined correlation
    assert effective_num_tests(trials) == 2.0
