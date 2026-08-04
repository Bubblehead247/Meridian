"""Tests for validation/deflated_sharpe.py — formulas verified against external
sources (see the module docstring), so these pin exact numeric values, not just
shape/sign checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from meridian.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio_stdev,
)


def _normal_returns(n=2000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n)


# --- sharpe_ratio_stdev -----------------------------------------------------

def test_sharpe_ratio_stdev_matches_manual_formula_with_raw_kurtosis():
    """Pins the exact formula, including raw (not excess/Fisher) kurtosis."""
    arr = _normal_returns()
    sr = arr.mean() / arr.std(ddof=1)
    skew = sps.skew(arr)
    kurt_raw = sps.kurtosis(arr, fisher=False)  # 3.0 = normal
    n = arr.size
    expected = np.sqrt((1 + 0.5 * sr**2 - skew * sr + ((kurt_raw - 3) / 4) * sr**2) / (n - 1))
    assert sharpe_ratio_stdev(arr) == pytest.approx(expected, rel=1e-9)


def test_sharpe_ratio_stdev_would_differ_under_excess_kurtosis_convention():
    """Regression guard for the off-by-3 bug: excess kurtosis (fisher=True) would
    give a materially different answer for a fat-tailed sample; confirm the
    function does NOT match that (wrong) convention."""
    rng = np.random.default_rng(1)
    # Heavy-tailed sample (Student-t, low df) so excess vs raw kurtosis diverge a lot.
    arr = rng.standard_t(df=3, size=3000) * 0.01
    sr = arr.mean() / arr.std(ddof=1)
    skew = sps.skew(arr)
    kurt_excess = sps.kurtosis(arr, fisher=True)  # 0.0 = normal (the wrong convention)
    n = arr.size
    wrong = np.sqrt((1 + 0.5 * sr**2 - skew * sr + ((kurt_excess - 3) / 4) * sr**2) / (n - 1))
    actual = sharpe_ratio_stdev(arr)
    assert actual != pytest.approx(wrong, rel=1e-6)


def test_sharpe_ratio_stdev_too_few_observations_is_nan():
    assert sharpe_ratio_stdev([0.01, -0.01, 0.02]) != sharpe_ratio_stdev([0.01, -0.01, 0.02])


# --- probabilistic_sharpe_ratio ---------------------------------------------

def test_psr_at_benchmark_equals_the_estimate_is_50_50():
    assert probabilistic_sharpe_ratio(sr=0.1, sr_benchmark=0.1, sr_std=0.05) == pytest.approx(0.5)


def test_psr_above_benchmark_exceeds_half():
    assert probabilistic_sharpe_ratio(sr=0.3, sr_benchmark=0.1, sr_std=0.05) > 0.5


def test_psr_nan_std_is_nan():
    assert probabilistic_sharpe_ratio(0.1, 0.0, float("nan")) != probabilistic_sharpe_ratio(
        0.1, 0.0, float("nan")
    )


# --- expected_max_sharpe -----------------------------------------------------

def test_expected_max_sharpe_matches_published_reference_value():
    """Bailey & Lopez de Prado's own worked example: N=1000 trials, V[SR]=1,
    E[SR]=0 -> expected max SR ~= 3.26."""
    rng = np.random.default_rng(2)
    trial_sharpes = rng.normal(0.0, 1.0, 100_000)  # large sample so std(ddof=1) ~= 1.0
    result = expected_max_sharpe(trial_sharpes, n_trials=1000)
    assert result == pytest.approx(3.255, abs=0.05)


def test_expected_max_sharpe_increases_with_more_trials():
    rng = np.random.default_rng(3)
    trial_sharpes = rng.normal(0.0, 1.0, 5000)
    small = expected_max_sharpe(trial_sharpes, n_trials=10)
    large = expected_max_sharpe(trial_sharpes, n_trials=1000)
    assert large > small


def test_expected_max_sharpe_degenerate_is_nan():
    assert expected_max_sharpe([0.5], n_trials=1) != expected_max_sharpe([0.5], n_trials=1)
    assert expected_max_sharpe([0.5, 0.5, 0.5], n_trials=3) != expected_max_sharpe(
        [0.5, 0.5, 0.5], n_trials=3
    )  # zero variance -> nan


# --- deflated_sharpe_ratio (integration of the three pieces) ---------------

def test_deflated_sharpe_ratio_shape_and_fields():
    arr = _normal_returns(seed=5)
    trial_sharpes = np.random.default_rng(6).normal(0.0, 0.5, 50)
    out = deflated_sharpe_ratio(arr, trial_sharpes=trial_sharpes, n_trials=50)
    assert set(out) == {"sr", "sr_std", "expected_max_sr", "dsr_pvalue"}
    assert 0.0 <= out["dsr_pvalue"] <= 1.0


def test_deflated_sharpe_ratio_high_sharpe_beats_low_trial_count_more_easily():
    """A high, real Sharpe should clear a small trial count's expected-max bar
    more easily than a large one (more trials -> higher bar -> lower DSR)."""
    arr = np.random.default_rng(7).normal(0.002, 0.01, 3000)  # a real, positive-mean series
    trial_sharpes = np.random.default_rng(8).normal(0.0, 0.3, 500)
    few = deflated_sharpe_ratio(arr, trial_sharpes=trial_sharpes, n_trials=2)
    many = deflated_sharpe_ratio(arr, trial_sharpes=trial_sharpes, n_trials=500)
    assert few["dsr_pvalue"] >= many["dsr_pvalue"]
