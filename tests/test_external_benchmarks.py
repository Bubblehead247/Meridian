"""Tests for the Harvey-Liu-Zhu external t-statistic benchmark (P3)."""

from __future__ import annotations

import numpy as np
import pytest

from meridian.validation.external_benchmarks import (
    HARVEY_LIU_ZHU_T_THRESHOLD,
    classical_t_stat,
    clears_harvey_liu_zhu_bar,
)


def test_classical_t_stat_matches_hand_computation():
    # t = mean/std * sqrt(n), elementary formula — verify directly, no library needed.
    returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008])
    n = len(returns)
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(n)
    assert classical_t_stat(returns) == pytest.approx(expected)


def test_classical_t_stat_nan_below_two_observations():
    assert classical_t_stat([0.01]) != classical_t_stat([0.01])  # NaN


def test_classical_t_stat_nan_for_zero_variance():
    assert classical_t_stat([0.01, 0.01, 0.01]) != classical_t_stat([0.01, 0.01, 0.01])


def test_classical_t_stat_grows_with_sample_size_at_fixed_sharpe():
    # t = SR * sqrt(n): holding the per-period Sharpe fixed, more observations must
    # produce a larger |t| — this is exactly the "large-N t-stat inflation" property
    # Harvey-Liu-Zhu's stricter bar exists to guard against.
    rng = np.random.default_rng(0)
    small = rng.normal(0.001, 0.01, 200)
    big = np.concatenate([small] * 10)  # same distribution, 10x the observations
    assert abs(classical_t_stat(big)) > abs(classical_t_stat(small))


def test_default_threshold_is_three():
    assert HARVEY_LIU_ZHU_T_THRESHOLD == pytest.approx(3.0)


def test_clears_bar_true_for_strong_signal():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.003, 0.01, 2000)  # strong, persistent drift, large n
    assert clears_harvey_liu_zhu_bar(returns) is True


def test_clears_bar_false_for_pure_noise():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, 100)  # small sample, no drift
    assert clears_harvey_liu_zhu_bar(returns) is False


def test_clears_bar_checks_both_directions():
    rng = np.random.default_rng(3)
    returns = rng.normal(-0.003, 0.01, 2000)  # strong negative drift
    assert clears_harvey_liu_zhu_bar(returns) is True  # |t| > 3, not t > 3


def test_clears_bar_respects_custom_threshold():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.001, 0.01, 200)
    t = classical_t_stat(returns)
    assert clears_harvey_liu_zhu_bar(returns, threshold=abs(t) + 1) is False
    assert clears_harvey_liu_zhu_bar(returns, threshold=abs(t) - 0.01) is True
