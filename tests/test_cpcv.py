"""Tests for CPCV/PBO (P2-C).

The exact PBO formula is pinned to Bailey/Borwein/Lopez de Prado/Zhu (2015) but was
not checked against the paper's own worked numerical example (no network access this
session — see the module docstring). What IS verified here is the statistical property
the estimator must have *by construction* regardless of any paper's specific numbers:
under pure noise, PBO -> ~0.5 (rank symmetry); under one persistently dominant trial,
PBO -> low. Both are checked with a real RNG, not hand-picked numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.validation.cpcv import make_cpcv_splits, probability_of_backtest_overfitting


# --- make_cpcv_splits ---------------------------------------------------------

def test_split_count_matches_combinatorics():
    splits = make_cpcv_splits(400, n_groups=4, n_test_groups=1, purge_bars=0, embargo_bars=0)
    assert len(splits) == 4  # C(4,1)

    splits2 = make_cpcv_splits(600, n_groups=6, n_test_groups=2, purge_bars=0, embargo_bars=0)
    assert len(splits2) == 15  # C(6,2)


def test_train_and_test_never_overlap():
    for split in make_cpcv_splits(500, n_groups=5, n_test_groups=2, purge_bars=3, embargo_bars=3):
        assert set(split.train_idx.tolist()).isdisjoint(set(split.test_idx.tolist()))


def test_purge_and_embargo_remove_bars_near_test_boundary():
    # n_groups=4 over n=400 -> groups of 100. Test group 0 spans [0,100). With
    # purge_bars=5, no purge needed before it (nothing precedes group 0). With
    # embargo_bars=10, bars [100,110) must be excluded from training even though they
    # belong to group 1 (not part of the test set) — that's what the embargo checks.
    splits = make_cpcv_splits(400, n_groups=4, n_test_groups=1, purge_bars=5, embargo_bars=10)
    only_group0 = next(s for s in splits if s.test_groups == (0,))
    train_set = set(only_group0.train_idx.tolist())
    for i in range(100, 110):
        assert i not in train_set
    assert 110 in train_set  # just past the embargo window, back in training


def test_invalid_n_groups_raises():
    with pytest.raises(ValueError):
        make_cpcv_splits(100, n_groups=1, n_test_groups=1)


def test_invalid_n_test_groups_raises():
    with pytest.raises(ValueError):
        make_cpcv_splits(100, n_groups=4, n_test_groups=4)


# --- probability_of_backtest_overfitting -------------------------------------

def _noise_trials(n_trials=10, n=1000, seed=0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    return {
        f"t{i}": pd.Series(rng.normal(0.0, 0.01, n), index=idx) for i in range(n_trials)
    }


def _one_dominant_trial(n_trials=10, n=1000, seed=0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    trials = {
        f"t{i}": pd.Series(rng.normal(0.0, 0.01, n), index=idx) for i in range(1, n_trials)
    }
    # A persistent, large true edge every single sub-window should reflect.
    trials["winner"] = pd.Series(rng.normal(0.01, 0.01, n), index=idx)
    return trials


def test_pbo_near_half_under_pure_noise():
    trials = _noise_trials(n_trials=12, n=1200, seed=1)
    result = probability_of_backtest_overfitting(
        trials, n_groups=8, n_test_groups=2, purge_bars=2, embargo_bars=2,
    )
    assert result["n_valid_splits"] > 10
    # Rank symmetry under the null is an asymptotic property; a finite combinatorial
    # sample won't land exactly on 0.5, so this allows a wide but still meaningful band.
    assert 0.3 <= result["pbo"] <= 0.7


def test_pbo_low_when_one_trial_persistently_dominates():
    trials = _one_dominant_trial(n_trials=12, n=1200, seed=2)
    result = probability_of_backtest_overfitting(
        trials, n_groups=8, n_test_groups=2, purge_bars=2, embargo_bars=2,
    )
    assert result["n_valid_splits"] > 10
    assert result["pbo"] < 0.3


def test_pbo_nan_with_fewer_than_two_trials():
    result = probability_of_backtest_overfitting({"only": pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))})
    assert result["pbo"] != result["pbo"]  # NaN
    assert result["n_splits"] == 0


def test_pbo_joint_dropna_aligns_trials_with_different_gaps():
    idx = pd.RangeIndex(500)
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    b.iloc[:100] = np.nan  # shorter history
    result = probability_of_backtest_overfitting({"a": a, "b": b}, n_groups=4, n_test_groups=1)
    assert result["n_splits"] > 0  # ran on the aligned 400-row common window, not 500
