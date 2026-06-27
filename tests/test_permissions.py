"""Tests for the family regime-permission matrix (PLAN.md §5)."""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.families import (
    PERMISSIONS,
    cash_reserve_boost,
    gate_positions,
    is_permitted,
    permission_mask,
)
from meridian.regimes.labeler import RegimeLabel


def L(trend="neutral", vol="normal", breadth="neutral") -> RegimeLabel:
    return RegimeLabel(trend=trend, volatility=vol, breadth=breadth)


# --- per-family rules -----------------------------------------------------

def test_trend_following_and_momentum_need_bull_and_not_extreme():
    for fam in ("trend_following", "momentum"):
        assert is_permitted(fam, L(trend="bull", vol="low"))
        assert not is_permitted(fam, L(trend="bull", vol="extreme"))   # VIX modifier
        assert not is_permitted(fam, L(trend="neutral"))
        assert not is_permitted(fam, L(trend="bear"))


def test_breakouts_need_bull_and_breadth_expansion_or_neutral():
    assert is_permitted("breakouts", L(trend="bull", breadth="expansion"))
    assert is_permitted("breakouts", L(trend="bull", breadth="neutral"))
    assert not is_permitted("breakouts", L(trend="bull", breadth="contraction"))
    assert not is_permitted("breakouts", L(trend="neutral", breadth="expansion"))


def test_pullback_needs_bull_and_expansion_or_neutral_breadth():
    assert is_permitted("pullback_continuation", L(trend="bull", breadth="neutral"))
    assert is_permitted("pullback_continuation", L(trend="bull", breadth="expansion"))
    assert not is_permitted("pullback_continuation", L(trend="bull", breadth="contraction"))
    assert not is_permitted("pullback_continuation", L(trend="neutral", breadth="neutral"))


def test_mean_reversion_neutral_or_bear_trend_and_neutral_or_contraction_breadth():
    assert is_permitted("mean_reversion", L(trend="neutral", breadth="contraction"))
    assert is_permitted("mean_reversion", L(trend="bear", breadth="neutral"))
    assert not is_permitted("mean_reversion", L(trend="bull", breadth="contraction"))
    assert not is_permitted("mean_reversion", L(trend="bear", breadth="expansion"))


def test_always_families_trade_in_every_regime_including_unknown():
    for fam in ("sector_rotation", "long_term_etf", "cash_reserve"):
        for tr in ("bull", "neutral", "bear", "unknown"):
            assert is_permitted(fam, L(trend=tr, vol="extreme", breadth="contraction"))


def test_deferred_families_never_trade():
    for fam in ("event_driven", "volatility"):
        assert not is_permitted(fam, L(trend="bull", vol="low", breadth="expansion"))


def test_unknown_regime_restricts_gated_family_but_not_always():
    assert not is_permitted("trend_following", L(trend="unknown", vol="unknown"))
    assert is_permitted("long_term_etf", L(trend="unknown", vol="unknown"))


def test_unknown_family_raises():
    with pytest.raises(KeyError, match="Unknown family"):
        is_permitted("nope", L())


def test_permission_matrix_covers_all_families():
    assert set(PERMISSIONS) == {
        "trend_following", "momentum", "breakouts", "pullback_continuation",
        "mean_reversion", "sector_rotation", "long_term_etf", "cash_reserve",
        "event_driven", "volatility",
    }


# --- frame mask + gating --------------------------------------------------

def _frame(rows: list[tuple[str, str, str]], dates) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["trend", "volatility", "breadth"], index=dates)


def test_permission_mask_vectorizes_the_rule():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = _frame(
        [("bull", "low", "neutral"), ("neutral", "low", "neutral"), ("bull", "extreme", "neutral")],
        idx,
    )
    mask = permission_mask("trend_following", frame)
    assert mask.tolist() == [True, False, False]
    assert mask.index.equals(idx)


def test_gate_positions_flattens_when_restricted():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    positions = pd.Series([1, -1, 1, -1], index=idx)
    frame = _frame(
        [("bull", "low", "neutral"),     # permit
         ("neutral", "low", "neutral"),  # restrict (not bull)
         ("bull", "extreme", "neutral"), # restrict (extreme vol)
         ("bull", "low", "neutral")],    # permit
        idx,
    )
    gated = gate_positions(positions, "trend_following", frame)
    assert gated.tolist() == [1, 0, 0, -1]


def test_gate_positions_missing_regime_date_is_unknown_restricted():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    positions = pd.Series([1, 1, 1], index=idx)
    # regime frame only covers the first date -> others reindex to unknown -> restricted
    frame = _frame([("bull", "low", "neutral")], idx[:1])
    gated = gate_positions(positions, "trend_following", frame)
    assert gated.tolist() == [1, 0, 0]


def test_integrates_with_labeler_regime_frame():
    """A real regime_frame (from the labeler) feeds the permission mask end to end."""
    import numpy as np

    from meridian.regimes.labeler import regime_frame

    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    close = pd.Series(np.linspace(100, 200, 300), index=idx)        # clean uptrend -> bull
    bars = pd.DataFrame({"high": close + 0.5, "low": close - 0.5, "close": close})
    vix = pd.Series(12.0, index=idx)                                 # low vol
    breadth = pd.Series(70.0, index=idx)                            # expansion
    frame = regime_frame(bars, vix, breadth)

    # bull + low + expansion: trend_following permitted, mean_reversion not.
    assert permission_mask("trend_following", frame).iloc[-1]
    assert not permission_mask("mean_reversion", frame).iloc[-1]


def test_cash_reserve_boost_on_bear_extreme_only():
    assert cash_reserve_boost(L(trend="bear", vol="extreme"))
    assert not cash_reserve_boost(L(trend="bear", vol="low"))
    assert not cash_reserve_boost(L(trend="bull", vol="extreme"))
